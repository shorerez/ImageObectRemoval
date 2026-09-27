"""Image loading and export (TIFF / JPG) with metadata preservation.

Color policy (REQUIREMENTS.md §7): pixels keep their source colorimetry; the
pipeline only manages bit depth (8 -> 16 on load). ICC profiles are preserved
exactly. EXIF is preserved exactly for JPG (raw blob passthrough); for TIFF it
is preserved as individual TIFF tags (best-effort, see §7). JPG export converts
to sRGB when the source carries a non-sRGB ICC profile.
"""
from __future__ import annotations

import io
import logging
import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image, ImageCms

from .errors import ImageExportError, ImageLoadError

log = logging.getLogger(__name__)

# TIFF tag numbers
TAG_ICC_PROFILE = 34675
TAG_EXIF_IFD = 34665

# Common photography tags we round-trip into TIFF exports.
# (Tag 305 Software is intentionally excluded: export always writes its own.)
_TIFF_EXPORT_TAGS = {
    271,   # Make
    272,   # Model
    274,   # Orientation
    306,   # DateTime
    315,   # Artist
    33432, # Copyright
    36867, # DateTimeOriginal
    37377, # ShutterSpeedValue (SRATIONAL -> handled as rational pair)
    37378, # ApertureValue
    37380, # ExposureBiasValue
    37381, # MaxApertureValue
    37383, # MeteringMode
    37385, # Flash
    37386, # FocalLength
    34855, # ISO
    41987, # FocalLengthIn35mmFilm
    42036, # LensModel
}

_FORMAT_TIFF = "TIFF"
_FORMAT_JPEG = "JPEG"

# EXIF/TIFF value type codes used in exif_tags values
#   'A' str | 'S' short int | 'L' long int | 'R' rational flat (num, den, ...)
#   'U' raw bytes (UNDEFINED)
MetaValue = object


@dataclass
class WorkingImage:
    """The single internal image representation.

    pixels: uint16 (H, W, 3) RGB.
    alpha:  uint16 (H, W) or None — preserved untouched through editing.
    icc:    raw ICC profile bytes or None.
    exif:   raw EXIF blob (JPG sources) — written back verbatim to JPG export.
    exif_tags: parsed EXIF/TIFF entries used for TIFF export / rebuilt EXIF.
    """

    pixels: np.ndarray
    alpha: np.ndarray | None = None
    icc: bytes | None = None
    exif: bytes | None = None
    exif_tags: dict[int, MetaValue] = field(default_factory=dict)
    source_path: str | None = None
    source_format: str | None = None

    @property
    def height(self) -> int:
        return int(self.pixels.shape[0])

    @property
    def width(self) -> int:
        return int(self.pixels.shape[1])

    @property
    def size(self) -> tuple[int, int]:
        """(width, height)"""
        return (self.width, self.height)

    def copy_pixels(self) -> np.ndarray:
        return self.pixels.copy()


# --- EXIF parsing / rebuilding ---------------------------------------------

_EXIF_TYPES = {1: ("B", "B"), 2: ("A", "s"), 3: ("S", "H"), 4: ("L", "I"),
               5: ("R", "I"), 7: ("U", "B"), 9: ("L", "i"), 10: ("R", "i")}
_EXIF_STRUCT = {1: "B", 2: "s", 3: "H", 4: "I", 5: None, 7: "B", 9: "i", 10: None}


def parse_exif_blob(data: bytes) -> dict[int, MetaValue]:
    """Parse an EXIF blob (TIFF-header + IFD0 + Exif sub-IFD) into a tag dict."""
    tags: dict[int, MetaValue] = {}
    try:
        if len(data) < 8:
            return tags
        order = data[:2]
        if order == b"II":
            end = "<"
        elif order == b"MM":
            end = ">"
        else:  # JPEG APP1 style with "Exif\0\0" prefix
            if data[:6] == b"Exif\x00\x00":
                return parse_exif_blob(data[6:])
            return tags

        def read_ifd(offset: int, out: dict) -> None:
            if offset <= 0 or offset + 2 > len(data):
                return
            (count,) = struct.unpack_from(end + "H", data, offset)
            pos = offset + 2
            for _ in range(count):
                if pos + 12 > len(data):
                    return
                tag, typ, cnt = struct.unpack_from(end + "HHI", data, pos)
                pos += 12
                if typ not in _EXIF_TYPES:
                    continue
                code, _ = _EXIF_TYPES[typ]
                size = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 7: 1, 9: 4, 10: 8}[typ]
                nbytes = size * cnt
                if nbytes <= 4:
                    raw = data[pos - 4 : pos]
                else:
                    (off,) = struct.unpack_from(end + "I", data, pos - 4)
                    if off + nbytes > len(data):
                        continue
                    raw = data[off : off + nbytes]
                if typ == 2:
                    out[tag] = raw.split(b"\x00", 1)[0].decode("ascii", "replace")
                elif typ in (5, 10):
                    fmt = end + ("II" if typ == 5 else "ii") * cnt
                    flat = struct.unpack(fmt, raw[: 8 * cnt])
                    out[tag] = ("R",) + flat
                elif typ == 7:
                    out[tag] = ("U", raw)
                else:
                    fmt = end + {1: "B", 3: "H", 4: "I", 9: "i"}[typ] * cnt
                    vals = struct.unpack(fmt, raw[: nbytes])
                    out[tag] = vals[0] if cnt == 1 else list(vals)
                if tag == TAG_EXIF_IFD and typ == 4 and cnt == 1:
                    (sub,) = struct.unpack_from(end + "I", raw, 0)
                    read_ifd(sub, out)

        read_ifd(struct.unpack_from(end + "I", data, 4)[0], tags)
    except Exception as exc:  # pragma: no cover - defensive
        log.info("EXIF parse skipped (%s)", exc)
    return tags


def build_exif_blob(tags: dict[int, MetaValue]) -> bytes | None:
    """Rebuild an EXIF blob from parsed tags via Pillow's Exif writer."""
    try:
        exif = Image.Exif()
        for tag, value in tags.items():
            if isinstance(value, tuple) and value and value[0] in ("R", "U"):
                if value[0] == "U":
                    exif[tag] = value[1]
                else:
                    flat = value[1:]
                    if len(flat) == 2:
                        exif[tag] = (flat[0], flat[1])
                    else:
                        exif[tag] = tuple(
                            (flat[i], flat[i + 1]) for i in range(0, len(flat) - 1, 2)
                        )
            else:
                exif[tag] = value
        blob = exif.tobytes()
        return blob or None
    except Exception as exc:  # pragma: no cover - defensive
        log.info("EXIF rebuild skipped (%s)", exc)
        return None


def _tiff_tag_to_meta(tag) -> tuple[int, MetaValue] | None:
    """Convert a tifffile-parsed tag into our meta representation."""
    code = int(tag.code)
    if code not in _TIFF_EXPORT_TAGS:
        return None
    try:
        value = tag.value
        dtype = tag.dtype
        name = getattr(dtype, "name", str(dtype))
        if name == "ASCII" or isinstance(value, str):
            return code, str(value)
        if name in ("BYTE", "UNDEFINED"):
            raw = bytes(value) if not isinstance(value, (bytes, bytearray)) else bytes(value)
            return code, ("U", raw)
        if name in ("SHORT", "SSHORT"):
            if isinstance(value, (tuple, list)):
                return code, [int(v) for v in value]
            return code, int(value)
        if name in ("LONG", "SLONG", "IFD"):
            if isinstance(value, (tuple, list)):
                return code, [int(v) for v in value]
            return code, int(value)
        if name in ("RATIONAL", "SRATIONAL"):
            pairs = value if isinstance(value, (tuple, list)) else [(value[0], value[1])]
            flat: list[int] = []
            for p in pairs:
                if isinstance(p, (tuple, list)) and len(p) == 2:
                    flat.extend([int(p[0]), int(p[1])])
                else:
                    flat.append(int(p))
            return code, ("R", *flat)
    except Exception:  # pragma: no cover - defensive
        return None
    return None


# --- load -------------------------------------------------------------------

def load_image(path: str | Path) -> WorkingImage:
    """Load a TIFF or JPG into a 16-bit RGB working buffer."""
    path = Path(path)
    if not path.is_file():
        raise ImageLoadError(f"File not found: {path}")
    suffix = path.suffix.lower()
    if suffix in {".tif", ".tiff"}:
        return _load_tiff(path)
    if suffix in {".jpg", ".jpeg"}:
        return _load_jpeg(path)
    raise ImageLoadError(
        f"Unsupported file type '{path.suffix}'. ObjectRemover opens TIFF and JPG files."
    )


def _to_uint16_rgb(arr: np.ndarray, path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    """Normalize an HxW[xC] array to uint16 RGB + optional uint16 alpha."""
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)  # grayscale -> RGB
    if arr.ndim != 3 or arr.shape[2] not in (3, 4):
        raise ImageLoadError(
            f"Unsupported channel layout in {path.name} "
            f"(shape {arr.shape}). Only RGB/RGBA images are supported."
        )
    if arr.dtype == np.uint8:
        arr = arr.astype(np.uint16) * 257
    elif arr.dtype == np.uint16:
        arr = arr.copy()
    elif arr.dtype in (np.float32, np.float64):
        raise ImageLoadError(
            f"{path.name}: floating-point TIFF is not supported in v1. "
            "Please save as 8/16-bit integer TIFF or JPG."
        )
    else:
        raise ImageLoadError(f"{path.name}: unsupported sample format '{arr.dtype}'.")
    rgb = np.ascontiguousarray(arr[:, :, :3])
    alpha = np.ascontiguousarray(arr[:, :, 3]) if arr.shape[2] == 4 else None
    return rgb, alpha


def _tag_bytes(page, code: int) -> bytes | None:
    try:
        tag = page.tags.get(code)
    except Exception:  # pragma: no cover - defensive
        return None
    if tag is None:
        return None
    try:
        value = tag.value
        if isinstance(value, (bytes, bytearray)):
            return bytes(value)
        if isinstance(value, memoryview):
            return value.tobytes()
        if isinstance(value, np.ndarray):
            return value.tobytes()
    except Exception:  # pragma: no cover - defensive
        return None
    return None


def _load_tiff(path: Path) -> WorkingImage:
    try:
        with tifffile.TiffFile(path) as tf:
            if not tf.pages:
                raise ImageLoadError(f"{path.name}: no pages found in TIFF.")
            page = tf.pages[0]
            rgb, alpha = _to_uint16_rgb(np.asarray(page.asarray()), path)
            icc = _tag_bytes(page, TAG_ICC_PROFILE)
            exif_tags: dict[int, MetaValue] = {}
            for tag in page.tags:
                meta = _tiff_tag_to_meta(tag)
                if meta is not None:
                    exif_tags[meta[0]] = meta[1]
    except ImageLoadError:
        raise
    except Exception as exc:
        raise ImageLoadError(f"Could not read TIFF '{path.name}': {exc}") from exc
    return WorkingImage(
        pixels=rgb,
        alpha=alpha,
        icc=icc,
        exif=None,
        exif_tags=exif_tags,
        source_path=str(path),
        source_format=_FORMAT_TIFF,
    )


def _load_jpeg(path: Path) -> WorkingImage:
    try:
        with Image.open(path) as img:
            img.load()
            if img.mode not in ("RGB", "RGBA"):
                img = img.convert("RGB")
            arr = np.asarray(img)
            exif = img.info.get("exif") or None
            if exif is not None:
                exif = bytes(exif)
            icc = img.info.get("icc_profile") or None
            if icc is not None:
                icc = bytes(icc)
    except Exception as exc:
        raise ImageLoadError(f"Could not read JPG '{path.name}': {exc}") from exc
    rgb, alpha = _to_uint16_rgb(arr, path)
    return WorkingImage(
        pixels=rgb,
        alpha=alpha,
        icc=icc,
        exif=exif,
        exif_tags=parse_exif_blob(exif) if exif else {},
        source_path=str(path),
        source_format=_FORMAT_JPEG,
    )


# --- export -----------------------------------------------------------------

def _u16_to_u8(pixels16: np.ndarray) -> np.ndarray:
    return np.rint(pixels16.astype(np.float32) / 257.0).astype(np.uint8)


def _srgb_profile_bytes() -> bytes:
    return ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()


def _profile_is_srgb(icc: bytes) -> bool:
    try:
        desc = ImageCms.getProfileDescription(ImageCms.ImageCmsProfile(io.BytesIO(icc)))
        return "sRGB" in desc
    except Exception:
        return False


def export_jpg(image: WorkingImage, path: str | Path, quality: int = 95) -> None:
    """Export as 8-bit sRGB JPG, preserving EXIF and embedding sRGB ICC.

    If the source carries a non-sRGB ICC profile the pixels are converted to
    sRGB so the JPG is display-correct everywhere.
    """
    path = Path(path)
    quality = int(min(100, max(1, quality)))
    rgb8 = _u16_to_u8(image.pixels)
    img = Image.fromarray(rgb8, mode="RGB")

    if image.icc:
        try:
            if not _profile_is_srgb(image.icc):
                src = ImageCms.ImageCmsProfile(io.BytesIO(image.icc))
                img = ImageCms.profileToProfile(
                    img, src, ImageCms.createProfile("sRGB"), outputMode="RGB"
                )
        except Exception as exc:
            log.warning("ICC conversion to sRGB failed (%s); exporting raw pixels.", exc)

    kwargs: dict = {
        "format": "JPEG",
        "quality": quality,
        "subsampling": 0 if quality >= 90 else 2,
        "icc_profile": _srgb_profile_bytes(),
    }
    exif_blob = image.exif or build_exif_blob(image.exif_tags)
    if exif_blob:
        kwargs["exif"] = exif_blob
    try:
        img.save(path, **kwargs)
    except Exception as exc:
        raise ImageExportError(f"Could not write JPG '{path.name}': {exc}") from exc


def _extratags_from_meta(exif_tags: dict[int, MetaValue]) -> list[tuple]:
    """Build tifffile extratags entries from parsed metadata."""
    entries: list[tuple] = []
    for code, value in exif_tags.items():
        try:
            if isinstance(value, str):
                entries.append((code, "s", len(value) + 1, value, True))
            elif isinstance(value, int):
                dtype = "H" if code in (274, 34855, 37383, 37385, 41987) else "I"
                entries.append((code, dtype, 1, int(value), True))
            elif isinstance(value, list):
                if all(isinstance(v, int) for v in value):
                    entries.append((code, "I", len(value), [int(v) for v in value], True))
            elif isinstance(value, tuple) and value:
                if value[0] == "U":
                    raw = value[1]
                    entries.append((code, "B", len(raw), raw, True))
                elif value[0] == "R":
                    flat = [int(v) for v in value[1:]]
                    entries.append((code, "I", len(flat), flat, True))
            elif isinstance(value, tuple) and all(isinstance(v, int) for v in value):
                entries.append((code, "I", len(value), list(value), True))
        except Exception as exc:  # pragma: no cover - defensive
            log.info("EXIF tag %s not written to TIFF (%s)", code, exc)
    return entries


def export_tiff(image: WorkingImage, path: str | Path) -> None:
    """Export as 16-bit deflate TIFF.

    Bit depth and colorimetry are preserved exactly (perfect round-trip for
    16-bit TIFF sources). ICC is re-embedded natively; EXIF tags are written
    as individual TIFF tags (best-effort, REQUIREMENTS.md §7).
    """
    path = Path(path)
    arr = image.pixels
    extrasamples = None
    if image.alpha is not None:
        arr = np.dstack([arr, image.alpha])
        extrasamples = 2  # unassociated alpha
    try:
        tifffile.imwrite(
            path,
            arr,
            photometric="rgb",
            compression="deflate",
            predictor=True,
            software="ObjectRemover",
            iccprofile=image.icc,
            extrasamples=extrasamples,
            extratags=_extratags_from_meta(image.exif_tags),
        )
    except Exception as exc:
        raise ImageExportError(f"Could not write TIFF '{path.name}': {exc}") from exc
