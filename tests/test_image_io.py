"""Tests for image loading/export and metadata round-trips."""
from __future__ import annotations

import numpy as np
import pytest
import tifffile
from PIL import Image

from object_remover.errors import ImageLoadError
from object_remover.image_io import (
    WorkingImage,
    export_jpg,
    export_tiff,
    load_image,
    parse_exif_blob,
)


def test_load_tiff_16bit_roundtrip(tiff_path, rgb16):
    img = load_image(tiff_path)
    assert img.pixels.dtype == np.uint16
    assert img.pixels.shape == rgb16.shape
    np.testing.assert_array_equal(img.pixels, rgb16)
    assert img.source_format == "TIFF"
    assert img.alpha is None


def test_load_tiff_8bit_upgrades_to_16(tmp_path, gradient8):
    path = tmp_path / "g.tif"
    tifffile.imwrite(path, gradient8, photometric="rgb")
    img = load_image(path)
    np.testing.assert_array_equal(img.pixels, gradient8.astype(np.uint16) * 257)


def test_load_tiff_rgba_preserves_alpha(tmp_path, rgb16):
    rgba = np.dstack([rgb16, (rgb16[:, :, 0] // 2).astype(np.uint16)])
    path = tmp_path / "a.tif"
    tifffile.imwrite(path, rgba, photometric="rgb", extrasamples=2)
    img = load_image(path)
    assert img.alpha is not None
    np.testing.assert_array_equal(img.alpha, rgba[:, :, 3])


def test_load_jpg_upgrades_and_reads_exif(tmp_path):
    exif = Image.Exif()
    exif[271] = "TestMake"
    exif[272] = "TestModel"
    path = tmp_path / "e.jpg"
    Image.fromarray(np.zeros((32, 32, 3), np.uint8), "RGB").save(
        path, quality=95, exif=exif.tobytes()
    )
    img = load_image(path)
    assert img.pixels.dtype == np.uint16
    assert img.exif is not None
    assert img.exif_tags.get(271) == "TestMake"
    assert img.exif_tags.get(272) == "TestModel"


def test_export_jpg_preserves_exif_and_embeds_srgb(tmp_path):
    exif = Image.Exif()
    exif[271] = "TestMake"
    src = tmp_path / "src.jpg"
    Image.fromarray(np.full((16, 16, 3), 128, np.uint8), "RGB").save(
        src, quality=95, exif=exif.tobytes()
    )
    img = load_image(src)
    out = tmp_path / "out.jpg"
    export_jpg(img, out, quality=90)
    with Image.open(out) as re:
        assert re.info.get("exif")
        assert parse_exif_blob(re.info["exif"]).get(271) == "TestMake"
        assert re.info.get("icc_profile")  # sRGB embedded


def test_export_tiff_16bit_exact_roundtrip(tmp_path, rgb16):
    img = WorkingImage(pixels=rgb16.copy(), icc=b"\x00" * 128)
    out = tmp_path / "out.tif"
    export_tiff(img, out)
    re = load_image(out)
    np.testing.assert_array_equal(re.pixels, rgb16)
    assert re.icc is not None and len(re.icc) == 128


def test_export_tiff_alpha_roundtrip(tmp_path, rgb16):
    alpha = (rgb16[:, :, 0] // 3).astype(np.uint16)
    img = WorkingImage(pixels=rgb16.copy(), alpha=alpha)
    out = tmp_path / "out.tif"
    export_tiff(img, out)
    re = load_image(out)
    assert re.alpha is not None
    np.testing.assert_array_equal(re.alpha, alpha)


def test_export_tiff_exif_tags(tmp_path):
    img = WorkingImage(
        pixels=np.zeros((8, 8, 3), np.uint16),
        exif_tags={271: "TestMake", 272: "TestModel", 34855: 200},
    )
    out = tmp_path / "m.tif"
    export_tiff(img, out)
    with tifffile.TiffFile(out) as tf:
        make = tf.pages[0].tags.get(271)
        assert make is not None and make.value.rstrip("\x00") == "TestMake"
        iso = tf.pages[0].tags.get(34855)
        assert iso is not None and int(iso.value) == 200


def test_unsupported_format_raises(tmp_path):
    path = tmp_path / "x.png"
    Image.fromarray(np.zeros((8, 8, 3), np.uint8)).save(path)
    with pytest.raises(ImageLoadError):
        load_image(path)


def test_missing_file_raises(tmp_path):
    with pytest.raises(ImageLoadError):
        load_image(tmp_path / "nope.jpg")


def test_grayscale_jpg_converts_to_rgb(tmp_path):
    path = tmp_path / "g.jpg"
    Image.fromarray(np.full((16, 16), 100, np.uint8), "L").save(path)
    img = load_image(path)
    assert img.pixels.shape == (16, 16, 3)
    np.testing.assert_array_equal(img.pixels[:, :, 0], img.pixels[:, :, 1])
