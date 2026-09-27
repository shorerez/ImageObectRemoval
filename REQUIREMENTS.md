# ObjectRemover — Requirements (approved spec)

Local Windows tool for removing user-painted objects from large images, filling the
holes from surrounding context with an AI inpainting model.

## 1. Platform & performance

- **Windows desktop application, fully local.** The only network event is a
  one-time AI-model download on first launch; all processing is local.
- **Target hardware:** NVIDIA RTX 3070 Ti Laptop (8 GB VRAM), 32 GB RAM,
  Intel i9-12900H. The app must **probe free VRAM at startup** and auto-tune;
  it must degrade gracefully when other applications occupy the GPU.
- **Responsiveness target:** typical removal (selection up to ~2K px across)
  **≤ ~5 s**; very large selections ~10–30 s, always with progress + cancel.

## 2. Inputs & pipeline

- **TIFF and JPG only.**
  - TIFF: 8/16-bit RGB/RGBA, uncompressed/LZW/deflate, BigTIFF read supported.
  - JPG: standard 8-bit.
- Anything else → clear error message, never a crash. Grayscale inputs are
  converted to RGB on load (lossless trivial case).
- Single **16-bit internal working buffer**. Embedded ICC profiles and EXIF are
  captured on load and preserved on export; alpha channels are preserved
  untouched. Multi-page TIFF → first page only.

## 3. Editing model

- **Two masks** painted on the image:
  - **Removal mask** (red overlay) — area to erase and regenerate.
  - **Protect / ignore mask** (green overlay) — constraint on fill generation.
- Both masks support **paint and erase** brushes (size / hardness / feather),
  are freely editable before committing, with **full stroke-level undo/redo**.
- **Remove** = context-based AI inpainting (LaMa-class, quality-first): the fill
  is synthesized from the surrounding area. Protected pixels are **never
  overwritten and never used as fill source**.
- **Overlap rule:** if the two masks overlap when Remove is clicked, the
  overlapping region is highlighted and the user chooses each time:
  **Protect wins / Removal wins / Cancel and fix by hand**. No silent behavior.
- After a successful Remove: masks clear, the step joins the undo history.
  Unlimited removal passes per image; unlimited session history.

## 4. Export

- Export the current state to **any local path** at any time:
  - **JPG** — sRGB, quality slider, EXIF + ICC preserved.
  - **16-bit TIFF** — deflate-compressed, ICC preserved, EXIF best-effort.
- Export never modifies the source image (unless the user explicitly chooses
  the source path in the save dialog).

## 5. Delivery

- Small installer; AI weights (~200–300 MB) **downloaded once on first launch**
  into `%LOCALAPPDATA%\ObjectRemover\models`, checksum-verified
  (trust-on-first-use hash recorded, pinned hash enforced when provided).
  Fully offline afterwards. Graceful retry/error if the download fails; the app
  remains usable via the classical fallback engine until the model is present.

## 6. Scope decisions (frozen for v1)

| Topic | Decision |
|---|---|
| Input formats | TIFF + JPG only (CR3 and DNG dropped — no RAW/demosaic module) |
| PNG | Not supported anywhere |
| Sessions | Export-only; no save/load of edit sessions (post-v1 candidate) |
| Stack | Python 3.11+, PySide6, Pillow + tifffile + OpenCV, ONNX Runtime |
| Inference | LaMa ONNX (Carve/LaMa-ONNX, Apache-2.0 weights), fixed 512×512 I/O |
| Model delivery | Small installer + one-time first-launch download |
| Export formats | JPG (sRGB, quality) + 16-bit TIFF |

## 7. Deliberate implementation deltas (documented deviations)

1. **Color policy = preserve source colorimetry.** The discussion mentioned an
   Adobe RGB working space; implementation keeps pixels in their source color
   space and manages bit depth only (8→16-bit on load). Rationale: zero gamut
   loss on round-trip, no extra ICC assets, inpainting is color-space agnostic.
   JPG export converts to sRGB when a non-sRGB ICC profile is present.
2. **TIFF EXIF is best-effort.** The chosen TIFF writer preserves ICC natively
   but has no first-class EXIF writer; JPG export preserves EXIF fully. If a
   TIFF EXIF write is not possible, export still succeeds (logged).
3. **Fixed-shape model tiling.** The LaMa ONNX export is fixed at 512×512, so
   inference runs in native-resolution 512×512 windows with 128 px overlap and
   feathered blending. No rescaling → no detail loss on large images.
4. **Checksum is trust-on-first-use.** The model hash recorded after the first
   successful download is verified on every later load; a pinned hash in
   `config.MODEL_SHA256` is enforced when set.
5. **Runtime quality tuning via `quality.ini`.** `quality.ini` in
   `app_data_dir()` (`%LOCALAPPDATA%\ObjectRemover` on Windows) tunes the fill
   — keys: `context_margin`, `multiscale`, `freq_sigma`, `harmonize`,
   `blend_band`, `feather_sigma` (plus `multiscale_trigger`/`min`). The file is
   re-read on every removal pass, so tuning needs no restart.
