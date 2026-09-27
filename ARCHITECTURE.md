# ObjectRemover — Architecture

## Layer map

```
┌─ UI (PySide6) ────────────────────────────────────────────────────────┐
│ MainWindow: toolbar (open, mask mode, eraser, brush, remove, undo,    │
│             redo, export), status bar, model download/progress        │
│ MaskCanvas: QGraphicsView + preview pyramid, brush strokes, overlays, │
│             zoom/pan, brush cursor                                    │
│ OverlapDialog / ExportDialog / download prompts                       │
├─ Document & History ──────────────────────────────────────────────────┤
│ ImageDocument: 16-bit working buffer + removal/protect MaskLayers     │
│ History + Commands: StrokeCommand, RemovalCommand, ClearMasksCommand  │
├─ Core Services (Qt-free, fully unit-tested) ──────────────────────────┤
│ image_io  : TIFF/JPG load, JPG/TIFF export, EXIF/ICC handling         │
│ inpaint   : InpaintService (tiling, compositing) + engines            │
│ models    : ModelManager (first-run download, checksum, atomic write) │
│ runtime   : ONNX session factory, VRAM probe, provider selection      │
├─ Runtime ─────────────────────────────────────────────────────────────┤
│ ONNX Runtime (CUDAExecutionProvider → CPUExecutionProvider)           │
│ Classical fallback engine (cv2.inpaint, Telea) when model is absent   │
└───────────────────────────────────────────────────────────────────────┘
```

Everything below the UI layer is Qt-free and tested headlessly. Long-running
work (inpainting, model download) runs in `QThread` workers via `jobs.py`; the
UI never blocks and every job is cancellable.

## Data flow — one removal pass

1. User paints/edits both masks (full-res uint8 buffers; preview pyramid only
   for display).
2. **Remove** clicked:
   - empty removal → status hint, no-op.
   - overlap detected → `OverlapDialog` (protect wins / removal wins / cancel).
   - `ImageDocument` snapshots are taken lazily by the command, not here.
3. `InpaintService.inpaint(pixels, removal, protect)` on a worker thread:
   - defensive rule: protect wins on any residual overlap inside the service.
   - bounding box of removal + context margin, clamped to the image.
   - **effective model mask = removal ∪ protect** — protected pixels are
     invisible to the model (excluded as fill source).
   - the region is covered by native-resolution 512×512 windows
     (step 384 = 512 − 128 overlap); each window goes through the engine;
     outputs are accumulated with linear edge ramps and normalized.
   - windows are `float32 [1,3,512,512]` in 0..1 for the ONNX engine.
   - composite: only removal-mask pixels are written (feathered one/two px
     inside the mask edge); protected and unselected pixels are bit-exact.
   - progress callback per window; `CancelFlag` checked between windows.
4. Result returns to the UI thread → `RemovalCommand` pushed (before/after
   pixel patch + before mask patches), masks cleared, view refreshed.
5. Repeat as many passes as desired; undo walks back through strokes and
   removals (undoing a removal restores its masks too).

## Memory model

- Full image: `uint16 (H, W, 3)` in RAM (100 MP ≈ 600 MB — fine on 32 GB).
- Masks: `uint8 (H, W)` each.
- Preview pyramid: one downsampled RGBA plane (≤ 4096 px side) + dirty-rect
  updates during strokes.
- History: commands store **bounding-box patches only**, so memory stays
  bounded across many passes.

## Inference engines

| Engine | When | Notes |
|---|---|---|
| `OnnxLamaEngine` | model present | Carve/LaMa-ONNX `lama_fp32.onnx`; inputs `image`/`mask`, fixed 512², output `output`; CUDA EP preferred, CPU fallback |
| `ClassicalEngine` | model missing / download pending | `cv2.inpaint` (Telea) — lower quality, keeps app usable and tests deterministic |

`runtime.create_session()` prefers `CUDAExecutionProvider`, falls back to CPU;
`build_default_engine()` tries each provider in turn (CUDA, then CPU) and ends
on `ClassicalEngine`.

**Output scale is measured, not assumed.** The Carve `lama_fp32.onnx` export
returns pixels in **0..255** (its own demo casts the raw output straight to
`uint8`), while the PyTorch model and most re-exports stay in **0..1**;
clipping a 0..255 output to [0, 1] turns every fill solid white. On
construction `OnnxLamaEngine` therefore runs a synthetic self-test (gray 512²
tile, centred hole) and derives the scale from the untouched context level,
normalising the raw output in `fill()`. A session whose self-test output is
non-finite or matches no plausible scale (1x/255x) is rejected, so a broken
provider degrades instead of painting white.

`runtime.probe_vram()` (pynvml if available) feeds the status bar; the
fixed 512² window keeps inference memory tiny regardless of image size.

## Model delivery

`ModelManager` downloads `lama_fp32.onnx` from the pinned URL into
`%LOCALAPPDATA%\ObjectRemover\models` (or `$XDG_DATA_HOME/ObjectRemover/models`):

- chunked download with progress + cancel → temp file → SHA-256
  (pinned hash if configured, else trust-on-first-use sidecar) → atomic rename.
- `status()` drives UI state (missing / downloading / ready / error); failure
  leaves the app on the classical engine with a clear message and retry.

## Threading rules

- Core services never touch Qt.
- UI thread only marshals results; workers emit `progress/succeeded/failed/
  cancelled`; `JobSupervisor` keeps handles alive and enables cancel.
- Document mutations happen only on the UI thread (command apply/undo/redo).

## Module map

```
src/object_remover/
├── __init__.py        version
├── __main__.py        python -m object_remover
├── app.py             QApplication bootstrap
├── config.py          constants, platform paths
├── errors.py          exception types
├── image_io.py        WorkingImage, load_image, export_jpg, export_tiff
├── document.py        ImageDocument, MaskLayer, commands, History, stroke math
├── inpaint.py         InpaintService, OnnxLamaEngine, ClassicalEngine
├── models.py          ModelManager
├── runtime.py         ONNX session factory, VRAM probe
├── jobs.py            QThread job runner (Qt side only)
└── ui/
    ├── main_window.py
    ├── canvas.py      MaskCanvas (brush, overlays, zoom/pan)
    ├── dialogs.py     OverlapDialog, ExportDialog, model prompts
```

## Packaging

- `packaging/object_remover.spec` — PyInstaller one-folder build.
- `packaging/installer.iss` — Inno Setup script producing the small installer
  (weights are *not* bundled; they download on first launch).
