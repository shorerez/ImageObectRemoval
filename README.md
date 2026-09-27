# ObjectRemover

Local Windows tool for removing objects from large images. Paint over what
should go away, optionally paint "protect" areas the fill must respect, hit
Remove, and export the result. Everything runs locally on your NVIDIA GPU.

Supported inputs: **TIFF, JPG** (large, up to ~100 MP). Exports: **JPG** (sRGB,
quality slider) and **16-bit TIFF**, metadata preserved.

![ObjectRemover UI](docs/ui-screenshot.png)

See [REQUIREMENTS.md](REQUIREMENTS.md) for the approved spec and
[ARCHITECTURE.md](ARCHITECTURE.md) for the design.

## Install (Windows, target machine)

```bat
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -r requirements-gpu.txt
pip install -e . --no-deps
python -m object_remover
```

`requirements-gpu.txt` installs `onnxruntime-gpu` (CUDA). If you only want the
CPU/classical engine for a smoke test, use `requirements-cpu.txt` instead.
Optional: `pip install pynvml` to show free VRAM in the status bar.

On **first launch** the app offers to download the LaMa AI model (~208 MB) into
`%LOCALAPPDATA%\ObjectRemover\models`. After that, everything is fully offline.
Until the model is present the app still works with a classical inpainting
fallback (noticeably lower quality).

## Develop / test (any OS)

```bash
python3.11 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements-cpu.txt
pip install -e . --no-deps
pip install pytest onnx            # onnx only needed for engine contract tests
pytest                             # core tests run headless
QT_QPA_PLATFORM=offscreen pytest tests/test_gui_smoke.py   # GUI smoke test
python -m object_remover           # run the app
```

## Usage

1. **Open** a TIFF or JPG.
2. Paint the **removal mask** (red). Switch to the **protect mask** (green) to
   paint areas the fill must not touch or copy from.
3. Both masks support paint/erase, brush size (`[` / `]`) and hardness; every
   stroke is undoable (`Ctrl+Z` / `Ctrl+Y`).
4. Click **Remove**. If the masks overlap you are asked which one wins.
5. Repeat for as many areas as you like, then **Export** as JPG or 16-bit TIFF.

## Build a Windows installer

```bat
pip install pyinstaller
pyinstaller packaging\object_remover.spec
iscc packaging\installer.iss
```

## Notes

- Inference runs in native-resolution 512×512 windows (LaMa ONNX fixed input),
  with 128 px overlap and feathered blending — no detail loss on huge images.
- Protected pixels are never overwritten and never used as fill source.
- Export never modifies the source image.
