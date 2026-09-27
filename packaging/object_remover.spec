# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for ObjectRemover (one-folder build).

Build:  pyinstaller packaging/object_remover.spec
Output: dist/ObjectRemover/ObjectRemover.exe
"""
import sys
from pathlib import Path

PROJECT = Path(SPECPATH).parent
src = PROJECT / "src"

a = Analysis(
    [str(src / "object_remover" / "__main__.py")],
    pathex=[str(src)],
    binaries=[],
    datas=[],
    hiddenimports=[
        "onnxruntime",
        "onnxruntime.capi",
        "object_remover",
    ],
    excludes=["tkinter", "matplotlib", "scipy"],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="ObjectRemover",
    debug=False,
    strip=False,
    upx=False,
    console=False,          # GUI app
    disable_windowed_traceback=False,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="ObjectRemover",
)
