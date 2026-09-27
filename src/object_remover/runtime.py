"""ONNX Runtime session factory and hardware probing."""
from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger(__name__)


def available_providers() -> list[str]:
    try:
        import onnxruntime as ort

        return list(ort.get_available_providers())
    except Exception:  # pragma: no cover - defensive
        return []


def create_onnx_session(model_path: str | Path, providers: list[str] | None = None):
    """Create an inference session preferring CUDA, falling back to CPU.

    ``providers`` restricts the session to the given execution providers (in
    order). Providers absent from the build are skipped; if none of the
    requested ones are available a RuntimeError is raised so the caller can
    try the next candidate.
    """
    import onnxruntime as ort

    avail = ort.get_available_providers()
    if providers is None:
        preferred = ("CUDAExecutionProvider", "CPUExecutionProvider")
        selected = [p for p in preferred if p in avail]
        if not selected:  # pragma: no cover - defensive
            selected = ["CPUExecutionProvider"]
    else:
        selected = [p for p in providers if p in avail]
        if not selected:
            raise RuntimeError(f"None of {list(providers)} available (have {avail}).")
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.log_severity_level = 3
    session = ort.InferenceSession(str(model_path), so, providers=selected)
    log.info(
        "ONNX session: providers=%s (available=%s)",
        session.get_providers(),
        avail,
    )
    return session


def probe_vram() -> dict | None:
    """Best-effort VRAM probe via pynvml/NVML. None when unavailable."""
    try:
        import pynvml  # type: ignore

        pynvml.nvmlInit()
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8", "replace")
            return {
                "name": str(name),
                "total_mb": int(info.total) // (1024 * 1024),
                "free_mb": int(info.free) // (1024 * 1024),
                "used_mb": int(info.used) // (1024 * 1024),
            }
        finally:
            pynvml.nvmlShutdown()
    except Exception:
        return None


def device_summary() -> str:
    """Human-readable device summary for the status bar."""
    providers = available_providers()
    if "CUDAExecutionProvider" in providers:
        line = "GPU acceleration: CUDA"
    elif providers:
        line = f"GPU acceleration: none ({providers[0]})"
    else:
        line = "ONNX Runtime not available"
    vram = probe_vram()
    if vram:
        line += (
            f" — {vram['name']} ({vram['free_mb']} MB free / {vram['total_mb']} MB)"
        )
    return line
