"""Model weights manager: first-run download, checksum verification, caching.

Download protocol:
- chunked HTTP GET to MODEL_URL with progress + cancel;
- SHA-256 computed while streaming;
- verification: pinned hash when config.MODEL_SHA256 is set, otherwise
  trust-on-first-use — the hash recorded on the first successful download is
  enforced on every later download;
- atomic temp-file -> final rename.
"""
from __future__ import annotations

import hashlib
import logging
import os
import threading
import urllib.request
from enum import Enum
from pathlib import Path
from typing import Callable

from .config import MODEL_FILENAME, MODEL_SHA256, MODEL_URL, models_dir
from .errors import ModelDownloadError

log = logging.getLogger(__name__)

ProgressFn = Callable[[int, int], None]  # (bytes done, bytes total or 0)


class ModelStatus(str, Enum):
    MISSING = "missing"
    DOWNLOADING = "downloading"
    READY = "ready"
    ERROR = "error"


class ModelManager:
    def __init__(
        self,
        directory: Path | None = None,
        url: str = MODEL_URL,
        filename: str = MODEL_FILENAME,
        pinned_sha256: str | None = MODEL_SHA256,
    ) -> None:
        self._dir = Path(directory) if directory is not None else models_dir()
        self._dir.mkdir(parents=True, exist_ok=True)
        self._url = url
        self._filename = filename
        self._pinned = (pinned_sha256 or "").lower() or None
        self._downloading = False
        self._last_error: str | None = None

    # -- paths ---------------------------------------------------------------
    @property
    def path(self) -> Path:
        return self._dir / self._filename

    @property
    def hash_sidecar(self) -> Path:
        return self._dir / (self._filename + ".sha256")

    @property
    def last_error(self) -> str | None:
        return self._last_error

    # -- status --------------------------------------------------------------
    def status(self) -> ModelStatus:
        if self._downloading:
            return ModelStatus.DOWNLOADING
        if self.path.is_file() and self.path.stat().st_size > 1_000_000:
            return ModelStatus.READY
        return ModelStatus.MISSING

    def is_ready(self) -> bool:
        return self.status() == ModelStatus.READY

    # -- download ------------------------------------------------------------
    def download(
        self,
        progress: ProgressFn | None = None,
        cancel: threading.Event | None = None,
    ) -> Path:
        """Download the model weights. Raises ModelDownloadError on failure."""
        if self._downloading:
            raise ModelDownloadError("A model download is already in progress.")
        self._downloading = True
        self._last_error = None
        tmp = self.path.with_suffix(".part")
        try:
            req = urllib.request.Request(
                self._url, headers={"User-Agent": "ObjectRemover/1.0"}
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                total = int(resp.headers.get("Content-Length") or 0)
                digest = hashlib.sha256()
                done = 0
                with open(tmp, "wb") as out:
                    while True:
                        if cancel is not None and cancel.is_set():
                            raise ModelDownloadError("Model download cancelled.")
                        chunk = resp.read(1 << 20)
                        if not chunk:
                            break
                        out.write(chunk)
                        digest.update(chunk)
                        done += len(chunk)
                        if progress is not None:
                            progress(done, total)
            sha = digest.hexdigest()
            self._verify(sha)
            self.hash_sidecar.write_text(sha + "\n", encoding="ascii")
            os.replace(tmp, self.path)
            log.info("Model ready at %s (sha256=%s)", self.path, sha)
            return self.path
        except ModelDownloadError as exc:
            self._last_error = str(exc)
            raise
        except Exception as exc:
            self._last_error = str(exc)
            raise ModelDownloadError(
                f"Could not download the AI model: {exc}. "
                "Check your internet connection and retry."
            ) from exc
        finally:
            self._downloading = False
            if tmp.exists():
                try:
                    tmp.unlink()
                except OSError:  # pragma: no cover - defensive
                    pass

    def _verify(self, sha: str) -> None:
        expected = self._pinned
        if expected is None and self.hash_sidecar.is_file():
            expected = self.hash_sidecar.read_text(encoding="ascii").strip().lower() or None
        if expected is not None and sha != expected:
            raise ModelDownloadError(
                "Model checksum mismatch "
                f"(expected {expected[:12]}…, got {sha[:12]}…). "
                "The download may be corrupted — please retry."
            )
