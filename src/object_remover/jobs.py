"""Qt job runner: long operations on worker threads with progress + cancel.

Core services stay Qt-free; this module is the only place that bridges them to
QThread. Terminal signals are delivered to the supervisor's thread via queued
connections; a JobSupervisor keeps handles alive until completion.
"""
from __future__ import annotations

import threading
from typing import Callable

from PySide6.QtCore import QObject, QThread, Signal, Slot

from .errors import CancelledError

JobFn = Callable[[Callable[[int, int], None], threading.Event], object]


class _Worker(QObject):
    progress = Signal(int, int)  # (done, total)
    succeeded = Signal(object)
    failed = Signal(str)
    cancelled = Signal()

    def __init__(self, fn: JobFn) -> None:
        super().__init__()
        self._fn = fn
        self._cancel = threading.Event()
        self.handle: "JobHandle | None" = None

    def request_cancel(self) -> None:
        self._cancel.set()

    def is_cancelled(self) -> bool:
        return self._cancel.is_set()

    @Slot()
    def run(self) -> None:
        def progress(done: int, total: int) -> None:
            self.progress.emit(done, total)

        try:
            result = self._fn(progress, self._cancel)
        except CancelledError:
            self.cancelled.emit()
        except Exception as exc:  # noqa: BLE001 - report any worker failure
            self.failed.emit(str(exc))
        else:
            self.succeeded.emit(result)


class JobHandle:
    def __init__(self, thread: QThread, worker: _Worker) -> None:
        self.thread = thread
        self.worker = worker

    def cancel(self) -> None:
        if self.thread.isRunning():
            self.worker.request_cancel()

    @property
    def running(self) -> bool:
        return self.thread.isRunning()


class JobSupervisor(QObject):
    """Owns running jobs; emits terminal signals with the originating handle."""

    job_succeeded = Signal(object, object)   # (handle, result)
    job_failed = Signal(object, str)         # (handle, message)
    job_cancelled = Signal(object)           # (handle,)
    job_progress = Signal(object, int, int)  # (handle, done, total)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._active: set[JobHandle] = set()

    @property
    def busy(self) -> bool:
        return bool(self._active)

    def run(self, fn: JobFn) -> JobHandle:
        thread = QThread(self)
        worker = _Worker(fn)
        worker.moveToThread(thread)
        handle = JobHandle(thread, worker)
        worker.handle = handle

        thread.started.connect(worker.run)
        worker.progress.connect(self._on_progress)
        worker.succeeded.connect(self._on_succeeded)
        worker.failed.connect(self._on_failed)
        worker.cancelled.connect(self._on_cancelled)
        worker.succeeded.connect(thread.quit)
        worker.failed.connect(thread.quit)
        worker.cancelled.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)

        self._active.add(handle)
        thread.start()
        return handle

    # -- queued slot receivers (run in the supervisor's thread) --------------
    @Slot(int, int)
    def _on_progress(self, done: int, total: int) -> None:
        handle = self._handle_of_sender()
        if handle is not None:
            self.job_progress.emit(handle, done, total)

    @Slot(object)
    def _on_succeeded(self, result: object) -> None:
        handle = self._handle_of_sender()
        self._finish(handle)
        if handle is not None:
            self.job_succeeded.emit(handle, result)

    @Slot(str)
    def _on_failed(self, message: str) -> None:
        handle = self._handle_of_sender()
        self._finish(handle)
        if handle is not None:
            self.job_failed.emit(handle, message)

    @Slot()
    def _on_cancelled(self) -> None:
        handle = self._handle_of_sender()
        self._finish(handle)
        if handle is not None:
            self.job_cancelled.emit(handle)

    def _handle_of_sender(self) -> JobHandle | None:
        sender = self.sender()
        return getattr(sender, "handle", None)

    def _finish(self, handle: JobHandle | None) -> None:
        if handle is not None:
            self._active.discard(handle)

    def cancel_all(self) -> None:
        for handle in list(self._active):
            handle.cancel()
