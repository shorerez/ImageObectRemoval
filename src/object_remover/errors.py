"""Exception types for ObjectRemover."""
from __future__ import annotations


class ObjectRemoverError(Exception):
    """Base class for all application errors."""


class ImageLoadError(ObjectRemoverError):
    """The input file cannot be loaded (unsupported format, corrupt, etc.)."""


class ImageExportError(ObjectRemoverError):
    """The output file cannot be written."""


class ModelDownloadError(ObjectRemoverError):
    """Model weights could not be downloaded or verified."""


class InpaintError(ObjectRemoverError):
    """Inpainting failed."""


class CancelledError(ObjectRemoverError):
    """A long-running operation was cancelled by the user."""
