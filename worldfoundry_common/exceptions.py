"""Domain-specific exception types for WorldFoundry."""
from __future__ import annotations


class WorldFoundryError(Exception):
    """Base exception for all WorldFoundry errors."""


class VersionMismatchError(WorldFoundryError):
    """Raised when the installed package version does not match the locked matrix."""


class ModelNotFoundError(WorldFoundryError):
    """Raised when a model path / HF repo is not found."""


class OMCompileError(WorldFoundryError):
    """Raised when MindIE-SD compile of a model fails."""


class AdapterNotFoundError(WorldFoundryError):
    """Raised when a LoRA adapter name is not registered."""


class MaskingError(WorldFoundryError):
    """Raised when a mask generator cannot satisfy its target ratio within budget."""
