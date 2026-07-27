"""Domain-specific exception types for WorldForge."""
from __future__ import annotations


class WorldForgeError(Exception):
    """Base exception for all WorldForge errors."""


class VersionMismatchError(WorldForgeError):
    """Raised when the installed package version does not match the locked matrix."""


class ModelNotFoundError(WorldForgeError):
    """Raised when a model path / HF repo is not found."""


class OMCompileError(WorldForgeError):
    """Raised when MindIE-SD compile of a model fails."""


class AdapterNotFoundError(WorldForgeError):
    """Raised when a LoRA adapter name is not registered."""
