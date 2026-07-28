"""Central registry of upstream models and weights."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class ModelEntry:
    name: str
    hf_repo: str
    local_path: Optional[Path]
    default_adapter_dir: Path


MODEL_REGISTRY: dict[str, ModelEntry] = {
    "vjepa2-vitl-1.2b": ModelEntry(
        name="vjepa2-vitl-1.2b",
        hf_repo="facebook/vjepa2-vitl-1.2b",
        local_path=None,  # populated by env var WORLDFORGE_MODELS_DIR
        default_adapter_dir=Path("models/vjepa2-vitl-1.2b/lora_adapters"),
    ),
}


def resolve_model_path(name: str) -> Path:
    """Resolve a registered model to its local path. Falls back to HF hub id."""
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Model {name!r} not in MODEL_REGISTRY")
    entry = MODEL_REGISTRY[name]
    if entry.local_path is not None and entry.local_path.exists():
        return entry.local_path
    return Path(entry.hf_repo)  # HF hub id (downloaded on first use)
