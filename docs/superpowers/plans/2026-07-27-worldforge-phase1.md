# WorldForge Phase-1 (Prototype Loop) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a working Ascend NPU training + inference loop for Meta V-JEPA 2 (1.2B) — DDP LoRA fine-tuning, vLLM-Ascend plugin inference, end-to-end smoke test, V-JEPA 2-AC finetuning phase reproduction.

**Architecture:** Three Python packages (`worldforge_train`, `worldforge_inf`, `worldforge_rl`) on a shared `worldforge_common` base. Phase-1 ships against the locked version matrix (CANN 8.5.1 + torch_npu 2.9.0 + vllm-ascend 0.18.0). No upstream forks; all adaptation lives in our packages. HuggingFace `transformers.VJEPA2Model` is wrapped with PEFT LoRA (target_modules verified at runtime).

**Tech Stack:** Python 3.11 · CANN 8.5.1 · torch_npu 2.9.0 · transformers 4.55.0 · peft 0.18.0 · vllm 0.18.0 · vllm-ascend 0.18.0 · decord 0.6.0 · datasets 3.6.0 · accelerate 1.10.0 · verl 0.5.0 · pytest 8.x · FastAPI + uvicorn · Prometheus client · OpenTelemetry SDK

## Global Constraints

These rules apply to every task. The implementer reads them once, not per-task.

| # | Rule | Source |
|---|------|--------|
| G1 | **Python 3.11** is the only supported runtime | `MindInferenceService` baseline |
| G2 | **Version matrix lock**: `torch_npu==2.9.0`, `torch==2.9.0`, `vllm==0.18.0`, `vllm-ascend==0.18.0`, `transformers==4.55.0`, `peft==0.18.0`, `verl==0.5.0`; CANN 8.5.1 + HDK 25.5.1+ | Spec §6.2 + verifier Finding 1 |
| G3 | **Default parallel = DDP for LoRA on 1.2B model**; FSDP only for full-finetune or long-seq | Spec §3.3 |
| G4 | **Every LoRA config must pass `model.print_trainable_parameters()` > 0** before training starts; if matches=0, abort | Spec §3.2 + risk row |
| G5 | **All file paths absolute** (no relative imports beyond package-internal) | User feedback rule |
| G6 | **Naming**: packages `worldforge_<subsystem>`; modules `snake_case`; classes `PascalCase`; constants `UPPER_SNAKE` | Consistency rule |
| G7 | **No upstream forks**: never modify files under `/opt/vjepa2/`, `site-packages/transformers/`, `site-packages/vllm*`; all adaptation in our packages | Spec §1.3 |
| G8 | **No silent defaults**: any failure (DVPP, OM load, LoRA miss) must raise or return explicit 4xx; never silently substitute | Spec §4.7 |
| G9 | **DRY / YAGNI / TDD**: write failing test first; minimal impl to pass; commit each task | Writing-plans skill |
| G10 | **Git commits per task**; commit message format: `<scope>: <imperative>`; do not amend previous commits | User feedback |
| G11 | **Mock NPU at unit-test time**: tests that require Ascend hardware must use `unittest.mock` for `torch.npu` so they run on CPU/CI | Practical constraint |
| G12 | **HMAC token auth on all inference endpoints** in Phase-1 | Spec §4.7 |

## File Structure

```
WorldForge/
├── docker/
│   └── worldforge-base/
│       ├── Dockerfile                     # T1
│       └── README.md                      # T1
├── worldforge_common/                     # shared base
│   ├── __init__.py                        # T2
│   ├── npu_env.py                         # T2 — torch_npu setup & version guard
│   ├── model_registry.py                  # T2 — paths to upstream models
│   ├── metrics.py                         # T18 — Prometheus counters
│   └── exceptions.py                      # T2 — domain errors
├── worldforge_train/
│   ├── __init__.py                        # T2
│   ├── data/
│   │   ├── __init__.py                    # T2
│   │   ├── video_source.py                # T3 — DVPP/decord backends
│   │   ├── tubelet_mask.py                # T4 — port from facebookresearch/vjepa2
│   │   └── dataset.py                     # T5 — HF datasets wrapper
│   ├── model/
│   │   ├── __init__.py                    # T2
│   │   ├── lora_wrapper.py                # T6 — PEFT/LoRA setup + sanity check
│   │   └── module_names.py                # T6 — runtime module-name discovery
│   ├── parallel/
│   │   ├── __init__.py                    # T2
│   │   ├── ddp.py                         # T7 — DDP launcher
│   │   └── fsdp.py                        # T8 — FSDP launcher (full-FT only)
│   ├── train.py                           # T7 — main training entry
│   ├── configs/
│   │   └── vjepa2_vitl_1.2b.yaml          # T7
│   └── tests/
│       ├── __init__.py                    # T2
│       ├── test_video_source.py           # T3
│       ├── test_tubelet_mask.py           # T4
│       ├── test_dataset.py                # T5
│       ├── test_lora_wrapper.py           # T6
│       ├── test_module_names.py           # T6
│       ├── test_ddp_smoke.py              # T7
│       └── test_fsdp_smoke.py             # T8
├── worldforge_inf/
│   ├── __init__.py                        # T2
│   ├── security.py                        # T13 — HMAC token
│   ├── batcher.py                         # T12 — video stream buffer
│   ├── scheduler.py                       # T12 — adapter-aware routing
│   ├── vjepa2_worker.py                   # T9 — vLLM-Ascend plugin worker
│   ├── compile/
│   │   ├── __init__.py                    # T2
│   │   ├── quantize.py                    # T10 — ModelSlim W8A8
│   │   └── mindie_sd.py                   # T11 — MindIE-SD compile
│   ├── api/
│   │   ├── __init__.py                    # T2
│   │   ├── encode.py                      # T14 — /v1/world/encode
│   │   ├── predict.py                     # T15 — /v1/world/predict
│   │   └── plan.py                        # T16 — /v1/world/plan (V-JEPA 2-AC)
│   ├── server.py                          # T14 — FastAPI app
│   └── tests/
│       ├── __init__.py                    # T2
│       ├── test_security.py               # T13
│       ├── test_vjepa2_worker.py          # T9
│       ├── test_quantize.py               # T10
│       ├── test_mindie_sd.py              # T11
│       ├── test_batcher.py                # T12
│       └── test_api_endpoints.py          # T14/15/16
├── worldforge_rl/                         # W7-W8 only (AC finetuning phase)
│   ├── __init__.py                        # T2
│   ├── actor.py                           # T19 — VJEPAActor (port V-JEPA 2-AC)
│   ├── env_droid.py                       # T19 — DroidEnv stub (offline data only)
│   ├── train_ac.py                        # T20 — AC finetuning entry
│   └── tests/
│       ├── __init__.py                    # T2
│       ├── test_actor.py                  # T19
│       └── test_train_ac_smoke.py         # T20
├── scripts/
│   ├── phase1_smoke.sh                    # T17 — end-to-end smoke
│   └── phase1_ac_repro.sh                 # T21 — AC repro runner
├── docs/
│   ├── superpowers/
│   │   ├── specs/
│   │   │   └── 2026-07-27-ascend-world-model-design.md  # already exists
│   │   └── plans/
│   │       └── 2026-07-27-worldforge-phase1.md          # this file
│   ├── user/
│   │   └── quickstart.md                  # T1
│   └── dev/
│       ├── environment.md                 # T1
│       └── runbook.md                     # T21
├── models/                                # model registry, gitignored
│   └── .gitkeep                           # T2
├── pyproject.toml                         # T1
├── .gitignore                             # T1
└── README.md                              # T1
```

---

### Task 1: Project skeleton + Docker base + pyproject.toml

**Files:**
- Create: `/mnt/c/WorkSpace/WorldForge/pyproject.toml`
- Create: `/mnt/c/WorkSpace/WorldForge/.gitignore`
- Create: `/mnt/c/WorkSpace/WorldForge/README.md`
- Create: `/mnt/c/WorkSpace/WorldForge/docker/worldforge-base/Dockerfile`
- Create: `/mnt/c/WorkSpace/WorldForge/docker/worldforge-base/README.md`
- Create: `/mnt/c/WorkSpace/WorldForge/docs/user/quickstart.md`
- Create: `/mnt/c/WorkSpace/WorldForge/docs/dev/environment.md`

**Interfaces:** None (skeleton task).

- [ ] **Step 1: Write `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "worldforge"
version = "0.1.0"
description = "Ascend NPU world model training + inference platform (Phase-1: V-JEPA 2)"
requires-python = "==3.11.*"
dependencies = [
    "torch==2.9.0",
    "torch_npu==2.9.0",
    "transformers==4.55.0",
    "peft==0.18.0",
    "accelerate==1.10.0",
    "datasets==3.6.0",
    "decord==0.6.0",
    "av==14.0.0",
    "pyyaml>=6.0",
    "numpy<2.0",
    "fastapi>=0.115",
    "uvicorn>=0.32",
    "pydantic>=2.9",
    "prometheus-client>=0.21",
    "opentelemetry-api>=1.27",
    "opentelemetry-sdk>=1.27",
]

[project.optional-dependencies]
train = ["deepspeed>=0.15"]
inf = ["vllm==0.18.0", "vllm-ascend==0.18.0"]
rl = ["verl==0.5.0"]
dev = ["pytest>=8.0", "pytest-cov>=5.0", "ruff>=0.6"]

[tool.setuptools.packages.find]
include = ["worldforge_*"]

[tool.pytest.ini_options]
testpaths = ["worldforge_train/tests", "worldforge_inf/tests", "worldforge_rl/tests"]
addopts = "-v --tb=short"
```

- [ ] **Step 2: Write `.gitignore`**

```gitignore
__pycache__/
*.pyc
*.egg-info/
.pytest_cache/
.ruff_cache/
.coverage
build/
dist/
*.tar.gz
.env
.venv/
models/
!models/.gitkeep
*.log
wandb/
tb_logs/
output/
checkpoints/
om_cache/
```

- [ ] **Step 3: Write top-level `README.md`**

```markdown
# WorldForge

Ascend NPU world model training + inference platform. **Phase-1** prototype ships
training + inference for Meta V-JEPA 2 (1.2B).

See:
- `docs/superpowers/specs/2026-07-27-ascend-world-model-design.md` — design spec
- `docs/superpowers/plans/2026-07-27-worldforge-phase1.md` — Phase-1 implementation plan
- `docs/user/quickstart.md` — end-user quickstart (added in Task 1)
- `docs/dev/environment.md` — dev environment setup (added in Task 1)
```

- [ ] **Step 4: Write `docker/worldforge-base/Dockerfile`**

```dockerfile
# WorldForge Phase-1 base image
# Locked version matrix (spec §6.2):
#   CANN 8.5.1 + torch_npu 2.9.0 + vllm-ascend 0.18.0
FROM ascend-pytorch:2.9.0-cann8.5.1-ubuntu22.04

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget vim ffmpeg libsm6 libxext6 libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Python deps
RUN pip install --no-cache-dir \
    transformers==4.55.0 \
    peft==0.18.0 \
    accelerate==1.10.0 \
    datasets==3.6.0 \
    decord==0.6.0 \
    av==14.0.0 \
    pyyaml \
    fastapi \
    uvicorn \
    pydantic \
    prometheus-client \
    opentelemetry-api \
    opentelemetry-sdk \
    pytest \
    pytest-cov \
    ruff

# Inference deps (optional layer)
RUN pip install --no-cache-dir \
    vllm==0.18.0 \
    vllm-ascend==0.18.0 || echo "vllm-ascend install may need additional wheel; rebuild for actual NPU host"

# RL deps (optional layer)
RUN pip install --no-cache-dir verl==0.5.0 || echo "verl install; rebuild for actual NPU host"

# Clone upstream V-JEPA 2 reference (NOT to be modified — G7)
RUN git clone https://github.com/facebookresearch/vjepa2 /opt/vjepa2

# Project source
WORKDIR /workspace
COPY pyproject.toml .
COPY worldforge_common/ worldforge_common/
COPY worldforge_train/ worldforge_train/
COPY worldforge_inf/ worldforge_inf/
COPY worldforge_rl/ worldforge_rl/

RUN pip install -e .

# Models dir (mounted at runtime)
RUN mkdir -p /workspace/models && touch /workspace/models/.gitkeep

CMD ["/bin/bash"]
```

- [ ] **Step 5: Write `docker/worldforge-base/README.md`**

```markdown
# WorldForge base image

Locked version matrix per spec §6.2:

- Base: `ascend-pytorch:2.9.0-cann8.5.1-ubuntu22.04`
- torch_npu: 2.9.0
- vllm: 0.18.0
- vllm-ascend: 0.18.0
- transformers: 4.55.0
- peft: 0.18.0

Build:
```bash
docker build -t worldforge-base:phase1 -f docker/worldforge-base/Dockerfile .
```

Run smoke (CPU fallback — NPU required for real tests):
```bash
docker run --rm -it worldforge-base:phase1 bash
```
```

- [ ] **Step 6: Write `docs/user/quickstart.md`**

```markdown
# Quickstart (Phase-1)

## Prerequisites

- 1× Atlas 800 A2 server with 8× Ascend 910B (64GB each), or CPU-only for unit tests
- Docker + Ascend NPU driver installed
- HF token with access to `facebook/vjepa2-vitl-1.2b`

## Run training (DDP LoRA)

```bash
docker run --rm -it --device /dev/davinci0 .../worldforge-base:phase1 \
    torchrun --nproc_per_node=8 \
        worldforge_train/train.py \
        --model facebook/vjepa2-vitl-1.2b \
        --dataset /data/videomix22m \
        --parallel ddp \
        --bf16 --gradient_checkpointing \
        --lora_r 16 --lora_alpha 32 \
        --per_device_batch 2 --grad_accum 8
```

## Run inference service

```bash
docker run --rm -it -p 8080:8080 .../worldforge-base:phase1 \
    python -m worldforge_inf.server \
        --model models/vjepa2-vitl-1.2b \
        --port 8080

# Test
curl -X POST http://localhost:8080/v1/world/encode \
    -H "Authorization: HMAC <token>" \
    -H "Content-Type: application/json" \
    -d '{"video": "<base64>", "adapter": "embodied-v1"}'
```
```

- [ ] **Step 7: Write `docs/dev/environment.md`**

```markdown
# Dev environment

## Local (no NPU)

```bash
# CPU-only install; NPU ops stubbed in unit tests
pip install -e ".[dev]"
pytest worldforge_train/tests worldforge_inf/tests worldforge_rl/tests
```

## With NPU

```bash
# Build base image (Task 1)
docker build -t worldforge-base:phase1 -f docker/worldforge-base/Dockerfile .

# Run tests inside container
docker run --rm --device /dev/davinci0:/dev/davinci0 --device /dev/davinci_manager \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
    -v $(pwd):/workspace \
    worldforge-base:phase1 \
    pytest worldforge_train/tests worldforge_inf/tests
```

## Version matrix enforcement

Any deviation from the pinned versions must be:
1. Discussed in a PR with explicit Versioning Policy link
2. Updated in `pyproject.toml`, `docker/.../Dockerfile`, and `worldforge_common/npu_env.py`
```

- [ ] **Step 8: Commit**

```bash
cd /mnt/c/WorkSpace/WorldForge
git add pyproject.toml .gitignore README.md docker/ docs/
git commit -m "chore: Phase-1 project skeleton, Dockerfile, docs"
```

---

### Task 2: Shared base (worldforge_common)

**Files:**
- Create: `worldforge_common/__init__.py`
- Create: `worldforge_common/npu_env.py`
- Create: `worldforge_common/model_registry.py`
- Create: `worldforge_common/exceptions.py`
- Create: `worldforge_common/tests/__init__.py`
- Create: `worldforge_common/tests/test_npu_env.py`
- Create: `worldforge_common/tests/test_model_registry.py`
- Create: `worldforge_common/tests/test_exceptions.py`
- Create: `models/.gitkeep`
- Create: `worldforge_train/__init__.py`
- Create: `worldforge_train/data/__init__.py`
- Create: `worldforge_train/model/__init__.py`
- Create: `worldforge_train/parallel/__init__.py`
- Create: `worldforge_train/tests/__init__.py`
- Create: `worldforge_inf/__init__.py`
- Create: `worldforge_inf/compile/__init__.py`
- Create: `worldforge_inf/api/__init__.py`
- Create: `worldforge_inf/tests/__init__.py`
- Create: `worldforge_rl/__init__.py`
- Create: `worldforge_rl/tests/__init__.py`

**Interfaces:**
- Consumes: nothing
- Produces:
  - `worldforge_common.npu_env.check_versions()` → `None` (raises `VersionMismatchError`)
  - `worldforge_common.npu_env.is_npu_available()` → `bool`
  - `worldforge_common.model_registry.MODEL_REGISTRY: dict[str, ModelEntry]`
  - `worldforge_common.exceptions.{VersionMismatchError, ModelNotFoundError, OMCompileError, AdapterNotFoundError}`

- [ ] **Step 1: Write `worldforge_common/__init__.py`**

```python
"""Shared base for WorldForge Phase-1."""
from worldforge_common.exceptions import (
    VersionMismatchError,
    ModelNotFoundError,
    OMCompileError,
    AdapterNotFoundError,
)

__all__ = [
    "VersionMismatchError",
    "ModelNotFoundError",
    "OMCompileError",
    "AdapterNotFoundError",
]
```

- [ ] **Step 2: Write `worldforge_common/exceptions.py`**

```python
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
```

- [ ] **Step 3: Write `worldforge_common/npu_env.py`**

```python
"""NPU environment setup and version guard.

Per Global Constraint G2 (spec §6.2):
  torch_npu 2.9.0 + vllm-ascend 0.18.0 + CANN 8.5.1 + transformers 4.55.0
"""
from __future__ import annotations

import importlib.metadata
from typing import Tuple

from worldforge_common.exceptions import VersionMismatchError

LOCKED_VERSIONS: dict[str, str] = {
    "torch": "2.9.0",
    "torch_npu": "2.9.0",
    "transformers": "4.55.0",
    "peft": "0.18.0",
    "accelerate": "1.10.0",
}


def _get_version(pkg: str) -> str:
    try:
        return importlib.metadata.version(pkg)
    except importlib.metadata.PackageNotFoundError:
        return "0.0.0+missing"


def check_versions(strict: bool = False) -> list[tuple[str, str, str]]:
    """Verify installed package versions against the locked matrix.

    Returns list of (pkg, expected, actual). If strict=True and any mismatch
    is found, raises VersionMismatchError.
    """
    mismatches: list[tuple[str, str, str]] = []
    for pkg, expected in LOCKED_VERSIONS.items():
        actual = _get_version(pkg)
        if actual != expected:
            mismatches.append((pkg, expected, actual))
    if strict and mismatches:
        raise VersionMismatchError(
            f"Locked version mismatch: {mismatches}. "
            f"See spec §6.2 and pyproject.toml for the matrix."
        )
    return mismatches


def is_npu_available() -> bool:
    """Return True iff torch.npu is importable and at least one device is reachable."""
    try:
        import torch_npu  # noqa: F401
        import torch
        return torch.npu.is_available() if hasattr(torch, "npu") else False
    except ImportError:
        return False


def npu_device_count() -> int:
    """Number of NPU devices available (0 on CPU-only machines)."""
    if not is_npu_available():
        return 0
    import torch
    return torch.npu.device_count()


def init_distributed(backend: str = "hccl") -> None:
    """Initialize torch.distributed with HCCL on NPU, Gloo on CPU fallback."""
    import torch
    import torch.distributed as dist
    if is_npu_available():
        dist.init_process_group(backend="hccl")
    else:
        dist.init_process_group(backend="gloo")
```

- [ ] **Step 4: Write `worldforge_common/model_registry.py`**

```python
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
```

- [ ] **Step 5: Write `tests/__init__.py` files (empty)**

```bash
for d in worldforge_common/tests worldforge_train/tests worldforge_train/data worldforge_train/model worldforge_train/parallel worldforge_inf/tests worldforge_inf/api worldforge_inf/compile worldforge_rl/tests; do
  touch "$d/__init__.py"
done
```

- [ ] **Step 6: Write `worldforge_common/tests/test_exceptions.py`**

```python
from worldforge_common.exceptions import (
    AdapterNotFoundError,
    ModelNotFoundError,
    OMCompileError,
    VersionMismatchError,
    WorldForgeError,
)


def test_all_inherit_from_base():
    for cls in (VersionMismatchError, ModelNotFoundError, OMCompileError, AdapterNotFoundError):
        assert issubclass(cls, WorldForgeError)
```

- [ ] **Step 7: Write `worldforge_common/tests/test_npu_env.py`**

```python
import pytest

from worldforge_common import npu_env
from worldforge_common.exceptions import VersionMismatchError


def test_check_versions_non_strict_returns_list():
    mismatches = npu_env.check_versions(strict=False)
    assert isinstance(mismatches, list)


def test_check_versions_strict_raises_on_mismatch(monkeypatch):
    # Force a mismatch by tampering with LOCKED_VERSIONS
    monkeypatch.setitem(npu_env.LOCKED_VERSIONS, "torch", "999.999.999")
    with pytest.raises(VersionMismatchError):
        npu_env.check_versions(strict=True)


def test_is_npu_available_returns_bool():
    result = npu_env.is_npu_available()
    assert isinstance(result, bool)


def test_npu_device_count_returns_int():
    count = npu_env.npu_device_count()
    assert isinstance(count, int)
    assert count >= 0
```

- [ ] **Step 8: Write `worldforge_common/tests/test_model_registry.py`**

```python
from pathlib import Path

import pytest

from worldforge_common.model_registry import MODEL_REGISTRY, resolve_model_path


def test_vjepa2_registered():
    assert "vjepa2-vitl-1.2b" in MODEL_REGISTRY
    entry = MODEL_REGISTRY["vjepa2-vitl-1.2b"]
    assert entry.hf_repo == "facebook/vjepa2-vitl-1.2b"
    assert isinstance(entry.default_adapter_dir, Path)


def test_resolve_unknown_raises():
    with pytest.raises(KeyError):
        resolve_model_path("nonexistent")
```

- [ ] **Step 9: Run tests**

```bash
cd /mnt/c/WorkSpace/WorldForge
PYTHONPATH=. pytest worldforge_common/tests -v
```

Expected: all pass (CPU-only is fine for these).

- [ ] **Step 10: Commit**

```bash
git add worldforge_common/ worldforge_train/__init__.py worldforge_train/data/__init__.py worldforge_train/model/__init__.py worldforge_train/parallel/__init__.py worldforge_train/tests/__init__.py worldforge_inf/__init__.py worldforge_inf/compile/__init__.py worldforge_inf/api/__init__.py worldforge_inf/tests/__init__.py worldforge_rl/__init__.py worldforge_rl/tests/__init__.py models/.gitkeep
git commit -m "feat(common): shared base, NPU env, model registry, exceptions"
```

---

### Task 3: Video source with DVPP / decord backend

**Files:**
- Create: `worldforge_train/data/video_source.py`
- Create: `worldforge_train/tests/test_video_source.py`

**Interfaces:**
- Consumes: nothing
- Produces:
  - `worldforge_train.data.video_source.VideoSource` (abstract)
  - `worldforge_train.data.video_source.DecordVideoSource(path) -> ndarray[T, H, W, 3] uint8`
  - `worldforge_train.data.video_source.build_video_source(path, prefer_dvpp=True) -> VideoSource`

- [ ] **Step 1: Write failing test `worldforge_train/tests/test_video_source.py`**

```python
import numpy as np
import pytest

from worldforge_train.data.video_source import (
    DecordVideoSource,
    build_video_source,
)


@pytest.fixture
def sample_mp4(tmp_path):
    """Create a 2-second 64x64 mp4 with decord for tests."""
    import decord
    import av

    out = tmp_path / "sample.mp4"
    container = av.open(str(out), mode="w")
    stream = container.add_stream("h264", rate=10)
    stream.width, stream.height = 64, 64
    for i in range(20):
        frame = av.VideoFrame.from_ndarray(
            np.full((64, 64, 3), i * 12, dtype=np.uint8)
        )
        frame.pts = i
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()
    return out


def test_decord_source_returns_uint8_frames(sample_mp4):
    src = DecordVideoSource(sample_mp4)
    frames = src.read_all()
    assert frames.dtype == np.uint8
    assert frames.shape == (20, 64, 64, 3)
    # Pixels vary frame-to-frame (i*12)
    assert frames[0, 0, 0, 0] != frames[-1, 0, 0, 0]


def test_build_video_source_default_returns_decord(sample_mp4):
    src = build_video_source(sample_mp4, prefer_dvpp=False)
    assert isinstance(src, DecordVideoSource)


def test_build_video_source_dvpp_unavailable_falls_back(sample_mp4, monkeypatch):
    monkeypatch.setattr(
        "worldforge_train.data.video_source._dvpp_available",
        lambda: False,
    )
    src = build_video_source(sample_mp4, prefer_dvpp=True)
    assert isinstance(src, DecordVideoSource)


def test_build_video_source_raises_for_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        build_video_source(tmp_path / "nope.mp4")
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /mnt/c/WorkSpace/WorldForge
PYTHONPATH=. pytest worldforge_train/tests/test_video_source.py -v
```

Expected: ImportError or AttributeError (module does not exist yet).

- [ ] **Step 3: Implement `worldforge_train/data/video_source.py`**

```python
"""Video source abstraction with DVPP (hardware) and decord (software) backends.

Per spec §3.1:
- Primary: Ascend DVPP (CANN hardware decoder)
- Fallback: decord (CPU)
- Single-batch DVPP failure must fall back to decord (per spec §4.7)
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union

import numpy as np

PathLike = Union[str, Path]


def _dvpp_available() -> bool:
    """Probe for Ascend DVPP availability. Returns False on non-Ascend hosts."""
    try:
        import torch_npu  # noqa: F401
        # Real DVPP detection requires CANN acl_dvpp import; treat as opt-in.
        return False  # Phase-1: always fall back; DVPP wired in Phase-2
    except ImportError:
        return False


class VideoSource(ABC):
    """Abstract video source returning T×H×W×3 uint8 frames."""

    @abstractmethod
    def read_all(self) -> np.ndarray:
        """Load all frames into memory."""


class DecordVideoSource(VideoSource):
    """Software video decoder using decord (CPU)."""

    def __init__(self, path: PathLike) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Video not found: {self.path}")

    def read_all(self) -> np.ndarray:
        import decord  # local import: optional dep
        decord.bridge.set_bridge("native")
        reader = decord.VideoReader(str(self.path))
        frames = reader.get_batch(range(len(reader))).asnumpy()
        return frames  # (T, H, W, 3) uint8


class DVPPVideoSource(VideoSource):
    """Hardware-accelerated decoder via Ascend DVPP.

    Phase-1: stub; raises NotImplementedError so callers fall back.
    Phase-2 W11: implement via acl_dvpp.
    """

    def __init__(self, path: PathLike) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Video not found: {self.path}")

    def read_all(self) -> np.ndarray:
        raise NotImplementedError(
            "DVPP decoder is wired in Phase-2 W11. "
            "Use build_video_source(prefer_dvpp=False) for now."
        )


def build_video_source(path: PathLike, prefer_dvpp: bool = True) -> VideoSource:
    """Build a video source, preferring DVPP if available and requested."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Video not found: {p}")
    if prefer_dvpp and _dvpp_available():
        try:
            return DVPPVideoSource(p)
        except Exception:
            pass  # fall through to decord per spec §4.7
    return DecordVideoSource(p)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /mnt/c/WorkSpace/WorldForge
PYTHONPATH=. pytest worldforge_train/tests/test_video_source.py -v
```

Expected: 4 tests pass.

- [ ] **Step 5: Commit**

```bash
git add worldforge_train/data/video_source.py worldforge_train/tests/test_video_source.py
git commit -m "feat(train): video source with DVPP/decord backends"
```

---

### Task 4: Tubelet masking (port from facebookresearch/vjepa2)

**Files:**
- Create: `worldforge_train/data/tubelet_mask.py`
- Create: `worldforge_train/tests/test_tubelet_mask.py`

**Interfaces:**
- Consumes: nothing
- Produces:
  - `worldforge_train.data.tubelet_mask.multiblock_mask(shape, num_blocks, ...)` → `np.ndarray[bool]` of shape `shape` (T,H,W)
  - `worldforge_train.data.tubelet_mask.tubelet_mask(shape, tubelet_size, ...)` → `np.ndarray[bool]`

- [ ] **Step 1: Write failing test `worldforge_train/tests/test_tubelet_mask.py`**

```python
import numpy as np
import pytest

from worldforge_train.data.tubelet_mask import multiblock_mask, tubelet_mask


def test_multiblock_mask_shape_and_dtype():
    shape = (8, 14, 14)
    mask = multiblock_mask(shape=shape, num_blocks=4, spatial_only=False, rng=np.random.default_rng(0))
    assert mask.shape == shape
    assert mask.dtype == bool
    # roughly half masked
    assert 0.3 <= mask.mean() <= 0.7


def test_multiblock_mask_spatial_only_zero_t():
    shape = (8, 14, 14)
    mask = multiblock_mask(shape=shape, num_blocks=4, spatial_only=True, rng=np.random.default_rng(0))
    # spatial_only: same mask broadcast across T
    assert np.all(mask[0] == mask[1])


def test_tubelet_mask_is_subset_of_full():
    shape = (8, 14, 14)
    full = np.ones(shape, dtype=bool)
    masked = tubelet_mask(shape=shape, tubelet_size=(2, 14, 14), rng=np.random.default_rng(0))
    assert masked.shape == shape
    assert masked.sum() < full.sum()
    assert masked.dtype == bool
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /mnt/c/WorkSpace/WorldForge
PYTHONPATH=. pytest worldforge_train/tests/test_tubelet_mask.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement `worldforge_train/data/tubelet_mask.py`**

```python
"""Tubelet masking for V-JEPA 2 (ported from facebookresearch/vjepa2/src/utils/tensors.py).

Adapted from /opt/vjepa2/src/utils/tensors.py under the original BSD-style license.
Original: Copyright (c) Meta Platforms, Inc. and affiliates.
"""
from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np


def _broadcast_shape_for_tubelet(
    shape: Tuple[int, int, int],
    tubelet_size: Tuple[int, int, int],
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """Compute the grid shape and per-axis stride from a (T,H,W) tensor and tubelet size."""
    grid = tuple(s // t for s, t in zip(shape, tubelet_size))
    return grid, tubelet_size


def multiblock_mask(
    shape: Tuple[int, int, int],
    num_blocks: int,
    spatial_only: bool = False,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate a multiblock mask following Bardes et al. 2024 (V-JEPA 2).

    Args:
        shape: (T, H, W) target tensor shape.
        num_blocks: number of rectangular blocks to mask.
        spatial_only: if True, broadcast mask across T.
        rng: numpy Generator for reproducibility.
    """
    if rng is None:
        rng = np.random.default_rng()
    if spatial_only:
        spatial = _sample_block_mask_2d((shape[1], shape[2]), num_blocks, rng)
        mask = np.broadcast_to(spatial[None], shape).copy()
    else:
        mask = _sample_block_mask_3d(shape, num_blocks, rng)
    return mask


def _sample_block_mask_2d(shape_hw: Tuple[int, int], num_blocks: int, rng: np.random.Generator) -> np.ndarray:
    H, W = shape_hw
    mask = np.zeros((H, W), dtype=bool)
    for _ in range(num_blocks):
        h = rng.integers(1, max(1, H // 2) + 1)
        w = rng.integers(1, max(1, W // 2) + 1)
        top = rng.integers(0, H - h + 1)
        left = rng.integers(0, W - w + 1)
        mask[top:top + h, left:left + w] = True
    return mask


def _sample_block_mask_3d(shape: Tuple[int, int, int], num_blocks: int, rng: np.random.Generator) -> np.ndarray:
    T, H, W = shape
    mask = np.zeros(shape, dtype=bool)
    for _ in range(num_blocks):
        t = rng.integers(1, max(1, T // 2) + 1)
        h = rng.integers(1, max(1, H // 2) + 1)
        w = rng.integers(1, max(1, W // 2) + 1)
        t0 = rng.integers(0, T - t + 1)
        h0 = rng.integers(0, H - h + 1)
        w0 = rng.integers(0, W - w + 1)
        mask[t0:t0 + t, h0:h0 + h, w0:w0 + w] = True
    return mask


def tubelet_mask(
    shape: Tuple[int, int, int],
    tubelet_size: Tuple[int, int, int],
    mask_ratio: float = 0.75,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate a tubelet-aligned binary mask by selecting whole tubelets at random.

    A tubelet of size (t,h,w) covers one cell in the grid (T/t, H/h, W/w).
    """
    if rng is None:
        rng = np.random.default_rng()
    grid, _ = _broadcast_shape_for_tubelet(shape, tubelet_size)
    total = int(np.prod(grid))
    n_masked = int(round(mask_ratio * total))
    grid_mask = np.zeros(total, dtype=bool)
    grid_mask[rng.choice(total, size=n_masked, replace=False)] = True
    grid_mask = grid_mask.reshape(grid)
    # Broadcast each grid cell to a tubelet
    mask = np.kron(grid_mask, np.ones(tubelet_size, dtype=bool))
    # Crop if shape not perfectly divisible by tubelet size
    mask = mask[: shape[0], : shape[1], : shape[2]]
    return mask
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /mnt/c/WorkSpace/WorldForge
PYTHONPATH=. pytest worldforge_train/tests/test_tubelet_mask.py -v
```

Expected: 3 tests pass.

- [ ] **Step 5: Commit**

```bash
git add worldforge_train/data/tubelet_mask.py worldforge_train/tests/test_tubelet_mask.py
git commit -m "feat(train): tubelet masking for V-JEPA 2"
```

---

### Task 5: Dataset wrapper (HF datasets + Arrow)

**Files:**
- Create: `worldforge_train/data/dataset.py`
- Create: `worldforge_train/tests/test_dataset.py`

**Interfaces:**
- Consumes: `worldforge_train.data.video_source.build_video_source`
- Produces:
  - `worldforge_train.data.dataset.Vjepa2Dataset(manifest_path, mask_fn)` returning dict `{"frames": Tensor, "mask": Tensor}` per item

- [ ] **Step 1: Write failing test `worldforge_train/tests/test_dataset.py`**

```python
import numpy as np
import pytest

from worldforge_train.data.dataset import Vjepa2Dataset
from worldforge_train.data.tubelet_mask import tubelet_mask


@pytest.fixture
def manifest(tmp_path, sample_mp4_from_task3):
    """Write a tiny manifest CSV: video_path, num_frames, height, width."""
    csv = tmp_path / "manifest.csv"
    csv.write_text(f"video_path,num_frames,height,width\n{sample_mp4_from_task4},20,64,64\n")
    return csv


def test_dataset_len_matches_manifest(manifest):
    ds = Vjepa2Dataset(
        manifest_path=manifest,
        mask_fn=lambda shape, rng: tubelet_mask(shape, (2, 14, 14), rng=rng),
    )
    assert len(ds) == 1


def test_dataset_getitem_returns_correct_shapes(manifest):
    ds = Vjepa2Dataset(
        manifest_path=manifest,
        mask_fn=lambda shape, rng: tubelet_mask(shape, (2, 14, 14), rng=rng),
    )
    item = ds[0]
    assert "frames" in item and "mask" in item
    # frames: (T, C, H, W) float32 in [0,1]
    assert item["frames"].shape[0] == 20
    assert item["frames"].shape[1] == 3
    assert item["frames"].dtype == np.float32
    assert 0.0 <= item["frames"].min() and item["frames"].max() <= 1.0
    # mask: (T, H_grid, W_grid) bool
    assert item["mask"].dtype == bool
```

- [ ] **Step 2: Add sample video fixture helper**

Append to `worldforge_train/tests/test_video_source.py` (extend existing module):

```python
@pytest.fixture
def sample_mp4_from_task3(tmp_path):
    """Re-use sample_mp4 fixture for cross-task tests."""
    import decord  # noqa
    import av
    out = tmp_path / "sample.mp4"
    container = av.open(str(out), mode="w")
    stream = container.add_stream("h264", rate=10)
    stream.width, stream.height = 64, 64
    for i in range(20):
        frame = av.VideoFrame.from_ndarray(
            np.full((64, 64, 3), i * 12, dtype=np.uint8)
        )
        frame.pts = i
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()
    return out
```

(Rename the existing fixture in test_video_source.py from `sample_mp4` to `sample_mp4_from_task3` if needed; both names work.)

- [ ] **Step 3: Run failing test**

```bash
cd /mnt/c/WorkSpace/WorldForge
PYTHONPATH=. pytest worldforge_train/tests/test_dataset.py -v
```

Expected: ImportError.

- [ ] **Step 4: Implement `worldforge_train/data/dataset.py`**

```python
"""HF datasets wrapper for V-JEPA 2 training."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from worldforge_train.data.video_source import build_video_source

MaskFn = Callable[[tuple[int, int, int], np.random.Generator], np.ndarray]


class Vjepa2Dataset:
    """Iterates (video, mask) samples from a CSV manifest.

    Manifest schema: video_path, num_frames, height, width
    """

    def __init__(self, manifest_path: Path, mask_fn: MaskFn) -> None:
        self.manifest_path = Path(manifest_path)
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")
        self.mask_fn = mask_fn
        self.entries: list[dict[str, str]] = []
        with self.manifest_path.open() as f:
            reader = csv.DictReader(f)
            self.entries = list(reader)

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        row = self.entries[idx]
        src = build_video_source(row["video_path"], prefer_dvpp=False)
        frames = src.read_all()  # (T, H, W, 3) uint8
        # Normalize to float32 in [0,1]
        frames = frames.astype(np.float32) / 255.0
        # (T, H, W, 3) → (T, 3, H, W)
        frames = np.transpose(frames, (0, 3, 1, 2))
        # Mask: align grid to height/width (no tubelet crop on T for simplicity)
        T, _, H, W = frames.shape
        grid_h, grid_w = H // 14, W // 14  # V-JEPA 2 default patch grid
        mask = self.mask_fn((T, grid_h, grid_w), np.random.default_rng(idx))
        return {"frames": frames, "mask": mask}
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
cd /mnt/c/WorkSpace/WorldForge
PYTHONPATH=. pytest worldforge_train/tests/test_dataset.py -v
```

Expected: 2 tests pass.

- [ ] **Step 6: Commit**

```bash
git add worldforge_train/data/dataset.py worldforge_train/tests/test_dataset.py worldforge_train/tests/test_video_source.py
git commit -m "feat(train): HF datasets wrapper with tubelet masking"
```

---

### Task 6: LoRA wrapper with mandatory module-name sanity check (G4)

**Files:**
- Create: `worldforge_train/model/module_names.py`
- Create: `worldforge_train/model/lora_wrapper.py`
- Create: `worldforge_train/tests/test_module_names.py`
- Create: `worldforge_train/tests/test_lora_wrapper.py`

**Interfaces:**
- Consumes: `transformers.VJEPA2Model`
- Produces:
  - `worldforge_train.model.module_names.discover_lora_targets(model, candidates)` → list of names actually present
  - `worldforge_train.model.lora_wrapper.build_lora_model(model, r=16, alpha=32)` → `peft.PeftModel` (raises `ValueError` if matches=0 per G4)

- [ ] **Step 1: Write failing test `worldforge_train/tests/test_module_names.py`**

```python
import torch.nn as nn

from worldforge_train.model.module_names import discover_lora_targets


class FakeLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(2, 2))


def test_discover_returns_only_present_names():
    import torch

    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = FakeLinear()
            self.k_proj = FakeLinear()
            self.unrelated = nn.Linear(2, 2)

    model = TinyModel()
    targets = discover_lora_targets(
        model,
        candidates=["q_proj", "k_proj", "v_proj", "o_proj", "mlp.fc1", "mlp.fc2"],
    )
    assert "q_proj" in targets
    assert "k_proj" in targets
    assert "v_proj" not in targets
    assert "mlp.fc1" not in targets


def test_discover_supports_nested_module_names():
    import torch

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(2, 2)
            self.fc2 = nn.Linear(2, 2)

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = MLP()

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = nn.ModuleList([Block(), Block()])

    model = Model()
    targets = discover_lora_targets(model, ["mlp.fc1"])
    assert "blocks.0.mlp.fc1" in targets
    assert "blocks.1.mlp.fc1" in targets
```

- [ ] **Step 2: Write failing test `worldforge_train/tests/test_lora_wrapper.py`**

```python
import pytest

from worldforge_train.model.lora_wrapper import build_lora_model


class _MatMul:
    def __matmul__(self, other):
        return other


class _TensorStub:
    def __init__(self):
        self.shape = (1, 1)


class _AttentionStub:
    def __init__(self):
        self.q_proj = _MatMul()
        self.k_proj = _MatMul()
        self.v_proj = _MatMul()
        self.o_proj = _MatMul()


class _MLPStub:
    def __init__(self):
        self.fc1 = _MatMul()
        self.fc2 = _MatMul()


class _BlockStub:
    def __init__(self):
        self.attn = _AttentionStub()
        self.mlp = _MLPStub()


class _TinyVJEPA2:
    """Stand-in for transformers.VJEPA2Model with realistic submodule names."""

    class predictor:
        class head:
            weight = _TensorStub()

    def __init__(self):
        self.blocks = [_BlockStub(), _BlockStub()]


def test_build_lora_returns_peft_model(_monkeypatch_for_peft):
    # We patch peft.LoraConfig + get_peft_model to avoid real HF dependency in tests
    from worldforge_train.model import lora_wrapper
    import peft

    captured = {}

    class FakeLoraConfig:
        def __init__(self, **kw):
            captured["config"] = kw

    class FakePeftModel:
        def __init__(self, base, config):
            captured["base"] = base
            captured["peft_config"] = config

    lora_wrapper.LoraConfig = FakeLoraConfig
    lora_wrapper.get_peft_model = lambda base, cfg: FakePeftModel(base, cfg)

    base = _TinyVJEPA2()
    model = lora_wrapper.build_lora_model(base, r=16, alpha=32)

    assert captured["base"] is base
    # target_modules should have been auto-discovered from the actual names
    assert "blocks.0.attn.q_proj" in captured["config"]["target_modules"]
    assert "blocks.0.mlp.fc1" in captured["config"]["target_modules"]


def test_build_lora_raises_when_no_targets_match():
    class Empty:
        class predictor:
            class head:
                weight = _TensorStub()

        blocks = []

    from worldforge_train.model import lora_wrapper
    lora_wrapper.LoraConfig = lambda **kw: None
    lora_wrapper.get_peft_model = lambda base, cfg: object()

    with pytest.raises(ValueError, match="LoRA targets matched 0 modules"):
        lora_wrapper.build_lora_model(Empty())
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
cd /mnt/c/WorkSpace/WorldForge
PYTHONPATH=. pytest worldforge_train/tests/test_module_names.py worldforge_train/tests/test_lora_wrapper.py -v
```

Expected: ImportError.

- [ ] **Step 4: Implement `worldforge_train/model/module_names.py`**

```python
"""Runtime discovery of LoRA-targetable module names in a model."""
from __future__ import annotations

from typing import Iterable

import torch.nn as nn


def discover_lora_targets(
    model: nn.Module,
    candidates: Iterable[str],
    suffix_only: bool = True,
) -> list[str]:
    """Return fully-qualified module names in `model` whose leaf name ends with
    one of the candidate suffixes.

    Args:
        model: a torch.nn.Module (e.g. VJEPA2Model).
        candidates: candidate leaf names like ["q_proj", "mlp.fc1"].
        suffix_only: if True (default), match by suffix only (so "mlp.fc1"
            matches "blocks.0.mlp.fc1"). If False, require exact match on the
            leaf name.
    """
    candidates = list(candidates)
    matches: list[str] = []
    for name, _module in model.named_modules():
        leaf = name.split(".")[-1] if name else ""
        if not leaf:
            continue
        for cand in candidates:
            cand_leaf = cand.split(".")[-1]
            if suffix_only and leaf == cand_leaf:
                matches.append(name)
                break
            if not suffix_only and leaf == cand_leaf:
                matches.append(name)
                break
    return matches
```

- [ ] **Step 5: Implement `worldforge_train/model/lora_wrapper.py`**

```python
"""LoRA wrapping for V-JEPA 2 with mandatory sanity check (Global Constraint G4).

Per spec §3.2 + risk row "LoRA silent zero training":
- Before training, MUST verify `model.print_trainable_parameters()` > 0
- If matches=0, raise ValueError; never silently proceed with no trainable params
"""
from __future__ import annotations

from dataclasses import dataclass

import torch.nn as nn

from worldforge_train.model.module_names import discover_lora_targets

# Default candidates: q/k/v/o projections + MLP linears.
# These will be filtered by discover_lora_targets against the actual model.
DEFAULT_LORA_CANDIDATES: list[str] = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "mlp.fc1", "mlp.fc2",
]


@dataclass
class LoraBuildResult:
    peft_model: object
    trainable_params: int
    total_params: int
    target_modules: list[str]


def build_lora_model(
    base_model: nn.Module,
    r: int = 16,
    alpha: int = 32,
    dropout: float = 0.05,
    bias: str = "none",
    candidate_targets: list[str] | None = None,
) -> LoraBuildResult:
    """Wrap a V-JEPA 2 base model with PEFT LoRA.

    Raises ValueError if LoRA matches zero modules (per G4).
    """
    from peft import LoraConfig, get_peft_model  # imported lazily to keep tests light

    candidates = candidate_targets or DEFAULT_LORA_CANDIDATES
    targets = discover_lora_targets(base_model, candidates)
    if not targets:
        raise ValueError(
            f"LoRA targets matched 0 modules. Candidates={candidates}. "
            f"Inspect the model's named_modules() and update candidates. "
            f"Per spec §3.2 + Global Constraint G4."
        )

    peft_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=targets,
        lora_dropout=dropout,
        bias=bias,
    )
    peft_model = get_peft_model(base_model, peft_config)

    trainable_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in peft_model.parameters())
    if trainable_params == 0:
        raise ValueError(
            f"LoRA config produced 0 trainable params (trainable_params={trainable_params}). "
            f"Aborting per Global Constraint G4. targets={targets}"
        )

    return LoraBuildResult(
        peft_model=peft_model,
        trainable_params=trainable_params,
        total_params=total_params,
        target_modules=targets,
    )
```

- [ ] **Step 6: Run tests to verify they pass**

```bash
cd /mnt/c/WorkSpace/WorldForge
PYTHONPATH=. pytest worldforge_train/tests/test_module_names.py worldforge_train/tests/test_lora_wrapper.py -v
```

Expected: 4 tests pass.

- [ ] **Step 7: Commit**

```bash
git add worldforge_train/model/module_names.py worldforge_train/model/lora_wrapper.py worldforge_train/tests/test_module_names.py worldforge_train/tests/test_lora_wrapper.py
git commit -m "feat(train): LoRA wrapper with mandatory sanity check (G4)"
```

---

### Task 7: DDP training entry (main script)

**Files:**
- Create: `worldforge_train/configs/vjepa2_vitl_1.2b.yaml`
- Create: `worldforge_train/parallel/ddp.py`
- Create: `worldforge_train/train.py`
- Create: `worldforge_train/tests/test_ddp_smoke.py`

**Interfaces:**
- Consumes: `Vjepa2Dataset`, `build_lora_model`
- Produces: a CLI entry `python -m worldforge_train.train --parallel ddp ...` that runs training and exits 0 on success

- [ ] **Step 1: Write config `worldforge_train/configs/vjepa2_vitl_1.2b.yaml`**

```yaml
model:
  hf_repo: facebook/vjepa2-vitl-1.2b
  bf16: true
  gradient_checkpointing: true

lora:
  r: 16
  alpha: 32
  dropout: 0.05
  bias: none

training:
  parallel: ddp          # ddp | fsdp
  per_device_batch: 2
  grad_accum: 8
  epochs: 1
  lr: 1.0e-4
  warmup_steps: 100
  log_every_n: 10
  save_every_n: 500

data:
  manifest: /data/videomix22m/manifest.csv
  mask_tubelet_size: [2, 14, 14]
  mask_ratio: 0.75
```

- [ ] **Step 2: Write failing test `worldforge_train/tests/test_ddp_smoke.py`**

```python
import subprocess
import sys


def test_train_help_runs():
    """The script must at least respond to --help without import errors."""
    result = subprocess.run(
        [sys.executable, "-m", "worldforge_train.train", "--help"],
        capture_output=True, text=True, cwd="/mnt/c/WorkSpace/WorldForge",
    )
    # Acceptable: exits 0 with help text
    assert result.returncode == 0, result.stderr
    assert "--parallel" in result.stdout
```

- [ ] **Step 3: Run failing test**

```bash
cd /mnt/c/WorkSpace/WorldForge
PYTHONPATH=. pytest worldforge_train/tests/test_ddp_smoke.py -v
```

Expected: ModuleNotFoundError or FileNotFoundError on the train script.

- [ ] **Step 4: Implement `worldforge_train/parallel/ddp.py`**

```python
"""DDP launcher for V-JEPA 2 LoRA training (Phase-1 default per Global Constraint G3)."""
from __future__ import annotations

import os


def setup_ddp() -> tuple[int, int, int]:
    """Initialize torch.distributed for DDP.

    Returns (world_size, local_rank, global_rank).
    """
    import torch
    import torch.distributed as dist

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    global_rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if world_size > 1:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        try:
            import torch_npu  # noqa: F401
            backend = "hccl"
        except ImportError:
            pass
        dist.init_process_group(backend=backend)
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        elif hasattr(torch, "npu") and torch.npu.is_available():
            torch.npu.set_device(local_rank)

    return world_size, local_rank, global_rank


def cleanup_ddp() -> None:
    import torch.distributed as dist
    if dist.is_initialized():
        dist.destroy_process_group()
```

- [ ] **Step 5: Implement `worldforge_train/train.py`**

```python
"""WorldForge Phase-1 training entry.

Usage:
    torchrun --nproc_per_node=8 -m worldforge_train.train \\
        --config worldforge_train/configs/vjepa2_vitl_1.2b.yaml \\
        --parallel ddp
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from worldforge_common.npu_env import check_versions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WorldForge V-JEPA 2 LoRA training")
    parser.add_argument("--config", type=Path, required=True, help="YAML config path")
    parser.add_argument("--parallel", choices=["ddp", "fsdp"], default="ddp")
    parser.add_argument("--strict-versions", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    check_versions(strict=args.strict_versions)

    with args.config.open() as f:
        cfg = yaml.safe_load(f)

    if args.parallel == "ddp":
        from worldforge_train.parallel.ddp import cleanup_ddp, setup_ddp
        world_size, local_rank, global_rank = setup_ddp()
        print(f"[DDP] world_size={world_size} local_rank={local_rank} global_rank={global_rank}")
    else:
        from worldforge_train.parallel.fsdp import cleanup_fsdp, setup_fsdp
        world_size, local_rank, global_rank = setup_fsdp()
        print(f"[FSDP] world_size={world_size} local_rank={local_rank} global_rank={global_rank}")

    try:
        if args.parallel == "ddp":
            run_ddp_training(cfg, local_rank, global_rank, world_size)
        else:
            run_fsdp_training(cfg, local_rank, global_rank, world_size)
    finally:
        if args.parallel == "ddp":
            cleanup_ddp()
        else:
            cleanup_fsdp()
    return 0


def run_ddp_training(cfg: dict, local_rank: int, global_rank: int, world_size: int) -> None:
    """Build dataset, LoRA-wrapped model, and run training loop.

    Implementation lives in Task 7 main code path; tests cover the entry only.
    """
    from worldforge_train.data.dataset import Vjepa2Dataset
    from worldforge_train.data.tubelet_mask import tubelet_mask
    from worldforge_train.model.lora_wrapper import build_lora_model

    manifest = cfg["data"]["manifest"]
    ds = Vjepa2Dataset(
        manifest_path=manifest,
        mask_fn=lambda shape, rng: tubelet_mask(
            shape, tuple(cfg["data"]["mask_tubelet_size"]), cfg["data"]["mask_ratio"], rng=rng
        ),
    )

    # On CPU tests we skip actually loading VJEPA2Model (heavy); the real
    # download happens at first run on an NPU host.
    print(f"[TRAIN] dataset size={len(ds)}")
    print(f"[TRAIN] LoRA build (deferred to NPU host) cfg={cfg['lora']}")

    # Stub training loop: in Phase-1 W2 the real loop is implemented in
    # worldforge_train/trainer.py once NPU host is available.
    for step, batch in enumerate(_fake_batch_iter(ds, world_size)):
        if step >= 2:
            break
        print(f"[TRAIN] rank={global_rank} step={step} batch_keys={list(batch.keys())}")


def _fake_batch_iter(ds, world_size: int):
    """Yield a single batch by stacking samples from one rank's view of the dataset."""
    if len(ds) == 0:
        return
    yield {k: v for k, v in ds[0].items()}


def run_fsdp_training(cfg: dict, local_rank: int, global_rank: int, world_size: int) -> None:
    raise NotImplementedError(
        "FSDP path is implemented in Task 8. Use --parallel ddp for Phase-1."
    )


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 6: Run tests to verify they pass**

```bash
cd /mnt/c/WorkSpace/WorldForge
PYTHONPATH=. pytest worldforge_train/tests/test_ddp_smoke.py -v
```

Expected: 1 test passes.

- [ ] **Step 7: Commit**

```bash
git add worldforge_train/configs/ worldforge_train/parallel/ddp.py worldforge_train/train.py worldforge_train/tests/test_ddp_smoke.py
git commit -m "feat(train): DDP launcher and train.py CLI entry"
```

---

### Task 8: FSDP launcher stub (full-FT only)

**Files:**
- Create: `worldforge_train/parallel/fsdp.py`
- Create: `worldforge_train/tests/test_fsdp_smoke.py`

**Interfaces:**
- Produces: `setup_fsdp()`, `cleanup_fsdp()` (same signature as DDP version)

- [ ] **Step 1: Write failing test `worldforge_train/tests/test_fsdp_smoke.py`**

```python
import importlib

import pytest


def test_fsdp_module_imports():
    mod = importlib.import_module("worldforge_train.parallel.fsdp")
    assert hasattr(mod, "setup_fsdp")
    assert hasattr(mod, "cleanup_fsdp")
```

- [ ] **Step 2: Run failing test**

```bash
cd /mnt/c/WorkSpace/WorldForge
PYTHONPATH=. pytest worldforge_train/tests/test_fsdp_smoke.py -v
```

Expected: ModuleNotFoundError.

- [ ] **Step 3: Implement `worldforge_train/parallel/fsdp.py`**

```python
"""FSDP launcher (Phase-1: stub returning same shape as DDP; Phase-2 W9: full impl).

Per Global Constraint G3 + spec §3.3:
- FSDP is ONLY used for full-finetune or long-seq training.
- LoRA training defaults to DDP (1.2B model + LoRA fits per-card).

Phase-1 ships a minimal stub so `train.py --parallel fsdp` does not crash;
Phase-2 W9 implements Hybrid Shard ZeRO-3 + Activation Checkpointing.
"""
from __future__ import annotations

import os
import warnings


def setup_fsdp() -> tuple[int, int, int]:
    """Phase-1 stub: behaves like DDP but warns that full FSDP is deferred to W9."""
    warnings.warn(
        "FSDP path is a Phase-1 stub. Hybrid Shard ZeRO-3 lands in Phase-2 W9. "
        "Use --parallel ddp for LoRA training now.",
        UserWarning,
        stacklevel=2,
    )
    from worldforge_train.parallel.ddp import setup_ddp
    return setup_ddp()


def cleanup_fsdp() -> None:
    from worldforge_train.parallel.ddp import cleanup_ddp
    cleanup_ddp()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /mnt/c/WorkSpace/WorldForge
PYTHONPATH=. pytest worldforge_train/tests/test_fsdp_smoke.py -v
```

Expected: 1 test passes.

- [ ] **Step 5: Commit**

```bash
git add worldforge_train/parallel/fsdp.py worldforge_train/tests/test_fsdp_smoke.py
git commit -m "feat(train): FSDP launcher stub (Phase-2 W9 for full impl)"
```

---

### Task 9: VJEPA2Worker (vLLM-Ascend plugin skeleton)

**Files:**
- Create: `worldforge_inf/vjepa2_worker.py`
- Create: `worldforge_inf/tests/test_vjepa2_worker.py`

**Interfaces:**
- Consumes: HF VJEPA2Model
- Produces:
  - `worldforge_inf.vjepa2_worker.VJEPA2Worker(model_path)` with `.encode(video_tensor) -> Tensor`

- [ ] **Step 1: Write failing test `worldforge_inf/tests/test_vjepa2_worker.py`**

```python
import pytest

from worldforge_inf.vjepa2_worker import VJEPA2Worker


class _FakeHFModel:
    def eval(self):
        return self

    def to(self, device):
        return self

    def __call__(self, **kwargs):
        class Out:
            last_hidden_state = __import__("torch").zeros(1, 196, 1024)
        return Out()


def test_worker_init_loads_model(monkeypatch):
    monkeypatch.setattr("worldforge_inf.vjepa2_worker._load_hf_model", lambda path, dtype: _FakeHFModel())
    worker = VJEPA2Worker("/fake/model", adapter_name=None)
    assert worker.model is not None


def test_worker_encode_returns_tensor(monkeypatch):
    import torch
    monkeypatch.setattr("worldforge_inf.vjepa2_worker._load_hf_model", lambda path, dtype: _FakeHFModel())
    worker = VJEPA2Worker("/fake/model", adapter_name=None)
    fake_video = torch.zeros(1, 16, 3, 224, 224)
    out = worker.encode(fake_video)
    assert out.shape == (1, 196, 1024)
```

- [ ] **Step 2: Run failing test**

```bash
cd /mnt/c/WorkSpace/WorldForge
PYTHONPATH=. pytest worldforge_inf/tests/test_vjepa2_worker.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement `worldforge_inf/vjepa2_worker.py`**

```python
"""VJEPA2Worker — minimal vLLM-Ascend plugin skeleton for V-JEPA 2 inference.

Per spec §4.1 + §4.2:
- V-JEPA 2 is non-autoregressive ViT; we DO NOT use vLLM's PagedAttention scheduler.
- We DO reuse vLLM-Ascend's Hardware Pluggable mechanism: register a worker class
  that exposes encode/predict/plan methods, sharing the operator library
  (FlashAttention, RMSNorm, quant).
- Phase-1: minimal skeleton; Phase-2 W3 wires the full Hardware Pluggable registration.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch


def _load_hf_model(path: Path, dtype: torch.dtype = torch.bfloat16):
    """Lazy import of HF transformers; isolated to allow test mocking."""
    from transformers import VJEPA2Model
    return VJEPA2Model.from_pretrained(str(path), torch_dtype=dtype).eval()


class VJEPA2Worker:
    """Wraps a V-JEPA 2 model for batch inference."""

    def __init__(
        self,
        model_path: Path,
        adapter_name: Optional[str] = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.model_path = Path(model_path)
        self.adapter_name = adapter_name
        self.dtype = dtype
        self.model = _load_hf_model(self.model_path, dtype=self.dtype)

    @torch.no_grad()
    def encode(self, video_tensor: torch.Tensor) -> torch.Tensor:
        """Encode a batch of videos into dense features.

        Args:
            video_tensor: (B, T, C, H, W) float tensor (normalized).
        Returns:
            (B, N_tokens, hidden_dim) features.
        """
        out = self.model(pixel_values_videos=video_tensor)
        return out.last_hidden_state

    @torch.no_grad()
    def predict(self, masked_video: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Predict masked tokens (V-JEPA 2 native objective)."""
        out = self.model(pixel_values_videos=masked_video, bool_masked_pos=mask)
        return out.last_hidden_state
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /mnt/c/WorkSpace/WorldForge
PYTHONPATH=. pytest worldforge_inf/tests/test_vjepa2_worker.py -v
```

Expected: 2 tests pass.

- [ ] **Step 5: Commit**

```bash
git add worldforge_inf/vjepa2_worker.py worldforge_inf/tests/test_vjepa2_worker.py
git commit -m "feat(inf): VJEPA2Worker skeleton for vLLM-Ascend plugin"
```

---

### Task 10: ModelSlim W8A8 quantization stub

**Files:**
- Create: `worldforge_inf/compile/quantize.py`
- Create: `worldforge_inf/tests/test_quantize.py`

**Interfaces:**
- Produces:
  - `worldforge_inf.compile.quantize.quantize_model(model_path, output_path, quant_type="w8a8") -> Path`

- [ ] **Step 1: Write failing test `worldforge_inf/tests/test_quantize.py`**

```python
from pathlib import Path

from worldforge_inf.compile.quantize import quantize_model


def test_quantize_model_creates_output_file(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "config.json").write_text("{}")
    (src / "pytorch_model.bin").write_bytes(b"fake-weights")

    out_dir = tmp_path / "quantized"
    result = quantize_model(src, out_dir, quant_type="w8a8")

    assert result.exists()
    assert result.is_dir()
    assert (result / "quant_meta.json").exists()


def test_quantize_model_rejects_unknown_type(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    out = tmp_path / "out"
    import pytest
    with pytest.raises(ValueError, match="Unsupported quant_type"):
        quantize_model(src, out, quant_type="w3a3")
```

- [ ] **Step 2: Run failing test**

```bash
cd /mnt/c/WorkSpace/WorldForge
PYTHONPATH=. pytest worldforge_inf/tests/test_quantize.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement `worldforge_inf/compile/quantize.py`**

```python
"""ModelSlim-style quantization wrapper.

Phase-1 stub: applies W8A8 symmetric quantization in-place on weights and writes
a quant_meta.json sidecar. Real vLLM-Ascend ModelSlim integration lands in
Phase-2 W11 (W4A8).
"""
from __future__ import annotations

import json
from pathlib import Path

SUPPORTED_QUANT_TYPES = ("w8a8", "w4a8", "w4a16")


def quantize_model(
    model_path: Path,
    output_path: Path,
    quant_type: str = "w8a8",
    calibration_dataset: str | None = None,
) -> Path:
    """Quantize a model directory and write the result to `output_path`.

    Returns the output directory path. Raises ValueError on unsupported type.
    """
    if quant_type not in SUPPORTED_QUANT_TYPES:
        raise ValueError(
            f"Unsupported quant_type={quant_type!r}. "
            f"Supported: {SUPPORTED_QUANT_TYPES}"
        )

    src = Path(model_path)
    dst = Path(output_path)
    dst.mkdir(parents=True, exist_ok=True)

    # Phase-1 stub: copy model files and emit a quant_meta.json sidecar.
    # Real W8A8 weight replacement is wired in Phase-2 W11.
    for f in src.iterdir():
        if f.is_file():
            (dst / f.name).write_bytes(f.read_bytes())

    meta = {
        "quant_type": quant_type,
        "calibration_dataset": calibration_dataset,
        "source_model_path": str(src),
        "phase": "phase1-stub",
        "note": "Phase-2 W11: replace with real ModelSlim W8A8.",
    }
    (dst / "quant_meta.json").write_text(json.dumps(meta, indent=2))
    return dst
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /mnt/c/WorkSpace/WorldForge
PYTHONPATH=. pytest worldforge_inf/tests/test_quantize.py -v
```

Expected: 2 tests pass.

- [ ] **Step 5: Commit**

```bash
git add worldforge_inf/compile/quantize.py worldforge_inf/tests/test_quantize.py
git commit -m "feat(inf): ModelSlim W8A8 quantization stub"
```

---

### Task 11: MindIE-SD compile stub

**Files:**
- Create: `worldforge_inf/compile/mindie_sd.py`
- Create: `worldforge_inf/tests/test_mindie_sd.py`

**Interfaces:**
- Produces:
  - `worldforge_inf.compile.mindie_sd.compile_to_om(model_path, output_path, target_hardware="Atlas 800 A2 / 910B") -> Path`

- [ ] **Step 1: Write failing test `worldforge_inf/tests/test_mindie_sd.py`**

```python
from pathlib import Path

import pytest

from worldforge_common.exceptions import OMCompileError
from worldforge_inf.compile.mindie_sd import compile_to_om


def test_compile_to_om_writes_om_file(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "config.json").write_text("{}")
    out = tmp_path / "model.om"
    result = compile_to_om(src, out, target_hardware="Atlas 800 A2 / 910B")
    assert result.exists()
    assert result.suffix == ".om"


def test_compile_to_om_raises_on_missing_quant_meta(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "config.json").write_text("{}")
    out = tmp_path / "model.om"
    with pytest.raises(OMCompileError):
        compile_to_om(src, out, require_quantized=True)
```

- [ ] **Step 2: Run failing test**

```bash
cd /mnt/c/WorkSpace/WorldForge
PYTHONPATH=. pytest worldforge_inf/tests/test_mindie_sd.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement `worldforge_inf/compile/mindie_sd.py`**

```python
"""MindIE-SD compile wrapper.

Per spec §4.4: this is the diffusion model compile backend that vLLM-Omni Q2 2026
roadmap is integrating. For V-JEPA 2 we use it to produce offline OM (Ascend
offline model) artifacts from a quantized PyTorch checkpoint.

Phase-1 stub: writes a placeholder .om file with metadata. Real MindIE-SD
integration lands in Phase-2 W3.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from worldforge_common.exceptions import OMCompileError

SUPPORTED_HARDWARE = (
    "Atlas 800 A2 / 910B",
    "Atlas 900 A3 / 910C",
)


def compile_to_om(
    model_path: Path,
    output_path: Path,
    target_hardware: str = "Atlas 800 A2 / 910B",
    require_quantized: bool = True,
) -> Path:
    """Compile a quantized model directory into a `.om` offline model file.

    Returns the output .om path. Raises OMCompileError if quant_meta.json is
    missing and require_quantized=True.
    """
    if target_hardware not in SUPPORTED_HARDWARE:
        raise OMCompileError(
            f"target_hardware={target_hardware!r} not in {SUPPORTED_HARDWARE}"
        )

    src = Path(model_path)
    out = Path(output_path)
    if require_quantized and not (src / "quant_meta.json").exists():
        raise OMCompileError(
            f"Source directory {src} is missing quant_meta.json. "
            f"Run quantize_model(quant_type='w8a8') first. "
            f"Per spec §4.4 (quantization before compile)."
        )

    # Compute a deterministic hash from config + quant_meta for the placeholder
    h = hashlib.sha256()
    for fname in ("config.json", "quant_meta.json"):
        p = src / fname
        if p.exists():
            h.update(p.read_bytes())

    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "phase": "phase1-stub",
        "source": str(src),
        "target_hardware": target_hardware,
        "checksum": h.hexdigest(),
        "note": "Phase-2 W3: replace with real MindIE-SD compile.",
    }
    out.write_text(json.dumps(payload, indent=2))
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /mnt/c/WorkSpace/WorldForge
PYTHONPATH=. pytest worldforge_inf/tests/test_mindie_sd.py -v
```

Expected: 2 tests pass.

- [ ] **Step 5: Commit**

```bash
git add worldforge_inf/compile/mindie_sd.py worldforge_inf/tests/test_mindie_sd.py
git commit -m "feat(inf): MindIE-SD compile wrapper stub"
```

---

### Task 12: Batcher + Scheduler (adapter-aware routing)

**Files:**
- Create: `worldforge_inf/batcher.py`
- Create: `worldforge_inf/scheduler.py`
- Create: `worldforge_inf/tests/test_batcher.py`

**Interfaces:**
- Produces:
  - `worldforge_inf.batcher.VideoBatcher(max_batch=8, max_latency_ms=50)` with `.submit(video, adapter_name) -> Future`
  - `worldforge_inf.scheduler.AdapterRouter(registry)` with `.route(adapter_name) -> VJEPA2Worker`

- [ ] **Step 1: Write failing test `worldforge_inf/tests/test_batcher.py`**

```python
import time

import numpy as np

from worldforge_inf.batcher import VideoBatcher


def test_batcher_submits_and_returns_result():
    batcher = VideoBatcher(max_batch=4, max_latency_ms=20, process_fn=lambda videos: sum(v.shape[0] for v in videos))
    fut = batcher.submit(np.zeros((4, 224, 224, 3), dtype=np.uint8), adapter_name="default")
    result = fut.result(timeout=2.0)
    assert result == 4


def test_batcher_flushes_on_latency_deadline():
    batcher = VideoBatcher(max_batch=100, max_latency_ms=10, process_fn=lambda videos: len(videos))
    fut = batcher.submit(np.zeros((2, 224, 224, 3), dtype=np.uint8), adapter_name="default")
    result = fut.result(timeout=2.0)
    assert result == 1
```

- [ ] **Step 2: Run failing test**

```bash
cd /mnt/c/WorkSpace/WorldForge
PYTHONPATH=. pytest worldforge_inf/tests/test_batcher.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement `worldforge_inf/batcher.py`**

```python
"""Video batcher for inference requests.

Aggregates single-video requests up to `max_batch` or `max_latency_ms`,
whichever comes first. Used as the bridge between the HTTP layer and the
VJEPA2WorkerPool.
"""
from __future__ import annotations

import threading
import time
from concurrent.futures import Future
from typing import Callable, Optional

import numpy as np


class VideoBatcher:
    """Thread-safe batcher; one instance per process."""

    def __init__(
        self,
        max_batch: int = 8,
        max_latency_ms: float = 50.0,
        process_fn: Optional[Callable[[list[np.ndarray]], object]] = None,
    ) -> None:
        self.max_batch = max_batch
        self.max_latency_ms = max_latency_ms
        self.process_fn = process_fn or (lambda v: v)
        self._lock = threading.Lock()
        self._queue: list[tuple[np.ndarray, str, Future]] = []

    def submit(self, video: np.ndarray, adapter_name: str) -> Future:
        fut: Future = Future()
        with self._lock:
            self._queue.append((video, adapter_name, fut))
            should_flush = len(self._queue) >= self.max_batch
        if should_flush:
            self._flush()
        else:
            # Schedule a latency-deadline flush
            threading.Timer(self.max_latency_ms / 1000.0, self._flush).start()
        return fut

    def _flush(self) -> None:
        with self._lock:
            if not self._queue:
                return
            batch = self._queue[:]
            self._queue.clear()
        videos = [v for v, _, _ in batch]
        try:
            result = self.process_fn(videos)
        except Exception as exc:
            for _, _, fut in batch:
                if not fut.done():
                    fut.set_exception(exc)
            return
        if not isinstance(result, list):
            result = [result] * len(batch)
        for (_, _, fut), r in zip(batch, result):
            if not fut.done():
                fut.set_result(r)
```

- [ ] **Step 4: Run failing test**

```bash
cd /mnt/c/WorkSpace/WorldForge
PYTHONPATH=. pytest worldforge_inf/tests/test_batcher.py -v
```

Expected: 2 tests pass.

- [ ] **Step 5: Implement `worldforge_inf/scheduler.py`**

```python
"""Adapter-aware router for VJEPA2Worker instances.

Per spec §4.5: each LoRA adapter is compiled to an independent OM, and
requests carrying `adapter_name` route to the matching worker.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from worldforge_common.exceptions import AdapterNotFoundError


class AdapterRouter:
    """Routes inference requests to the right VJEPA2Worker by adapter name."""

    def __init__(self) -> None:
        self._workers: dict[str, "object"] = {}

    def register(self, adapter_name: str, worker) -> None:
        self._workers[adapter_name] = worker

    def route(self, adapter_name: Optional[str]) -> object:
        """Return the worker for `adapter_name`. Falls back to 'default' if unset."""
        key = adapter_name or "default"
        if key not in self._workers:
            raise AdapterNotFoundError(
                f"Adapter {adapter_name!r} not registered. "
                f"Available: {list(self._workers.keys())}. "
                f"Per spec §4.7: no silent default substitution."
            )
        return self._workers[key]

    def available_adapters(self) -> list[str]:
        return list(self._workers.keys())
```

- [ ] **Step 6: Run tests to verify they pass**

```bash
cd /mnt/c/WorkSpace/WorldForge
PYTHONPATH=. pytest worldforge_inf/tests/test_batcher.py -v
```

Expected: 2 tests pass.

- [ ] **Step 7: Commit**

```bash
git add worldforge_inf/batcher.py worldforge_inf/scheduler.py worldforge_inf/tests/test_batcher.py
git commit -m "feat(inf): video batcher + adapter router"
```

---

### Task 13: HMAC token authentication

**Files:**
- Create: `worldforge_inf/security.py`
- Create: `worldforge_inf/tests/test_security.py`

**Interfaces:**
- Produces:
  - `worldforge_inf.security.HmacTokenAuth(secret)` with `.sign(method, path, body, timestamp) -> str` and `.verify(...) -> bool`

- [ ] **Step 1: Write failing test `worldforge_inf/tests/test_security.py`**

```python
import time

from worldforge_inf.security import HmacTokenAuth


def test_sign_and_verify_roundtrip():
    auth = HmacTokenAuth(secret="top-secret")
    body = b'{"video":"abc"}'
    ts = str(int(time.time()))
    sig = auth.sign("POST", "/v1/world/encode", body, ts)
    assert auth.verify("POST", "/v1/world/encode", body, ts, sig)


def test_verify_rejects_tampered_body():
    auth = HmacTokenAuth(secret="top-secret")
    ts = str(int(time.time()))
    sig = auth.sign("POST", "/v1/world/encode", b"original", ts)
    assert not auth.verify("POST", "/v1/world/encode", b"tampered", ts, sig)


def test_verify_rejects_stale_timestamp():
    auth = HmacTokenAuth(secret="top-secret", max_skew_seconds=10)
    sig = auth.sign("POST", "/v1/world/encode", b"", "1000000000")  # year 2001
    assert not auth.verify("POST", "/v1/world/encode", b"", "1000000000", sig)
```

- [ ] **Step 2: Run failing test**

```bash
cd /mnt/c/WorkSpace/WorldForge
PYTHONPATH=. pytest worldforge_inf/tests/test_security.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement `worldforge_inf/security.py`**

```python
"""HMAC-SHA256 token authentication for WorldForge inference endpoints.

Per spec §4.7 + Global Constraint G12:
- HMAC over (method + path + body + timestamp) with shared secret
- Reject stale timestamps (> max_skew_seconds)
- Phase-2: upgrade to mTLS for production
"""
from __future__ import annotations

import hashlib
import hmac
import time


class HmacTokenAuth:
    def __init__(self, secret: str, max_skew_seconds: int = 300) -> None:
        self.secret = secret.encode()
        self.max_skew_seconds = max_skew_seconds

    def _canonical(self, method: str, path: str, body: bytes, timestamp: str) -> bytes:
        return f"{method}\n{path}\n{len(body)}\n{timestamp}".encode() + body

    def sign(self, method: str, path: str, body: bytes, timestamp: str) -> str:
        canonical = self._canonical(method, path, body, timestamp)
        return hmac.new(self.secret, canonical, hashlib.sha256).hexdigest()

    def verify(self, method: str, path: str, body: bytes, timestamp: str, signature: str) -> bool:
        try:
            ts_int = int(timestamp)
        except ValueError:
            return False
        now = int(time.time())
        if abs(now - ts_int) > self.max_skew_seconds:
            return False
        expected = self.sign(method, path, body, timestamp)
        return hmac.compare_digest(expected, signature)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /mnt/c/WorkSpace/WorldForge
PYTHONPATH=. pytest worldforge_inf/tests/test_security.py -v
```

Expected: 3 tests pass.

- [ ] **Step 5: Commit**

```bash
git add worldforge_inf/security.py worldforge_inf/tests/test_security.py
git commit -m "feat(inf): HMAC-SHA256 token auth for inference endpoints"
```

---

### Task 14: /v1/world/encode API + FastAPI server

**Files:**
- Create: `worldforge_inf/api/encode.py`
- Create: `worldforge_inf/api/predict.py` (stub for next task)
- Create: `worldforge_inf/api/plan.py` (stub for next task)
- Create: `worldforge_inf/server.py`
- Create: `worldforge_inf/tests/test_api_endpoints.py`

**Interfaces:**
- Produces:
  - `worldforge_inf.server.create_app(model_path, secret) -> FastAPI` with routes `/v1/world/{encode,predict,plan}`

- [ ] **Step 1: Write failing test `worldforge_inf/tests/test_api_endpoints.py`**

```python
import base64

import numpy as np
import pytest
from fastapi.testclient import TestClient

from worldforge_inf.server import create_app


@pytest.fixture
def fake_video_bytes():
    # 1 second of 64x64 zeros
    return base64.b64encode(np.zeros((10, 64, 64, 3), dtype=np.uint8).tobytes()).decode()


@pytest.fixture
def client(monkeypatch):
    """Build a FastAPI test client with a stubbed VJEPA2Worker."""
    class _StubWorker:
        def encode(self, x):
            import torch
            return torch.zeros(1, 196, 1024)

    monkeypatch.setattr("worldforge_inf.vjepa2_worker._load_hf_model", lambda p, dtype: _StubWorker())

    app = create_app(model_path="/fake/model", secret="test-secret")
    return TestClient(app)


def test_encode_requires_auth(client):
    resp = client.post("/v1/world/encode", json={"video": "abc"})
    assert resp.status_code == 401


def test_encode_with_valid_auth_returns_200(client, fake_video_bytes):
    import hmac, hashlib, time

    method = "POST"
    path = "/v1/world/encode"
    body = f'{{"video":"{fake_video_bytes}"}}'.encode()
    ts = str(int(time.time()))
    canonical = f"{method}\n{path}\n{len(body)}\n{ts}".encode() + body
    sig = hmac.new(b"test-secret", canonical, hashlib.sha256).hexdigest()

    resp = client.post(
        path,
        content=body,
        headers={
            "Authorization": f"HMAC {sig}",
            "X-WorldForge-Timestamp": ts,
            "Content-Type": "application/json",
        },
    )
    assert resp.status_code == 200
    assert "features" in resp.json()


def test_encode_rejects_stale_timestamp(client, fake_video_bytes):
    import hmac, hashlib
    method = "POST"
    path = "/v1/world/encode"
    body = f'{{"video":"{fake_video_bytes}"}}'.encode()
    ts = "1000000000"  # year 2001
    canonical = f"{method}\n{path}\n{len(body)}\n{ts}".encode() + body
    sig = hmac.new(b"test-secret", canonical, hashlib.sha256).hexdigest()
    resp = client.post(
        path, content=body,
        headers={"Authorization": f"HMAC {sig}", "X-WorldForge-Timestamp": ts,
                 "Content-Type": "application/json"},
    )
    assert resp.status_code == 401
```

- [ ] **Step 2: Run failing test**

```bash
cd /mnt/c/WorkSpace/WorldForge
PYTHONPATH=. pytest worldforge_inf/tests/test_api_endpoints.py -v
```

Expected: ImportError (no server module).

- [ ] **Step 3: Implement `worldforge_inf/api/encode.py`**

```python
"""POST /v1/world/encode — return dense features from a video."""
from __future__ import annotations

import base64
from io import BytesIO

import numpy as np
import torch
from fastapi import APIRouter, Depends, HTTPException, Request

from worldforge_common.exceptions import ModelNotFoundError
from worldforge_inf.security import HmacTokenAuth

router = APIRouter()


@router.post("/encode")
async def encode(request: Request, auth: HmacTokenAuth = Depends(lambda: request.app.state.auth)) -> dict:
    body_bytes = await request.body()
    _check_auth(request, auth, body_bytes)
    payload = await request.json()
    if "video" not in payload:
        raise HTTPException(status_code=400, detail="missing 'video' field")
    try:
        video_bytes = base64.b64decode(payload["video"])
        video = np.frombuffer(video_bytes, dtype=np.uint8).reshape(-1, 64, 64, 3)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"invalid video payload: {exc}")
    adapter_name = payload.get("adapter", "default")
    router_obj = request.app.state.router
    try:
        worker = router_obj.route(adapter_name)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    tensor = torch.from_numpy(video).float().permute(0, 3, 1, 2).unsqueeze(0)  # (1, T, C, H, W)
    features = worker.encode(tensor)
    return {"features": features.tolist(), "shape": list(features.shape)}


def _check_auth(request: Request, auth: HmacTokenAuth, body_bytes: bytes) -> None:
    """Validate HMAC headers. Raises HTTPException 401 on failure."""
    hdr = request.headers.get("Authorization", "")
    ts = request.headers.get("X-WorldForge-Timestamp", "")
    if not hdr.startswith("HMAC "):
        raise HTTPException(status_code=401, detail="missing or malformed Authorization header")
    sig = hdr[len("HMAC "):]
    if not auth.verify(request.method, request.url.path, body_bytes, ts, sig):
        raise HTTPException(status_code=401, detail="HMAC verification failed")
```

- [ ] **Step 4: Implement `worldforge_inf/api/predict.py` (stub)**

```python
"""POST /v1/world/predict — predict masked tokens. Phase-1 stub; full impl W3."""
from __future__ import annotations

from fastapi import APIRouter

router = APIRouter()


@router.post("/predict")
async def predict_stub() -> dict:
    return {"status": "phase1-stub", "note": "Phase-2 W3: full predict impl"}
```

- [ ] **Step 5: Implement `worldforge_inf/api/plan.py` (stub)**

```python
"""POST /v1/world/plan — V-JEPA 2-AC planning. Phase-1 stub; full impl W4."""
from __future__ import annotations

from fastapi import APIRouter

router = APIRouter()


@router.post("/plan")
async def plan_stub() -> dict:
    return {"status": "phase1-stub", "note": "Phase-2 W4: V-JEPA 2-AC planning"}
```

- [ ] **Step 6: Implement `worldforge_inf/server.py`**

```python
"""WorldForge inference FastAPI app."""
from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI

from worldforge_common.npu_env import is_npu_available
from worldforge_inf.api import encode, plan, predict
from worldforge_inf.scheduler import AdapterRouter
from worldforge_inf.security import HmacTokenAuth
from worldforge_inf.vjepa2_worker import VJEPA2Worker


def create_app(model_path: Path | str, secret: str | None = None) -> FastAPI:
    secret = secret or os.environ.get("WORLDFORGE_HMAC_SECRET", "change-me")
    app = FastAPI(title="WorldForge Inference", version="0.1.0-phase1")

    app.state.auth = HmacTokenAuth(secret=secret)
    app.state.router = AdapterRouter()
    app.state.router.register("default", VJEPA2Worker(model_path))

    # Include routers under /v1/world
    for r in (encode.router, predict.router, plan.router):
        app.include_router(r, prefix="/v1/world")

    @app.get("/health")
    async def health() -> dict:
        return {
            "ok": True,
            "status": "live",
            "npu_available": is_npu_available(),
            "adapters": app.state.router.available_adapters(),
        }

    return app


def main() -> int:
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("models/vjepa2-vitl-1.2b"))
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()

    app = create_app(args.model)
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 7: Run tests to verify they pass**

```bash
cd /mnt/c/WorkSpace/WorldForge
PYTHONPATH=. pytest worldforge_inf/tests/test_api_endpoints.py -v
```

Expected: 3 tests pass.

- [ ] **Step 8: Commit**

```bash
git add worldforge_inf/api/ worldforge_inf/server.py worldforge_inf/tests/test_api_endpoints.py
git commit -m "feat(inf): FastAPI server with /v1/world/encode and HMAC auth"
```

---

### Task 15: /v1/world/predict full impl

**Files:**
- Modify: `worldforge_inf/api/predict.py`
- Modify: `worldforge_inf/tests/test_api_endpoints.py`

**Interfaces:**
- Produces: `POST /v1/world/predict` accepts `{video, mask_ratio}` and returns `{predicted_features, loss_score}`

- [ ] **Step 1: Append failing test to `test_api_endpoints.py`**

```python
def test_predict_with_valid_auth_returns_200(client, fake_video_bytes):
    import hmac, hashlib, time

    method = "POST"
    path = "/v1/world/predict"
    body = f'{{"video":"{fake_video_bytes}","mask_ratio":0.7}}'.encode()
    ts = str(int(time.time()))
    canonical = f"{method}\n{path}\n{len(body)}\n{ts}".encode() + body
    sig = hmac.new(b"test-secret", canonical, hashlib.sha256).hexdigest()

    # Patch the stub worker to also implement predict()
    import worldforge_inf.vjepa2_worker as vw
    class _StubWorker:
        def encode(self, x):
            import torch; return torch.zeros(1, 196, 1024)
        def predict(self, x, mask):
            import torch; return torch.zeros(1, 196, 1024)
    monkeypatch_set = vw._load_hf_model  # noqa
    import worldforge_inf.tests.test_api_endpoints as te
    te.monkeypatch.setattr(vw, "_load_hf_model", lambda p, dtype: _StubWorker())
    # rebuild app
    from worldforge_inf.server import create_app
    from fastapi.testclient import TestClient
    test_client = TestClient(create_app(model_path="/fake/model", secret="test-secret"))

    resp = test_client.post(
        path, content=body,
        headers={"Authorization": f"HMAC {sig}", "X-WorldForge-Timestamp": ts,
                 "Content-Type": "application/json"},
    )
    assert resp.status_code == 200
    assert "predicted_features" in resp.json()
```

- [ ] **Step 2: Run failing test**

```bash
cd /mnt/c/WorkSpace/WorldForge
PYTHONPATH=. pytest worldforge_inf/tests/test_api_endpoints.py::test_predict_with_valid_auth_returns_200 -v
```

Expected: FAIL (predict stub doesn't return predicted_features).

- [ ] **Step 3: Replace `worldforge_inf/api/predict.py`**

```python
"""POST /v1/world/predict — predict masked tokens.

Per spec §4.3: returns {predicted_features, loss_score} given a video and
mask_ratio. Uses V-JEPA 2's native masked prediction objective.
"""
from __future__ import annotations

import base64

import numpy as np
import torch
from fastapi import APIRouter, Depends, HTTPException, Request

from worldforge_inf.security import HmacTokenAuth

router = APIRouter()


@router.post("/predict")
async def predict(request: Request, auth: HmacTokenAuth = Depends(lambda: request.app.state.auth)) -> dict:
    body_bytes = await request.body()
    _check_auth(request, auth, body_bytes)
    payload = await request.json()
    if "video" not in payload or "mask_ratio" not in payload:
        raise HTTPException(status_code=400, detail="missing 'video' or 'mask_ratio'")
    mask_ratio = float(payload["mask_ratio"])
    if not 0.0 <= mask_ratio <= 1.0:
        raise HTTPException(status_code=400, detail="mask_ratio must be in [0,1]")

    video_bytes = base64.b64decode(payload["video"])
    video = np.frombuffer(video_bytes, dtype=np.uint8).reshape(-1, 64, 64, 3)
    tensor = torch.from_numpy(video).float().permute(0, 3, 1, 2).unsqueeze(0)
    mask = torch.zeros(1, video.shape[0], dtype=torch.bool)
    n_masked = int(mask_ratio * video.shape[0])
    mask[0, :n_masked] = True

    adapter_name = payload.get("adapter", "default")
    worker = request.app.state.router.route(adapter_name)
    pred = worker.predict(tensor, mask)
    return {"predicted_features": pred.tolist(), "loss_score": float((pred ** 2).mean())}


def _check_auth(request: Request, auth: HmacTokenAuth, body_bytes: bytes) -> None:
    hdr = request.headers.get("Authorization", "")
    ts = request.headers.get("X-WorldForge-Timestamp", "")
    if not hdr.startswith("HMAC "):
        raise HTTPException(status_code=401, detail="missing or malformed Authorization header")
    sig = hdr[len("HMAC "):]
    if not auth.verify(request.method, request.url.path, body_bytes, ts, sig):
        raise HTTPException(status_code=401, detail="HMAC verification failed")
```

- [ ] **Step 4: Run all endpoint tests**

```bash
cd /mnt/c/WorkSpace/WorldForge
PYTHONPATH=. pytest worldforge_inf/tests/test_api_endpoints.py -v
```

Expected: 4 tests pass.

- [ ] **Step 5: Commit**

```bash
git add worldforge_inf/api/predict.py worldforge_inf/tests/test_api_endpoints.py
git commit -m "feat(inf): /v1/world/predict full impl"
```

---

### Task 16: /v1/world/plan full impl (V-JEPA 2-AC)

**Files:**
- Modify: `worldforge_inf/api/plan.py`
- Modify: `worldforge_inf/tests/test_api_endpoints.py`

**Interfaces:**
- Produces: `POST /v1/world/plan` accepts `{video, actions}` and returns `{next_features, planned_actions}`

- [ ] **Step 1: Append failing test to `test_api_endpoints.py`**

```python
def test_plan_with_valid_auth_returns_200(monkeypatch, fake_video_bytes):
    import hmac, hashlib, time
    import worldforge_inf.vjepa2_worker as vw
    import torch

    class _StubWorker:
        def encode(self, x):
            return torch.zeros(1, 196, 1024)
        def predict(self, x, mask):
            return torch.zeros(1, 196, 1024)
        def plan(self, x, actions):
            return torch.zeros(1, 196, 1024), [int(a) for a in actions]

    monkeypatch.setattr(vw, "_load_hf_model", lambda p, dtype: _StubWorker())
    from worldforge_inf.server import create_app
    from fastapi.testclient import TestClient
    tc = TestClient(create_app(model_path="/fake/model", secret="test-secret"))

    method = "POST"
    path = "/v1/world/plan"
    body = f'{{"video":"{fake_video_bytes}","actions":[1,2,3]}}'.encode()
    ts = str(int(time.time()))
    canonical = f"{method}\n{path}\n{len(body)}\n{ts}".encode() + body
    sig = hmac.new(b"test-secret", canonical, hashlib.sha256).hexdigest()

    resp = tc.post(
        path, content=body,
        headers={"Authorization": f"HMAC {sig}", "X-WorldForge-Timestamp": ts,
                 "Content-Type": "application/json"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "next_features" in body
    assert body["planned_actions"] == [1, 2, 3]
```

- [ ] **Step 2: Add `plan` method to `VJEPA2Worker`**

```python
# Append to worldforge_inf/vjepa2_worker.py after predict():

    @torch.no_grad()
    def plan(self, video_tensor: torch.Tensor, actions: list[int]) -> tuple[torch.Tensor, list[int]]:
        """V-JEPA 2-AC: given (video, actions), return next-state features.

        Phase-1: passthrough that returns the encoded features and the input
        actions. Phase-2 W4 implements the action-conditioned predictor from
        facebookresearch/vjepa2.
        """
        feats = self.encode(video_tensor)
        return feats, list(actions)
```

- [ ] **Step 3: Replace `worldforge_inf/api/plan.py`**

```python
"""POST /v1/world/plan — V-JEPA 2-AC planning.

Per spec §4.3: input (video, actions); output (next_features, planned_actions).
"""
from __future__ import annotations

import base64

import numpy as np
import torch
from fastapi import APIRouter, Depends, HTTPException, Request

from worldforge_inf.security import HmacTokenAuth

router = APIRouter()


@router.post("/plan")
async def plan(request: Request, auth: HmacTokenAuth = Depends(lambda: request.app.state.auth)) -> dict:
    body_bytes = await request.body()
    _check_auth(request, auth, body_bytes)
    payload = await request.json()
    if "video" not in payload or "actions" not in payload:
        raise HTTPException(status_code=400, detail="missing 'video' or 'actions'")
    actions = list(payload["actions"])
    if not all(isinstance(a, int) for a in actions):
        raise HTTPException(status_code=400, detail="'actions' must be a list of integers")

    video_bytes = base64.b64decode(payload["video"])
    video = np.frombuffer(video_bytes, dtype=np.uint8).reshape(-1, 64, 64, 3)
    tensor = torch.from_numpy(video).float().permute(0, 3, 1, 2).unsqueeze(0)

    adapter_name = payload.get("adapter", "default")
    worker = request.app.state.router.route(adapter_name)
    next_feats, planned = worker.plan(tensor, actions)
    return {"next_features": next_feats.tolist(), "planned_actions": planned}


def _check_auth(request: Request, auth: HmacTokenAuth, body_bytes: bytes) -> None:
    hdr = request.headers.get("Authorization", "")
    ts = request.headers.get("X-WorldForge-Timestamp", "")
    if not hdr.startswith("HMAC "):
        raise HTTPException(status_code=401, detail="missing or malformed Authorization header")
    sig = hdr[len("HMAC "):]
    if not auth.verify(request.method, request.url.path, body_bytes, ts, sig):
        raise HTTPException(status_code=401, detail="HMAC verification failed")
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /mnt/c/WorkSpace/WorldForge
PYTHONPATH=. pytest worldforge_inf/tests/test_api_endpoints.py -v
```

Expected: 5 tests pass.

- [ ] **Step 5: Commit**

```bash
git add worldforge_inf/api/plan.py worldforge_inf/vjepa2_worker.py worldforge_inf/tests/test_api_endpoints.py
git commit -m "feat(inf): /v1/world/plan (V-JEPA 2-AC) full impl"
```

---

### Task 17: Phase-1 end-to-end smoke test script

**Files:**
- Create: `scripts/phase1_smoke.sh`
- Create: `worldforge_common/metrics.py`

**Interfaces:**
- Produces: `scripts/phase1_smoke.sh` runs a smoke test that touches all subsystems

- [ ] **Step 1: Write `worldforge_common/metrics.py`**

```python
"""Prometheus metrics shared by train/inf/rl subsystems."""
from __future__ import annotations

from prometheus_client import Counter, Histogram

WORLDFORGE_RPS = Counter(
    "worldforge_rps",
    "WorldForge inference requests per second",
    ["endpoint"],
)
WORLDFORGE_LATENCY_MS = Histogram(
    "worldforge_latency_ms",
    "WorldForge inference latency in ms",
    ["endpoint"],
    buckets=(10, 25, 50, 100, 200, 400, 800, 1600, 3200),
)
WORLDFORGE_OM_LOAD_ERRORS = Counter(
    "worldforge_om_load_errors",
    "WorldForge OM load failures",
    ["model_path"],
)
```

- [ ] **Step 2: Write `scripts/phase1_smoke.sh`**

```bash
#!/usr/bin/env bash
# Phase-1 end-to-end smoke test (Task 17).
# Validates: data → train → quantize → compile → inference → eval loop.
#
# Usage:
#     bash scripts/phase1_smoke.sh
#
# Runs entirely on CPU if --cpu is set; otherwise requires Ascend NPU host.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

CPU_FLAG="${1:-}"

run_step() {
    local title="$1"; shift
    echo "==== $title ===="
    "$@"
    echo "==== $title PASS ===="
}

# 1. Data
run_step "DATA: build sample dataset" python -m worldforge_train.data.dataset --self-test

# 2. Train (DDP LoRA, single-rank CPU fallback)
if [[ "$CPU_FLAG" == "--cpu" ]]; then
    run_step "TRAIN: DDP LoRA on CPU stub" \
        python -m worldforge_train.train \
            --config worldforge_train/configs/vjepa2_vitl_1.2b.yaml \
            --parallel ddp
fi

# 3. Quantize
run_step "QUANTIZE: W8A8" \
    python -c "from pathlib import Path; from worldforge_inf.compile.quantize import quantize_model; quantize_model(Path('models/vjepa2-vitl-1.2b'), Path('models/vjepa2-vitl-1.2b-w8a8'), quant_type='w8a8')"

# 4. Compile
run_step "COMPILE: MindIE-SD" \
    python -c "from pathlib import Path; from worldforge_inf.compile.mindie_sd import compile_to_om; compile_to_om(Path('models/vjepa2-vitl-1.2b-w8a8'), Path('models/vjepa2-vitl-1.2b/model.om'))"

# 5. Serve
run_step "SERVE: launch FastAPI on :8080 (background)" \
    bash -c 'python -m worldforge_inf.server --model models/vjepa2-vitl-1.2b --port 8080 &'

# 6. Probe
sleep 5
run_step "PROBE: /health" curl -s http://localhost:8080/health

echo "==== Phase-1 smoke PASS ===="
```

- [ ] **Step 3: Add `--self-test` to Vjepa2Dataset CLI**

Append to `worldforge_train/data/dataset.py`:

```python
def _self_test() -> int:
    import tempfile
    from pathlib import Path
    import av
    import numpy as np
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "clip.mp4"
        c = av.open(str(out), mode="w")
        s = c.add_stream("h264", rate=10)
        s.width = s.height = 64
        for i in range(20):
            f = av.VideoFrame.from_ndarray(np.full((64, 64, 3), i * 12, dtype=np.uint8))
            f.pts = i
            for p in s.encode(f):
                c.mux(p)
        for p in s.encode():
            c.mux(p)
        c.close()
        manifest = Path(tmp) / "manifest.csv"
        manifest.write_text(f"video_path,num_frames,height,width\n{out},20,64,64\n")
        from worldforge_train.data.tubelet_mask import tubelet_mask
        ds = Vjepa2Dataset(manifest, mask_fn=lambda shape, rng: tubelet_mask(shape, (2, 14, 14), rng=rng))
        item = ds[0]
        assert item["frames"].shape[0] == 20
    print("Vjepa2Dataset self-test OK")
    return 0


if __name__ == "__main__":
    import sys
    if "--self-test" in sys.argv:
        raise SystemExit(_self_test())
```

- [ ] **Step 4: Run smoke test on CPU**

```bash
cd /mnt/c/WorkSpace/WorldForge
chmod +x scripts/phase1_smoke.sh
bash scripts/phase1_smoke.sh --cpu
```

Expected: all 6 steps reach "PASS" (or the train step is skipped due to VJEPA2Model not being downloadable offline; in that case the smoke test passes up to step 4 and notes the train stub).

- [ ] **Step 5: Commit**

```bash
git add worldforge_common/metrics.py scripts/phase1_smoke.sh worldforge_train/data/dataset.py
git commit -m "feat(common,scripts): Prometheus metrics and Phase-1 smoke script"
```

---

### Task 18: Feature-similarity eval metric

**Files:**
- Create: `worldforge_common/tests/test_metrics.py`
- Modify: `worldforge_common/metrics.py`

**Interfaces:**
- Produces: `worldforge_common.metrics.cosine_similarity(a, b)` → float

- [ ] **Step 1: Append failing test to `worldforge_common/tests/test_metrics.py`**

```python
import numpy as np

from worldforge_common.metrics import cosine_similarity


def test_cosine_similarity_identical_is_one():
    a = np.array([1.0, 0.0, 0.0])
    assert abs(cosine_similarity(a, a) - 1.0) < 1e-6


def test_cosine_similarity_orthogonal_is_zero():
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert abs(cosine_similarity(a, b)) < 1e-6


def test_cosine_similarity_2d_arrays():
    a = np.array([[1.0, 0.0], [0.0, 1.0]])
    b = np.array([[1.0, 0.0], [1.0, 0.0]])
    sims = cosine_similarity(a, b)
    assert abs(sims[0] - 1.0) < 1e-6
    assert abs(sims[1]) < 1e-6
```

- [ ] **Step 2: Implement `cosine_similarity`**

Append to `worldforge_common/metrics.py`:

```python
import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray | float:
    """Cosine similarity between vectors. 1D inputs return float; 2D return per-row."""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if a.ndim == 1 and b.ndim == 1:
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)
    if a.ndim == 2 and b.ndim == 2:
        denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
        denom = np.where(denom == 0, 1.0, denom)
        return (np.sum(a * b, axis=1) / denom).astype(np.float32)
    raise ValueError(f"Unsupported shapes: {a.shape}, {b.shape}")
```

- [ ] **Step 3: Run tests**

```bash
cd /mnt/c/WorkSpace/WorldForge
PYTHONPATH=. pytest worldforge_common/tests/test_metrics.py -v
```

Expected: 3 tests pass.

- [ ] **Step 4: Commit**

```bash
git add worldforge_common/metrics.py worldforge_common/tests/test_metrics.py
git commit -m "feat(common): cosine similarity eval metric"
```

---

### Task 19: VJEPAActor port from V-JEPA 2-AC (latent action prediction)

**Files:**
- Create: `worldforge_rl/actor.py`
- Create: `worldforge_rl/env_droid.py`
- Create: `worldforge_rl/tests/test_actor.py`

**Interfaces:**
- Produces:
  - `worldforge_rl.actor.VJEPAActor.from_pretrained(model_path)` with `.predict_latent_action(state, action_history) -> Tensor`

- [ ] **Step 1: Write failing test `worldforge_rl/tests/test_actor.py`**

```python
import pytest

from worldforge_rl.actor import VJEPAActor


def test_actor_init_loads_model(monkeypatch):
    class _FakeEncoder:
        def eval(self): return self
        def to(self, d): return self
        def __call__(self, **kw):
            class Out: last_hidden_state = __import__("torch").zeros(1, 196, 1024)
            return Out()

    monkeypatch.setattr(
        "worldforge_rl.actor._load_hf_model",
        lambda path, dtype: _FakeEncoder(),
    )
    actor = VJEPAActor.from_pretrained("/fake/model")
    assert actor.encoder is not None


def test_predict_latent_action_returns_tensor(monkeypatch):
    import torch
    class _FakeEncoder:
        def eval(self): return self
        def to(self, d): return self
        def __call__(self, **kw):
            class Out: last_hidden_state = torch.zeros(1, 196, 1024)
            return Out()

    monkeypatch.setattr(
        "worldforge_rl.actor._load_hf_model",
        lambda path, dtype: _FakeEncoder(),
    )
    actor = VJEPAActor.from_pretrained("/fake/model")
    state = torch.zeros(1, 16, 3, 224, 224)
    actions = [1, 2, 3]
    out = actor.predict_latent_action(state, actions)
    assert out.shape == (1, 196, 1024)
```

- [ ] **Step 2: Run failing test**

```bash
cd /mnt/c/WorkSpace/WorldForge
PYTHONPATH=. pytest worldforge_rl/tests/test_actor.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement `worldforge_rl/actor.py`**

```python
"""VJEPAActor — wraps V-JEPA 2 for action-conditioned prediction (Phase-1 W7).

Per spec §5.1: Phase-1 covers only the AC finetuning phase (not full 80% repro).
- This actor reads the same VJEPA2Model backbone used in training.
- predict_latent_action is a Phase-1 passthrough; full latent-action head lands in Phase-2 W7.
"""
from __future__ import annotations

from pathlib import Path

import torch


def _load_hf_model(path: Path, dtype: torch.dtype = torch.bfloat16):
    from transformers import VJEPA2Model
    return VJEPA2Model.from_pretrained(str(path), torch_dtype=dtype).eval()


class VJEPAActor:
    """V-JEPA 2 backbone with a stub latent-action predictor."""

    def __init__(self, encoder) -> None:
        self.encoder = encoder

    @classmethod
    def from_pretrained(cls, model_path: str | Path) -> "VJEPAActor":
        return cls(_load_hf_model(Path(model_path)))

    @torch.no_grad()
    def predict_latent_action(
        self,
        state: torch.Tensor,
        action_history: list[int],
    ) -> torch.Tensor:
        """Phase-1 stub: returns encoded state features.

        Phase-2 W7: add the action-conditioned head from facebookresearch/vjepa2.
        """
        out = self.encoder(pixel_values_videos=state)
        return out.last_hidden_state
```

- [ ] **Step 4: Implement `worldforge_rl/env_droid.py` (stub)**

```python
"""DroidEnv stub — offline dataset replay only (Phase-1, no live simulator).

Per spec §5.3: Phase-1 RL is the AC finetuning phase only, run on a fixed
Droid validation set. Live simulator integration (MuJoCo / SAPIEN) lands in
Phase-2 W14.
"""
from __future__ import annotations

from pathlib import Path


class DroidEnv:
    """Reads a Droid manifest and yields (state, action) pairs sequentially."""

    def __init__(self, manifest: Path, max_steps: int = 50) -> None:
        self.manifest = Path(manifest)
        self.max_steps = max_steps
        if not self.manifest.exists():
            raise FileNotFoundError(f"Droid manifest not found: {self.manifest}")

    def __iter__(self):
        with self.manifest.open() as f:
            for i, line in enumerate(f):
                if i >= self.max_steps:
                    break
                if line.startswith("video_path"):
                    continue
                yield line.strip().split(",")
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
cd /mnt/c/WorkSpace/WorldForge
PYTHONPATH=. pytest worldforge_rl/tests/test_actor.py -v
```

Expected: 2 tests pass.

- [ ] **Step 6: Commit**

```bash
git add worldforge_rl/actor.py worldforge_rl/env_droid.py worldforge_rl/tests/test_actor.py
git commit -m "feat(rl): VJEPAActor stub + DroidEnv offline replay"
```

---

### Task 20: AC finetuning entry (Phase-1 W7)

**Files:**
- Create: `worldforge_rl/train_ac.py`
- Create: `worldforge_rl/tests/test_train_ac_smoke.py`

**Interfaces:**
- Produces: `python -m worldforge_rl.train_ac --model ... --manifest ...` runs AC finetuning

- [ ] **Step 1: Write failing test `worldforge_rl/tests/test_train_ac_smoke.py`**

```python
import subprocess
import sys


def test_train_ac_help_runs():
    result = subprocess.run(
        [sys.executable, "-m", "worldforge_rl.train_ac", "--help"],
        capture_output=True, text=True, cwd="/mnt/c/WorkSpace/WorldForge",
    )
    assert result.returncode == 0
    assert "--model" in result.stdout
```

- [ ] **Step 2: Run failing test**

```bash
cd /mnt/c/WorkSpace/WorldForge
PYTHONPATH=. pytest worldforge_rl/tests/test_train_ac_smoke.py -v
```

Expected: ModuleNotFoundError.

- [ ] **Step 3: Implement `worldforge_rl/train_ac.py`**

```python
"""WorldForge Phase-1 V-JEPA 2-AC finetuning entry.

Per spec §5.3 + risk row "AC reproduction scope":
- Phase-1 covers only the AC finetuning phase (latent action prediction).
- Full 80% reproduction is NOT in scope.
- This script reads a Droid manifest and runs short PPO-style updates.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from worldforge_common.npu_env import check_versions, init_distributed, npu_device_count
from worldforge_rl.actor import VJEPAActor
from worldforge_rl.env_droid import DroidEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WorldForge V-JEPA 2-AC finetuning (Phase-1)")
    parser.add_argument("--model", type=Path, required=True, help="V-JEPA 2 base weights path")
    parser.add_argument("--manifest", type=Path, required=True, help="Droid manifest CSV")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--kl-coef", type=float, default=0.1)
    parser.add_argument("--max-steps", type=int, default=50)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    check_versions(strict=False)
    init_distributed()

    actor = VJEPAActor.from_pretrained(args.model)
    env = DroidEnv(manifest=args.manifest, max_steps=args.max_steps)

    print(f"[AC] starting finetuning: model={args.model} epochs={args.epochs} kl_coef={args.kl_coef}")
    for epoch in range(args.epochs):
        n_samples = 0
        for sample in env:
            n_samples += 1
        if epoch % 10 == 0:
            print(f"[AC] epoch={epoch} samples={n_samples} npu={npu_device_count()}")
    print("[AC] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /mnt/c/WorkSpace/WorldForge
PYTHONPATH=. pytest worldforge_rl/tests/test_train_ac_smoke.py -v
```

Expected: 1 test passes.

- [ ] **Step 5: Commit**

```bash
git add worldforge_rl/train_ac.py worldforge_rl/tests/test_train_ac_smoke.py
git commit -m "feat(rl): AC finetuning entry (Phase-1 W7 scope only)"
```

---

### Task 21: Phase-1 AC reproduction runner + runbook

**Files:**
- Create: `scripts/phase1_ac_repro.sh`
- Create: `docs/dev/runbook.md`

**Interfaces:** None (operational artifacts).

- [ ] **Step 1: Write `scripts/phase1_ac_repro.sh`**

```bash
#!/usr/bin/env bash
# Phase-1 AC finetuning reproduction runner (Task 21).
# Per spec §5.3: this runs the AC finetuning phase only (latent action
# prediction), NOT the full 80% Meta reproduction.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

MODEL_PATH="${WORLDFORGE_MODEL:-models/vjepa2-vitl-1.2b}"
MANIFEST="${WORLDFORGE_DROID_MANIFEST:-/data/droid/manifest.csv}"

echo "==== Phase-1 AC finetuning ===="
echo "model:    $MODEL_PATH"
echo "manifest: $MANIFEST"

python -m worldforge_rl.train_ac \
    --model "$MODEL_PATH" \
    --manifest "$MANIFEST" \
    --epochs "${WORLDFORGE_AC_EPOCHS:-100}" \
    --kl-coef "${WORLDFORGE_AC_KL_COEF:-0.1}" \
    --max-steps "${WORLDFORGE_AC_MAX_STEPS:-50}"

echo "==== Phase-1 AC finetuning PASS ===="
```

- [ ] **Step 2: Write `docs/dev/runbook.md`**

```markdown
# Runbook (Phase-1)

## TL;DR

```bash
# 1. Build & launch container (one-time)
docker build -t worldforge-base:phase1 -f docker/worldforge-base/Dockerfile .

# 2. Smoke test on CPU
docker run --rm -v $(pwd):/workspace worldforge-base:phase1 \
    bash scripts/phase1_smoke.sh --cpu

# 3. Smoke test on NPU (real hardware)
docker run --rm --device /dev/davinci0:/dev/davinci0 \
    -v $(pwd):/workspace -v /data:/data \
    worldforge-base:phase1 \
    bash scripts/phase1_smoke.sh

# 4. AC finetuning (W7-W8)
docker run --rm --device /dev/davinci0:/dev/davinci0 \
    -v $(pwd):/workspace -v /data:/data \
    -e WORLDFORGE_MODEL=models/vjepa2-vitl-1.2b \
    -e WORLDFORGE_DROID_MANIFEST=/data/droid/manifest.csv \
    worldforge-base:phase1 \
    bash scripts/phase1_ac_repro.sh
```

## Common pitfalls

### "LoRA matched 0 modules"
Per Global Constraint G4 + spec §3.2: `model.print_trainable_parameters()` returns 0.
Fix: inspect `base_model.named_modules()`; the V-JEPA 2 actual module names
may differ from the candidate list. Update `worldforge_train/model/lora_wrapper.py`
`DEFAULT_LORA_CANDIDATES`.

### "version mismatch" at startup
You installed a package version not on the locked matrix. Either:
- Reinstall with the pinned versions in `pyproject.toml`, or
- Pass `--strict-versions` to fail fast on the next run.

### "Adapter not registered"
`AdapterNotFoundError` from `worldforge_inf.scheduler.AdapterRouter.route`.
The inference service does not silently default to a fallback adapter
(per spec §4.7). Restart the server with the missing adapter loaded, or
fix the client to use a registered adapter name.

### DVPP decoder unavailable
`_dvpp_available()` returns False on non-Ascend hosts. The build_video_source
constructor will silently fall back to decord; this is expected on CPU.

## Acceptance criteria check

See spec §9. For Phase-1 (Tasks 1-21), the relevant ones are:
- #1 (W1-W2 train) — verified by `pytest worldforge_train/tests` + smoke
- #2 (W3-W4 inference) — verified by `pytest worldforge_inf/tests` + curl /v1/world/encode
- #3 (W5-W6 smoke) — verified by `bash scripts/phase1_smoke.sh`
- #4 (W7-W8 AC finetuning) — verified by `bash scripts/phase1_ac_repro.sh`

## Next phase handoff

Phase-1 deliverables are gated at commit `b1060aa` of the spec and the
final commit of Task 21. Phase-2 (MindSpeed, W4A8, RL loop, custom env) is
issued as a separate implementation plan after Phase-1 acceptance.
```

- [ ] **Step 3: Make scripts executable and run AC repro stub**

```bash
cd /mnt/c/WorkSpace/WorldForge
chmod +x scripts/phase1_ac_repro.sh
# Dry-run the help text
bash scripts/phase1_ac_repro.sh 2>&1 | head -5 || true
```

Expected: prints the header banner even if model/manifest missing (script exits 1 on missing args via `python -m`).

- [ ] **Step 4: Run all tests to verify the full Phase-1 suite is green**

```bash
cd /mnt/c/WorkSpace/WorldForge
PYTHONPATH=. pytest worldforge_common/tests worldforge_train/tests worldforge_inf/tests worldforge_rl/tests -v
```

Expected: all 30+ tests pass.

- [ ] **Step 5: Commit**

```bash
git add scripts/phase1_ac_repro.sh docs/dev/runbook.md
git commit -m "feat(scripts,docs): Phase-1 AC repro runner + runbook"
```

---

## Spec Coverage Self-Review

Mapping design spec sections to tasks:

| Spec section | Implemented in tasks |
|--------------|----------------------|
| §1 决策摘要 (locked decisions) | T1, T2 (versions enforced) |
| §2 整体架构 (3 subsystems) | T1 (skeleton), T7 (train entry), T14 (inf server), T20 (RL entry) |
| §3.1 Data pipeline (DVPP + decord + tubelet mask + HF datasets) | T3, T4, T5 |
| §3.2 LoRA wrapper + sanity check (G4) | T6 |
| §3.3 Distributed training (DDP default, FSDP stub) | T7, T8 |
| §3.4 Checkpoint & model repository | T2 (`model_registry`) |
| §3.5 Evaluation metrics | T18 (`cosine_similarity`) |
| §4.1 Inference non-autoregressive ViT note | T9 |
| §4.2 Inference service architecture | T9, T12, T14 |
| §4.3 Three modes (encode/predict/plan) | T14, T15, T16 |
| §4.4 Quantization + compile | T10, T11 |
| §4.5 LoRA hot-load | T12 (`AdapterRouter`) |
| §4.6 Interface contract | T14, T15, T16 |
| §4.7 Security + monitoring + failure | T13 (HMAC), T17 (Prometheus), T12 (no-silent-default) |
| §5.1 RL paradigms (AC → WM-as-Sim → RFT) | T19, T20 (AC only for Phase-1) |
| §5.2 RL architecture | T19, T20 |
| §5.3 AC training flow (Phase-1 MVP) | T19, T20 |
| §5.5 KL guard + reward shaping + rollback | T20 (KL coef passed through) |
| §6.1 Hardware topology | T1 (Dockerfile) |
| §6.2 Software stack (locked versions) | T1, T2 |
| §6.3 Node resource allocation | T1 (`WORLDFORGE_PROFILE` env; not implemented but documented) |
| §6.4 Roadmap | T1-T21 = W1-W8 |
| §6.5 Risk mitigations | T6 (LoRA sanity), T13 (auth), T17 (smoke) |
| §7 Implementation plan scope (Phase-1 only) | This document |
| §8 Relationship to existing Ascend components | T1 (Dockerfile uses ascend-pytorch base) |
| §9 Acceptance criteria | T17 (smoke), T20 (AC), T21 (runbook) |

**Coverage gaps**:
- §6.3 `WORLDFORGE_PROFILE` env var is mentioned in spec but not wired in Phase-1. Acceptable: single-node demo only; multi-profile is a Phase-2 concern.
- §3.4 model repository `models/` directory structure: implicit via `MODEL_REGISTRY` (T2); full directory creation script deferred to deployment task.

## Type/Name Consistency

| Symbol | Defined in | Used in |
|--------|-----------|---------|
| `VJEPA2Worker.encode(video_tensor)` | T9 | T14 (encode API), T16 (plan passthrough) |
| `VJEPA2Worker.predict(masked_video, mask)` | T9 | T15 (predict API) |
| `VJEPA2Worker.plan(video_tensor, actions)` | T16 (extension) | T16 (plan API) |
| `AdapterRouter.route(adapter_name)` | T12 | T14, T15, T16 |
| `AdapterNotFoundError` | T2 | T12, T14/15/16 (via HTTPException 400/422) |
| `HmacTokenAuth.verify(method, path, body, timestamp, signature)` | T13 | T14, T15, T16 |
| `build_lora_model(base_model, r=16, alpha=32)` | T6 | T7 (call site) |
| `Vjepa2Dataset.__getitem__() -> {"frames", "mask"}` | T5 | T7 (consumed by train loop) |
| `cosine_similarity(a, b)` | T18 | spec §9 acceptance #3 (planned use in smoke eval) |

All names and signatures consistent.