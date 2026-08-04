# WorldFoundry

Ascend NPU world model training + inference platform. **Phase-1** prototype ships
training + inference for Meta V-JEPA 2 (1.2B).

## Prerequisites

WorldFoundry installs **only inside the Ascend NPU base image**:

```dockerfile
FROM ascend-pytorch:2.9.0-cann8.5.1-ubuntu22.04
```

The image ships `torch==2.9.0+cpu` and `torch_npu==2.9.0` together. `pip install -e .[...]`
from a stock PyPI + Python 3.11 host fails at dependency resolution (no `torch==2.9.0`
wheel for that environment). For dev/test workflows, install the optional extras from
inside the base image:

```bash
pip install -e .[dev,video]   # pytest + av + decord for video-decode tests
```

Other extras: `[inf]` (vllm-ascend), `[quant]` (msmodelslim).

See:
- `docs/superpowers/specs/2026-07-27-ascend-world-model-design.md` — design spec
- `docs/superpowers/plans/2026-07-27-worldforge-phase1.md` — Phase-1 implementation plan
- `docs/user/quickstart.md` — end-user quickstart (added in Task 1)
- `docs/dev/environment.md` — dev environment setup (added in Task 1)