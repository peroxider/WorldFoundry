# WorldFoundry NPU image set (C-T3)

Three tagged images share a common `worldfoundry-npu:phase1` base. The
base builds the locked matrix from `requirements/npu.txt` (which chains
`-r base.txt`); the two child images add only their stack-specific
extras from `requirements/{inf,diffusion}.txt` so Docker cache reuse
between them is exact.

| Tag | Built from | Adds | Purpose |
|---|---|---|---|
| `worldfoundry-npu:phase1` | `docker/worldfoundry-npu/Dockerfile` (FROM `ascend-pytorch:2.9.0-cann8.5.1-ubuntu22.04`) | locked matrix (`requirements/npu.txt` chains `base.txt`): torch 2.9.0, torch_npu 2.9.0, transformers 4.55.0, peft 0.18.0, accelerate 1.14.0, triton 3.7.1, datasets 3.6.0, fastapi/uvicorn/pydantic/opentelemetry | Canonical C-T3 base; the FROM for the two child tags. |
| `worldfoundry-npu-vllm:phase1` | `docker/worldfoundry-npu-vllm/Dockerfile` (FROM `worldfoundry-npu:phase1`) | `vllm==0.18.0` + `vllm-ascend==0.18.0` from `requirements/inf.txt` | M2 mainline — V-JEPA2 worker + Atlas serving path. |
| `worldfoundry-npu-diffusion:phase1` | `docker/worldfoundry-npu-diffusion/Dockerfile` (FROM `worldfoundry-npu:phase1`) | `diffusers>=0.30` + `transformers[sentencepiece]>=4.55` from `requirements/diffusion.txt` | M3 mainline scaffold (v1 CPU pixel-decoder today; M3-T4 swaps in real MindIE-SD). |

## Build

All three images are built from the repo root so the `COPY
requirements/...` paths resolve. On Atlas:

```bash
# 1. Base first (the two children need it).
docker build -t worldfoundry-npu:phase1 \
    -f docker/worldfoundry-npu/Dockerfile .

# 2. Children in any order; their FROM resolves via the local image.
docker build -t worldfoundry-npu-vllm:phase1 \
    -f docker/worldfoundry-npu-vllm/Dockerfile .

docker build -t worldfoundry-npu-diffusion:phase1 \
    -f docker/worldfoundry-npu-diffusion/Dockerfile .
```

`docker buildx build` with a registry cache is recommended for CI; the
child images benefit most because they share their base's pip-installed
layer.

## Acceptance gate

Per roadmap §9.2 P1 C-T3:

```bash
docker run --rm worldfoundry-npu-vllm:phase1 python -c "import vllm; print(vllm.__version__)"
# expect: 0.18.0
```

Smoke checks for the other two:

```bash
docker run --rm worldfoundry-npu:phase1 \
    python -c "import torch, torch_npu; print(torch.__version__, torch_npu.__version__)"
# expect: 2.9.0 2.9.0

docker run --rm worldfoundry-npu-diffusion:phase1 \
    python -c "import diffusers; print(diffusers.__version__)"
# expect: >=0.30
```

## Run

```bash
# Interactive shell (CPU fallback works; NPU needs --device /dev/davinciX).
docker run --rm -it worldfoundry-npu:phase1 bash

# M2 mainline smoke on Atlas card 7:
docker run --rm -it \
    --device /dev/davinci7 --device /dev/davinci_manager \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
    -v $PWD/bench_results:/workspace/bench_results \
    worldfoundry-npu-vllm:phase1 bash
```

## Legacy / sibling images

- `worldfoundry-base:phase1` (W2.18-era, `docker/worldfoundry-base/Dockerfile`) — kept as
  the legacy single-tag base; built directly from `ascend-pytorch:2.9.0-cann8.5.1-ubuntu22.04` without the new cache
  hierarchy.
- `worldfoundry-franka:atlas-cann9` (W2.24, `docker/worldfoundry-franka/Dockerfile`) —
  the Atlas-validated MuJoCo + EGL container for the `--env franka_kitchen`
  render path; uses a different upstream base (`quay.io/ascend/vllm-ascend:v0.18.0`)
  because of CANN9 graphics stack requirements.

## Cross-references

- `requirements/README.md` — split lockfile layout; maps each `requirements/<stack>.txt` to a C-T3 image tag.
- `docs/roadmap/2026-08-14-opensource-extensions-loop-todo.md` §9.2 P1 — C-T3 entry.
- `docs/dev/environment.md` — Atlas CANN / Ascend device / driver mount conventions.
