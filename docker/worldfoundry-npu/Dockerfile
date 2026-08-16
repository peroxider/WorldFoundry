# WorldFoundry NPU base image (C-T3).
#
# Builds the locked version matrix (master pins per W35.2 ADR §2.1).
# This is the FROM base for ``worldfoundry-npu-vllm`` and
# ``worldfoundry-npu-diffusion``; sharing it gives the three C-T3 tags
# a single dependency layer in the Docker cache.
#
# Locked matrix (mirrors ``requirements/base.txt`` + ``requirements/npu.txt``):
#   torch==2.9.0   torch_npu==2.9.0   transformers==4.55.0
#   peft==0.18.0   accelerate==1.14.0 triton==3.7.1
#   datasets==3.6.0   fastapi/uvicorn/pydantic/opentelemetry service stack
# CANN 8.5.1 toolkit is host-provided (Atlas driver); see
# ``docs/dev/environment.md``.

FROM ascend-pytorch:2.9.0-cann8.5.1-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System deps shared by all downstream images. Kept narrow on purpose
# so the base cache stays slim; downstream Dockerfiles can add more
# (e.g. libegl1/libgl1 for Franka).
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        wget \
        curl \
        ffmpeg \
        libsm6 \
        libxext6 \
        libxrender-dev \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Copy the split requirements first so this layer is stable across
# source-only changes. ``requirements/npu.txt`` chains ``-r base.txt``
# which is the locked master matrix.
COPY requirements/ requirements/

# Install the NPU base matrix. ``npu.txt`` is currently a documentation
# anchor (cann-toolkit is host-provided, see the file comment); the
# real install is ``base.txt``. Pip is idempotent.
RUN pip install --no-cache-dir -r requirements/npu.txt

# Now copy the project source. Source changes bust only this layer.
COPY pyproject.toml README.md ./
COPY worldfoundry_common/ worldfoundry_common/
COPY worldfoundry_train/ worldfoundry_train/
COPY worldfoundry_inf/ worldfoundry_inf/
COPY worldfoundry_rl/ worldfoundry_rl/

RUN pip install --no-cache-dir -e .

# Runtime scratch dirs for benchmark outputs / model snapshots.
RUN mkdir -p /workspace/models /workspace/bench_results

CMD ["/bin/bash"]
