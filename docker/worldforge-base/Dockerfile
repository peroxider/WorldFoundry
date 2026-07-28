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

# Clone upstream V-JEPA 2 reference (NOT to be modified — G7).
# Pin the commit because upstream currently publishes no release tags.
RUN git clone --depth 1 --branch main --single-branch https://github.com/facebookresearch/vjepa2 /opt/vjepa2 \
    && cd /opt/vjepa2 \
    && git fetch --depth 1 origin 5a58d8661b8972106712142ebbd07fe9fb119c8f \
    && git checkout --detach 5a58d8661b8972106712142ebbd07fe9fb119c8f

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