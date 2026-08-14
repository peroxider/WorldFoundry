FROM quay.io/ascend/vllm-ascend:v0.18.0

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN sed -i 's|http://ports.ubuntu.com/ubuntu-ports|https://mirrors.aliyun.com/ubuntu-ports|g' /etc/apt/sources.list \
    && apt-get update && apt-get install -y --no-install-recommends \
    libegl1 \
    libgl1 \
    libgl1-mesa-dri \
    libgles2 \
    libglfw3 \
    libosmesa6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

RUN python -m pip install --no-cache-dir \
        --index-url https://mirrors.aliyun.com/pypi/simple \
        transformers==4.55.0 \
        peft==0.18.0 \
        accelerate==1.10.0 \
        datasets==3.6.0 \
        einops \
        'pyyaml>=6.0' \
        'numpy<2.0' \
        'opencv-python-headless<5' \
        av==14.0.0 \
        dm-control==1.0.44 \
        mujoco==3.11.0

CMD ["/bin/bash"]
