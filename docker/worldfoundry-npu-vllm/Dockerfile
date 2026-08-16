# WorldFoundry NPU + vLLM-Ascend image (C-T3).
#
# Extends ``worldfoundry-npu:phase1`` with ``vllm==0.18.0`` +
# ``vllm-ascend==0.18.0`` from ``requirements/inf.txt``. The M2 mainline
# (V-JEPA2 worker + Atlas serving path).
#
# Acceptance gate (C-T3):
#   docker run --rm worldfoundry-npu-vllm:phase1 python -c "import vllm"
# exits 0; vLLM is registered, the editable ``worldfoundry-inf`` install
# is inherited from the parent image.

FROM worldfoundry-npu:phase1

ENV PIP_NO_CACHE_DIR=1

# Single-file COPY keeps the image's surface area narrow and avoids
# pinning to the entire ``requirements/`` dir layout. ``inf.txt`` chains
# ``-r base.txt`` which is already installed in the parent; pip skips
# re-installs of already-satisfied constraints.
COPY requirements/inf.txt /tmp/requirements-inf.txt
RUN pip install --no-cache-dir -r /tmp/requirements-inf.txt \
    && rm -f /tmp/requirements-inf.txt

# Inherit WORKDIR / CMD from the parent image.

LABEL org.opencontainers.image.title="worldfoundry-npu-vllm" \
      org.opencontainers.image.source="https://github.com/WorldFoundry/WorldFoundry" \
      org.opencontainers.image.description="C-T3 image: locked NPU base + vLLM-Ascend (M2 mainline)"
