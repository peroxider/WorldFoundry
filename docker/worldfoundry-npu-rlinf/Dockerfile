# WorldFoundry NPU + RLinf optional layer (C-T4).
#
# Extends ``worldfoundry-npu-vllm:phase1`` with ``rlinf==0.3.0`` +
# ``ray==2.56.1`` from ``requirements/rlinf.txt``. Used by the M1-T4 P2
# Ray-cluster e2e on Atlas card 7 (``--rl-backend rlinf`` path).
#
# CAVEAT (W35.2 §5.2): RLinf v0.3.0 pulls torch 2.10.0 transitively, while
# the C-T1 lock pins torch 2.9.0 for torch_npu compatibility. This image
# inherits torch 2.9.0 from the parent and accepts the constraint
# mismatch as a known risk; the real RLinf+Atlas e2e validates whether
# the minor-version conflict is benign or blocks the runner. Host smoke
# does NOT need this image -- the orchestrator's ``--rl-backend rlinf``
# accepts a mock factory without importing RLinf.
#
# Acceptance gate (C-T4):
#   docker run --rm worldfoundry-npu-rlinf:phase1 \
#     python -c "import ray, rlinf; print(ray.__version__, rlinf.__version__)"
# exits 0; ray is registered, RLinf loads.

FROM worldfoundry-npu-vllm:phase1

ENV PIP_NO_CACHE_DIR=1

# Single-file COPY keeps the surface narrow; ``rlinf.txt`` chains
# ``-r base.txt`` which is already installed in the parent image.
COPY requirements/rlinf.txt /tmp/requirements-rlinf.txt
RUN pip install --no-cache-dir -r /tmp/requirements-rlinf.txt \
    && rm -f /tmp/requirements-rlinf.txt

LABEL org.opencontainers.image.title="worldfoundry-npu-rlinf" \
      org.opencontainers.image.source="https://github.com/WorldFoundry/WorldFoundry" \
      org.opencontainers.image.description="C-T4 image: optional RLinf + Ray layer on top of worldfoundry-npu-vllm (M1-T4 P2 path)"