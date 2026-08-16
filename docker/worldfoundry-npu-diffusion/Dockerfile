# WorldFoundry NPU + diffusion scaffold image (C-T3).
#
# Extends ``worldfoundry-npu:phase1`` with ``diffusers>=0.30`` +
# ``transformers[sentencepiece]>=4.55`` from ``requirements/diffusion.txt``.
# This is the M3 mainline scaffold; today's active path is the v1 CPU
# pixel-decoder stub at ``worldfoundry_inf.diffusion.backends.cpu``, and
# the v2 NPU diffusion path (M3-T4) will replace this stub with real
# MindIE-SD inside this same image.

FROM worldfoundry-npu:phase1

ENV PIP_NO_CACHE_DIR=1

# Single-file COPY keeps the surface narrow; ``diffusion.txt`` chains
# ``-r base.txt`` which is already installed in the parent image.
COPY requirements/diffusion.txt /tmp/requirements-diffusion.txt
RUN pip install --no-cache-dir -r /tmp/requirements-diffusion.txt \
    && rm -f /tmp/requirements-diffusion.txt

# Inherit WORKDIR / CMD from the parent image.

LABEL org.opencontainers.image.title="worldfoundry-npu-diffusion" \
      org.opencontainers.image.source="https://github.com/WorldFoundry/WorldFoundry" \
      org.opencontainers.image.description="C-T3 image: locked NPU base + diffusers scaffold (M3 mainline)"
