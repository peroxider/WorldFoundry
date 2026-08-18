"""vLLM-Omni multimodal output orchestration (M3-T7).

Registers :class:`LatentToFramePipeline` as Omni's multimodal output
component via :func:`build_omni_output`.
"""

from worldfoundry_inf.omni.output import (
    OmniOutputComponent,
    build_omni_output,
)

__all__ = ["OmniOutputComponent", "build_omni_output"]
