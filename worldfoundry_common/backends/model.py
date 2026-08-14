"""Model-layer axis — which world-model encoder stack backs the pipeline.

Backends:
* ``self``         — WorldFoundry's self-developed tiny JEPA stack
                     (``TinyVJEPA2Encoder``; alias ``tiny``).
* ``transformers`` — HuggingFace ``VJEPA2Model`` via the HF shim
                     (``HfVJEPA2EncoderShim``; alias ``hf_vjepa2``).

This formalises the ``encoder_kind`` switch that the RTI orchestrator already
exposes (``tiny`` vs ``hf_vjepa2``). Every backend is a module that satisfies
the shim surface (``.config`` / ``.parameters()`` / ``.eval()`` /
``.forward(...)``), so downstream planner/actor code treats them as opaque.
"""
from __future__ import annotations

from typing import Any

from worldfoundry_common.backends.registry import Backend, BackendRegistry

registry = BackendRegistry("model")


def _build_self(**kwargs: Any) -> Any:
    from worldfoundry_train.model.tiny_jepa import (
        TinyEncoderConfig,
        TinyVJEPA2Encoder,
    )

    return TinyVJEPA2Encoder(TinyEncoderConfig(**kwargs))


def _build_transformers(**kwargs: Any) -> Any:
    from worldfoundry_train.model.hf_vjepa_shim import HfVJEPA2EncoderShim

    return HfVJEPA2EncoderShim.from_random_init(**kwargs)


registry.register(
    "self",
    Backend(
        "self",
        "Self-developed tiny JEPA encoder (TinyVJEPA2Encoder).",
        build=_build_self,
    ),
)
registry.register(
    "tiny",
    Backend(
        "tiny",
        "Alias for the 'self' tiny JEPA encoder stack.",
        build=_build_self,
    ),
)
registry.register(
    "transformers",
    Backend(
        "transformers",
        "HuggingFace VJEPA2Model via HfVJEPA2EncoderShim.",
        build=_build_transformers,
    ),
)
registry.register(
    "hf_vjepa2",
    Backend(
        "hf_vjepa2",
        "Alias for the 'transformers' HF VJEPA2 encoder.",
        build=_build_transformers,
    ),
)

__all__ = ["registry"]
