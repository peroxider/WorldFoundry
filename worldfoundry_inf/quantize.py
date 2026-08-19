"""Thin forwarder — W8A8 pack/unpack primitives now live in compile/.

Historical path: ``worldfoundry_inf.quantize`` (W26 read-side unpack
utilities). The primitives were consolidated into the quantization
package at ``worldfoundry_inf.compile.quantize_primitives`` (facade +
primitives + backends all under ``compile/``), so this module only
re-exports them for backward compatibility. New code should import
from :mod:`worldfoundry_inf.compile.quantize_primitives` (or the
:mod:`worldfoundry_inf.compile.quantize` facade for the public entry
point).
"""
from worldfoundry_inf.compile.quantize_primitives import *  # noqa: F401,F403
from worldfoundry_inf.compile.quantize_primitives import __all__  # noqa: F401
