"""Thin forwarder — VJEPA2 torch worker moved to worldfoundry_inf.workers.

Historical path: ``worldfoundry_inf.workers.vjepa2_worker``. The worker modules
were consolidated under :mod:`worldfoundry_inf.workers` (package root
keeps only service entry points). New code should import from
``worldfoundry_inf.workers.vjepa2_worker``.
"""
from worldfoundry_inf.workers.vjepa2_worker import *  # noqa: F401,F403
