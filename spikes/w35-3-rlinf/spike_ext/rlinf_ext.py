# W35.3 S2 spike: RLinf user extension module (official channel, PR #920).
# Loaded once per Worker process via the RLINF_EXT_MODULE environment variable:
# rlinf/scheduler/cluster/utils.py:81 load_user_extension_module imports this
# module and calls register().

from rlinf.models import register_model


def register():
    from spike_ext.vjepa2_policy import get_model

    register_model(
        "vjepa2_policy",
        get_model,
        category="embodied",
        force=True,  # idempotent: register() may run several times per process
    )
