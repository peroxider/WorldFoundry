# W35.3 spike: process-wide patch so every python process on this PYTHONPATH
# (including Ray/spawn worker subprocesses) gets the custom env type registered.
# Guarded: never break interpreter startup if rlinf is unavailable.

try:
    import rlinf.envs as _rlinf_envs

    _orig_get_env_cls = _rlinf_envs.get_env_cls

    def _spike_get_env_cls(env_type, env_cfg=None):
        if env_type == "gym_cartpole":
            from spike_ext.gym_cartpole_env import GymnasiumCartPoleEnv

            return GymnasiumCartPoleEnv
        return _orig_get_env_cls(env_type, env_cfg)

    _rlinf_envs.get_env_cls = _spike_get_env_cls

    # rlinf.envs.SupportedEnvType is a real Enum with hardcoded members;
    # action_utils.prepare_actions (action_utils.py:282) dispatches on it.
    # Map "gym_cartpole" onto EMBODICHAIN so all enum dispatch sites take the
    # passthrough branch (our GymnasiumCartPoleEnv mirrors EmbodiChainEnv).
    _real_envs_supported = _rlinf_envs.SupportedEnvType

    class _EnvsSupportedEnvTypeShim:
        def __call__(self, value):
            try:
                return _real_envs_supported(value)
            except ValueError:
                if value == "gym_cartpole":
                    return _real_envs_supported.EMBODICHAIN
                raise

        def __getattr__(self, name):
            return getattr(_real_envs_supported, name)

    _envs_shim = _EnvsSupportedEnvTypeShim()
    _rlinf_envs.SupportedEnvType = _envs_shim
    try:
        import rlinf.envs.action_utils as _action_utils

        _action_utils.SupportedEnvType = _envs_shim
    except Exception:
        pass
except Exception:
    pass

try:
    import rlinf.config as _rlinf_config

    _orig_supported_env_type = _rlinf_config.SupportedEnvType

    class _SupportedEnvTypeShim:
        def __call__(self, value):
            try:
                return _orig_supported_env_type(value)
            except ValueError:
                if value == "gym_cartpole":
                    return "gym_cartpole"
                raise

        def __getattr__(self, name):
            return getattr(_orig_supported_env_type, name)

    _rlinf_config.SupportedEnvType = _SupportedEnvTypeShim()
except Exception:
    pass

# --- CPU (NO_ACCEL) platform shim -------------------------------------------
# RLinf worker layer has no built-in CPU fallback:
# AcceleratorUtil.get_torch_platform()/get_device_type() return None for
# NO_ACCEL (accelerator.py:442/:452), which crashes workers immediately
# (e.g. huggingface_worker.py:52 current_device()). Patch both to serve a
# minimal torch.cuda-like shim backed by CPU. Must be installed before
# rlinf.scheduler.worker.worker is imported (class attrs computed at import).
def _spike_debug(msg):
    try:
        import os as _os

        with open("/home/chad/w35-3/sitecustomize-debug.log", "a") as _f:
            _f.write(f"[pid={_os.getpid()}] {msg}\n")
    except Exception:
        pass


try:
    import torch as _torch

    class _CpuPlatformShim:
        """Minimal stand-in for the torch.cuda-like platform module on CPU."""

        @staticmethod
        def current_device():
            return _torch.device("cpu")

        @staticmethod
        def set_device(device):
            return None

        @staticmethod
        def empty_cache():
            return None

        @staticmethod
        def synchronize(device=None):
            return None

        @staticmethod
        def is_initialized():
            return True

        @staticmethod
        def is_available():
            return True

        @staticmethod
        def device_count():
            return 1

        @staticmethod
        def ipc_collect():
            return None

        class Stream:
            def __init__(self, *args, **kwargs):
                pass

            def synchronize(self, *args, **kwargs):
                pass

            def wait_event(self, *args, **kwargs):
                pass

            def wait_stream(self, *args, **kwargs):
                pass

            def record_event(self, *args, **kwargs):
                pass

            def query(self):
                return True

        @staticmethod
        def current_stream(device=None):
            return _CpuPlatformShim.Stream()

        @staticmethod
        def default_stream(device=None):
            return _CpuPlatformShim.Stream()

        @staticmethod
        def stream(stream):
            import contextlib as _cl

            return _cl.nullcontext()

        @staticmethod
        def manual_seed_all(seed):
            _torch.manual_seed(seed)

        class Event:
            def __init__(self, *args, **kwargs):
                pass

            def record(self, *args, **kwargs):
                pass

            def wait(self, *args, **kwargs):
                pass

            def synchronize(self, *args, **kwargs):
                pass

            def query(self):
                return True

    from rlinf.scheduler.hardware.accelerators.accelerator import (
        AcceleratorUtil as _AcceleratorUtil,
    )

    _orig_get_torch_platform = _AcceleratorUtil.get_torch_platform
    _orig_get_device_type = _AcceleratorUtil.get_device_type

    def _spike_get_torch_platform(accelerator_type=None):
        platform = _orig_get_torch_platform(accelerator_type)
        return platform if platform is not None else _CpuPlatformShim

    def _spike_get_device_type(accelerator_type=None):
        device_type = _orig_get_device_type(accelerator_type)
        return device_type if device_type is not None else "cpu"

    _AcceleratorUtil.get_torch_platform = staticmethod(_spike_get_torch_platform)
    _AcceleratorUtil.get_device_type = staticmethod(_spike_get_device_type)

    # Importing AcceleratorUtil above can pull in rlinf.scheduler.worker.worker
    # via package __init__ chains BEFORE this patch is applied, leaving the
    # already-created Worker class attributes at None. Repair them directly.
    try:
        from rlinf.scheduler.worker.worker import Worker as _Worker

        if getattr(_Worker, "torch_platform", None) is None:
            _Worker.torch_platform = _CpuPlatformShim
        if getattr(_Worker, "torch_device_type", None) is None:
            _Worker.torch_device_type = "cpu"
        _spike_debug("worker class attrs repaired")
    except Exception:
        _spike_debug("worker class not yet imported; patch will apply at import")

    _spike_debug("cpu platform patch applied")
except Exception:
    import traceback as _tb

    _spike_debug("cpu platform patch FAILED:\n" + _tb.format_exc())

# --- FSDP1 int device_id fix -------------------------------------------------
# rlinf/hybrid_engines/fsdp/strategy/fsdp.py passes device_id=int(LOCAL_RANK)
# into FSDP; torch.device(<int>) means "accelerator by index" and raises
# "Cannot access accelerator device when none is available" on CPU-only builds.
# Rewrite int device_id -> torch.device("cpu") when no accelerator exists.
try:
    import rlinf.hybrid_engines.fsdp.strategy.fsdp as _fsdp_strategy

    _RealFSDP = _fsdp_strategy.FSDP

    def _spike_FSDP(*args, **kwargs):
        import torch as _t

        if not _t.cuda.is_available():
            device_id = kwargs.get("device_id")
            if isinstance(device_id, int):
                kwargs["device_id"] = _t.device("cpu")
            # torch FSDP forbids CPU params with sync_module_states=True;
            # pointless anyway on the single-rank CPU spike setup.
            if kwargs.get("sync_module_states"):
                kwargs["sync_module_states"] = False
        return _RealFSDP(*args, **kwargs)

    _fsdp_strategy.FSDP = _spike_FSDP
    _spike_debug("fsdp int device_id patch applied")
except Exception:
    import traceback as _tb

    _spike_debug("fsdp patch FAILED:\n" + _tb.format_exc())

# --- pin_memory guard ---------------------------------------------------------
# patch_syncer.py:886 calls snapshot_value.pin_memory(), which raises
# "Cannot access accelerator device when none is available" on CPU-only torch.
# pin_memory is a CPU->GPU transfer optimization; a no-op self-return preserves
# CPU semantics. (Covers all rlinf pin_memory call sites at once.)
try:
    import torch as _torch

    if not _torch.cuda.is_available():
        _orig_pin_memory = _torch.Tensor.pin_memory

        def _spike_pin_memory(self, *args, **kwargs):
            try:
                return _orig_pin_memory(self, *args, **kwargs)
            except RuntimeError:
                return self

        _torch.Tensor.pin_memory = _spike_pin_memory

        # record_stream is a CUDA caching-allocator hint (patch_syncer.py:469);
        # meaningless on CPU and rejects our dummy Stream objects.
        def _spike_record_stream(self, stream):
            return None

        _torch.Tensor.record_stream = _spike_record_stream
        _spike_debug("pin_memory + record_stream guards applied")
except Exception:
    import traceback as _tb

    _spike_debug("pin_memory guard FAILED:\n" + _tb.format_exc())

# --- patch_syncer copy-stream fix ---------------------------------------------
# CPUSnapshotPatchBuilder._get_copy_stream (patch_syncer.py:534) rebuilds an
# index-less device via torch.device(device.type, current_device()), which
# requires current_device() to be an int index (accelerator assumption) and
# crashes on CPU. On CPU the stream is a dummy anyway; return it directly.
try:
    import torch as _t2

    if not _t2.cuda.is_available():
        import rlinf.hybrid_engines.weight_syncer.patch_syncer as _ps

        def _spike_get_copy_stream(self, device):
            key = device if device is not None else _t2.device("cpu")
            copy_stream = self._copy_streams.get(key)
            if copy_stream is None:
                from rlinf.scheduler.worker.worker import Worker as _W

                copy_stream = _W.torch_platform.Stream(device=key)
                self._copy_streams[key] = copy_stream
            return copy_stream

        _ps.CPUSnapshotPatchBuilder._get_copy_stream = _spike_get_copy_stream
        _spike_debug("patch_syncer copy-stream fix applied")
except Exception:
    import traceback as _tb

    _spike_debug("patch_syncer fix FAILED:\n" + _tb.format_exc())
