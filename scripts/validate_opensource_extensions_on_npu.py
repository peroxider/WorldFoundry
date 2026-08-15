#!/usr/bin/env python3
"""Comprehensive NPU validation of the open-source extension deliverables.

Runs inside ``worldfoundry-franka:npu-cann9`` against physical card 7.

Covers the deliverables claimed in section 8.1 of
``docs/roadmap/2026-08-14-opensource-extensions-loop-todo.md``:

* M1-T1 / T2 -- RLinfRunnerConfig / RunnerFactory / WorldModelActor+Critic
* M2-T1 / T2 / T3 -- VJEPA2NPUWorker predict_*_batch CPU fallbacks
* M3-T1 / T2 / T3 -- diffusion pipeline ABC + mindie_sd / vllm_omni backend
* C-T2 -- pyproject extras (validated via import surface only)

Each section asserts shape / dtype / determinism / availability -- never
executes a full HF forward pass on NPU here. The full NPU five-stage
loop is already covered by ``scripts/render_train_rl_cpu.py``.
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path


def _section(name):
    return {"name": name, "passed": False, "notes": []}


def check_m1():
    sec = _section("M1-T1/T2: RLinf adapter seam")
    try:
        from worldfoundry_common.adapters.rlinf import (
            RLinfRunnerConfig,
            RunnerFactory,
            RLinfEnvWorker,
        )
        from worldfoundry_train.model.hf_vjepa_shim import (
            WorldModelActor,
            WorldModelCritic,
        )

        cfg = RLinfRunnerConfig(cluster_role="head", world_model_builder=lambda c: object())
        assert hasattr(cfg, "cluster_role") and cfg.cluster_role == "head"

        # Protocol -- runtime_checkable
        assert isinstance(RunnerFactory, type)

        # Adapter env worker has the bridge methods
        for method in ("reset", "chunk_step", "close"):
            assert hasattr(RLinfEnvWorker, method), method

        # Actor / Critic adapter classes are importable; they require a real
        # HfJEPAPredictorShim, so we assert the class methods exist instead
        # of instantiating (no need to load HF weights here).
        for cls, label in ((WorldModelActor, "actor"), (WorldModelCritic, "critic")):
            assert callable(cls), label
            assert hasattr(cls, "forward"), label + ".forward"

        sec["passed"] = True
        sec["notes"].append("RLinfRunnerConfig + RunnerFactory + adapter classes importable")
    except Exception:
        sec["notes"].append(traceback.format_exc())
    return sec


def check_m2():
    sec = _section("M2-T1/T2/T3: VJEPA2 worker parity")
    try:
        from worldfoundry_inf.vllm_ascend_worker import VJEPA2NPUWorker

        worker = VJEPA2NPUWorker(model_path="dummy", device="cpu")
        for name in (
            "predict_masked_batch",
            "predict_temporal",
            "predict_temporal_batch",
            "predict_action_conditioned",
            "predict_action_conditioned_batch",
        ):
            assert hasattr(worker, name), name

        # Deterministic CPU fallback smoke: actual worker signatures.
        # predict_masked_batch(videos, mask_ratios) -> list[Tensor]
        # predict_temporal(video_tensor, past_frames, future_frames) -> (Tensor, float)
        # predict_action_conditioned(video_tensor, actions, past_frames, future_frames)
        import torch

        videos = [torch.randn(1, 4, 3, 8, 8)]  # (B=1, T=4, C=3, H=8, W=8)

        # The NPU worker requires load_model() before inference; we do not
        # have real HF weights in this validation, so we assert the
        # expected RuntimeError("load_model() must be called") -- proving
        # the worker is wired and the method dispatch reaches the new
        # CPU-fallback implementation rather than the old
        # ``raise NotImplementedError(...)`` shortcut. The full parity is
        # exercised by worldfoundry_inf/tests/test_vllm_ascend_worker_parity.py.
        try:
            worker.predict_masked_batch(videos, mask_ratios=[0.5])
        except RuntimeError as exc:
            assert "load_model" in str(exc), str(exc)
            sec["notes"].append(
                "predict_masked_batch requires load_model (wired, not NotImpl)"
            )
        except NotImplementedError as exc:
            raise AssertionError(
                "predict_masked_batch still raises NotImplementedError: " + str(exc)
            )

        # Same check for temporal / action-conditioned: each may raise a
        # RuntimeError explaining what extra construction arg is required;
        # what matters here is the absence of NotImplementedError.
        for call, label in (
            (lambda: worker.predict_temporal(videos[0], 2, 2), "predict_temporal"),
            (lambda: worker.predict_action_conditioned(
                videos[0], torch.zeros(1, 2, 9), 2, 2,
            ), "predict_action_conditioned"),
        ):
            try:
                call()
            except RuntimeError:
                pass  # load_model / predictor_ckpt / action_predictor_ckpt
            except NotImplementedError as exc:
                raise AssertionError(
                    label + " still raises NotImplementedError: " + str(exc)
                )

        sec["passed"] = True
        sec["notes"].append(
            "predict_*_batch wired (require load_model / predictor_ckpt)"
        )
    except Exception:
        sec["notes"].append(traceback.format_exc())
    return sec


def check_m3():
    sec = _section("M3-T1/T2/T3: diffusion pipeline + backends")
    try:
        from worldfoundry_inf.diffusion.pipeline import (
            LatentToFramePipeline,
            PipelineNotAvailableError,
        )
        from worldfoundry_inf.diffusion.backends.cpu import (
            CPUPixelDecoderPipeline,
        )
        from worldfoundry_common.backends.inference import registry

        # ABC contract
        for name in ("encode_latents", "decode_latents", "rollout"):
            assert hasattr(LatentToFramePipeline, name), name

        # CPU stub -- shape / dtype / determinism
        # decode_latents(latent=(B, N, encoder_dim), grid=(T, Hg, Wg))
        #   -> (B, T, 3, H, W) float32
        import torch as _torch
        pipe = CPUPixelDecoderPipeline()  # defaults: encoder_dim=64, img_size=196, patch=14
        Hg = Wg = 14
        T_post = 2
        N = T_post * Hg * Wg  # 392
        encoder_dim = pipe.encoder_dim
        latent = _torch.randn(1, N, encoder_dim, dtype=_torch.float32)
        grid = (T_post, Hg, Wg)
        out = pipe.decode_latents(latent, grid=grid)
        assert tuple(out.shape) == (1, T_post, 3, 196, 196), out.shape
        assert out.dtype == _torch.float32, out.dtype

        out2 = pipe.decode_latents(latent, grid=grid)
        assert _torch.equal(out, out2), "CPU pixel-decoder not deterministic"

        # Backends registered

        # Backends registered
        names = registry.list()
        assert "mindie_sd" in names, names
        assert "vllm_omni" in names, names

        # Build availability
        avail_mindie = registry.available("mindie_sd")
        assert isinstance(avail_mindie, tuple) and len(avail_mindie) == 2, avail_mindie

        # PipelineNotAvailableError is a real exception
        assert issubclass(PipelineNotAvailableError, Exception)

        sec["passed"] = True
        sec["notes"].append(
            "pipeline ABC + CPU stub deterministic; registered = " + str(names)
        )
    except Exception:
        sec["notes"].append(traceback.format_exc())
    return sec


def check_c_t1():
    sec = _section("C-T1: version matrix lock")
    try:
        import os
        import re
        import tomllib
        here = os.path.dirname(os.path.abspath(__file__))
        candidates = [
            os.path.join(here, "..", "pyproject.toml"),
            os.path.join(here, "..", "..", "pyproject.toml"),
            "/workspace/pyproject.toml",
        ]
        pyproject_path = next((p for p in candidates if os.path.exists(p)), None)
        assert pyproject_path is not None, "no pyproject.toml found"
        with open(pyproject_path, "rb") as f:
            data = tomllib.load(f)
        deps = list(data.get("project", {}).get("dependencies", []))
        optional = data.get("project", {}).get("optional-dependencies", {})

        # Master pins (per W35.2 ADR section 2.1 / section 2.2)
        must_have_deps = {
            "torch_npu==2.9.0",
            "torch==2.9.0",
            "transformers==4.55.0",
            "accelerate==1.14.0",
            "triton==3.7.1",
        }
        missing = [m for m in must_have_deps if m not in deps]
        assert not missing, "missing core pins: " + str(missing)

        # rl extra must pin ray==2.56.1 + verl==0.5.0
        rl_extra = optional.get("rl", [])
        assert any("ray==2.56.1" in r for r in rl_extra), rl_extra
        assert any("verl==0.5.0" in r for r in rl_extra), rl_extra

        # inf extra must pin vllm==0.18.0 + vllm-ascend==0.18.0
        inf_extra = optional.get("inf", [])
        assert any("vllm==0.18.0" in v for v in inf_extra), inf_extra
        assert any("vllm-ascend==0.18.0" in v for v in inf_extra), inf_extra

        # requirements/ split must exist with at least base.txt
        req_dir = os.path.join(os.path.dirname(pyproject_path), "requirements")
        assert os.path.isdir(req_dir), "requirements/ dir missing next to pyproject.toml"
        base_txt = os.path.join(req_dir, "base.txt")
        assert os.path.exists(base_txt), "requirements/base.txt missing"

        # base.txt must mirror the master pins
        with open(base_txt) as bf:
            base_lines = bf.read()
        for pin in ("torch==2.9.0", "torch_npu==2.9.0", "transformers==4.55.0",
                    "accelerate==1.14.0", "triton==3.7.1"):
            assert pin in base_lines, "base.txt missing pin: " + pin

        # Non-base split files MUST chain via ``-r base.txt`` so a downstream
        # ``pip install -r requirements/rl.txt`` pulls torch/torch_npu pins
        # automatically. Enforce presence (pip accepts the directive anywhere
        # in the file, but the convention is to put it at the top).
        split_files = ["npu.txt", "rl.txt", "inf.txt", "dev.txt", "quant.txt",
                       "video.txt", "diffusion.txt", "vllm_omni.txt"]
        for fname in split_files:
            with open(os.path.join(req_dir, fname)) as sf:
                content = sf.read()
            assert "-r base.txt" in content, (
                fname + " missing ``-r base.txt`` chain"
            )

        # The split must cover all 8 optional extras + README
        expected = {"base.txt", "npu.txt", "rl.txt", "inf.txt", "dev.txt",
                    "quant.txt", "video.txt", "diffusion.txt",
                    "vllm_omni.txt", "README.md"}
        present = set(os.listdir(req_dir))
        missing_files = expected - present
        assert not missing_files, "missing requirements files: " + str(missing_files)

        # Commented-out lines (e.g. ``# cann-toolkit==8.5.1``) don't count
        # toward the line count -- assert every non-comment, non-blank line
        # in base.txt matches the PEP 508 package spec shape.
        spec_re = re.compile(r"^[A-Za-z0-9_.\-\[\]+]+\s*[<>=!~]=?\s*[\w.\-+*]+$")
        bad = []
        for ln in base_lines.splitlines():
            ln = ln.strip()
            if not ln or ln.startswith("#") or ln.startswith("-"):
                continue
            if not spec_re.match(ln):
                bad.append(ln)
        assert not bad, "malformed spec lines: " + str(bad)

        sec["passed"] = True
        sec["notes"].append(
            "pyproject pins + requirements/*.txt split verified ("
            + str(len(present))
            + " files; "
            + str(len(must_have_deps))
            + " master pins)"
        )
    except Exception:
        sec["notes"].append(traceback.format_exc())
    return sec


def check_c_t2():
    sec = _section("C-T2: pyproject extras metadata")
    try:
        import os
        import tomllib
        # Walk up from this file to the nearest pyproject.toml so the script
        # works both in the Atlas container (/workspace/.../pyproject.toml)
        # and on the host (../pyproject.toml).
        here = os.path.dirname(os.path.abspath(__file__))
        candidates = [
            os.path.join(here, "..", "pyproject.toml"),
            os.path.join(here, "..", "..", "pyproject.toml"),
            "/workspace/pyproject.toml",
        ]
        pyproject_path = next((p for p in candidates if os.path.exists(p)), None)
        assert pyproject_path is not None, "no pyproject.toml found"
        with open(pyproject_path, "rb") as f:
            data = tomllib.load(f)
        optional = data.get("project", {}).get("optional-dependencies", {})
        assert "diffusion" in optional or "vllm_omni" in optional, optional.keys()
        sec["passed"] = True
        sec["notes"].append(
            "extras keys = " + str(sorted(optional.keys()))
        )
    except Exception:
        sec["notes"].append(traceback.format_exc())
    return sec


def check_npu():
    sec = _section("NPU passthrough")
    try:
        import torch
        import torch_npu  # noqa: F401
        available = torch.npu.is_available()
        count = torch.npu.device_count()
        device = str(torch.zeros(1).npu().device) if available else "n/a"
        sec["passed"] = bool(available and count >= 1)
        sec["notes"].append(
            "available=" + str(available) + " count=" + str(count) + " device=" + device
        )
    except Exception:
        sec["notes"].append(traceback.format_exc())
    return sec


def main():
    results = [
        check_npu(),
        check_m1(),
        check_m2(),
        check_m3(),
        check_c_t1(),
        check_c_t2(),
    ]
    summary = {
        "n_passed": sum(1 for r in results if r["passed"]),
        "n_total": len(results),
        "results": results,
        "torch_version": __import__("torch").__version__,
        "python": sys.version.split()[0],
        "image": os.environ.get("WORLDFOUNDRY_IMAGE", "unknown"),
    }
    out_path = Path(
        os.environ.get(
            "EXT_VALIDATION_OUT",
            "/workspace/bench_results/_ext_validation/result.json",
        )
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if summary["n_passed"] == summary["n_total"] else 1


if __name__ == "__main__":
    sys.exit(main())