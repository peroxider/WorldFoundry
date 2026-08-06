"""POST /v1/world/predict — predict masked tokens or future frame latents.

Per spec §4.3 + W24 (Phase-3):

- ``mode="masked"`` (default, byte-compatible with W22): V-JEPA 2's native
  masked-prediction objective; returns ``{predicted_features, loss_score}``.
- ``mode="temporal"``: world-model roll-out; ``past_frames`` + ``future_frames``
  slice the request video into past / future windows; the worker encodes
  the past with its online encoder and runs the temporal predictor to
  produce features for the next H frames; the EMA target encoder scores
  confidence. Returns ``{future_features, confidence, mode}``.
- ``mode="action"`` (W33): action-conditioned roll-out; same shape as
  ``mode="temporal"`` but additionally takes a 2-D ``actions`` field of
  shape ``(past_frames, action_dim)`` (one action vector per past frame).
  Returns ``{future_features, confidence, mode}``.

All modes share the same HMAC contract (``_check_auth`` is unchanged).

W25: each request is routed through the per-adapter ``PredictBatcher``
which aggregates up to ``max_batch`` requests or ``max_latency_ms``
deadline before dispatching a single batched worker call.
"""
from __future__ import annotations

import base64
import binascii

import numpy as np
import torch
from fastapi import APIRouter, Depends, HTTPException, Request

from worldfoundry_common.exceptions import AdapterNotFoundError
from worldfoundry_inf.predict_batcher import PredictBatcher, PredictRequest
from worldfoundry_inf.security import HmacTokenAuth

router = APIRouter()


@router.post("/predict")
async def predict(request: Request, auth: HmacTokenAuth = Depends(lambda: None)) -> dict:
    auth = request.app.state.auth
    body_bytes = await request.body()
    _check_auth(request, auth, body_bytes)
    payload = await request.json()
    if "video" not in payload:
        raise HTTPException(status_code=400, detail="missing 'video' field")

    mode = str(payload.get("mode", "masked"))
    if mode not in ("masked", "temporal", "action"):
        raise HTTPException(
            status_code=400,
            detail=f"mode must be 'masked', 'temporal', or 'action', got {mode!r}",
        )

    try:
        video_bytes = base64.b64decode(payload["video"], validate=True)
        video = np.frombuffer(video_bytes, dtype=np.uint8).reshape(-1, 64, 64, 3)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"invalid video payload: {exc}") from exc

    adapter_name = payload.get("adapter", "default")
    try:
        worker = request.app.state.router.route(adapter_name)
    except AdapterNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"adapter {adapter_name!r} not registered") from exc

    tensor = torch.from_numpy(video).float().permute(0, 3, 1, 2).unsqueeze(0)
    # W25: /predict uses the per-adapter batcher to amortize GPU compute;
    # /encode and /plan still call the worker directly. The batcher is
    # optional — fall back to direct calls when not configured.
    batcher = getattr(request.app.state, "predict_batcher", None)

    if mode == "masked":
        return _handle_masked(payload, tensor, worker, batcher)
    if mode == "temporal":
        return _handle_temporal(payload, tensor, worker, batcher)
    return _handle_action(payload, tensor, worker, batcher)


def _handle_masked(payload: dict, tensor: torch.Tensor, worker: object, batcher: PredictBatcher | None) -> dict:
    if "mask_ratio" not in payload:
        raise HTTPException(status_code=400, detail="missing 'mask_ratio'")
    mask_ratio = float(payload["mask_ratio"])
    if not 0.0 <= mask_ratio <= 1.0:
        raise HTTPException(status_code=400, detail="mask_ratio must be in [0,1]")
    if batcher is not None and hasattr(worker, "predict_masked_batch"):
        req = PredictRequest(video=tensor[0], mode="masked", mask_ratio=mask_ratio)
        try:
            pred = batcher.submit(req).result(timeout=2.0)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"predict batcher failed: {exc}") from exc
    else:
        mask = torch.zeros(1, tensor.shape[1], dtype=torch.bool)
        n_masked = int(mask_ratio * tensor.shape[1])
        mask[0, :n_masked] = True
        pred = worker.predict(tensor, mask)
    return {
        "predicted_features": pred.tolist(),
        "loss_score": float((pred ** 2).mean()),
    }


def _handle_temporal(payload: dict, tensor: torch.Tensor, worker: object, batcher: PredictBatcher | None) -> dict:
    if "past_frames" not in payload or "future_frames" not in payload:
        raise HTTPException(
            status_code=400,
            detail="mode='temporal' requires both 'past_frames' and 'future_frames'",
        )
    past_frames = int(payload["past_frames"])
    future_frames = int(payload["future_frames"])
    if past_frames < 1 or future_frames < 1:
        raise HTTPException(
            status_code=400,
            detail=f"past_frames and future_frames must be >= 1, got "
            f"past_frames={past_frames}, future_frames={future_frames}",
        )
    if past_frames + future_frames > tensor.shape[1]:
        raise HTTPException(
            status_code=400,
            detail=f"past_frames + future_frames ({past_frames + future_frames}) "
            f"exceeds video length ({tensor.shape[1]} frames)",
        )

    if not hasattr(worker, "predict_temporal"):
        raise HTTPException(
            status_code=503,
            detail="temporal mode is not enabled for this worker "
            "(construct the worker with predictor_ckpt or use the "
            "CPU VJEPA2Worker on a worker that supports it)",
        )

    if batcher is not None and hasattr(worker, "predict_temporal_batch"):
        req = PredictRequest(
            video=tensor[0],
            mode="temporal",
            past_frames=past_frames,
            future_frames=future_frames,
        )
        try:
            future_features, confidence = batcher.submit(req).result(timeout=2.0)
        except NotImplementedError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"predict batcher failed: {exc}") from exc
    else:
        try:
            future_features, confidence = worker.predict_temporal(
                tensor, past_frames, future_frames,
            )
        except NotImplementedError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    return {
        "future_features": future_features.tolist(),
        "confidence": float(confidence),
        "mode": "temporal",
    }


def _handle_action(payload: dict, tensor: torch.Tensor, worker: object, batcher: PredictBatcher | None) -> dict:
    """W33: action-conditioned roll-out via the W27 v2 bundle.

    Validates ``actions`` (2-D list shape ``(past_frames, action_dim)``),
    then routes through the per-adapter ``PredictBatcher`` (when the worker
    exposes ``predict_action_conditioned_batch``) or calls the worker
    directly. Returns ``{future_features, confidence, mode: "action"}``.
    """
    if "past_frames" not in payload or "future_frames" not in payload:
        raise HTTPException(
            status_code=400,
            detail="mode='action' requires both 'past_frames' and 'future_frames'",
        )
    past_frames = int(payload["past_frames"])
    future_frames = int(payload["future_frames"])
    if past_frames < 1 or future_frames < 1:
        raise HTTPException(
            status_code=400,
            detail=f"past_frames and future_frames must be >= 1, got "
            f"past_frames={past_frames}, future_frames={future_frames}",
        )
    if past_frames + future_frames > tensor.shape[1]:
        raise HTTPException(
            status_code=400,
            detail=f"past_frames + future_frames ({past_frames + future_frames}) "
            f"exceeds video length ({tensor.shape[1]} frames)",
        )

    if "actions" not in payload:
        raise HTTPException(
            status_code=400,
            detail="mode='action' requires 'actions' (a list of past_frames "
            "action vectors)",
        )
    actions_raw = payload["actions"]
    if not isinstance(actions_raw, list):
        raise HTTPException(
            status_code=400,
            detail="'actions' must be a list of action vectors (one per past "
            "frame)",
        )
    if len(actions_raw) != past_frames:
        raise HTTPException(
            status_code=400,
            detail=f"'actions' length {len(actions_raw)} != past_frames "
            f"{past_frames}",
        )
    worker_action_dim = int(getattr(worker, "action_dim", 0))
    for i, vec in enumerate(actions_raw):
        if not isinstance(vec, list):
            raise HTTPException(
                status_code=400,
                detail=f"actions[{i}] must be a list of floats, got {type(vec).__name__}",
            )
        if len(vec) != worker_action_dim:
            raise HTTPException(
                status_code=400,
                detail=f"actions[{i}] length {len(vec)} != worker's "
                f"action_dim {worker_action_dim}",
            )
        for j, a in enumerate(vec):
            if isinstance(a, bool) or not isinstance(a, (int, float)):
                raise HTTPException(
                    status_code=400,
                    detail=f"actions[{i}][{j}] must be a number, got {type(a).__name__}",
                )

    if not hasattr(worker, "predict_action_conditioned"):
        raise HTTPException(
            status_code=503,
            detail="action mode is not enabled for this worker "
            "(construct the worker with action_predictor_ckpt)",
        )

    actions_tensor = torch.tensor(actions_raw, dtype=torch.float32).unsqueeze(0)

    if batcher is not None and hasattr(worker, "predict_action_conditioned_batch"):
        req = PredictRequest(
            video=tensor[0],
            mode="action",
            past_frames=past_frames,
            future_frames=future_frames,
            actions=actions_tensor,
        )
        try:
            future_features, confidence = batcher.submit(req).result(timeout=2.0)
        except NotImplementedError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"predict batcher failed: {exc}") from exc
    else:
        try:
            future_features, confidence = worker.predict_action_conditioned(
                tensor, actions_tensor, past_frames, future_frames,
            )
        except NotImplementedError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    return {
        "future_features": future_features.tolist(),
        "confidence": float(confidence),
        "mode": "action",
    }


def _check_auth(request: Request, auth: HmacTokenAuth, body_bytes: bytes) -> None:
    hdr = request.headers.get("Authorization", "")
    ts = request.headers.get("X-WorldFoundry-Timestamp", "")
    if not hdr.startswith("HMAC "):
        raise HTTPException(status_code=401, detail="missing or malformed Authorization header")
    sig = hdr[len("HMAC "):]
    if not auth.verify(request.method, request.url.path, body_bytes, ts, sig):
        raise HTTPException(status_code=401, detail="HMAC verification failed")

