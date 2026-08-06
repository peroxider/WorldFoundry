# Next-Steps Todo — post-W27 v2 (2026-08-06)

## Background

W27 v2 (`0d8910f`) landed on master: the W27 v2 action-conditioned inference
loader + worker API closes the I2b (action-conditioned prediction) path on CPU.
W22/W24/W25/W26/W27 + W27 v2 are now stacked on master; the only remaining
W27 cleanup is the W33 plan-endpoint wire-up (an API-side task, not a model
task).

There are 4 candidate next steps that don't require an Atlas hardware window
and are independently shippable. The table below ranks them by value/risk;
the body of each section is the working plan.

## Master commit graph (recap)

```
e8150be  chore: W22 cleanup (ruff)
cb112fa  feat(w25): dynamic predict batcher
338ded8  feat(w26): temporal-objective (past-K → future-H loss)
796fb50  feat(w27): ActionConditionedPredictor + worker stub
0d8910f  feat(w27 v2): action-conditioned inference bundle + worker API   ← HEAD
```

## Candidate ranking

| # | Task | Closes | Hardware | Effort | Risk |
|---|---|---|---|---|---|
| 1 | **W33 — `/v1/world/predict` `mode=action` + plan endpoint接入** | I2b end-to-end, roadmap W33 | CPU-only | ~1 session | low |
| 2 | **W28 — 训推一体编排设计文档** | G14 (resource orchestration) | none | ~½ session | low |
| 3 | **W25-roadmap — 全参微调 + SFT 数据格式** | G5 (no full-FT entry) | CPU-only | ~1 session | low |
| 4 | **W32 — AC latent-action head (`actor.predict_latent_action`)** | G8 T2c side, AC actor | CPU-only | ~1–2 sessions | medium |

**Recommendation:** start with **#1 (W33)** — natural close-out of the W27 v2
chain (loader → worker → API). Then **#2 (W28)** to land the design piece
before any further plumbing weights in. **#3 + #4** can run in either order.

---

## #1 — W33: `/v1/world/predict` `mode=action` + plan endpoint接入

### Goal
The W27 v2 bundle is wired into the worker but the API surface still returns
503 for action-conditioned requests. W33 lands the FastAPI endpoint contract
for `mode=action` and the `/v1/world/plan` endpoint that uses the same bundle.

### Closes
- G8 (T2c side partial) — actor passthrough is still roadmap W32; this is the
  inference half.
- G10 (I2b part) — predict endpoint returns real action-conditioned latents.
- Roadmap W33 row.

### Deliverables
1. `worldfoundry_inf/api/predict.py` — extend `_handle_temporal` to accept
   `mode="action"`; new `_handle_action(payload, tensor, worker, batcher)` that
   validates `actions` (base64-decoded `(K, action_dim)` float16/float32 tensor
   or structured list), batches via the per-adapter `PredictBatcher` when
   available, and returns `{future_features, confidence, mode: "action"}`.
2. `worldfoundry_inf/api/plan.py` — new FastAPI router for `POST /v1/world/plan`
   that reuses the action bundle's `predict_action_conditioned` under the
   same HMAC contract as `/encode` / `/predict`. Response shape mirrors
   `/predict` for parity.
3. `worldfoundry_inf/server.py` — register the new plan router; expose
   `--action-predictor-ckpt` and `--action-dim` CLI args (default `None` / `7`).
4. `worldfoundry_inf/predict_batcher.py` — extend `PredictRequest` to carry
   `actions`; add a `_batched_action_dispatch` method that batches actions
   the same way the temporal batcher batches `past_frames` / `future_frames`.
5. Tests:
   - `tests/test_api_endpoints.py` — `test_predict_mode_action_returns_200`,
     `test_predict_mode_action_rejects_missing_actions`,
     `test_predict_mode_action_rejects_action_dim_mismatch`,
     `test_plan_with_valid_auth_returns_200`.
   - `tests/test_predict_batcher.py` — `test_batcher_action_mode_dispatch`,
     `test_batcher_action_mode_shape_mismatch_raises`.

### Acceptance criteria
- `POST /v1/world/predict { mode: "action", video, past_frames, future_frames, actions }`
  → 200 with `{future_features, confidence, mode: "action"}`.
- `POST /v1/world/plan { video, actions, past_frames, future_frames }` → 200
  with the same response shape.
- Missing/invalid actions → 400 with structured error.
- Action dim mismatch → 400 with structured error.
- Worker without `action_predictor_ckpt` → 503 (preserve existing semantics).

### Out of scope
- NPU path (`vllm_ascend_worker.py`) — Atlas G11 boundary; keep the
  `NotImplementedError` on `predict_action_conditioned`.
- Multi-step rollout (past K, predict H, take as input for next H, …) — W26 v2
  scope, separate PR.
- Training-side `ActionConditionedObjective` — separate W27+ scope; the loader
  is forward-compatible.

### Estimated effort
**1 session** (single feature, mirrors W24 predict/dual-mode work).

---

## #2 — W28: 训推一体编排设计文档

### Goal
Roadmap G14 (训推一体资源编排) currently has no design doc. W28 lands the
design piece that gates W37 (RL training + main training stack integration)
and the Phase-4 RL architecture落地.

### Closes
- G14 (resource orchestration — same-cluster training/inference mixing, weight
  hot-update, vRAM quota).

### Deliverables
1. `docs/superpowers/specs/2026-08-06-w28-train-inference-orchestration-design.md`
   — design doc covering:
   - **Topology options** (split cluster, shared-NPU mixed, scheduler-mediated
     pool) with trade-offs and the recommended option.
   - **Weight hot-update** mechanism (polling vs push; manifest versioning; the
     adapter router's contract).
   - **vRAM quota** policy (fraction reserved for training vs inference; OOM
     fallback semantics).
   - **Failure modes** (inference worker holds stale weights; training needs
     exclusive NPU access during a checkpoint save; rollback path).
   - **Cross-references** to existing precedent (`docs/superpowers/specs/2026-07-27-ascend-world-model-design.md`,
     W28 RL framework ADR `2026-08-04-w35-rl-framework-adr.md`).
2. ADR-style decision log entry: which orchestration model is the chosen
   baseline, and why.

### Acceptance criteria
- Doc reviewed by the user; design decision recorded.
- Doc explicitly enumerates the integration points with W22/W24/W25/W26/W27
  (so any Phase-3 impl PR can reference the orchestration contract).
- Doc identifies the W37 implementation prerequisites.

### Out of scope
- Implementation — design only; W37+ is the implementation gate.

### Estimated effort
**½ session** (single design doc, no code).

---

## #3 — W25-roadmap: 全参微调入口 + SFT 数据格式

### Goal
Roadmap G5 (no full-FT entry / SFT format) is the only remaining Phase-3
loose end that's CPU-only. Land a `full` mode in the trainer that bypasses
the LoRA wrapper but reuses checkpoint / EMA / FSDP machinery, and document
the SFT data schema.

### Closes
- G5 (no full-FT entry / SFT data format).

### Deliverables
1. `worldfoundry_train/trainer.py` — `--mode {lora, full}` CLI arg; in `full`
   mode, the trainer omits the LoRA wrapper and trains every encoder
   parameter directly (existing `optimizer_state` / `checkpoint` flow is
   reused). The `'lora_signature'` field in checkpoint payloads is allowed
   to be `None` for `full` mode (already tolerant).
2. `docs/superpowers/specs/2026-08-XX-w25-full-finetune-sft-schema.md` — SFT
   data schema: video clip manifest JSONL (one JSON per line) with fields
   `{"video_path": str, "start_frame": int, "num_frames": int, "action":
   Optional[List[float]], "label": Optional[str]}`. Validation rules
   (action_dim if present must match the model's expected action_dim;
   num_frames must satisfy `past_frames + future_frames` if temporal mode).
3. `scripts/w25_full_finetune_smoke.sh` — CPU-runnable smoke that runs the
   trainer in `full` mode for 5 steps on a synthetic clip and asserts the
   checkpoint is reloadable.
4. Tests:
   - `tests/test_trainer.py` — `test_trainer_full_mode_no_lora_wrapper`,
     `test_trainer_full_mode_checkpoint_round_trip`.
   - `tests/test_trainer_cpu.py` — `test_trainer_full_mode_loss_decreases`
     (parallel to the existing LoRA variant).

### Acceptance criteria
- `python -m worldfoundry_train.trainer --mode full --ckpt /tmp/w_full.pt ...`
  runs end-to-end on CPU and saves a checkpoint that the W24 inference loader
  can load.
- SFT schema doc reviewed by the user; one example manifest attached.
- Smoke script passes in CI.

### Out of scope
- Distributed full-FT (FSDP+full FT) — that's W29 integration scope; this
  task is CPU-only + single-GPU NPU smoke.

### Estimated effort
**1 session** (trainer change + doc + smoke).

---

## #4 — W32: AC latent-action head (`actor.predict_latent_action`)

### Goal
`worldfoundry_rl/actor.py:29-40`'s `predict_latent_action` is currently a
passthrough that returns the encoder feature unchanged. W32 lands the
real `latent_action` head: encoder latent + past transitions → next action
prediction. Sister to the W27 v2 inference wiring but on the RL training side.

### Closes
- G8 (T2c side) — AC latent-action head replaces the passthrough.
- Roadmap W32 row.

### Deliverables
1. `worldfoundry_train/model/latent_action_head.py` (new) — `LatentActionHead`
   `nn.Module`: a small MLP `nn.Sequential(Linear(encoder_dim + action_dim,
   hidden), GELU, Linear(hidden, action_dim))` that predicts the next
   action conditioned on the current encoder latent + current action.
2. `worldfoundry_train/rl_losses.py` (new) — `action_prediction_loss` (smooth_l1
   between predicted and ground-truth next action).
3. `worldfoundry_rl/actor.py` — replace passthrough with the new head
   (`actor.predict_latent_action(latent, current_action) -> next_action_pred`).
4. Tests:
   - `tests/test_latent_action_head.py` — construction, forward shape,
     gradient flow, bf16 dtype preservation.
   - `tests/test_actor.py` (new) — `test_actor_predict_latent_action_is_not_passthrough`,
     `test_actor_predict_latent_action_shape`.

### Acceptance criteria
- `predict_latent_action` no longer returns the input latent byte-for-byte.
- Forward shape: `(B, action_dim)`.
- The actor + a synthetic dataset can train a 5-step loop where loss
  decreases.

### Out of scope
- Real MuJoCo/SAPIEN/CARLA environment — that's W31 (env wiring).
- PPO/GRPO — that's W34. W32 is the head only; the loss function is for
  head-only supervised pretraining.

### Estimated effort
**1–2 sessions** (model + actor + tests + smoke).

---

## Tracking

| # | Task | Committed on master? | Owner | Status |
|---|---|---|---|---|
| 1 | W33 plan endpoint | `28e8a6b` (2026-08-06) | — | **done** |
| 2 | W28 orchestration design | `d284809` (2026-08-06) | — | **done** |
| 3 | W25-roadmap full-FT + SFT | `1e5882d` (2026-08-06) | — | **done** |
| 4 | W32 latent-action head | `c9e7197` (2026-08-06) | — | **done** |

All four candidates shipped on 2026-08-06 in this order: W33 → W28 → W25-roadmap → W32.
The CPU-only Phase-3 path described by this document is **closed**. Atlas-gated
work (Phase-4) is tracked separately in `docs/dev/next-steps-phase4-atlas.md`
(see the "What's next" section below).

## What's next

The next planning document is `docs/dev/next-steps-phase4-atlas.md`. It
catalogs the remaining items that **require an Atlas hardware window** to
land (FSDP+full-FT integration, NPU worker parity, PPO/GRPO losses, real
env wiring, scheduler-pool W37), so the next dev session can pick up
without re-discovering which work is blocked on hardware vs. blocked on
spec.

## Cross-reference

- W27 v2 commit: `0d8910f`
- Phase-3 final HEAD: `c9e7197` (W32)
- Roadmap: `docs/roadmap/2026-08-03-jepa2-ascend-gap-analysis-roadmap.md`
- W22 design review addendum: `docs/superpowers/specs/2026-08-05-w22-design-review-addendum.md`
- W35 RL framework ADR: `docs/superpowers/specs/2026-08-04-w35-rl-framework-adr.md`
- W25 full-FT + SFT schema: `docs/superpowers/specs/2026-08-06-w25-full-finetune-sft-schema.md`
- W28 训推一体编排设计: `docs/superpowers/specs/2026-08-06-w28-train-inference-orchestration-design.md`
- **Next plan (Atlas-gated):** `docs/dev/next-steps-phase4-atlas.md`
