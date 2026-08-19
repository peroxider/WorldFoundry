# W25: Full-Parameter Fine-Tuning + SFT Data Schema

Status: shipped (closes roadmap G5)
Date: 2026-08-06
Parent plan: `docs/dev/next-steps-w27-v2.md` § #3

## Why this exists

Roadmap G5 is the only remaining Phase-3 loose end that's CPU-only and
shippable without Atlas hardware: there is no entry point for **full
parameter fine-tuning** (the trainer is LoRA-only), and there is no
documented **SFT data format** for non-LoRA adaptation. W25 closes both
gaps with a single, narrow change set:

- A `--mode {lora, full}` knob on `worldfoundry_train.train` that
  bypasses the PEFT/LoRA wrapper and trains every encoder parameter
  directly. Existing checkpoint / EMA / FSDP machinery is reused.
- A JSONL manifest schema for SFT data that the W24 inference loader
  can consume.

Full mode is intentionally **not** a new training algorithm: it is the
same supervised loop as LoRA mode, minus the low-rank adapter. The
predictor + EMA target encoder (W22) are orthogonal and continue to
work in either mode.

## Mode switch

### CLI

```bash
python -m worldfoundry_train.train --config <yaml> --mode full
```

`--mode` accepts `lora` (default, byte-for-byte identical to pre-W25
behavior) or `full`. Unknown values fail-fast (ValueError) at startup,
mirroring the `--parallel` guard. The mode may also be encoded in the
YAML under `training.mode`; the CLI flag wins.

### Programmatic

```python
from worldfoundry_train.trainer import run_training

losses = run_training(
    cfg, strategy, mode="full",
    base_model=encoder,        # any nn.Module
    num_steps=100,
    batch_fn=...,
)
```

The `mode` kwarg is **keyword-only** and defaults to `"lora"`. The
existing public signature is otherwise unchanged; all LoRA-mode call
sites continue to work without edits.

### Internal branching

In `full` mode the trainer:

1. **Skips** `build_lora_model`. The base model is wrapped by
   `strategy.wrap` directly (the same call site as LoRA mode, so FSDP /
   DDP coverage is unchanged).
2. **Enables** `requires_grad=True` for every parameter via
   `_enable_full_finetune(base_model)`. The downstream
   `_build_optimizer` already filters by `requires_grad`, so the
   optimizer correctly sees **all** parameters (not just LoRA adapters).
3. **Stores** `lora_signature=None` in the checkpoint payload. The
   `save_checkpoint` / `load_checkpoint` / `validate_lora_signature`
   triple is already tolerant of `None`:
   - `save_checkpoint` writes `"lora_signature": None` when
     `lora_signature` is `None` (no `to_dict()` call).
   - `load_checkpoint` returns `lora_signature=None` on a full-FT
     payload.
   - `validate_lora_signature` short-circuits when the checkpoint
     signature is `None`, so full-FT resume never raises a LoRA-shape
     mismatch.

`_trainable_state_dict` (the W16 helper) filters by
`requires_grad=True`, so a full-FT checkpoint contains **every
parameter** (vs. a LoRA checkpoint, which only contains the adapter
weights). This is the same checkpoint format the W24 inference loader
already consumes for masked-mode inference.

## SFT data schema

### Manifest (one JSON per line)

```
{"video_path": "<abs or repo-relative path>",
 "start_frame": 0,
 "num_frames": 16,
 "action": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
 "label": "swing_high"}
```

### Field spec

| Field         | Type                       | Required | Notes |
|---------------|----------------------------|----------|-------|
| `video_path`  | `str`                      | yes      | Absolute path or path resolvable relative to the manifest. Decoder must accept the container/codec at load time. |
| `start_frame` | `int`                      | yes      | 0-indexed inclusive offset into the video file. |
| `num_frames`  | `int`                      | yes      | Number of consecutive frames to sample. Must be `>= 1`. |
| `action`      | `Optional[list[float]]`    | no       | Per-frame action vector. If present, length **must** match the model's `action_dim` (default 7) **per row** in the temporal target grid. The trainer does not validate this at the manifest level — the JEPA action-conditioned objective (W22) does. |
| `label`       | `Optional[str]`            | no       | Free-form label for downstream SFT. The trainer does not consume it; it is preserved for evaluation and RL-actor warm-start use cases. |

### Temporal-mode validation

When `jepa.enabled=true` and the trainer runs in `temporal` mode, every
manifest row must satisfy:

```
num_frames >= past_frames + future_frames
```

where `past_frames` and `future_frames` are read from the JEPA config.
The dataset constructor enforces this at sample time; a row that fails
the check raises a `ValueError` keyed on the offending field
(`num_frames` < required, or `start_frame + num_frames` exceeds the
video length).

When `jepa.enabled=false` (the default), only `num_frames >= 1` and
`start_frame >= 0` are checked; the rest is decoder territory.

### Example manifest (`examples/w25_sft_smoke/manifest.jsonl`)

```
{"video_path": "examples/w25_sft_smoke/clip_000.mp4", "start_frame": 0,   "num_frames": 16, "action": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], "label": "straight"}
{"video_path": "examples/w25_sft_smoke/clip_001.mp4", "start_frame": 4,   "num_frames": 16, "action": [0.1, 0.0,-0.2, 0.0, 0.0, 0.0, 0.0], "label": "turn_left"}
{"video_path": "examples/w25_sft_smoke/clip_002.mp4", "start_frame": 12,  "num_frames": 16}
```

Rows without `action` and `label` are valid (both are optional). The
collator treats missing actions as a masked-predict sample (no AC
objective term) and missing labels as unlabeled.

## Acceptance

- `python -m worldfoundry_train.train --mode full --steps 5 --config <yaml>`
  runs end-to-end on CPU, produces a `.pt` checkpoint, and the
  checkpoint is loadable by `load_checkpoint` (round-trip test in
  `tests/test_trainer.py::test_run_training_full_mode_checkpoint_round_trip`).
- `tests/test_trainer_cpu.py::test_trainer_full_mode_loss_decreases_on_real_vjepa2`
  exercises the gradient flow end-to-end on a real (tiny) HF VJEPA 2
  encoder and asserts the loss decreases across 5 steps.
- `scripts/archive/w25/w25_full_finetune_smoke.sh` runs both of the above and
  exits 0.

## Cross-references

- W22 design: `docs/superpowers/specs/2026-07-27-ascend-world-model-design.md`
- W25 LoRA wrapper: `worldfoundry_train/model/lora_wrapper.py`
- W16 checkpoint: `worldfoundry_train/checkpoint.py` (line 308,
  line 426, line 441)
- W25-roadmap parent: `docs/dev/next-steps-w27-v2.md` § #3

## Out of scope

- Distributed full-FT (FSDP+full FT) is W29 integration scope; this
  schema and mode work is CPU-only + single-GPU NPU smoke.
- Multi-step rollout SFT (past K, predict H, take as input for next H,
  …) is W26 v2; the schema here is a flat per-clip manifest.
- The training-side `ActionConditionedObjective` (W27+) is orthogonal;
  the manifest's `action` field is already consumed by the W22
  collator when `jepa.enabled=true`.
