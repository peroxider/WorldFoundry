# Quickstart (Phase-1)

## Prerequisites

- 1× Atlas 800 A2 server with 8× Ascend 910B (64GB each), or CPU-only for unit tests
- Docker + Ascend NPU driver installed
- HF token with access to `facebook/vjepa2-vitl-1.2b`

## Run training (DDP LoRA)

```bash
docker run --rm -it --device /dev/davinci0 .../worldforge-base:phase1 \
    torchrun --nproc_per_node=8 \
        worldforge_train/train.py \
        --model facebook/vjepa2-vitl-1.2b \
        --dataset /data/videomix22m \
        --parallel ddp \
        --bf16 --gradient_checkpointing \
        --lora_r 16 --lora_alpha 32 \
        --per_device_batch 2 --grad_accum 8
```

## Run inference service

```bash
docker run --rm -it -p 8080:8080 .../worldforge-base:phase1 \
    python -m worldforge_inf.server \
        --model models/vjepa2-vitl-1.2b \
        --port 8080

# Test
curl -X POST http://localhost:8080/v1/world/encode \
    -H "Authorization: HMAC <token>" \
    -H "Content-Type: application/json" \
    -d '{"video": "<base64>", "adapter": "embodied-v1"}'
```