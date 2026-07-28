# WorldForge base image

Locked version matrix per spec §6.2:

- Base: `ascend-pytorch:2.9.0-cann8.5.1-ubuntu22.04`
- torch_npu: 2.9.0
- vllm: 0.18.0
- vllm-ascend: 0.18.0
- transformers: 4.55.0
- peft: 0.18.0

Build:
```bash
docker build -t worldforge-base:phase1 -f docker/worldforge-base/Dockerfile .
```

Run smoke (CPU fallback — NPU required for real tests):
```bash
docker run --rm -it worldforge-base:phase1 bash
```