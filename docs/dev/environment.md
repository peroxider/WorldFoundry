# Dev environment

## Local (no NPU)

```bash
# CPU-only install; NPU ops stubbed in unit tests
pip install -e ".[dev]"
pytest worldfoundry_train/tests worldfoundry_inf/tests worldfoundry_rl/tests
```

## With NPU

```bash
# Build base image (Task 1)
docker build -t worldfoundry-base:phase1 -f docker/worldfoundry-base/Dockerfile .

# Run tests inside container
docker run --rm --device /dev/davinci0:/dev/davinci0 --device /dev/davinci_manager \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
    -v $(pwd):/workspace \
    worldfoundry-base:phase1 \
    pytest worldfoundry_train/tests worldfoundry_inf/tests
```

## Version matrix enforcement

Any deviation from the pinned versions must be:
1. Discussed in a PR with explicit Versioning Policy link
2. Updated in `pyproject.toml`, `docker/.../Dockerfile`, and `worldfoundry_common/npu_env.py`