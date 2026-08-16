# W43 — M2-T5 vLLM-Ascend e2e (partial closure)

## Context

The W41 spec (`docs/superpowers/specs/2026-08-15-w41-vllm-ascend-integration.md`)
gates **M2-T5** on `registry.available("vllm_ascend") == (True, ...)` inside
the `worldfoundry-npu-vllm:phase1` Docker image. On 2026-08-16, two blockers
were discovered while running the M2-T5 acceptance:

1. **Atlas host has zero internet egress.** `/etc/resolv.conf` nameserver
   `178.27.1.100` is reachable but **DNS resolution returns NXDOMAIN** for
   every mirror tried (`mirrors.aliyun.com`, `pypi.tuna.tsinghua.edu.cn`,
   `hf-mirror.com`, `huggingface.co`) — Atlas is fully air-gapped from the
   public internet. This blocks the "build a fresh `worldfoundry-npu-vllm:phase1`
   image with vllm==0.18.0 + vllm-ascend==0.18.0 wheels" path from C-T3.
2. **The local uv wheel cache only has metadata** (`.msgpack` files) for vllm
   0.18.0 / vllm-ascend 0.18.0 / verl 0.5.0 — `uv pip install --offline`
   reports "vllm<0.18.1 cannot be used" because no actual `.whl` is cached.
   `httpx` and `openai` (transitively-incomplete aarch64 wheels for
   `pydantic-core`) install cleanly; vllm-related wheels do not.

What this means: the M2-T5 acceptance gate as written *cannot* run on the
Atlas project `.venv` today. The state recorded in this summary is the
**partial closure** that is achievable from WSL2 without violating the
"no network installs on Atlas" constraint.

## What was actually validated

### A. Atlas already runs real vllm-ascend Docker containers

```
$ docker images | grep vllm-ascend
quay.io/ascend/vllm-ascend   v0.23.0rc1   13315b656180   3 weeks ago   17.2GB
quay.io/ascend/vllm-ascend   nightly-main-310p  9ddb2231dd02  6 weeks ago 15.7GB
```

`docker ps` shows three vllm-ascend containers actively serving on Atlas:

| Container | Port | Model | Backend |
|---|---|---|---|
| `vllm-auto-6-1025` | 1025 | Llama3-8B-Chinese-Chat | chat |
| `auto-bge-large-5-9123` | 9123 | bge-large-zh-v1.5 | embedding |
| `auto-bge-reranker-5-9124` | 9124 | bge-reranker-large | rerank |

Each exposes the OpenAI-compatible HTTP surface (`/v1/chat/completions`,
`/v1/embeddings`, `/v1/rerank`). All three are **genuine Ascend NPU inference**
served from `quay.io/ascend/vllm-ascend:v0.23.0rc1` containers — confirmed
by the `system_fingerprint: vllm-0.23.0-35793fbf` and the `--additional-config
{"ascend_compilation_config": {"fuse_norm_quant": false}}` flag in their
serve commands.

### B. Direct HTTP probes from the project `.venv`

`uv pip install --offline httpx` succeeds in the Atlas project `.venv`
without network. A minimal httpx probe from the project venv Python
(snapshot in `openai_probe.json`) hits all three live endpoints with
HTTP 200 and finite, well-shaped responses:

| Endpoint | Status | Sample payload |
|---|---|---|
| `:1025/v1/chat/completions` | 200 | `"reply": "Hello! How can I assist you today"`, `system_fingerprint: vllm-0.23.0-35793fbf` |
| `:9123/v1/embeddings` | 200 | `embedding_dim: 1024`, `embedding_sample: [0.0056, -0.0464, -0.0751]` |
| `:9124/v1/rerank` | 200 | `top_score: 0.0343` |

### C. Registry probe — M2-T4 integration is intact

From the same Python session, `worldfoundry_common.backends.list_backends("inference")`
returns `[mindie_sd, sglang, torch, vllm, vllm_ascend, vllm_omni]`, and the
availability probe for each backend matches the M2-T4 spec exactly:

| Backend | `available()` | Hint |
|---|---|---|
| `torch` | True | — |
| `mindie_sd` | True | — |
| `vllm` | False | `pip install -e '.[inf]'; alias of 'vllm_ascend' (canonical name; 'vllm' retained for backward compatibility).` |
| `vllm_ascend` | False | `pip install -e '.[inf]' (requires torch_npu + vllm + vllm_ascend wheels); available only inside the worldfoundry-npu-vllm Docker image.` |
| `sglang` | False | `Install sglang (github.com/sgl-project/sglang) to use this backend.` |
| `vllm_omni` | False | `vllm-omni public wheel pending (W44+); tracked under TODO M2-T6.` |

This proves **the M2-T4 canonical-alias + triple-import probe (commit `c1d5d03`) is
wired correctly** — it reports `vllm_ascend` unavailable with an actionable hint
that points to the right container-image fix, exactly the failure mode M2-T4 was
designed to surface.

## What was NOT validated

- **In-process `VJEPA2NPUWorker` construction** with a real V-JEPA 2 weights
  path through vllm-ascend. This requires `import vllm_ascend` to succeed in
  the project `.venv`, which requires the wheels on disk, which requires
  network egress. Atlas has no egress. **Blocked on M2-T5 to be re-run inside
  the `worldfoundry-npu-vllm:phase1` Docker image once the image is built with
  wheels pre-baked** (C-T3 path; requires `docker buildx build` with a
  mirror-or-tarball of the wheels — not currently possible from WSL2).
- **`--inference-backend vllm_ascend` orchestrator run** producing the cosine
  parity gate from the W41 spec (`cosine` vs `torch` ≤ 0.01 drift, with
  `summary.mode == "npu"`). Same blocker — needs the image.

## Acceptance gate status

| Gate | Criterion | Result |
|---|---|---|
| HTTP-OpenAI-API probe | Atlas vllm-ascend containers reachable from project `.venv` Python | ✅ all 3 endpoints HTTP 200 |
| M2-T4 registry probe | `vllm_ascend` registered with correct hint when wheels absent | ✅ |
| In-process vllm_ascend | `import vllm_ascend` succeeds; `VJEPA2NPUWorker` constructs | ❌ blocked on network egress |
| W41 spec cosine parity | `summary.mode == "npu"`, cosine drift ≤ 0.01 | ❌ blocked on image build |

## How to apply / next steps

- When Atlas gets network egress (or wheels are sideloaded), the same
  probe script can be re-run inside the `worldfoundry-npu-vllm:phase1`
  image to close the in-process gates. The HTTP probe stays valid as the
  smoke baseline.
- The C-T3 follow-up should target `worldfoundry-npu-vllm:phase1` build
  with wheels baked from a sideloaded tarball (e.g., `uv pip download
  --dest wheels/ -r requirements/inf.txt` on a host that has egress,
  then `COPY wheels/` into the image).
- The `bench_results/w43_vllm_ascend_e2e/` directory is the canonical
  evidence location for both partial and full closure of M2-T5; future
  runs should append to `openai_probe.json` rather than overwrite.

## Cross-references

- `worldfoundry_common/backends/inference.py` — M2-T4 canonical alias + probe
- `docs/superpowers/specs/2026-08-15-w41-vllm-ascend-integration.md` — spec
- `docs/roadmap/2026-08-14-opensource-extensions-loop-todo.md` §9.3 — P2 list
- `docker/worldfoundry-npu-vllm/` — C-T3 image target (build pending egress)