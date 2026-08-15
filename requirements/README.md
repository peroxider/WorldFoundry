# `requirements/` — split lockfile

> C-T1 (per `docs/roadmap/2026-08-14-opensource-extensions-loop-todo.md` §9.1 P0):
> 平铺锁版，让 `uv pip check` 在 host / container 内均可独立校验矩阵，
> 同时给 C-T3 P1 多阶段 Dockerfile (`npu` / `npu-vllm` / `npu-diffusion`)
> 提供单一 `pip install -r` 入口。

## Layout

| 文件 | 作用 | 镜像目标 |
|---|---|---|
| `base.txt` | 镜像 `pyproject.toml [project] dependencies`；torch / transformers / accelerate / triton / fastapi | 全部镜像 |
| `npu.txt` | 文档化 CANN 8.5.1 约束；torch_npu 已在 base | `npu` |
| `rl.txt` | verl + ray（verl 是 shipped 路径，RLinf 推迟） | `npu-rl` |
| `inf.txt` | vllm + vllm-ascend | `npu-vllm` |
| `dev.txt` | pytest + ruff | dev 镜像 |
| `quant.txt` | msmodelslim + torchao | quant 镜像 |
| `video.txt` | av + decord + opencv-python | video 镜像 |
| `diffusion.txt` | diffusers + accelerate（Route B v2 待 M3-T4） | `npu-diffusion` |
| `vllm_omni.txt` | vllm-omni（公开 wheel 未发；M3-T7 / M2-T6 待 vllm-omni 公开后启用） | 推迟 |

## 与 `pyproject.toml` 的关系

`pyproject.toml [project.optional-dependencies]` 是**单一事实源**；
`requirements/*.txt` 是镜像构建时的**平铺视图**。
两者版本号必须保持一致；若其中一边改了，另一边必须同步修改并重跑
`uv pip check`（见 `scripts/validate_opensource_extensions_on_npu.py` 的
`check_c_t1`）。

## 安装示例

```bash
# 全部可选栈
.venv/bin/python -m pip install -r requirements/dev.txt

# 仅 RL 栈（用于 Atlas verl e2e 容器）
.venv/bin/python -m pip install -r requirements/rl.txt
```

## 版本矩阵来源

- `torch_npu==2.9.0` / `transformers==4.55.0` / `vllm-ascend==0.18.0`：
  项目 master pins（W35.2 §2.1）。
- `accelerate==1.14.0` / `triton==3.7.1` / `ray==2.56.1`：RLinf v0.3.0
  spike pip-freeze（W35.2 §2.2）。
- CANN 8.5.1：宿主驱动（详见 `docs/dev/environment.md`），非 pip 包。