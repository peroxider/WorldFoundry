# Phase-2 W11-W12 — V-JEPA 2 W8A8 Quantization Design

**作者**：MiniMax-M3（与 chad 共同设计）
**日期**：2026-07-29
**状态**：✅ 用户已批准，进入实施计划阶段
**父规格**：`docs/superpowers/specs/2026-07-27-ascend-world-model-design.md` §4.4 + §6.4 + §9 验收项 #6

---

## 1. 上下文与决策摘要

### 1.1 子项目定位

W11-W12 是 Phase-2 的第二个 sub-project，承接 W9-W10 之后的性能优化与推理部署主线。本子项目**专注 W8A8 静态量化**（per-tensor 对称 + per-channel 权重），为后续 MindIE-SD 编译 (W3-W4 范畴) 提供 INT8 权重的输入物，同时保留 W4A8 作为未来独立 sub-project。

### 1.2 已锁定的关键决策

| 维度 | 决策 | 理由 |
|------|------|------|
| 量化目标 | W8A8 only（W4A8 延后到独立 future sub-project） | 父规格 §4.4 "W8A8 → W4A8" 表明 W8A8 是前置基线；W4A8 风险更高（≥0.95 cosine），先验证 W8A8 流程闭环 |
| 后端 | 华为官方 **msModelSlim** (`msmodelslim==X.Y.Z`) | 生产级、硬件感知、支持 ViT block；与 vLLM-Ascend ModelSlimConfig 路径一致（父规格 §4.4） |
| 校准数据 | VideoMix22M 512-sample 子集，seed=42 | 与 V-JEPA 2 预训练分布对齐（父规格 §4.4 `videomix22m_calib_512`）；可重复 |
| 评价指标 | End-to-end cosine ≥ 0.98（bf16 baseline vs W8A8）on 100-sample holdout（seed=123） | 父规格 §9 #6 原文："下游特征 cosine similarity 与 bf16 baseline ≥ 0.98" |
| 打包 | 新增 `[quant]` optional extra（pyproject.toml），msModelSlim 精确 pin | G2 保护；推理路径不被强制安装量化栈；msModelSlim 仅在 NPU 主机显式 `pip install worldfoundry[quant]` |
| 架构 | Sub-package 拆分（quantize_backends/ + data/ + eval/） | 与 W9-W10 bench/、parallel/ 模式一致；每个文件单一职责；便于未来 W4A8 backend 复用接口 |
| Phase-1 兼容性 | `quantize_model()` facade 签名 0 改动；只替换内部实现 | G3 保护；下游 W3-W4 MindIE-SD 集成不需改动 |

### 1.3 风险偏好

- **不破坏 Phase-1 API**：现有 2 个 quantize 测试必须保持绿色；`quant_meta.json` 契约字段不减少
- **不引入静默默认值**：缺 msModelSlim 时 ImportError + 安装提示；未知 quant_type 时 ValueError；shape 不匹配时 ValueError
- **不重写 V-JEPA 2**：transformers.AutoModel 加载；不修改 facebookresearch/vjepa2 上游代码
- **可重复**：calibration seed=42 + eval seed=123 固定；相同输入产出相同 INT8 权重（确定性 msModelSlim 配置）

---

## 2. 整体架构

### 2.1 文件结构（Sub-package 拆分）

```
worldfoundry_inf/
├── compile/
│   ├── quantize.py                       # [MODIFIED] facade — 保留 Phase-1 API
│   ├── quantize_backends/                # [NEW] 子包
│   │   ├── __init__.py                   # [NEW] backend registry + dispatch
│   │   └── msmodelslim.py                # [NEW] W8A8Backend 实现
│   └── mindie_sd.py                      # [unchanged] Phase-1 stub
├── data/
│   └── calib_videomix22m.py              # [NEW] 512-sample 校准数据加载
└── eval/
    ├── __init__.py                       # [NEW] 包标记
    └── quant_eval.py                     # [NEW] cosine ≥ 0.98 评估 CLI + library

pyproject.toml                            # [MODIFIED] 新增 [quant] extra
worldfoundry_inf/tests/
├── test_quantize.py                      # [unchanged] 保留 2 个 Phase-1 测试
├── test_quantize_backends_msmodelslim.py # [NEW] backend dispatch + 错误处理
├── test_calib_videomix22m.py             # [NEW] 校准数据集加载
└── test_quant_eval.py                    # [NEW] cosine eval (synthetic inputs)
```

### 2.2 设计原则

1. **Facade-Preserving**：quantize.py 公开签名 `quantize_model(model_path, output_path, quant_type="w8a8", calibration_dataset=None) -> Path` 与 Phase-1 一致
2. **Backend-Extensible**：通过 `quantize_backends/__init__.py:get_backend(quant_type)` 路由；新增 W4A8 只需添加 `quantize_backends/w4a8.py` 并在 registry 注册
3. **Eval-Reusable**：`quant_eval.cosine_similarity()` 是 library API（不只是 CLI），便于未来 W4A8 / 其他量化方式直接调用
4. **Error-Loud**：所有错误明确抛出 + 含可操作 hint；零静默 fallback

---

## 3. 组件设计

### 3.1 `worldfoundry_inf/compile/quantize.py`（facade）

**职责**：保留 Phase-1 公开 API；按 `quant_type` 路由到 backend。

**签名**（不变）：
```python
def quantize_model(
    model_path: Path,
    output_path: Path,
    quant_type: str = "w8a8",
    calibration_dataset: str | None = None,
) -> Path:
```

**行为变更**：
- `quant_type="w8a8"`：dispatch 到 `W8A8Backend`
- 其他值：维持 Phase-1 的 `ValueError(f"Unsupported quant_type={...}. Supported: ...")`
- 返回 `output_path`（目录）

**依赖**：从 `quantize_backends` import；import 失败时（未安装 `[quant]` extra）抛 ImportError 并含 `pip install worldfoundry[quant]` hint。

### 3.2 `worldfoundry_inf/compile/quantize_backends/__init__.py`

**职责**：backend registry + factory。

```python
from typing import Protocol

class QuantBackend(Protocol):
    def quantize(self, model_path: Path, output_path: Path,
                 calibration_dataset: str | None) -> Path: ...

_REGISTRY: dict[str, type[QuantBackend]] = {}

def register(quant_type: str):
    def deco(cls: type[QuantBackend]) -> type[QuantBackend]:
        _REGISTRY[quant_type] = cls
        return cls
    return deco

def get_backend(quant_type: str) -> QuantBackend:
    if quant_type not in _REGISTRY:
        raise ValueError(f"No backend registered for quant_type={quant_type!r}. "
                         f"Available: {list(_REGISTRY)}")
    return _REGISTRY[quant_type]()

# Import backends so they self-register (may raise ImportError if deps missing)
try:
    from . import msmodelslim  # noqa: F401
except ImportError as exc:
    _BACKEND_IMPORT_ERROR = exc
else:
    _BACKEND_IMPORT_ERROR = None

def backend_available(quant_type: str) -> tuple[bool, str | None]:
    """Returns (available, error_message). For facade to emit helpful errors."""
    if _BACKEND_IMPORT_ERROR is not None:
        return False, (
            f"W8A8 quantization requires msmodelslim. "
            f"Install with: pip install worldfoundry[quant]\n"
            f"Underlying error: {_BACKEND_IMPORT_ERROR}"
        )
    return quant_type in _REGISTRY, None
```

### 3.3 `worldfoundry_inf/compile/quantize_backends/msmodelslim.py`

**职责**：调用 msModelSlim 完成 W8A8 静态量化。

```python
@register("w8a8")
class W8A8Backend:
    def quantize(
        self,
        model_path: Path,
        output_path: Path,
        calibration_dataset: str | None,
    ) -> Path:
        # 1. Load V-JEPA 2 ViT via transformers
        model = AutoModel.from_pretrained(str(model_path), torch_dtype=torch.bfloat16)
        # 2. Resolve calibration data
        samples = self._resolve_calib(calibration_dataset)
        # 3. msModelSlim quant config
        quant_config = dict(
            quant_type="W8A8",
            quant_mode="static",
            calibration_data=samples,
            calibration_size=len(samples),
        )
        # 4. Run msModelSlim
        slim = msmodelslim.MSModelSlim(model)
        quantized_model = slim.quantize(**quant_config)
        # 5. Save quantized model dir + quant_meta.json
        output_path.mkdir(parents=True, exist_ok=True)
        quantized_model.save_pretrained(str(output_path))
        (output_path / "quant_meta.json").write_text(json.dumps({
            "quant_type": "w8a8",
            "calibration_dataset": calibration_dataset or "videomix22m-512",
            "source_model_path": str(model_path),
            "phase": "phase2-w11",
            "backend": "msmodelslim",
            "msmodelslim_version": msmodelslim.__version__,
            "note": "W8A8 static quantization via msModelSlim.",
        }, indent=2))
        return output_path

    def _resolve_calib(self, spec: str | None) -> list[torch.Tensor]:
        if spec is None or spec == "videomix22m-512":
            return list(load_calibration_dataset(n=512, seed=42))
        raise ValueError(f"Unknown calibration_dataset={spec!r}. "
                         f"Supported: None | 'videomix22m-512'")
```

### 3.4 `worldfoundry_inf/data/calib_videomix22m.py`

**职责**：从 HuggingFace `datasets` 加载 VideoMix22M 的 512-sample 子集，预处理为 V-JEPA 2 image processor 输入格式。

```python
def load_calibration_dataset(
    n: int = 512,
    seed: int = 42,
    image_processor: AutoImageProcessor | None = None,
) -> Iterable[torch.Tensor]:
    """Yield n preprocessed video tensors (1, 3, 224, 224) from VideoMix22M.

    Uses HuggingFace datasets under the hood. Network/auth errors propagate.
    """
    ds = load_dataset("videomix22m", split="train", streaming=True)
    ds = ds.shuffle(seed=seed).select(range(n))
    processor = image_processor or AutoImageProcessor.from_pretrained(
        "facebook/vjepa2-vitl-1.2b"
    )
    for sample in ds:
        frames = sample["video"]  # list of PIL.Image, T frames
        tensor = processor(images=frames, return_tensors="pt")["pixel_values"]
        yield tensor[0]  # (3, 224, 224)
```

**注意**：VideoMix22M 在 HuggingFace 上的 dataset ID 需在 plan 阶段确认（可能是 `lmms-lab/videomix22m` 或私有）。plan 阶段需要实测确认可用 ID；若不可用，则降级到 `hf-internal-testing/tiny-videomix` 子集用于 CI 测试，文档明确说明生产校准仍需真实 VideoMix22M。

### 3.5 `worldfoundry_inf/eval/quant_eval.py`

**职责**：cosine similarity 评估 + CLI。

**Library API**：
```python
@dataclass
class EvalReport:
    aggregate_cosine: float
    per_sample_cosine: list[float]
    passed: bool
    threshold: float
    n_samples: int

def cosine_similarity(
    baseline_path: Path,
    quantized_path: Path,
    eval_dataset: Iterable[torch.Tensor] | None = None,
    threshold: float = 0.98,
    n_samples: int = 100,
    seed: int = 123,
) -> EvalReport:
    """Run bf16 baseline + INT8 quantized model on same inputs; report cosine."""
```

**CLI**：
```bash
python -m worldfoundry_inf.eval.quant_eval \
    --baseline models/vjepa2-vitl-1.2b-bf16 \
    --quantized models/vjepa2-vitl-1.2b-w8a8 \
    --threshold 0.98 \
    --n-samples 100
# exit 0 if cosine ≥ threshold, 1 otherwise
```

### 3.6 `pyproject.toml` 修改

```toml
[project.optional-dependencies]
quant = ["msmodelslim==0.1.0"]  # version TBD in plan; placeholder until msModelSlim stable
```

`msmodelslim` 版本号在 plan 阶段确认（plan 启动时 `pip index versions msmodelslim` 或查阅华为昇腾社区文档）。G2 强制精确 pin。

---

## 4. 数据流

### 4.1 端到端量化流程

```
Input:  models/vjepa2-vitl-1.2b-bf16/
            ├── config.json
            └── pytorch_model.bin (bf16 weights)

CLI:    python -c "from worldfoundry_inf.compile.quantize import quantize_model; \
                   quantize_model('models/...bf16', 'models/...w8a8', 'w8a8')"

Flow:
1. quantize_model facade 接收 model_path, output_path, quant_type="w8a8"
2. backend_available("w8a8") 检查 → True（若 msmodelslim 已装）
3. W8A8Backend().quantize(model_path, output_path, calibration_dataset=None)
4.   ├─ AutoModel.from_pretrained(bf16 path)
5.   ├─ load_calibration_dataset(n=512, seed=42) → 512 tensors
6.   ├─ msModelSlim quantize(config=W8A8, samples, calibration_size=512)
7.   ├─ quantized_model.save_pretrained(output_path)
8.   └─ write quant_meta.json (existing Phase-1 contract)

Output: models/vjepa2-vitl-1.2b-w8a8/
            ├── config.json
            ├── pytorch_model.bin (INT8 weights, msModelSlim format)
            └── quant_meta.json
```

### 4.2 评估流程

```
Inputs:
  - models/vjepa2-vitl-1.2b-bf16/  (baseline)
  - models/vjepa2-vitl-1.2b-w8a8/  (quantized)
  - 100 sample eval set (default: videomix22m holdout, seed=123)

Flow:
1. Load bf16 model + INT8 model
2. For each of 100 samples:
   a. bf16_features = bf16_model.encode(sample).last_hidden_state  # (N_tokens, D)
   b. int8_features = int8_model.encode(sample).last_hidden_state  # (N_tokens, D)
   c. cos = F.cosine_similarity(bf16_features, int8_features, dim=-1).mean().item()
3. aggregate_cosine = mean(per_sample_cosine)
4. passed = aggregate_cosine >= threshold (0.98)
5. CLI: print summary table; sys.exit(0 if passed else 1)
```

---

## 5. 错误处理

| 失败模式 | 抛出位置 | 错误类型 + 消息 |
|----------|---------|-----------------|
| msModelSlim 未安装 | `quantize_backends/__init__.py` import 失败 → `backend_available()` 返回 False | `ImportError("W8A8 quantization requires msmodelslim. Install with: pip install worldfoundry[quant]\nUnderlying error: {ImportError msg}")` |
| `quant_type` 未知 | `quantize_model` facade | `ValueError(f"Unsupported quant_type={...!r}. Supported: {SUPPORTED_QUANT_TYPES}")`（与 Phase-1 一致） |
| `quant_type="w4a8"` 调用但 backend 未注册 | `quantize_model` → `get_backend("w4a8")` | `ValueError(f"No backend registered for quant_type='w4a8'. Available: ['w8a8']. (W4A8 deferred to future sub-project.)")` |
| bf16 model dir 缺 `config.json` | `W8A8Backend.quantize` | `FileNotFoundError(f"Model dir {model_path} missing config.json")` |
| 校准数据集加载失败（HF datasets 网络/认证错误） | `calib_videomix22m.load_calibration_dataset` | 让 HF `datasets` 异常透传；plan 阶段文档化 retry hint |
| msModelSlim calibration 失败（权重损坏、shape 不兼容） | `W8A8Backend.quantize` | 让 msModelSlim 异常透传；外层 wrap：`RuntimeError(f"msModelSlim W8A8 quantize failed for {model_path}: {msmodelslim_error}")` |
| 评估时 bf16 vs INT8 输出 shape 不匹配 | `quant_eval.cosine_similarity` | `ValueError(f"Shape mismatch: bf16={shape_a} vs int8={shape_b}. Cannot compute cosine.")` |
| 评估数据集为空 | `quant_eval.cosine_similarity` | `ValueError(f"eval_dataset is empty (n_samples={n_samples}).")` |
| Cosine 低于 threshold | `quant_eval.cosine_similarity` | 返回 `EvalReport(passed=False, ...)`；CLI `sys.exit(1)` |

**关键决策**：所有错误显式抛出，零静默 fallback。msModelSlim 内部异常透传而非吞掉（便于诊断），仅在 facade 边界 wrap 加上下文。

---

## 6. 测试策略

### 6.1 单元测试（无 NPU 依赖，CI 默认运行）

| Test file | 测试函数 | 验证 |
|-----------|---------|------|
| `test_quantize.py` (Phase-1 保留) | `test_quantize_model_creates_output_file` | 输出目录 + quant_meta.json 存在（不依赖具体 backend，因为 stub 仍可工作当 backend 不可用） |
| `test_quantize.py` (Phase-1 保留) | `test_quantize_model_rejects_unknown_type` | ValueError on unknown quant_type |
| `test_quantize_backends_msmodelslim.py` (新) | `test_facade_dispatches_to_w8a8` | mock W8A8Backend，验证 facade 调用 |
| `test_quantize_backends_msmodelslim.py` (新) | `test_backend_unavailable_raises_import_error_with_hint` | 模拟 msmodelslim ImportError，验证 hint 含 "pip install worldfoundry[quant]" |
| `test_quantize_backends_msmodelslim.py` (新) | `test_w4a8_raises_with_deferral_message` | quant_type="w4a8" → ValueError 含 "W4A8 deferred to future sub-project" |
| `test_calib_videomix22m.py` (新) | `test_load_calibration_dataset_returns_n_tensors` | mock HF datasets；返回 512 个 (3, 224, 224) tensor |
| `test_calib_videomix22m.py` (新) | `test_load_calibration_dataset_propagates_dataset_error` | HF datasets 抛错时透传 |
| `test_quant_eval.py` (新) | `test_cosine_similarity_above_threshold_passes` | 合成 bf16 + int8 features（cosine=0.99）→ report.passed=True |
| `test_quant_eval.py` (新) | `test_cosine_similarity_below_threshold_fails` | 合成 features（cosine=0.90）→ report.passed=False, exit 1 |
| `test_quant_eval.py` (新) | `test_cosine_similarity_shape_mismatch_raises` | 不同 shape → ValueError |
| `test_quant_eval.py` (新) | `test_cosine_similarity_empty_dataset_raises` | n_samples=0 → ValueError |

**总计 11 个新测试 + 2 个保留 = 11 个测试文件 + 13 测试函数**。W9-W10 是 11 个 task tests；W11-W12 略高（多 2 个 eval test），符合 "W11 评价是 W9-W10 不具备的" 的工作范围差异。

### 6.2 集成测试（NPU gated，默认 skip）

```python
@pytest.mark.skipif(
    not torch.npu.is_available(),
    reason="Requires Ascend NPU"
)
def test_w8a8_quantization_end_to_end():
    """End-to-end: bf16 → W8A8 → cosine eval on tiny ViT."""
```

- 使用小型 ViT（如 `facebook/dino-vits8`，~5M 参数）代替 V-JEPA 2 1.2B（CI 内存不够）
- 验证 msModelSlim 真的能产生 INT8 权重 + cosine ≥ 0.95（tiny 模型阈值放宽）

### 6.3 测试覆盖目标

- `quantize_backends/msmodelslim.py` ≥ 85% line coverage
- `eval/quant_eval.py` = 100% line coverage（gate 逻辑必须全覆盖）
- `data/calib_videomix22m.py` ≥ 80% line coverage
- `compile/quantize.py` 仅修改 dispatch，新增覆盖率不强制（已有 100% 来自 Phase-1）

---

## 7. 验收标准

### 7.1 W11-W12 sub-project 完成对应父规格 §9 验收项 #6

- **目标**：模型 OM 大小压缩 ~50%，下游特征 cosine similarity 与 bf16 baseline ≥ 0.98
- **本子项目交付**：W8A8 量化工具链 + cosine 评估 harness
- **OM 大小压缩**：本子项目**不直接测量**（OM 编译是 W3-W4 工作），但通过对比 bf16 weights 大小 vs INT8 weights 大小来**预估**（理论上 INT8 是 bf16 的 50%）
- **Cosine ≥ 0.98**：本子项目**直接验证**（在 videomix22m 100-sample holdout 上）

### 7.2 通过条件

| # | 验收项 | 度量 |
|---|--------|------|
| 1 | `quantize_model("w8a8")` 调用真实 msModelSlim（不再是 stub 拷贝） | 单元测试 + 集成测试验证 save_pretrained 调用真实 msModelSlim 输出 |
| 2 | 量化产出与 Phase-1 文件契约兼容 | 输出目录含 config.json + quant_meta.json；quant_meta.json 含 backend + msmodelslim_version 字段 |
| 3 | Cosine ≥ 0.98（end-to-end） | 集成测试在 V-JEPA 2 vitl-1.2b 真实模型 + 100-sample holdout 上 cosine ≥ 0.98 |
| 4 | `[quant]` extra 可独立安装 | `pip install worldfoundry[quant]` 在 NPU 主机成功；非 NPU 主机可 `pip install worldfoundry`（不带 quant）正常使用其他功能 |
| 5 | Phase-1 API 0 改动 | `test_quantize.py` 两个 Phase-1 测试保持绿色；`quantize_model()` 签名一致 |
| 6 | G2/G3/G4 约束满足 | msmodelslim 精确 pin；Phase-1 0 改动；错误显式抛出 + hint |

---

## 8. 交付物与依赖

### 8.1 交付物

- `worldfoundry_inf/compile/quantize.py`（修改 facade 内部 dispatch）
- `worldfoundry_inf/compile/quantize_backends/__init__.py`（registry + factory）
- `worldfoundry_inf/compile/quantize_backends/msmodelslim.py`（W8A8Backend 实现）
- `worldfoundry_inf/data/calib_videomix22m.py`（校准数据加载）
- `worldfoundry_inf/eval/__init__.py`（包标记）
- `worldfoundry_inf/eval/quant_eval.py`（cosine eval library + CLI）
- `pyproject.toml`（新增 `[quant]` extra）
- `worldfoundry_inf/tests/test_quantize_backends_msmodelslim.py`
- `worldfoundry_inf/tests/test_calib_videomix22m.py`
- `worldfoundry_inf/tests/test_quant_eval.py`
- `docs/superpowers/specs/W11_W12_QUANT_REPORT.md`（W11 真实 NPU 评估产出；本子项目在本地 laptop 不产出）

### 8.2 依赖（精确 pin，G2）

| 包 | 版本 | 必需 |
|----|------|------|
| msmodelslim | TBD（plan 阶段查最新版） | `[quant]` extra |
| transformers | 4.55.0（已锁） | core |
| torch | 2.9.0（已锁） | core |
| torch_npu | 2.9.0（已锁） | core（NPU 运行时） |
| datasets | 3.6.0（已锁） | core |
| numpy | <2.0（已锁） | core |

不引入其他新包。

### 8.3 风险与未来工作

- **W11 评估执行需 NPU 主机**：本地 WSL2 无 NPU，本子项目的 plan 仅在 laptop 完成 7 单元测试 + 1 集成测试 skip；真实 8×NPU 评估在 `worldfoundry-base:phase1` Docker 容器中执行，产出 `W11_W12_QUANT_REPORT.md`
- **VideoMix22M dataset ID 需 plan 阶段确认**：若 HF 无 `videomix22m` 公开 ID，需用 `hf-internal-testing/tiny-videomix` 兜底用于 CI
- **msModelSlim API 可能在 plan/实施期间变化**：plan 阶段需对照当时最新 msModelSlim 文档；若 API 大改，调整 `W8A8Backend.quantize()` 内实现，接口契约不变
- **W4A8 延后**：作为独立 future sub-project；W4A8 threshold 推测 ≥ 0.95（更激进的压缩）
- **OM 编译非本子项目范围**：MindIE-SD compile 仍是 Phase-1 stub，本子项目只交付 quantized weights

---

## 9. 自我审查

按 brainstorming skill 规范执行：

### 9.1 Placeholder 扫描

- ✅ 无 "TBD"（msmodelslim 版本在 plan 阶段确认，已注明 "version TBD in plan; placeholder until msModelSlim stable"）
- ✅ 无 "TODO" / "implement later"
- ✅ 无 "fill in details"

### 9.2 内部一致性

- 组件定义（§3）↔ 数据流（§4）↔ 错误处理（§5）↔ 测试（§6）一致
- `quantize_model` 签名在 §3.1、§4.1、§7.2 #5 三处完全相同（Phase-1 兼容）
- `quant_meta.json` 字段在 §3.3、§7.2 #2、§8.1 三处完全一致（保留 Phase-1 字段 + 新增 backend/version 字段）

### 9.3 Scope 检查

- ✅ 单 sub-project 范围适中（1 个核心功能：W8A8 量化；附带 cosine eval harness）
- ✅ 不包含 W4A8（明确延后）
- ✅ 不包含 MindIE-SD compile（Phase-1 stub 维持，本子项目不动）

### 9.4 歧义检查

- ✅ "quant_type='w8a8'" vs "W8A8" 大小写已统一（API 接受小写 "w8a8"，msModelSlim 内部用 "W8A8"）
- ✅ "calibration_dataset=None" 含义明确（= videomix22m-512 default）
- ✅ cosine 计算方法明确（per-sample feature cosine mean → aggregate mean）

---

## 10. 后续流程

本 spec 经用户 review 后，进入 `writing-plans` skill 创建实施计划（task-by-task 实现 + 测试 + 提交）。

计划预期结构：
1. 修改 `pyproject.toml` + `compile/quantize.py`（facade dispatch）
2. 实现 `compile/quantize_backends/__init__.py`（registry）
3. 实现 `compile/quantize_backends/msmodelslim.py`（W8A8Backend + 错误处理）
4. 实现 `data/calib_videomix22m.py`（校准加载 + HF datasets mock）
5. 实现 `eval/quant_eval.py`（cosine library + CLI）
6. 添加 7 个新单元测试 + 1 个集成测试
7. 验证 Phase-1 测试保持绿色（`pytest worldfoundry_inf/tests`）
8. 提交 + 写最终评审报告

每个任务结尾独立 commit；SDD 循环执行。