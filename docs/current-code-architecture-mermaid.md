# WorldFoundry 当前代码架构与执行关系

> 分析基线：当前仓库代码，而非路线图目标。  
> 图例约定：红色实线表示当前主调用，红色虚线表示可插拔适配或可选路径，灰色线表示数据/制品传递。

## 1. 代码分析结论

WorldFoundry 当前有两条彼此关联、但执行方式不同的主线：

1. **离线 RTI 主线**：渲染/采集 → 世界模型训练 → 推理评估 → 可选图像生成 → 策略训练 → 世界模型与策略交替更新。
2. **在线推理主线**：HTTP/HMAC → Adapter 路由 → 动态批处理 → 编码、掩码预测、时序预测或动作条件规划。

平台的四个大模块分别承担单一类型的职责：

| 大模块 | 核心能力 | 不应混入该层的概念 |
|---|---|---|
| `worldfoundry_common` | 契约、后端发现、设备解析、版本校验 | 具体训练循环、具体推理算法 |
| `worldfoundry_train` | 数据加工、模型目标、优化、并行策略、Checkpoint | HTTP 服务、仿真进程调度 |
| `worldfoundry_inf` | 在线接口、Worker 路由、批处理、latent 推理、图像生成、量化编译 | PPO 更新、环境 step |
| `worldfoundry_rl` | 环境池、Actor rollout、PPO/GAE、Learner 更新、权重广播 | JEPA/VAE 预训练 |

当前实现的边界也需要明确：

- HTTP 服务公开 `/health`、`/v1/world/encode`、`/predict`、`/plan`；**当前没有 `/render` HTTP 接口**。
- `frame_gen_phase()` 是 RTI 编排中的可选阶段，通过 `LatentToFramePipeline` 执行图像生成。
- `vllm_omni` 当前优先尝试真实 `vllm` 包，但主机上的实际 encoder/diffusion 张量计算仍由 ONNX Runtime Worker/Pipeline 承担。
- RTI 编排选择 vLLM、SGLang 或 vLLM-Omni 时会记录 dispatch fingerprint；针对当前编排器本地训练出的 Tiny/HF 模型，`inference_phase()` 仍走进程内 PyTorch 路径。
- `diffusers` 后端仍是预留项；`MindIE-SD → ATC → stub` 是当前 `compile_to_om()` 的实际优先级。

## 2. 外部接口与提供能力

```mermaid
%%{init: {"theme":"base","themeVariables":{"primaryColor":"#fff7f7","primaryTextColor":"#222222","primaryBorderColor":"#a6192e","lineColor":"#a6192e","secondaryColor":"#ffffff","tertiaryColor":"#f7e3e6"}}}%%
flowchart LR
    Client["外部调用方<br/>CLI / HTTP Client / 运维探针"]

    subgraph CLI["命令行入口"]
        RTI["scripts/render_train_rl_cpu.py<br/>完整 RTI 与 RL 闭环"]
        Recipe["scripts/rti_recipe.py<br/>声明式阶段编排"]
        TrainCLI["worldfoundry_train.train<br/>JEPA 训练"]
        VAECLI["scripts/train_vae.py<br/>VAE 重建训练"]
        RLCLI["worldfoundry_rl.distributed_train<br/>多进程 Actor-Learner PPO"]
    end

    subgraph HTTP["在线 HTTP 接口：worldfoundry_inf.server"]
        Health["GET /health<br/>存活、NPU、Adapter、Batcher 状态"]
        Encode["POST /v1/world/encode<br/>视频 → 稠密 latent/features"]
        Predict["POST /v1/world/predict<br/>masked / temporal / action"]
        Plan["POST /v1/world/plan<br/>视频 + 动作 → 未来 latent 与置信度"]
    end

    Client --> RTI
    Client --> Recipe
    Client --> TrainCLI
    Client --> VAECLI
    Client --> RLCLI
    Client --> Health
    Client --> Encode
    Client --> Predict
    Client --> Plan

    Auth["HmacTokenAuth<br/>Authorization + Timestamp"]
    Router["AdapterRouter<br/>按 adapter 名选择 Worker"]
    Batcher["PredictBatcher<br/>max_batch / latency deadline"]
    Worker["InferenceWorker<br/>encode / predict / plan"]

    Encode --> Auth --> Router --> Worker
    Predict --> Auth
    Predict --> Router --> Batcher --> Worker
    Plan --> Auth
    Plan --> Router --> Worker

    classDef entry fill:#ffffff,stroke:#a6192e,stroke-width:2px,color:#222;
    classDef core fill:#fff3f4,stroke:#a6192e,stroke-width:2px,color:#222;
    class Client,RTI,Recipe,TrainCLI,VAECLI,RLCLI,Health,Encode,Predict,Plan entry;
    class Auth,Router,Batcher,Worker core;
```

### HTTP 输入输出语义

| 接口 | 输入 | 提供能力 | 输出 |
|---|---|---|---|
| `GET /health` | 无 | 服务、NPU、Adapter、时序/动作模式、Batcher 状态探测 | 状态 JSON |
| `POST /v1/world/encode` | Base64 视频、可选 `adapter` | 视频编码 | `features`、`shape` |
| `POST /v1/world/predict` | 视频、`mode` 及模式参数 | 掩码补全；未来 latent 预测；动作条件预测 | predicted/future features、score/confidence |
| `POST /v1/world/plan` | 视频、动作序列、过去/未来帧数 | 动作条件规划 | 未来 latent、置信度、动作信息 |

## 3. 离线 RTI 的真实执行顺序

以下顺序直接对应 `scripts/render_train_rl_cpu.py::main()`；可选后端探测发生在正式阶段之前，图像生成位于推理评估之后、策略训练之前。

```mermaid
%%{init: {"theme":"base","themeVariables":{"actorBkg":"#fff7f7","actorBorder":"#a6192e","actorTextColor":"#222222","signalColor":"#a6192e","signalTextColor":"#222222","noteBkgColor":"#f7e3e6","noteBorderColor":"#a6192e","activationBkgColor":"#fdecee","activationBorderColor":"#a6192e"}}}%%
sequenceDiagram
    autonumber
    participant U as CLI / YAML
    participant O as RTI Orchestrator
    participant B as BackendRegistry
    participant E as Render / Env
    participant T as World Model Train
    participant I as Inference Probe
    participant G as Frame Generation
    participant P as Policy / PPO
    participant S as Artifact Store

    U->>O: 参数与后端选择
    O->>B: resolve_config(inference / device / sim_env)
    B-->>O: 后端描述与 dispatch fingerprint

    O->>E: render_phase(cfg)
    E->>E: env.reset / env.step / physics.render
    E-->>S: MP4、Actions、CSV Manifest

    O->>T: train_phase(cfg, manifest, frames, actions)
    T->>T: 编码 → JEPA/重建损失 → backward → optimizer → EMA
    T-->>S: Checkpoint + train history

    O->>I: inference_phase(cfg, checkpoint)
    I->>I: 编码、latent 预测、相似度评估
    I-->>O: predicted pixels + latent + metrics

    opt frame_gen_backend != none
        O->>G: frame_gen_phase(predicted_video, latent_rollout)
        G->>G: latent → scheduler/DiT → VAE decode → RGB
        G-->>S: 生成帧、PSNR/SSIM、engine fingerprint
    end

    O->>P: policy_bootstrap_phase(cfg, checkpoint)
    loop environment rollout
        P->>E: action
        E-->>P: observation, reward, done
    end
    P->>P: GAE → PPO clipped update

    opt rl_backend = rlinf or verl
        O->>B: build_runner(...)
        B-->>P: 外部 RL 框架适配结果
    end

    loop num_outer_iters
        O->>T: 世界模型增量训练
        O->>P: 策略 rollout 与更新
    end
    O-->>S: render_train_rl.json
```

## 4. 在线推理的真实执行顺序

```mermaid
%%{init: {"theme":"base","themeVariables":{"actorBkg":"#fff7f7","actorBorder":"#a6192e","actorTextColor":"#222222","signalColor":"#a6192e","signalTextColor":"#222222","noteBkgColor":"#f7e3e6","noteBorderColor":"#a6192e"}}}%%
sequenceDiagram
    autonumber
    participant C as HTTP Client
    participant A as FastAPI Route
    participant H as HmacTokenAuth
    participant R as AdapterRouter
    participant Q as PredictBatcher
    participant W as VJEPA Worker

    C->>A: POST video + adapter + mode
    A->>H: verify(method, path, body, timestamp, signature)
    H-->>A: authenticated
    A->>A: Base64 decode → NumPy → Torch Tensor
    A->>R: route(adapter_name)
    R-->>A: selected worker

    alt /encode
        A->>W: encode(video)
        W-->>A: dense features
    else /predict and batch method exists
        A->>Q: submit(PredictRequest)
        Q->>Q: 按 mode、最大批量、延迟门限聚合
        Q->>W: predict_*_batch(...)
        W-->>Q: per-request results
        Q-->>A: future/predicted features + confidence
    else /predict direct fallback
        A->>W: predict / predict_temporal / predict_action_conditioned
        W-->>A: features + score/confidence
    else /plan
        A->>W: predict_action_conditioned(...)
        W-->>A: future latent + confidence
    end
    A-->>C: JSON response
```

## 5. 大模块能力与关系

这张图只表达“谁依赖谁、提供什么能力”，不表达执行时序。

```mermaid
%%{init: {"theme":"base","themeVariables":{"primaryColor":"#fff7f7","primaryTextColor":"#222222","primaryBorderColor":"#a6192e","lineColor":"#a6192e","secondaryColor":"#ffffff","tertiaryColor":"#f7e3e6"}}}%%
flowchart TB
    Entry["入口与编排<br/>CLI / Recipe / FastAPI"]

    Common["worldfoundry_common<br/>契约、注册发现、设备与版本治理"]
    Train["worldfoundry_train<br/>世界模型训练与 Checkpoint"]
    Inf["worldfoundry_inf<br/>在线推理、图像生成、量化编译"]
    RL["worldfoundry_rl<br/>环境交互、Rollout、PPO 更新"]

    Data[("视频 / 动作 / Manifest")]
    Weight[("Checkpoint / Adapter / OM / ONNX")]
    Metric[("Metrics / JSON Summary")]

    Entry -->|阶段调度| Train
    Entry -->|服务启动与请求路由| Inf
    Entry -->|策略训练调度| RL
    Entry -->|后端解析| Common

    Train -->|依赖设备、后端、适配协议| Common
    Inf -->|依赖异常、Worker 协议、后端注册| Common
    RL -->|依赖训练/回滚协议、环境后端| Common

    Data --> Train
    RL -->|环境帧与动作| Data
    Train -->|发布模型权重| Weight
    Weight --> Inf
    Weight --> RL
    Inf -->|latent / 生成帧| RL
    Train --> Metric
    Inf --> Metric
    RL --> Metric

    classDef module fill:#fff3f4,stroke:#a6192e,stroke-width:2px,color:#222;
    classDef store fill:#ffffff,stroke:#777777,stroke-width:1.5px,color:#222;
    class Entry,Common,Train,Inf,RL module;
    class Data,Weight,Metric store;
```

## 6. 大模块内部的重要组件

### 6.1 `worldfoundry_common`：统一契约与选择机制

```mermaid
flowchart LR
    Config["YAML / CLI backend config"] --> Resolve["resolve_config / Stack.compose"]
    Resolve --> Registry["BackendRegistry<br/>register / list / build"]
    Registry --> Device["device<br/>cpu / npu / auto"]
    Registry --> Training["training<br/>self / fsdp / mindspeed / rlinf / verl"]
    Registry --> Inference["inference<br/>torch / vllm_ascend / onnx / omni / diffusion"]
    Registry --> Model["model<br/>tiny / transformers"]
    Registry --> Sim["sim_env<br/>mock / mujoco / manipulator / droid"]

    Protocols["结构化协议<br/>InferenceWorker / RolloutWorker / TrainingRunner"]
    Runtime["NPU Runtime<br/>版本校验 / 设备探测 / HCCL 初始化"]
    Registry -. 构建结果遵守 .-> Protocols
    Device --> Runtime

    classDef c fill:#fff3f4,stroke:#a6192e,color:#222;
    class Resolve,Registry,Protocols,Runtime c;
```

### 6.2 `worldfoundry_train`：两条训练路径与统一优化机制

```mermaid
flowchart LR
    Input[("视频帧 + Actions + Manifest")]
    Dataset["Vjepa2Dataset / VideoSource"]
    Mask["TubeletMaskGenerator"]
    Collator["Vjepa2Collator"]

    subgraph JEPA["预测式世界模型训练"]
        Online["Online Encoder<br/>TinyVJEPA2 或 HF VJEPA2 Shim"]
        Target["EMA Target Encoder<br/>stop-gradient"]
        Predictor["JEPAPredictor / TemporalPredictor / ActionPredictor"]
        JLoss["masked / temporal / action-conditioned loss"]
        EMA["EMAUpdater.after_step"]
    end

    subgraph GEN["生成式训练"]
        VAE["VAEModel.encode / decode"]
        KLLoss["重建损失 + KL loss"]
    end

    Strategy["DistributedStrategy<br/>DDP / FSDP / MindSpeed-MM"]
    Optim["Optimizer + grad accumulation"]
    CKPT[("Checkpoint")]

    Input --> Dataset --> Mask --> Collator
    Collator --> Online --> Predictor --> JLoss
    Collator --> Target --> JLoss
    JLoss --> Strategy --> Optim --> EMA --> Target
    Collator --> VAE --> KLLoss --> Strategy
    Optim --> CKPT

    classDef model fill:#fff3f4,stroke:#a6192e,color:#222;
    classDef algo fill:#ffffff,stroke:#777777,color:#222;
    class Online,Target,Predictor,VAE model;
    class JLoss,KLLoss,EMA,Strategy,Optim algo;
```

### 6.3 `worldfoundry_inf`：在线推理与图像生成

```mermaid
flowchart LR
    HTTP["FastAPI Routes"] --> Auth["HMAC 校验"] --> Router["AdapterRouter"]
    Router --> Batcher["PredictBatcher"]

    subgraph WM["世界模型推理 Worker"]
        TorchW["VJEPA2Worker<br/>PyTorch"]
        NPUW["VJEPA2NPUWorker<br/>vLLM-Ascend"]
        ONNXW["OnnxVJEPAWorker<br/>ONNX Runtime"]
    end

    Batcher --> TorchW
    Batcher -. 可选后端 .-> NPUW
    Batcher -. 可选后端 .-> ONNXW

    subgraph ImageGen["图像生成：LatentToFramePipeline"]
        Latent["encode_latents / rollout"]
        DiT["DiffusionTransformer<br/>预测噪声或 x0"]
        Scheduler["DDIM / Flow Matching Scheduler"]
        VAE["VAEModel.decode"]
        RGB["RGB Frames"]
        Latent --> Scheduler
        Scheduler -->|逐步调用 denoiser| DiT
        DiT --> Scheduler
        Scheduler --> VAE --> RGB
    end

    TorchW -->|features / latent| Latent
    ONNXW -->|features / latent| Latent

    Quant["QuantBackend<br/>W8A8 等"] --> Compile["compile_to_om<br/>MindIE-SD / ATC / stub"]
    Compile --> Artifact[("OM / ONNX / quantized model")]

    classDef model fill:#fff3f4,stroke:#a6192e,color:#222;
    classDef infra fill:#ffffff,stroke:#777777,color:#222;
    class TorchW,NPUW,ONNXW,Latent,DiT,VAE model;
    class HTTP,Auth,Router,Batcher,Scheduler,Quant,Compile infra;
```

### 6.4 `worldfoundry_rl`：Actor–Environment–Learner 闭环

```mermaid
flowchart LR
    Learner["LearnerPool<br/>主策略与版本号"]
    ActorPool["ActorPool × N<br/>multiprocessing.Process"]
    Actor["Actor Policy<br/>VisionPolicy.act"]
    EnvPool["AsyncEnvPool × M<br/>Pipe IPC"]
    Env["Env.reset / step<br/>Mock / dm_control / Manipulator"]
    Chunk[("RolloutChunk<br/>obs, action, logp, value, reward, done")]
    GAE["compute_gae"]
    PPO["PPO.update<br/>clipped surrogate + value + entropy"]

    Learner -->|versioned CPU state_dict| ActorPool
    ActorPool --> Actor
    Actor -->|actions| EnvPool --> Env
    Env -->|observations / rewards / done| EnvPool --> Actor
    Actor --> Chunk --> ActorPool --> Learner
    Learner --> GAE --> PPO -->|更新主策略并 version + 1| Learner

    classDef runtime fill:#fff3f4,stroke:#a6192e,color:#222;
    classDef data fill:#ffffff,stroke:#777777,color:#222;
    class Learner,ActorPool,Actor,EnvPool,Env,GAE,PPO runtime;
    class Chunk data;
```

## 7. 自研模块调用第三方开源库的接口与能力

| 自研模块 | 第三方库及接口 | 使用的能力 | 当前状态 |
|---|---|---|---|
| `worldfoundry_common.npu_env`、`worldfoundry_train.device` | `torch_npu`；`torch.npu.is_available()`、`device_count()`、Tensor/Module `.to("npu")` | 昇腾设备发现、设备放置、NPU 张量计算 | 当前生产设备路径 |
| `worldfoundry_common.npu_env`、并行策略 | `torch.distributed.init_process_group(backend="hccl")`、`all_reduce` | 多进程通信、梯度同步 | NPU 用 HCCL，CPU 显式切为 Gloo |
| `worldfoundry_train.parallel.ddp` | `torch.nn.parallel.DistributedDataParallel`、`no_sync()` | 数据并行、梯度累积期间抑制同步 | 已实现 |
| `worldfoundry_train.parallel.fsdp` | `torch.distributed.fsdp.FullyShardedDataParallel` 及 state-dict API | 参数/梯度/优化器状态分片、Checkpoint 汇聚 | 已实现 |
| `worldfoundry_train.parallel.mindspeed` | MindSpeed-MM 包装入口与 PyTorch Optimizer 接口 | 昇腾大模型并行训练接入 | 可选后端、延迟导入 |
| `worldfoundry_train.model.hf_vjepa_shim` | `transformers.VJEPA2Config`、`VJEPA2Model`、`from_pretrained()` | V-JEPA 2 Backbone、HF Hub/本地权重加载 | 已接入；由 shim 统一输出协议 |
| `worldfoundry_train.model.lora_wrapper` | `peft.LoraConfig`、`get_peft_model()` | 对注意力/MLP 线性层实施 LoRA 参数高效微调 | 已实现，并检查非零可训练参数 |
| HF encoder 量化路径 | `torchao.quantization.Int8WeightOnlyConfig`、量化 Tensor 类型 | HF V-JEPA 权重仅 W8 量化 | 可选路径 |
| `worldfoundry_inf.server`、`api/*` | `fastapi.FastAPI`、`APIRouter`、`Request`、`HTTPException`；`uvicorn.run()` | HTTP 路由、依赖注入、错误响应与 ASGI 服务 | 已实现 |
| `worldfoundry_inf.workers.vllm_ascend_worker` | vLLM Worker 基类/模型注册；`vllm-ascend` NPU Platform | 将 VJEPA Worker 接入 vLLM 调度与 Ascend 平台 | Atlas 可选路径 |
| `worldfoundry_inf.omni.pipeline_real` | `vllm.entrypoints.llm.LLM`、构造签名探测 | 真实 vLLM 包接入和运行能力指纹 | 当前主机多为探测；张量计算仍走 ORT |
| `worldfoundry_inf.workers.onnx_vllm_worker` | `onnxruntime.InferenceSession.run()`、Execution Providers | 非 PyTorch encoder 推理；CPU EP 或 Ascend EP | 已实现 |
| `worldfoundry_inf.diffusion.backends.onnx_diffusion` | `torch.onnx.export()`、`onnxruntime.InferenceSession.run()` | DiT/decoder 导出与非 PyTorch 图像生成推理 | 已实现 |
| `worldfoundry_inf.compile.quantize_backends.msmodelslim` | `msmodelslim.MSModelSlim(model).quantize(...)` | W8A8 静态量化、校准数据驱动量化 | Ascend 可选依赖 |
| `worldfoundry_inf.compile.mindie_sd` | `mindie.sd.tools.compile(...)` | 量化模型编译为 Ascend OM | 优先编译路径；依赖可用时执行 |
| `worldfoundry_inf.compile.mindie_sd` | `atc --model ... --framework 5 --soc-version ...` | ONNX → Ascend OM 离线编译 | 次优先真实路径 |
| `worldfoundry_inf.compile.mindie_sd` | JSON metadata writer | 在 MindIE-SD/ATC 均不可用时保留制品契约 | fallback；不是可执行 OM |
| `worldfoundry_inf` 的 `diffusers` backend | `diffusers` 包可用性探测 | 预留 Hugging Face diffusion pipeline 接口 | 当前 `NotImplementedError`，不能算已提供能力 |
| `worldfoundry_rl.env_franka_kitchen` | `dm_control.manipulation.load()`、`env.reset()`、`env.step()`、`env.physics.render()` | MuJoCo 操控任务、奖励、状态与 RGB 渲染 | 已实现，重依赖延迟导入 |
| `worldfoundry_train.data.synthetic_video` | `cv2.VideoWriter` | 合成视频编码并写入 MP4 | 已实现 |
| `worldfoundry_rl.rlinf_factory` | `rlinf.runners.embodied_runner.EmbodiedRunner` | 检查构造签名并尝试构造真实 EmbodiedRunner；完整训练由 `RLinfRunner.run()` 适配面承接 | 依赖 live Ray cluster；构造失败会保留结构化指纹 |
| `worldfoundry_rl.verl_factory` | `verl.trainer.main_ppo.run_ppo`、`is_cuda_available` | 探测真实 verl 入口、签名、文件和运行环境 | 当前 factory 的 `RealVerlRunner.run()` 只记录接入指纹，不直接启动完整 `run_ppo` |
| `worldfoundry_common.adapters.verl.VerlAdapter` | `verl.trainer.main_ppo.run_ppo(cfg)` | 真正调用 verl 的程序化 PPO 入口 | 适配协议已实现；需要完整 verl 配置与运行环境 |
| `worldfoundry_rl.actor_pool`、`env_pool` | Python `multiprocessing.Process`、`Pipe`、`spawn` | Actor 与环境 Worker 多进程隔离、IPC、权重广播 | 已实现的 CPU-faithful 分布式骨架 |

## 8. 阅读图时的层次约束

- **数据**只画成圆柱或数据节点：视频、动作、Manifest、RolloutChunk。
- **模型**只画成模型节点：Encoder、Predictor、DiT、VAE、VisionPolicy。
- **算法/训练范式**只画成处理步骤：JEPA loss、KL loss、GAE、PPO。
- **分布式方法**只作为执行策略：DDP、FSDP、MindSpeed-MM、Actor/Env 多进程。
- **第三方库**只画在系统边界之外，并通过具体 API 名连接自研适配器。

因此，模块图回答“谁提供什么能力”，时序图回答“先执行什么、后执行什么”，第三方表回答“具体调用什么接口获得什么能力”，三类信息不再堆叠在一个框中。
