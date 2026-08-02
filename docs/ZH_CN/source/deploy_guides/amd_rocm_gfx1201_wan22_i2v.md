# AMD ROCm GFX1201 上的 Wan2.2 I2V

本文描述 `gfx1201-hvat-scratch` 分支中的 Wan2.2 I2V 管线。目标卡为 32 GB 显存、原生支持 E4M3/E5M2 FP8 矩阵计算的 AMD RDNA4/GFX1201。蒸馏 FP8 支持 1/2/4/8/16 卡：1/2/4/8 卡使用 Ulysses，16 卡使用 Ring，均关闭 CFG。

> GFX1201 attention 内核已经完成 720P 代表 shape 的正确性和性能验证；完整 Wan2.1/Wan2.2 生成仍需在模型机器上验证画质、显存和端到端耗时。

## 支持矩阵

| 管线 | `model_cls` | 步数 | DiT 权重 | 状态 |
| --- | --- | ---: | --- | --- |
| 标准 BF16 I2V | `wan2.2_moe` | 40 | `Wan-AI/Wan2.2-I2V-A14B` 中的 `high_noise_model` 和 `low_noise_model` | GFX1201 静态适配，待硬件验证 |
| 蒸馏 BF16 I2V | `wan2.2_moe_distill` | 4 | `Wan2.2-Distill-Models` 的 high/low BF16 合并权重 | 已配置，待硬件验证 |
| 蒸馏 FP8 I2V | `wan2.2_moe_distill` | 4 | `Wan2.2-Distill-Models` 的 high/low scaled FP8 权重 | 已配置，待硬件验证 |

当前没有为 GFX1201 提供 T2V、INT8 或 ComfyUI 管线。Wan2.2 high/low distill LoRA 可通过 `LightX2VRun` 一键合并、量化为 scaled FP8，并复用同一套多卡推理入口。

## Aiter 原生 SageAttention2 路径

AMD 平台的 `aiter_fav3_sage_bf16_attn` 调用 Aiter 的 `fav3_sage_wrapper_func`，并由 LightX2V 适配器显式传入 `backend=sage_attn_v2_gfx1201`，因此不会回退到 Triton Sage v1。输入和输出为 BF16，Q/K 按块量化为 INT8，V 按通道量化为 GFX1201 原生 `float8_e4m3fn`；计算内核使用 gfx1201 的 INT8 QK WMMA 和 FP8 PV WMMA，对位 4090 SageAttention2 的 Q/K INT8、V FP8 数据流。

现有 FlyDSL/Triton BF16 配置继续作为基线。以下独立配置只把 self-attention 切换到 Sage V2；text/image cross-attention 保持已调优的 `aiter_triton_bf16_flash_attn`，便于在不引入短序列 Sage 退化的前提下比较画质、显存和耗时：

- `configs/platforms/amd_rocm/wan22_moe_i2v_bf16_40step_r9600_sage_gfx1201.json`
- `configs/platforms/amd_rocm/wan22_moe_i2v_distill_bf16_4step_sage_gfx1201.json`
- `configs/platforms/amd_rocm/wan22_moe_i2v_distill_fp8_4step_sage_gfx1201.json`

Wan2.1 的 720P 同类入口为 `configs/platforms/amd_rocm/wan21_t2v_sage_gfx1201.json` 和 `configs/platforms/amd_rocm/wan21_i2v_sage_gfx1201.json`。

最终生产配置为 `BLOCK_M=128`、`BLOCK_N=32`、`waves_per_eu=2`、`LDS_PADDING=16`、K-only prefetch、`PRE_LOAD_V=false` 和 `USE_FP8_P_OFFSET=true`。720P 三 workload sweep 中，K-only prefetch、offset 关闭时的 kernel geomean 为调优后 FlyDSL BF16 的 `1.3840x`，full-call geomean 为 `1.3087x`；开启 offset 后 full-call geomean 约为 `1.3062x`。

720P sampled-row FP32 验证覆盖 Wan2.1 H5、Wan2.2 H5 和 Wan2.2 TI2V H3，共 6 个 workload/seed case。FP8 P offset 在 6/6 case 中提高 cosine 并降低 RMSE，平均归一化 P L1 error 从约 `0.03242` 降到 `0.02250`，相对 offset 关闭的 full-call 速度为 `0.99584x`，因此默认开启。

原生 V2 当前接受 dense、non-causal、D=128、Q/K/V 序列长度和 head 数相同的 self-attention，并可选返回经过 K-smoothing 修正的自然对数 LSE。LightX2V 适配器会在调用边界检查这些条件，并显式选择 `sage_attn_v2_gfx1201`；不支持的调用会直接报错，不会静默回退 Sage v1。Wan text/image cross-attention 继续使用已调优的 `aiter_triton_bf16_flash_attn`。

Wan A14B 有 40 个 attention heads，16 不能整除 40，因此 16 卡继续使用 Ring SP，不能改成 16 路 Ulysses。Ring 每轮调用原生 Sage V2 获得当前 K/V shard 的 output 和 LSE，再用 log-sum-exp 权重合并 16 个 shard；1/2/4/8 卡仍使用 Ulysses。该 LSE 路径已完成代码和 CPU 契约检查，正式作为 16 卡生产路径前还需要运行 gfx1201 sampled-row Ring merge correctness。

LightX2VRun 保留 `--infer-fp8` 作为 BF16 attention 基线，原生 Sage V2 使用独立入口。16 卡命令为：

```bash
GPU_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
  ./lightx2vModelrun --infer-sage-fp8
```

该入口生成 `seq_p_size=16`、`seq_p_attn_type=ring` 和 `self_attn_1_type=aiter_fav3_sage_bf16_attn` 的运行时配置。使用 1/2/4/8 个 GPU id 调用同一入口时自动改用 Ulysses，self-attention 仍为 Sage V2。

LightX2V 固定 Aiter revision 为 `630e9c108106dbf2e21125d687ea9fcc142d264c`（`feat(sage): expose gfx1201 V2 LSE for Ring SP`，包含已调优的 K-prefetch 和 FP8-P-offset 默认值）。更新该 revision 后必须重新构建镜像，现有镜像中的 Aiter wheel 和 LightX2V 源码不会自动变化。

## 镜像构建源码

镜像构建使用本地 LightX2V、AITER 和 LightX2VRun，不在 Dockerfile 中 clone 仓库。默认三个仓库互为 sibling：

```text
Code/
|-- LightX2V/
|-- aiter/
`-- LightX2VRun/
```

构建前在宿主机初始化 AITER submodule：

```bash
git -C ../aiter \
  -c http.proxy=http://127.0.0.1:10808 \
  -c https.proxy=http://127.0.0.1:10808 \
  submodule update --init --recursive
```

然后从 LightX2V 仓库执行：

```bash
HTTP_PROXY=http://127.0.0.1:10808/ \
HTTPS_PROXY=http://127.0.0.1:10808/ \
bash dockerfiles/platforms/build_gfx1201.sh
```

构建脚本会校验三个工作树均已提交且干净，并确认 AITER 和 LightX2VRun 的 HEAD 分别匹配可选的 `AITER_COMMIT`、`LIGHTX2V_RUN_COMMIT`。随后通过独立的 BuildKit named context 复制 AITER 源码和 LightX2VRun 运行脚本；最终镜像记录三个 revision。目录不为 sibling 时使用 `AITER_SOURCE_DIR` 和 `LIGHTX2V_RUN_SOURCE_DIR` 覆盖。`HTTP_PROXY`、`HTTPS_PROXY` 和 `NO_PROXY` 仅在调用者显式设置时转发，不会持久化到镜像。

Dockerfile 直接调用该固定 AITER 源码中的 `.github/scripts/install_triton.sh`，AITER wheel 构建层和最终运行层继承同一个 Triton 层。最终镜像同时保留 `hipcc` 所需的 C/C++、Python headers、CMake 和 Ninja 环境，用于 AITER 首次运行时 JIT；不会继承基础镜像中未经 AITER 选择的 Triton，也不会把完整 AITER 源码留在最终层。

## 32 GB GFX1201 验证顺序

1. 首先验证蒸馏 FP8 4 步。它使用 E4M3 scaled FP8 DiT、FP8 T5、Aiter FlyDSL/Triton BF16 Flash Attention 和 Aiter FP8 GEMM，是最匹配该卡硬件能力的管线。LightX2V 保留 per-channel/rowwise scale 契约并调用 Aiter 公共 A8W8 dispatcher；gfx1201 的已调优形状精确命中 CK lookup，未调优形状先使用 16x16 WMMA CK heuristic，只有 CK 的 `IsSupportedArgument` 明确拒绝该形状时才转到 rowwise Triton。FP8 指 GEMM 的 A/W 数据类型；`dtype=torch.bfloat16` 是累加后的输出类型，attention Q/K/V 也仍为 BF16。
2. FP8 通过后再验证蒸馏 BF16 4 步。BF16 配置使用 phase offload，速度会明显低于 FP8 主路径，并需要更多主机内存。
3. 最后验证标准 BF16 40 步。它主要用于确认非蒸馏原始模型兼容性，不适合作为性能或首次成功标准。

上游文档记录的 720p/81 帧蒸馏 FP8 单卡 offload 峰值为 `29250 MiB`，但该数据来自 H100，不能直接等同于 ROCm/AITER 的显存占用。相对 32 GB 显存，它只有约 3.5 GiB 的名义余量，ROCm allocator、AITER workspace 或算子实现差异都可能消耗这部分空间。

当前 FP8 配置使用 `offload_granularity=model`：每个阶段只把当前约 15 GB 的 high-noise 或 low-noise FP8 DiT 放到 GPU，另一份保留在 CPU。这是性能优先配置。如果首次运行 OOM，第一降级项是保持 FP8 权重并把配置临时改为 `offload_granularity=phase`；确认 phase offload 能完成后，再分析是否恢复 model offload。不要把切换 BF16 当作第一降级手段，因为 BF16 会增加权重存储、主机内存和传输量。

主机需要同时容纳模型权重、offload buffer、T5、VAE 和运行时内存。FP8 首测建议至少准备 64 GB 主机内存；BF16 phase offload 和标准 40 步建议优先使用 128 GB 主机内存。

## `wan_moe_i2v_4090.json` 的意义

该文件是标准 Wan2.2 MoE 40 步 I2V 配置，不是 4 步蒸馏配置。它的价值在于 `cpu_offload=true` 和 `offload_granularity=phase` 所代表的低显存卸载方式。

不能直接在 GFX1201 上使用它：

- `flash_attn3` 面向 NVIDIA Hopper。本分支的 self-attention 改为 `aiter_flydsl_bf16_flash_attn`，text/image cross-attention 改为 `aiter_triton_bf16_flash_attn`。
- 标准 `WanScheduler` 使用 `infer_steps` 和连续的 `boundary` 切换 high/low noise 模型。
- 文件中的 `boundary_step_index` 和 `denoising_step_list` 只由蒸馏调度器消费，对标准 40 步管线无效，因此 R9600 配置没有保留它们。

对应配置为：

`configs/platforms/amd_rocm/wan22_moe_i2v_bf16_40step_r9600_gfx1201.json`

该配置同时卸载 T5 和 VAE。phase offload 会显著增加主机内存占用和 PCIe 数据传输，建议镜像验证时同时记录 GPU 峰值显存、主机峰值内存和单步耗时。它是 32 GB 卡上的兼容性补充，不是推荐的首测或性能管线。

## 模型目录

建议把模型统一挂载到 `/models`，目录结构如下：

```text
/models/
|-- Wan-AI/Wan2.2-I2V-A14B/
|   |-- high_noise_model/
|   |-- low_noise_model/
|   |-- models_t5_umt5-xxl-enc-bf16.pth
|   `-- Wan2.1_VAE.pth
|-- lightx2v/Wan2.2-Distill-Models/
|   |-- wan2.2_i2v_A14b_high_noise_lightx2v_4step.safetensors
|   |-- wan2.2_i2v_A14b_low_noise_lightx2v_4step.safetensors
|   |-- wan2.2_i2v_A14b_high_noise_scaled_fp8_e4m3_lightx2v_4step.safetensors
|   `-- wan2.2_i2v_A14b_low_noise_scaled_fp8_e4m3_lightx2v_4step.safetensors
`-- encoders/t5/models_t5_umt5-xxl-enc-fp8.pth
```

三条管线都需要原始 `Wan-AI/Wan2.2-I2V-A14B` 根目录，因为 tokenizer、VAE 和其他支撑文件仍从这里加载。BF16 蒸馏管线额外使用两份约 28.58 GB 的非量化蒸馏权重；FP8 蒸馏管线使用两份约 15.01 GB 的 scaled FP8 权重和 FP8 T5。

`Wan2.2-Distill-Models` 中其他文件的处理方式：

- `*_int8_*`：当前 GFX1201 AITER 配置未接入，不要用于这三条启动脚本。
- `*_comfyui.safetensors` 和 `wan2.2_i2v_scale_fp8_comfyui.json`：面向 ComfyUI 打包，不用于原生 LightX2V 管线。
- `*_split/`：当前配置没有选择这些拆分目录。
- `*_1030.safetensors`：属于可替换的 high-noise 版本，但当前配置固定使用非 `1030` high/low 配对，替换后需要单独验证。
- `*_720p_260412.safetensors`：当前配置没有选择，单文件约 57.16 GB，不应在首次 R9600 验证中使用。
- LoRA：需要 `Wan2.2-Distill-Loras` 和对应 LoRA 配置，不属于当前 GFX1201 支持矩阵。

## 启动

### 容器脚本启动

镜像在 `/opt/LightX2VRun` 中保留构建时已提交的运行脚本快照，但不固定模型卷、不设置全局模型路径，也不替换容器入口。`/workspace/LightX2V/scripts/platforms/amd_rocm/` 只保留两个双卡兼容入口：Distill FP8 和 LoRA FP8。它们优先使用宿主 bind mount 到 `/workspace/LightX2VRun` 的对应脚本；没有检测到挂载时会打印 `WARNING`，再使用 `/opt/LightX2VRun` 快照。模型和结果目录由 `docker run` 挂载：

```bash
docker run --rm -it \
  --device=/dev/kfd \
  --device=/dev/dri \
  --ipc=host \
  -e GPU_ARCHS=gfx1201 \
  -e CU_NUM=24 \
  -v /path/to/models:/models:ro \
  -v /path/to/outputs:/workspace/LightX2V/save_results \
  lightx2v-rocm:gfx1201-hvat-scratch
```

进入容器后执行：

```bash
bash /workspace/LightX2V/scripts/platforms/amd_rocm/run_wan22_moe_i2v_distill_fp8_4step_gfx1201.sh
```

FP8 启动脚本会在挂载的 `AITER_JIT_DIR` 下使用 `gfx1201-cu24-rowwise-v2` 子目录。旧目录中的 `module_gemm_a8w8.so` 不包含新的 gfx1201 lookup，不能直接复用；新命名空间只会在首次启动时重新 JIT。

这两个镜像内兼容入口固定调用双卡脚本，默认使用逻辑 GPU `0,1`。1/4/8/16 卡从挂载的 `/workspace/LightX2VRun/scripts` 启动。LightX2V 默认为 `/workspace/LightX2V`，模型根目录为 `/models`，Wan2.2 基础模型默认为 `/models/Wan-AI/Wan2.2-I2V-A14B`。每步进度和最终 DiT `s/it` 默认开启。

### 源码脚本启动

进入容器后设置：

```bash
export LIGHTX2V_PATH=/workspace/LightX2V
export MODELS_ROOT=/models
export MODEL_PATH=/models/Wan-AI/Wan2.2-I2V-A14B
export HIP_VISIBLE_DEVICES=0
export GPU_ARCHS=gfx1201
export CU_NUM=24
```

标准 BF16 40 步 R9600 低显存配置：

```bash
bash /workspace/LightX2V/scripts/platforms/amd_rocm/run_wan22_moe_i2v_bf16_40step_r9600_gfx1201.sh
```

蒸馏 BF16 4 步：

```bash
bash /workspace/LightX2V/scripts/platforms/amd_rocm/run_wan22_moe_i2v_distill_bf16_4step_gfx1201.sh
```

蒸馏 FP8 4 步：

```bash
bash /workspace/LightX2V/scripts/platforms/amd_rocm/run_wan22_moe_i2v_distill_fp8_4step_gfx1201.sh
```

LoRA 合并、scaled FP8 量化并执行蒸馏 FP8 4 步：

```bash
bash /workspace/LightX2V/scripts/platforms/amd_rocm/run_wan22_moe_i2v_distill_lora_fp8_4step_gfx1201.sh
```

这两个镜像内入口固定为双卡。挂载 `LightX2VRun` 后，1/4/8/16 卡仍从
`/workspace/LightX2VRun/scripts` 中对应的带卡数脚本启动。

两个蒸馏脚本会切换到 `MODELS_ROOT` 后启动，以便配置中的 `lightx2v/...` 和 `encoders/...` 相对路径稳定解析。标准 40 步管线只读取 `MODEL_PATH` 下的原始 high/low noise 模型，不会读取 `Wan2.2-Distill-Models`。
