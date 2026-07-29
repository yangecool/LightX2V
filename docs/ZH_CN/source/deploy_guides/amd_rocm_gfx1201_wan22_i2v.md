# AMD ROCm GFX1201 上的 Wan2.2 I2V

本文只描述 `gfx1201-hvat-scratch` 分支中已经配置的单卡 Wan2.2 I2V 管线。目标卡为 32 GB 显存、原生支持 E4M3/E5M2 FP8 矩阵计算的 AMD RDNA4/GFX1201。所有配置均使用 Aiter FlyDSL BF16 Flash Attention 作为 self-attention，使用 Aiter Triton BF16 Flash Attention 作为 text cross-attention，并使用 Torch RMSNorm/RoPE、关闭多卡并行。

> 所有 GFX1201 配置尚未进行镜像或硬件验证。32 GB 和原生 FP8 能力使蒸馏 FP8 成为最合理的首测管线，但不代表已经确认能在 32 GB 内完成 720p/81 帧生成。

## 支持矩阵

| 管线 | `model_cls` | 步数 | DiT 权重 | 状态 |
| --- | --- | ---: | --- | --- |
| 标准 BF16 I2V | `wan2.2_moe` | 40 | `Wan-AI/Wan2.2-I2V-A14B` 中的 `high_noise_model` 和 `low_noise_model` | GFX1201 静态适配，待硬件验证 |
| 蒸馏 BF16 I2V | `wan2.2_moe_distill` | 4 | `Wan2.2-Distill-Models` 的 high/low BF16 合并权重 | 已配置，待硬件验证 |
| 蒸馏 FP8 I2V | `wan2.2_moe_distill` | 4 | `Wan2.2-Distill-Models` 的 high/low scaled FP8 权重 | 已配置，待硬件验证 |

当前没有为 GFX1201 提供 T2V、LoRA、INT8、ComfyUI 权重或多卡配置。源码中存在这些能力并不等于本分支的 GFX1201 配置已经覆盖它们。

## 镜像构建源码

镜像构建使用本地 LightX2V 和 AITER 源码，不在 Dockerfile 中 clone 这两个仓库。默认目录布局为：

```text
LightX2V-ROCm-GFX1201/
|-- LightX2V/
`-- aiter/
```

构建前在宿主机初始化 AITER submodule：

```bash
git -C ../aiter submodule update --init --recursive
```

然后从 LightX2V 仓库执行：

```bash
bash dockerfiles/platforms/build_gfx1201.sh
```

构建脚本会校验本地 AITER 工作树干净、HEAD 等于 Dockerfile 固定的 `AITER_COMMIT`，并确认所有 submodule 已初始化且处于固定 revision；随后通过独立的 BuildKit `aiter_source` context 将源码交给 Dockerfile。LightX2V 仍使用主 build context 的 `COPY`。`AITER_SOURCE_DIR` 只用于覆盖默认的 sibling 目录位置，不改变 commit 校验。

## 32 GB GFX1201 验证顺序

1. 首先验证蒸馏 FP8 4 步。它使用 E4M3 scaled FP8 DiT、FP8 T5、Aiter FlyDSL/Triton BF16 Flash Attention 和 Aiter FP8 GEMM，是最匹配该卡硬件能力的管线。FP8 指 DiT GEMM 权重和计算；attention Q/K/V 仍为 BF16。
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

与 NVIDIA 镜像保持相同的职责边界：Dockerfile 只提供运行环境和 LightX2V 源码，不固定模型卷、不设置全局模型路径，也不替换容器入口。模型和结果目录由 `docker run` 挂载：

```bash
docker run --rm -it \
  --device=/dev/kfd \
  --device=/dev/dri \
  --ipc=host \
  -v /path/to/models:/models:ro \
  -v /path/to/outputs:/workspace/LightX2V/save_results \
  lightx2v-rocm:gfx1201-hvat-scratch
```

进入容器后执行：

```bash
bash /workspace/LightX2V/scripts/platforms/amd_rocm/run_wan22_moe_i2v_distill_fp8_4step_gfx1201.sh
```

该 GFX1201 脚本提供容器内默认值：LightX2V 位于 `/workspace/LightX2V`，模型根目录为 `/models`，Wan2.2 基础模型位于 `/models/Wan-AI/Wan2.2-I2V-A14B`，默认使用逻辑 GPU 0。路径仍可通过 `LIGHTX2V_PATH`、`MODELS_ROOT`、`MODEL_PATH`、`HIP_VISIBLE_DEVICES` 或 `CUDA_VISIBLE_DEVICES` 覆盖，不影响 Dockerfile 的通用性。

### 源码脚本启动

进入容器后设置：

```bash
export LIGHTX2V_PATH=/workspace/LightX2V
export MODELS_ROOT=/models
export MODEL_PATH=/models/Wan-AI/Wan2.2-I2V-A14B
export HIP_VISIBLE_DEVICES=0
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

两个蒸馏脚本会切换到 `MODELS_ROOT` 后启动，以便配置中的 `lightx2v/...` 和 `encoders/...` 相对路径稳定解析。标准 40 步管线只读取 `MODEL_PATH` 下的原始 high/low noise 模型，不会读取 `Wan2.2-Distill-Models`。
