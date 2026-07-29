# AMD ROCm GFX1201 上的 Wan2.2 I2V

本文只描述 `gfx1201-hvat-scratch` 分支中已经配置的单卡 Wan2.2 I2V 管线。所有配置均使用 AITER 注意力、Torch RMSNorm/RoPE，并关闭多卡并行。

> `R9600/GFX1201` 标准 40 步配置来自上游 `wan_moe_i2v_4090.json` 的低显存思路，但尚未进行镜像或硬件验证。它是待验证配置，不代表已经确认能在目标显存容量内完成 720p/81 帧生成。

## 支持矩阵

| 管线 | `model_cls` | 步数 | DiT 权重 | 状态 |
| --- | --- | ---: | --- | --- |
| 标准 BF16 I2V | `wan2.2_moe` | 40 | `Wan-AI/Wan2.2-I2V-A14B` 中的 `high_noise_model` 和 `low_noise_model` | GFX1201 静态适配，待硬件验证 |
| 蒸馏 BF16 I2V | `wan2.2_moe_distill` | 4 | `Wan2.2-Distill-Models` 的 high/low BF16 合并权重 | 已配置，待硬件验证 |
| 蒸馏 FP8 I2V | `wan2.2_moe_distill` | 4 | `Wan2.2-Distill-Models` 的 high/low scaled FP8 权重 | 已配置，待硬件验证 |

当前没有为 GFX1201 提供 T2V、LoRA、INT8、ComfyUI 权重或多卡配置。源码中存在这些能力并不等于本分支的 GFX1201 配置已经覆盖它们。

## `wan_moe_i2v_4090.json` 的意义

该文件是标准 Wan2.2 MoE 40 步 I2V 配置，不是 4 步蒸馏配置。它的价值在于 `cpu_offload=true` 和 `offload_granularity=phase` 所代表的低显存卸载方式。

不能直接在 GFX1201 上使用它：

- `flash_attn3` 面向 NVIDIA Hopper，本分支改为 `aiter_attn`。
- 标准 `WanScheduler` 使用 `infer_steps` 和连续的 `boundary` 切换 high/low noise 模型。
- 文件中的 `boundary_step_index` 和 `denoising_step_list` 只由蒸馏调度器消费，对标准 40 步管线无效，因此 R9600 配置没有保留它们。

对应配置为：

`configs/platforms/amd_rocm/wan22_moe_i2v_bf16_40step_r9600_gfx1201.json`

该配置同时卸载 T5 和 VAE。phase offload 会显著增加主机内存占用和 PCIe 数据传输，建议镜像验证时同时记录 GPU 峰值显存、主机峰值内存和单步耗时。

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
