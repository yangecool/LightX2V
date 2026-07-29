#!/usr/bin/env bash
set -euo pipefail

lightx2v_path="${LIGHTX2V_PATH:-}"
models_root="${MODELS_ROOT:-}"
model_path="${MODEL_PATH:-}"
: "${lightx2v_path:?Set LIGHTX2V_PATH to the LightX2V checkout}"
: "${models_root:?Set MODELS_ROOT to the directory containing Wan-AI, lightx2v, and encoders}"
: "${model_path:?Set MODEL_PATH to the Wan2.2 model root}"

export PLATFORM=amd_rocm
export DTYPE=BF16
export TOKENIZERS_PARALLELISM=false
gpu_devices="${HIP_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-0}}"
export HIP_VISIBLE_DEVICES="${gpu_devices}"
export CUDA_VISIBLE_DEVICES="${gpu_devices}"

source "${lightx2v_path}/scripts/base/base.sh"

cd "${models_root}"

python -m lightx2v.infer \
    --model_cls wan2.2_moe_distill \
    --task i2v \
    --model_path "${model_path}" \
    --config_json "${lightx2v_path}/configs/platforms/amd_rocm/wan22_moe_i2v_distill_bf16_4step_gfx1201.json" \
    --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside." \
    --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
    --image_path "${lightx2v_path}/assets/inputs/imgs/img_0.jpg" \
    --save_result_path "${lightx2v_path}/save_results/output_lightx2v_wan22_moe_i2v_distill_bf16_4step_gfx1201.mp4"
