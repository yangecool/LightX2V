#!/usr/bin/env bash
set -euo pipefail

lightx2v_path="${LIGHTX2V_PATH:-/workspace/LightX2V}"
models_root="${MODELS_ROOT:-/models}"
model_path="${MODEL_PATH:-${models_root}/Wan-AI/Wan2.2-I2V-A14B}"
output_dir="${LIGHTX2V_OUTPUT_DIR:-/outputs}"
input_image="${LIGHTX2V_INPUT_IMAGE:-${lightx2v_path}/assets/inputs/imgs/img_0.jpg}"
gpu_device="${LIGHTX2V_GPU:-0}"
seed="${LIGHTX2V_SEED:-42}"

high_noise_ckpt="${models_root}/lightx2v/Wan2.2-Distill-Models/wan2.2_i2v_A14b_high_noise_scaled_fp8_e4m3_lightx2v_4step.safetensors"
low_noise_ckpt="${models_root}/lightx2v/Wan2.2-Distill-Models/wan2.2_i2v_A14b_low_noise_scaled_fp8_e4m3_lightx2v_4step.safetensors"
t5_ckpt="${models_root}/encoders/t5/models_t5_umt5-xxl-enc-fp8.pth"
config_path="${lightx2v_path}/configs/platforms/amd_rocm/wan22_moe_i2v_distill_fp8_4step_gfx1201.json"

default_prompt="Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."
default_negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
prompt="${LIGHTX2V_PROMPT:-${default_prompt}}"
negative_prompt="${LIGHTX2V_NEGATIVE_PROMPT:-${default_negative_prompt}}"

required_paths=(
    "${model_path}"
    "${model_path}/google/umt5-xxl"
    "${model_path}/Wan2.1_VAE.pth"
    "${high_noise_ckpt}"
    "${low_noise_ckpt}"
    "${t5_ckpt}"
    "${config_path}"
    "${input_image}"
)

missing_paths=()
for required_path in "${required_paths[@]}"; do
    if [[ ! -e "${required_path}" ]]; then
        missing_paths+=("${required_path}")
    fi
done

if (( ${#missing_paths[@]} > 0 )); then
    echo "LightX2V cannot start because required files are missing:" >&2
    printf '  - %s\n' "${missing_paths[@]}" >&2
    echo >&2
    echo "Mount the prepared model directory at /models." >&2
    exit 1
fi

if [[ ! -e /dev/kfd ]]; then
    echo "LightX2V cannot start because /dev/kfd is unavailable." >&2
    echo "Start the container with --device=/dev/kfd --device=/dev/dri." >&2
    exit 1
fi

if [[ -n "${LIGHTX2V_OUTPUT_PATH:-}" ]]; then
    output_path="${LIGHTX2V_OUTPUT_PATH}"
else
    output_path="${output_dir}/wan22_i2v_fp8_$(date +%Y%m%d_%H%M%S).mp4"
fi
mkdir -p "$(dirname "${output_path}")"

export PLATFORM=amd_rocm
export DTYPE=BF16
export TOKENIZERS_PARALLELISM=false
export HIP_VISIBLE_DEVICES="${gpu_device}"
export CUDA_VISIBLE_DEVICES="${gpu_device}"

source "${lightx2v_path}/scripts/base/base.sh"

echo "Starting Wan2.2 I2V 4-step FP8 generation"
echo "GPU: ${gpu_device}"
echo "Input: ${input_image}"
echo "Output: ${output_path}"

cd "${models_root}"
exec python3 -m lightx2v.infer \
    --model_cls wan2.2_moe_distill \
    --task i2v \
    --model_path "${model_path}" \
    --config_json "${config_path}" \
    --prompt "${prompt}" \
    --negative_prompt "${negative_prompt}" \
    --image_path "${input_image}" \
    --save_result_path "${output_path}" \
    --seed "${seed}"
