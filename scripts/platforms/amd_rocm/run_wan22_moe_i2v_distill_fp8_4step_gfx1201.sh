#!/usr/bin/env bash
set -euo pipefail

lightx2v_path="${LIGHTX2V_PATH:-/workspace/LightX2V}"
models_root="${MODELS_ROOT:-/models}"
model_path="${MODEL_PATH:-${models_root}/Wan-AI/Wan2.2-I2V-A14B}"

export PLATFORM=amd_rocm
export DTYPE=BF16
export TOKENIZERS_PARALLELISM=false
export GPU_ARCHS="${GPU_ARCHS:-gfx1201}"
export CU_NUM="${CU_NUM:-24}"
export AITER_LOG_TUNED_CONFIG="${AITER_LOG_TUNED_CONFIG:-1}"
if [[ "${GPU_ARCHS}" != "gfx1201" || "${CU_NUM}" != "24" ]]; then
    echo "gfx1201 FP8 requires GPU_ARCHS=gfx1201 and CU_NUM=24 to match the tuned AITER lookup" >&2
    exit 1
fi
# AITER imports an existing module_gemm_a8w8.so without checking whether its
# generated lookup came from an older CSV. Namespace this JIT revision once so
# persistent cache mounts cannot silently reuse the pre-tuning module.
aiter_jit_root="${AITER_JIT_DIR:-/workspace/.cache/aiter}"
export AITER_JIT_DIR="${aiter_jit_root%/}/gfx1201-cu24-rowwise-v2"
mkdir -p "${AITER_JIT_DIR}"
# On ROCm, setting HIP_VISIBLE_DEVICES and CUDA_VISIBLE_DEVICES to the same
# non-zero physical id can filter the already-filtered device list twice.
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-0,1}}"
unset CUDA_VISIBLE_DEVICES
IFS=',' read -r -a gpu_devices <<<"${HIP_VISIBLE_DEVICES}"
if [[ ! "${HIP_VISIBLE_DEVICES}" =~ ^[0-9]+,[0-9]+$ || "${gpu_devices[0]}" == "${gpu_devices[1]}" ]]; then
    echo "gfx1201 FP8 SP=2 requires two different HIP devices, for example HIP_VISIBLE_DEVICES=0,1" >&2
    exit 1
fi

export PYTHONPATH="${PYTHONPATH:-}"
source "${lightx2v_path}/scripts/base/base.sh"

cd "${models_root}"

torchrun --standalone --nproc_per_node=2 -m lightx2v.infer \
    --model_cls wan2.2_moe_distill \
    --task i2v \
    --model_path "${model_path}" \
    --config_json "${lightx2v_path}/configs/platforms/amd_rocm/wan22_moe_i2v_distill_fp8_4step_gfx1201.json" \
    --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside." \
    --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
    --image_path "${lightx2v_path}/assets/inputs/imgs/img_0.jpg" \
    --save_result_path "${lightx2v_path}/save_results/output_lightx2v_wan22_moe_i2v_distill_fp8_4step_gfx1201.mp4"
