#!/usr/bin/env bash
set -Eeuo pipefail

mounted_run_root="${LIGHTX2V_RUN_ROOT:-/workspace/LightX2VRun}"
bundled_run_root="${LIGHTX2V_BUNDLED_RUN_ROOT:-/opt/LightX2VRun}"
launcher_rel=scripts/run_wan22_moe_i2v_distill_lora_fp8_4step_gfx1201_2gpu.sh

if [[ -f "${mounted_run_root}/${launcher_rel}" ]]; then
    launcher="${mounted_run_root}/${launcher_rel}"
else
    echo "WARNING: LightX2VRun is not mounted at ${mounted_run_root}; using the bundled launcher snapshot from ${bundled_run_root}." >&2
    echo "WARNING: mount the current LightX2VRun checkout at /workspace/LightX2VRun to receive launcher-only updates without rebuilding the image." >&2
    launcher="${bundled_run_root}/${launcher_rel}"
fi

if [[ ! -f "${launcher}" ]]; then
    echo "ERROR: bundled GFX1201 LoRA launcher not found: ${launcher}" >&2
    exit 1
fi

exec bash "${launcher}" "$@"
