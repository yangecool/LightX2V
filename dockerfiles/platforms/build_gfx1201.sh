#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "$0")" && pwd)"
lightx2v_dir="$(cd "${script_dir}/../.." && pwd)"
aiter_source_dir="${AITER_SOURCE_DIR:-${lightx2v_dir}/../aiter}"
lightx2v_run_source_dir="${LIGHTX2V_RUN_SOURCE_DIR:-${lightx2v_dir}/../LightX2VRun}"

base_image_source="${BASE_IMAGE:-vllm/vllm-openai-rocm:v0.25.0-base}"
image_tag="${IMAGE_TAG:-lightx2v-rocm:gfx1201-hvat-scratch}"
apt_mirror="${APT_MIRROR:-https://mirrors.ctyun.cn/ubuntu}"
pypi_mirror="${PYPI_MIRROR:-https://pypi.tuna.tsinghua.edu.cn/simple}"
uv_version="${UV_VERSION:-0.11.32}"
pytorch_rocm_arch="${PYTORCH_ROCM_ARCH:-gfx1201}"
aiter_rocm_arch="${AITER_ROCM_ARCH:-gfx1201}"
aiter_cu_num="${AITER_CU_NUM:-24}"
aiter_repo="${AITER_REPO:-https://gitcode.com/hvat-ai/aiter.git}"
max_jobs="${MAX_JOBS:-$(nproc)}"
lightx2v_revision="$(git -C "${lightx2v_dir}" rev-parse HEAD)"

if [[ ! -d "${lightx2v_run_source_dir}" && -z "${LIGHTX2V_RUN_SOURCE_DIR:-}" ]]; then
    nested_workspace_run_dir="${lightx2v_dir}/../../LightX2VRun"
    if [[ -d "${nested_workspace_run_dir}" ]]; then
        lightx2v_run_source_dir="${nested_workspace_run_dir}"
    fi
fi

if [[ "${aiter_cu_num}" != "24" ]]; then
    echo "This gfx1201 image uses the LightX2V/AITER 24-CU dispatch contract; got AITER_CU_NUM=${aiter_cu_num}" >&2
    exit 1
fi

proxy_build_args=()
if [[ -n "${HTTP_PROXY:-}" ]]; then
    proxy_build_args+=(--build-arg "HTTP_PROXY=${HTTP_PROXY}")
fi
if [[ -n "${HTTPS_PROXY:-}" ]]; then
    proxy_build_args+=(--build-arg "HTTPS_PROXY=${HTTPS_PROXY}")
fi
if [[ -n "${NO_PROXY:-}" ]]; then
    proxy_build_args+=(--build-arg "NO_PROXY=${NO_PROXY}")
fi

if [[ ! -d "${aiter_source_dir}" ]]; then
    echo "Local Aiter source directory does not exist: ${aiter_source_dir}" >&2
    exit 1
fi
aiter_source_dir="$(cd "${aiter_source_dir}" && pwd)"
aiter_revision="$(git -C "${aiter_source_dir}" rev-parse HEAD)"
aiter_commit="${AITER_COMMIT:-${aiter_revision}}"
if [[ "${aiter_revision}" != "${aiter_commit}" ]]; then
    echo "Local Aiter revision ${aiter_revision} does not match requested revision ${aiter_commit}" >&2
    exit 1
fi
if [[ -n "$(git -C "${aiter_source_dir}" status --porcelain --untracked-files=all)" ]]; then
    echo "Local Aiter worktree must be clean before it is copied into the image" >&2
    exit 1
fi
aiter_submodule_status="$(git -C "${aiter_source_dir}" submodule status --recursive)"
if [[ "${aiter_submodule_status}" =~ (^|$'\n')[-+U] ]]; then
    echo "Local Aiter submodules must be initialized at their pinned revisions" >&2
    echo "Run with the local proxy:" >&2
    echo "git -C ${aiter_source_dir} -c http.proxy=http://127.0.0.1:10808 -c https.proxy=http://127.0.0.1:10808 submodule update --init --recursive" >&2
    exit 1
fi

if [[ -n "${AITER_VERSION:-}" ]]; then
    aiter_version="${AITER_VERSION}"
else
    aiter_tag="$(git -C "${aiter_source_dir}" describe --tags --abbrev=0 --match 'gfx1201-hvat-scratch-v*' 2>/dev/null || true)"
    if [[ -z "${aiter_tag}" ]]; then
        aiter_tag="$(git -C "${aiter_source_dir}" describe --tags --abbrev=0 --match 'v*' 2>/dev/null || true)"
    fi
    if [[ -z "${aiter_tag}" ]]; then
        echo "Unable to derive an Aiter base version from a reachable Git tag" >&2
        echo "Set AITER_VERSION explicitly." >&2
        exit 1
    fi
    aiter_base_version="${aiter_tag##*-v}"
    aiter_base_version="${aiter_base_version#v}"
    aiter_version="${aiter_base_version}+gfx1201.g${aiter_revision:0:10}"
fi

if ! base_image_id="$(docker image inspect --format '{{.Id}}' "${base_image_source}")"; then
    echo "Base image is not available locally: ${base_image_source}" >&2
    echo "Pull it before building: docker pull ${base_image_source}" >&2
    exit 1
fi
if [[ -z "${base_image_id}" || "${base_image_id}" != sha256:* ]]; then
    echo "Unable to resolve ${base_image_source} to an immutable local image ID" >&2
    exit 1
fi
if ! base_image_reference="$(docker image inspect --format '{{if .RepoDigests}}{{index .RepoDigests 0}}{{end}}' "${base_image_source}")"; then
    echo "Unable to inspect the registry digest for ${base_image_source}" >&2
    exit 1
fi
if [[ -z "${base_image_reference}" || "${base_image_reference}" != *@sha256:* ]]; then
    echo "The local base image has no usable RepoDigest: ${base_image_source}" >&2
    echo "Pull the tagged image from its registry before building." >&2
    exit 1
fi
base_image_digest="${base_image_reference##*@}"

if [[ -n "$(git -C "${lightx2v_dir}" status --porcelain)" ]]; then
    echo "Local LightX2V worktree must be clean before it is copied into the image" >&2
    exit 1
fi

if [[ ! -d "${lightx2v_run_source_dir}" ]]; then
    echo "Local LightX2VRun source directory does not exist: ${lightx2v_run_source_dir}" >&2
    exit 1
fi
lightx2v_run_source_dir="$(cd "${lightx2v_run_source_dir}" && pwd)"
lightx2v_run_revision="$(git -C "${lightx2v_run_source_dir}" rev-parse HEAD)"
lightx2v_run_commit="${LIGHTX2V_RUN_COMMIT:-${lightx2v_run_revision}}"
if [[ "${lightx2v_run_revision}" != "${lightx2v_run_commit}" ]]; then
    echo "Local LightX2VRun revision ${lightx2v_run_revision} does not match requested revision ${lightx2v_run_commit}" >&2
    exit 1
fi
if [[ -n "$(git -C "${lightx2v_run_source_dir}" status --porcelain --untracked-files=all)" ]]; then
    echo "Local LightX2VRun worktree must be clean before it is copied into the image" >&2
    exit 1
fi

required_run_files=(
    README.md
    scripts/report_dit_progress.py
    scripts/run_wan22_moe_i2v_distill_fp8_4step_gfx1201_common.sh
    scripts/run_wan22_moe_i2v_distill_lora_fp8_4step_gfx1201_common.sh
)
for gpu_count in 1 2 4 8 16; do
    required_run_files+=(
        "scripts/run_wan22_moe_i2v_distill_fp8_4step_gfx1201_${gpu_count}gpu.sh"
        "scripts/run_wan22_moe_i2v_distill_lora_fp8_4step_gfx1201_${gpu_count}gpu.sh"
    )
done
for required_run_file in "${required_run_files[@]}"; do
    if [[ ! -f "${lightx2v_run_source_dir}/${required_run_file}" ]]; then
        echo "Required LightX2VRun file is unavailable: ${lightx2v_run_source_dir}/${required_run_file}" >&2
        exit 1
    fi
done

echo "============================================================"
echo "BASE_IMAGE_SOURCE   = ${base_image_source}"
echo "BASE_IMAGE_REFERENCE= ${base_image_reference}"
echo "BASE_IMAGE_DIGEST   = ${base_image_digest}"
echo "BASE_IMAGE_ID       = ${base_image_id}"
echo "APT_MIRROR          = ${apt_mirror}"
echo "PYPI_MIRROR         = ${pypi_mirror}"
echo "UV_VERSION          = ${uv_version}"
echo "LIGHTX2V_REVISION   = ${lightx2v_revision}"
echo "LIGHTX2V_RUN_REVISION= ${lightx2v_run_revision}"
echo "PYTORCH_ROCM_ARCH   = ${pytorch_rocm_arch}"
echo "AITER_ROCM_ARCH     = ${aiter_rocm_arch}"
echo "AITER_CU_NUM        = ${aiter_cu_num}"
echo "AITER_REPO          = ${aiter_repo}"
echo "AITER_COMMIT        = ${aiter_commit}"
echo "AITER_VERSION       = ${aiter_version}"
echo "AITER_SOURCE_DIR    = ${aiter_source_dir}"
echo "LIGHTX2V_RUN_SOURCE_DIR= ${lightx2v_run_source_dir}"
echo "MAX_JOBS            = ${max_jobs}"
echo "IMAGE_TAG           = ${image_tag}"
echo "============================================================"

DOCKER_BUILDKIT=1 docker build \
    --network=host \
    --file "${script_dir}/Dockerfile_gfx1201" \
    --tag "${image_tag}" \
    --build-context "aiter_source=${aiter_source_dir}" \
    --build-context "lightx2v_run_source=${lightx2v_run_source_dir}" \
    --build-arg "BASE_IMAGE=${base_image_reference}" \
    --build-arg "BASE_IMAGE_SOURCE=${base_image_source}" \
    --build-arg "BASE_IMAGE_DIGEST=${base_image_digest}" \
    --build-arg "BASE_IMAGE_ID=${base_image_id}" \
    --build-arg "APT_MIRROR=${apt_mirror}" \
    --build-arg "PIP_INDEX_URL=${pypi_mirror}" \
    --build-arg "UV_VERSION=${uv_version}" \
    --build-arg "ARG_PYTORCH_ROCM_ARCH=${pytorch_rocm_arch}" \
    --build-arg "AITER_ROCM_ARCH=${aiter_rocm_arch}" \
    --build-arg "AITER_CU_NUM=${aiter_cu_num}" \
    --build-arg "AITER_REPO=${aiter_repo}" \
    --build-arg "AITER_COMMIT=${aiter_commit}" \
    --build-arg "AITER_VERSION=${aiter_version}" \
    --build-arg "MAX_JOBS=${max_jobs}" \
    --build-arg "LIGHTX2V_REVISION=${lightx2v_revision}" \
    --build-arg "LIGHTX2V_RUN_REVISION=${lightx2v_run_revision}" \
    "${proxy_build_args[@]}" \
    --progress=plain \
    "${lightx2v_dir}"

echo "Built ${image_tag} from ${base_image_reference} (${base_image_id})"
