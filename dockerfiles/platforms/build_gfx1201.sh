#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "$0")" && pwd)"
lightx2v_dir="$(cd "${script_dir}/../.." && pwd)"
aiter_source_dir="${AITER_SOURCE_DIR:-${lightx2v_dir}/../aiter}"

base_image_source="${BASE_IMAGE:-rocm/pytorch:rocm7.2.4_ubuntu24.04_py3.12_pytorch_release_2.10.0}"
image_tag="${IMAGE_TAG:-lightx2v-rocm:gfx1201-hvat-scratch}"
pytorch_rocm_arch="${PYTORCH_ROCM_ARCH:-gfx1201}"
aiter_rocm_arch="${AITER_ROCM_ARCH:-gfx1201}"
aiter_repo="${AITER_REPO:-https://github.com/yangecool/aiter.git}"
aiter_commit="${AITER_COMMIT:-1b37c33172ea807d528de91c7b4f8f74ff61ec44}"
aiter_version="${AITER_VERSION:-0.1.16.post3+gfx1201.g1b37c3317}"
accelerate_version="${ACCELERATE_VERSION:-1.14.0}"
peft_version="${PEFT_VERSION:-0.20.0}"
max_jobs="${MAX_JOBS:-$(nproc)}"
lightx2v_revision="$(git -C "${lightx2v_dir}" rev-parse HEAD)"

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
if [[ "${aiter_revision}" != "${aiter_commit}" ]]; then
    echo "Local Aiter revision ${aiter_revision} does not match pinned revision ${aiter_commit}" >&2
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

echo "============================================================"
echo "BASE_IMAGE_SOURCE   = ${base_image_source}"
echo "BASE_IMAGE_REFERENCE= ${base_image_reference}"
echo "BASE_IMAGE_DIGEST   = ${base_image_digest}"
echo "BASE_IMAGE_ID       = ${base_image_id}"
echo "LIGHTX2V_REVISION   = ${lightx2v_revision}"
echo "PYTORCH_ROCM_ARCH   = ${pytorch_rocm_arch}"
echo "AITER_ROCM_ARCH     = ${aiter_rocm_arch}"
echo "AITER_REPO          = ${aiter_repo}"
echo "AITER_COMMIT        = ${aiter_commit}"
echo "AITER_VERSION       = ${aiter_version}"
echo "AITER_SOURCE_DIR    = ${aiter_source_dir}"
echo "AITER_SOURCE_REV    = ${aiter_revision}"
echo "ACCELERATE_VERSION  = ${accelerate_version}"
echo "PEFT_VERSION        = ${peft_version}"
echo "MAX_JOBS            = ${max_jobs}"
echo "IMAGE_TAG           = ${image_tag}"
echo "============================================================"

DOCKER_BUILDKIT=1 docker build \
    --network=host \
    --file "${script_dir}/Dockerfile_gfx1201" \
    --tag "${image_tag}" \
    --build-context "aiter_source=${aiter_source_dir}" \
    --build-arg "BASE_IMAGE=${base_image_reference}" \
    --build-arg "BASE_IMAGE_SOURCE=${base_image_source}" \
    --build-arg "BASE_IMAGE_DIGEST=${base_image_digest}" \
    --build-arg "BASE_IMAGE_ID=${base_image_id}" \
    --build-arg "ARG_PYTORCH_ROCM_ARCH=${pytorch_rocm_arch}" \
    --build-arg "AITER_ROCM_ARCH=${aiter_rocm_arch}" \
    --build-arg "AITER_REPO=${aiter_repo}" \
    --build-arg "AITER_COMMIT=${aiter_commit}" \
    --build-arg "AITER_VERSION=${aiter_version}" \
    --build-arg "ACCELERATE_VERSION=${accelerate_version}" \
    --build-arg "PEFT_VERSION=${peft_version}" \
    --build-arg "MAX_JOBS=${max_jobs}" \
    --build-arg "LIGHTX2V_REVISION=${lightx2v_revision}" \
    "${proxy_build_args[@]}" \
    --progress=plain \
    "${lightx2v_dir}"

echo "Built ${image_tag} from ${base_image_reference} (${base_image_id})"
