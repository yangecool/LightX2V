"""
AMD ROCm Device implementation for LightX2V.

AMD ROCm provides CUDA-compatible APIs through HIP (Heterogeneous-computing Interface for Portability).
This module handles AMD-specific optimizations including:
- Disabling cudnn for faster VAE convolution
- sgl_kernel compatibility layer using aiter library (required on AMD)
"""

import os
import sys

import torch
import torch.distributed as dist
from loguru import logger

from lightx2v_platform.registry_factory import PLATFORM_DEVICE_REGISTER

# Detect AMD ROCm platform
IS_AMD_ROCM = hasattr(torch.version, "hip") and torch.version.hip is not None

# aiter installation info.
#
# AITER_REPO / AITER_BRANCH / AITER_COMMIT define the *standalone* pip-install
# fallback used when the user has not built the gfx1201 Docker image
# (which copies the local aiter source tree in at a known commit, and is the
# path that ships the tuned gfx1201 lookup + CK rowwise dispatch).
#
# The Docker image path is authoritative for gfx1201 Wan2.2 bring-up:
#   dockerfiles/platforms/build_gfx1201.sh
#   dockerfiles/platforms/Dockerfile_gfx1201
# It copies aiter from a sibling source dir, then build_gfx1201.sh verifies
# `aiter_revision == AITER_COMMIT` and that the worktree is clean.
#
# The values below are only consulted when aiter is `pip install`-ed outside
# the image (e.g. `pip install -e .` from a fresh clone, or the
# AITER_INSTALL_CMD snippet printed on ImportError). They are kept in sync
# with the gfx1201-compatible tip of the aiter fork:
#   repo:    https://gitcode.com/hvat-ai/aiter.git
#   branch:  gfx1201-hvat-scratch
#   commit:  53eb40c94 perf(sage): select tuned gfx1201 V2 defaults
# Older commits predate the tuned K-prefetch and FP8-P-offset defaults.
AITER_REPO = os.getenv("AITER_REPO", "https://gitcode.com/hvat-ai/aiter.git")
AITER_BRANCH = os.getenv("AITER_BRANCH", "gfx1201-hvat-scratch")
AITER_COMMIT = os.getenv(
    "AITER_COMMIT", "53eb40c9463de9b7efeae8f58b75d31e1d187c47"
)
AITER_INSTALL_CMD = f"""
# One-line install command for aiter (AMD ROCm optimized kernels).
# Bumps AITER_COMMIT to track the gfx1201-compatible tip; override via env
# if you need to pin a different revision.
git clone {AITER_REPO} /tmp/aiter && \\
cd /tmp/aiter && \\
git checkout {AITER_BRANCH} && \\
git checkout {AITER_COMMIT} && \\
pip install -e .
"""


def _runtime_gfx() -> str:
    """Return the normalized HIP architecture without requiring a GPU at import."""
    try:
        device = torch.cuda.current_device()
        name = torch.cuda.get_device_properties(device).gcnArchName
    except Exception:
        return ""
    return str(name).lower().split(":", 1)[0]


class AiterSglKernelCompat:
    """
    Compatibility layer to use aiter with sgl_kernel interface.

    This class wraps aiter functions to match sgl_kernel's API,
    allowing existing code to work seamlessly on AMD GPUs.

    Note: This is REQUIRED on AMD ROCm as the original sgl_kernel
    does not support AMD GPUs.
    """

    def __init__(self, aiter_module):
        self._aiter = aiter_module
        self._runtime_gfx = _runtime_gfx()
        missing = [
            name
            for name in ("gemm_a8w8", "pertoken_quant", "dtypes")
            if not hasattr(aiter_module, name)
        ]
        if missing:
            raise ImportError(
                "Installed aiter is missing required public APIs: " + ", ".join(missing)
            )

        # Do not bypass Aiter's architecture dispatch. The public rowwise A8W8
        # entry selects CK where its instances are supported and selects the
        # gfx11/gfx12 implementation before launching a kernel.
        self._gemm_a8w8 = aiter_module.gemm_a8w8
        self._gemm_backend = "aiter-rowwise-dispatch"
        self._pertoken_quant = aiter_module.pertoken_quant
        self._dtypes = aiter_module.dtypes
        self._rmsnorm2d_fwd = getattr(aiter_module, "rmsnorm2d_fwd", None)
        self._rms_norm = getattr(aiter_module, "rms_norm", None)
        if self._rmsnorm2d_fwd is None and self._rms_norm is None:
            raise ImportError("Installed aiter has no RMSNorm public API")

        self.rmsnorm_backend = os.getenv("LIGHTX2V_AMD_RMSNORM_BACKEND", "aiter")
        if self.rmsnorm_backend not in {"aiter", "ck"}:
            raise ValueError(
                "LIGHTX2V_AMD_RMSNORM_BACKEND must be 'aiter' or 'ck', "
                f"got {self.rmsnorm_backend!r}"
            )
        logger.info(
            "Using aiter as sgl_kernel backend (gfx={}, GEMM={}, "
            "per-token FP8 quantization, RMSNorm backend={})",
            self._runtime_gfx or "unknown",
            self._gemm_backend,
            self.rmsnorm_backend,
        )

    def rmsnorm(self, input, weight, eps, enable_pdl=False):
        """RMSNorm compatible with sgl_kernel, including NVIDIA's PDL keyword.

        PDL is not an AMD execution flag.  It is accepted at this boundary so
        the model does not need an architecture-specific call site.  Aiter's
        shape-aware public wrapper is preferred because it selects the portable
        HIP/Triton path for supported hidden sizes; the raw CK entry is opt-in.
        """
        del enable_pdl
        if self.rmsnorm_backend == "aiter" and self._rmsnorm2d_fwd is not None:
            return self._rmsnorm2d_fwd(input, weight, eps, 0)
        if self._rms_norm is None:
            raise RuntimeError("Aiter CK RMSNorm entry is unavailable")
        return self._rms_norm(input, weight, eps)

    @staticmethod
    def _normalize_scale(scale, expected, name):
        if not isinstance(scale, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor, got {type(scale).__name__}")
        if scale.ndim not in (1, 2) or scale.numel() != expected:
            raise ValueError(
                f"{name} must contain {expected} scale values, got shape={tuple(scale.shape)}"
            )
        if scale.ndim == 2 and 1 not in scale.shape:
            raise ValueError(
                f"{name} must be a vector or a row/column broadcast view, "
                f"got shape={tuple(scale.shape)}"
            )
        # CK consumes scale values as vectors; either rank is a zero-copy view.
        return scale

    def fp8_scaled_mm(self, input_quant, weight, input_scale, weight_scale, dtype, bias=None):
        """FP8 GEMM compatible with ``sgl_kernel.fp8_scaled_mm``.

        SGL exposes weights as ``[K, N]`` while Aiter's public A8W8 API uses
        ``[N, K]`` and transposes it internally.  The second transpose below
        restores the Aiter contract without copying the checkpoint tensor when
        LightX2V's normal ``[N, K].t()`` view is used.
        """
        if input_quant.ndim != 2 or weight.ndim != 2:
            raise ValueError(
                f"A8W8 expects 2D matrices, got input={input_quant.ndim}D weight={weight.ndim}D"
            )
        m, k = input_quant.shape
        k_weight, n = weight.shape
        if k != k_weight:
            raise ValueError(
                f"A8W8 K mismatch: input shape={tuple(input_quant.shape)}, "
                f"SGL weight shape={tuple(weight.shape)}"
            )
        input_scale = self._normalize_scale(input_scale, m, "input_scale")
        weight_scale = self._normalize_scale(weight_scale, n, "weight_scale")
        for tensor, name in (
            (weight, "weight"),
            (input_scale, "input_scale"),
            (weight_scale, "weight_scale"),
        ):
            if tensor.device != input_quant.device:
                raise ValueError(
                    f"{name} must be on {input_quant.device}, got {tensor.device}"
                )
        if input_scale.dtype != torch.float32 or weight_scale.dtype != torch.float32:
            raise ValueError(
                "A8W8 scales must be float32, got "
                f"input_scale={input_scale.dtype} weight_scale={weight_scale.dtype}"
            )
        if bias is not None and (bias.ndim != 1 or bias.numel() != n):
            raise ValueError(f"bias must have shape [{n}], got {tuple(bias.shape)}")
        if bias is not None and bias.device != input_quant.device:
            raise ValueError(f"bias must be on {input_quant.device}, got {bias.device}")

        weight_nk = weight.transpose(-2, -1)
        return self._gemm_a8w8(input_quant, weight_nk, input_scale, weight_scale, bias, dtype)

    def int8_scaled_mm(self, input_quant, weight, input_scale, weight_scale, dtype, bias=None):
        """INT8 GEMM compatible with sgl_kernel.int8_scaled_mm"""
        return self.fp8_scaled_mm(input_quant, weight, input_scale, weight_scale, dtype, bias)

    def sgl_per_token_quant_fp8(self, x, out, scale):
        """Per-token FP8 quantization compatible with sgl_kernel.sgl_per_token_quant_fp8"""
        q, s = self._pertoken_quant(x, quant_dtype=torch.float8_e4m3fn)
        out.copy_(q)
        scale.copy_(s)

    def sgl_per_token_group_quant_fp8(self, x, out, scale, group_size=128, eps=1e-10, fp8_min=-448.0, fp8_max=448.0):
        """Per-token per-group FP8 quantization compatible with sgl_kernel.sgl_per_token_group_quant_fp8"""
        m, k = x.shape
        x_view = x.view(m, -1, group_size)
        x_amax = x_view.abs().float().amax(dim=2).view(m, -1).clamp(eps)
        q = (x_view * (fp8_max / x_amax.unsqueeze(2))).to(torch.float8_e4m3fn).view(m, k)
        s = (x_amax / fp8_max).view(m, -1)
        out.copy_(q)
        scale.copy_(s)


def _get_aiter_sgl_kernel():
    """Get aiter-based sgl_kernel compatibility layer."""
    try:
        import aiter

        return AiterSglKernelCompat(aiter)
    except ImportError:
        logger.error(
            f"\n{'=' * 60}\nERROR: AMD ROCm detected but aiter is not installed.\naiter is REQUIRED for LightX2V to work on AMD GPUs.\n\nPlease install aiter:\n{AITER_INSTALL_CMD}\n{'=' * 60}\n"
        )
        raise ImportError(f"aiter is required for AMD ROCm support. Please install: pip install git+{AITER_REPO}@{AITER_COMMIT}")


@PLATFORM_DEVICE_REGISTER("amd_rocm")
class AmdRocmDevice:
    """
    AMD ROCm Device implementation for LightX2V.

    AMD ROCm uses CUDA-compatible APIs through HIP.
    This class provides AMD-specific optimizations.
    """

    name = "amd_rocm"

    @staticmethod
    def init_device_env():
        """
        Initialize AMD ROCm optimizations.

        This is called from lightx2v_platform.set_ai_device when platform is amd_rocm.
        1. Disable cudnn for faster VAE convolution
        2. Inject aiter as sgl_kernel compatibility layer (REQUIRED on AMD)
        """
        logger.info("AMD ROCm platform detected, initializing optimizations...")

        # Disable cudnn for faster VAE conv computation
        torch.backends.cudnn.enabled = False
        logger.info("  - cudnn disabled for faster VAE convolution")

        # Inject aiter as sgl_kernel compatibility layer (REQUIRED)
        sgl_kernel = _get_aiter_sgl_kernel()
        missing_attention = [
            name
            for name in ("flash_attn_func", "flash_attn_varlen_func")
            if not hasattr(sgl_kernel._aiter, name)
        ]
        if missing_attention:
            raise ImportError(
                "Installed aiter is missing required attention APIs: "
                + ", ".join(missing_attention)
            )
        sys.modules["sgl_kernel"] = sgl_kernel
        # Update any module that already imported sgl_kernel
        for mod_name, mod in list(sys.modules.items()):
            if mod is not None and hasattr(mod, "sgl_kernel"):
                setattr(mod, "sgl_kernel", sgl_kernel)
        logger.info(
            "  - aiter capability summary: commit={}, gfx={}, dense_attention={}, "
            "varlen_attention={}, gemm_a8w8={}, pertoken_quant={}, "
            "gemm_dispatch={}, rmsnorm={}",
            AITER_COMMIT,
            sgl_kernel._runtime_gfx or "unknown",
            hasattr(sgl_kernel._aiter, "flash_attn_func"),
            hasattr(sgl_kernel._aiter, "flash_attn_varlen_func"),
            hasattr(sgl_kernel._aiter, "gemm_a8w8"),
            hasattr(sgl_kernel._aiter, "pertoken_quant"),
            sgl_kernel._gemm_backend,
            sgl_kernel.rmsnorm_backend,
        )

    @staticmethod
    def is_available() -> bool:
        """Check if AMD ROCm is available."""
        return IS_AMD_ROCM and torch.cuda.is_available()

    @staticmethod
    def get_device() -> str:
        """Get the device type string. Returns 'cuda' for ROCm compatibility."""
        return "cuda"

    @staticmethod
    def init_parallel_env():
        """Initialize distributed parallel environment for AMD ROCm."""
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(dist.get_rank())


# Export constants
__all__ = [
    "IS_AMD_ROCM",
    "AITER_REPO",
    "AITER_COMMIT",
    "AITER_INSTALL_CMD",
    "AiterSglKernelCompat",
    "AmdRocmDevice",
]
