"""AMD ROCm attention adapter backed by Aiter.

The adapter mirrors the NVIDIA dense/varlen split.  Aiter's public dense
entry point owns the gfx1201 FlyDSL-vs-Triton decision; the varlen entry point
is reserved for packed or genuinely multi-sequence inputs.
"""

import os
from collections import Counter, OrderedDict

import torch
from loguru import logger

from lightx2v_platform.base.amd_rocm import (
    AITER_COMMIT,
    AITER_INSTALL_CMD,
    AITER_REPO,
)
from lightx2v_platform.ops.attn.template import AttnWeightTemplate
from lightx2v_platform.registry_factory import PLATFORM_ATTN_WEIGHT_REGISTER

IS_AMD_ROCM = hasattr(torch.version, "hip") and torch.version.hip is not None

aiter_flash_attn_func = None
aiter_flash_attn_varlen_func = None
AITER_AVAILABLE = False
AITER_IMPORT_ERROR = None

try:
    from aiter import flash_attn_func as aiter_flash_attn_func
    from aiter import flash_attn_varlen_func as aiter_flash_attn_varlen_func

    AITER_AVAILABLE = True
    logger.info("Aiter dense and varlen attention APIs found")
except ImportError as e:
    AITER_IMPORT_ERROR = str(e)
    if IS_AMD_ROCM:
        logger.warning(
            "Aiter attention APIs are unavailable on AMD ROCm. Install aiter:\n{}",
            AITER_INSTALL_CMD,
        )
    else:
        logger.debug("Aiter attention is only available on AMD ROCm")


_ROUTE_COUNTS = Counter()


def get_aiter_attention_route_counts(reset=False):
    """Return adapter route counters for profiling and bring-up reports."""
    counts = dict(_ROUTE_COUNTS)
    if reset:
        _ROUTE_COUNTS.clear()
    return counts


def _normalize_window_size(window_size):
    if window_size is None:
        return (-1, -1, 0)
    window_size = tuple(window_size)
    if len(window_size) == 2:
        return (*window_size, 0)
    if len(window_size) == 3:
        return window_size
    raise ValueError(f"window_size must contain two or three values, got {window_size!r}")


def _dense_flydsl_eligible(q, k, v, kwargs):
    """Mirror Aiter's public dense eligibility check for profiling only."""
    if q.ndim == 3:
        q, k, v = q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0)
    try:
        from aiter.ops.flydsl_fmha_config import (
            can_use_gfx1201_flydsl_dense_attention,
        )

        device_index = q.device.index if q.device.type == "cuda" else None
        arch = torch.cuda.get_device_properties(device_index).gcnArchName
        dtype_name = "bf16" if q.dtype == torch.bfloat16 else str(q.dtype)
        window_size = _normalize_window_size(kwargs.get("window_size", (-1, -1)))
        return can_use_gfx1201_flydsl_dense_attention(
            arch=arch,
            q_shape=tuple(q.shape),
            k_shape=tuple(k.shape),
            v_shape=tuple(v.shape),
            q_dtype=dtype_name,
            k_dtype=dtype_name if k.dtype == q.dtype else str(k.dtype),
            v_dtype=dtype_name if v.dtype == q.dtype else str(v.dtype),
            same_cuda_device=(
                q.is_cuda
                and k.is_cuda
                and v.is_cuda
                and q.device == k.device == v.device
            ),
            dropout_p=kwargs.get("dropout_p", 0.0),
            softmax_scale=kwargs.get("softmax_scale"),
            causal=kwargs.get("causal", False),
            window_size=window_size,
            has_bias=kwargs.get("bias") is not None,
            has_alibi=kwargs.get("alibi_slopes") is not None,
            return_lse=kwargs.get("return_lse", False),
            return_attn_probs=kwargs.get("return_attn_probs", False),
            has_cu_seqlens=False,
            has_sink=kwargs.get("sink_ptr", kwargs.get("sink")) is not None,
            num_splits=kwargs.get("num_splits", 0),
            requires_grad=(
                torch.is_grad_enabled()
                and (q.requires_grad or k.requires_grad or v.requires_grad)
            ),
        )
    except Exception:
        return False


def _restore_attention_output(output, token_count):
    """Flatten the primary output while preserving optional Aiter diagnostics."""
    if isinstance(output, tuple):
        if not output:
            return output
        return (output[0].reshape(token_count, -1), *output[1:])
    return output.reshape(token_count, -1)


def _single_sequence_shape(q, k, v, cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv):
    """Check the cheap structural part of Aiter's dense-attention contract."""
    if q.ndim not in (3, 4) or k.ndim != q.ndim or v.ndim != q.ndim:
        return False
    if q.ndim == 4:
        if q.shape[0] != 1 or k.shape[0] != 1 or v.shape[0] != 1:
            return False
        q_len, k_len, v_len = q.shape[1], k.shape[1], v.shape[1]
    else:
        q_len, k_len, v_len = q.shape[0], k.shape[0], v.shape[0]
    if k_len != v_len:
        return False
    if max_seqlen_q is not None and int(max_seqlen_q) != q_len:
        return False
    if max_seqlen_kv is not None and int(max_seqlen_kv) != k_len:
        return False

    # A two-element cumulative-length tensor is still a dense single sequence;
    # any longer tensor represents packed/multi-sequence input.
    for lengths in (cu_seqlens_q, cu_seqlens_kv):
        if lengths is not None and (
            not isinstance(lengths, torch.Tensor) or lengths.numel() != 2
        ):
            return False
    return True


@PLATFORM_ATTN_WEIGHT_REGISTER("aiter_attn")
class AiterAttnWeight(AttnWeightTemplate):
    """Aiter attention with dense self/cross and packed varlen routing."""

    def __init__(self):
        self.config = {}
        # CPU callers may reuse metadata across layers. Keep only a bounded
        # per-attention-instance cache so conversion does not retain every
        # transient tensor from a long-running service.
        self._normalized_varlen_cache = OrderedDict()

        if not IS_AMD_ROCM:
            raise RuntimeError(
                "aiter_attn is only available on AMD ROCm (torch.version.hip is not set)."
            )
        if not AITER_AVAILABLE:
            raise ImportError(
                "Aiter is not installed on AMD ROCm. "
                f"Import error: {AITER_IMPORT_ERROR}\n{AITER_INSTALL_CMD}"
            )

    def _normalize_cu_seqlens(self, values, device, total_tokens, name, validate_end=True):
        if values is None:
            raise ValueError(f"{name} is required for varlen Aiter attention")
        if not isinstance(values, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor, got {type(values).__name__}")
        key = (id(values), values.data_ptr(), str(device), total_tokens, validate_end)
        cached = self._normalized_varlen_cache.get(key)
        if cached is not None:
            self._normalized_varlen_cache.move_to_end(key)
            return cached

        normalized = values.to(device=device, dtype=torch.int32, non_blocking=True)
        if not normalized.is_contiguous():
            normalized = normalized.contiguous()
        if normalized.ndim != 1 or normalized.numel() < 2:
            raise ValueError(f"{name} must be a 1D int32 tensor with at least two entries")
        if os.getenv("LIGHTX2V_AITER_VALIDATE_ATTENTION", "1") == "1":
            if int(normalized[0].item()) != 0:
                raise ValueError(
                    f"{name} must start at 0, got {int(normalized[0].item())}"
                )
            if validate_end and int(normalized[-1].item()) != total_tokens:
                raise ValueError(
                    f"{name} must end at {total_tokens}, got {int(normalized[-1].item())}"
                )
            if bool(torch.any(normalized[1:] < normalized[:-1]).item()):
                raise ValueError(f"{name} must be monotonically non-decreasing")
        self._normalized_varlen_cache[key] = normalized
        self._normalized_varlen_cache.move_to_end(key)
        while len(self._normalized_varlen_cache) > 16:
            self._normalized_varlen_cache.popitem(last=False)
        return normalized

    @staticmethod
    def _flatten_qkv(q, k, v):
        if q.ndim == 4:
            return (
                q.reshape(-1, q.shape[-2], q.shape[-1]),
                k.reshape(-1, k.shape[-2], k.shape[-1]),
                v.reshape(-1, v.shape[-2], v.shape[-1]),
            )
        return q, k, v

    def apply(
        self,
        q,
        k,
        v,
        cu_seqlens_q=None,
        cu_seqlens_kv=None,
        max_seqlen_q=None,
        max_seqlen_kv=None,
        **kwargs,
    ):
        dropout_p = kwargs.get("dropout_p", 0.0)
        softmax_scale = kwargs.get("softmax_scale", None)
        causal = kwargs.get("causal", False)
        window_size = _normalize_window_size(kwargs.get("window_size", (-1, -1)))
        bias = kwargs.get("bias", None)
        alibi_slopes = kwargs.get("alibi_slopes", None)
        deterministic = kwargs.get("deterministic", None)
        sink_ptr = kwargs.get("sink_ptr", kwargs.get("sink", None))
        return_lse = kwargs.get("return_lse", False)
        return_attn_probs = kwargs.get("return_attn_probs", False)
        how_v3_bf16_cvt = kwargs.get("how_v3_bf16_cvt", 1)
        num_splits = kwargs.get("num_splits", 0)
        block_table = kwargs.get("block_table", None)
        output_buffer = kwargs.get("out", None)
        if _single_sequence_shape(q, k, v, cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv):
            q_len = q.shape[1] if q.ndim == 4 else q.shape[0]
            k_len = k.shape[1] if k.ndim == 4 else k.shape[0]
            if cu_seqlens_q is not None:
                self._normalize_cu_seqlens(
                    cu_seqlens_q, q.device, q_len, "cu_seqlens_q"
                )
            if cu_seqlens_kv is not None:
                self._normalize_cu_seqlens(
                    cu_seqlens_kv, k.device, k_len, "cu_seqlens_kv"
                )
            if _dense_flydsl_eligible(q, k, v, kwargs):
                _ROUTE_COUNTS["aiter_dense_flydsl_candidate"] += 1
            else:
                _ROUTE_COUNTS["aiter_dense_triton_fallback"] += 1
            if q.ndim == 3:
                q, k, v = q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0)
            output = aiter_flash_attn_func(
                q,
                k,
                v,
                dropout_p=dropout_p,
                softmax_scale=softmax_scale,
                causal=causal,
                window_size=window_size,
                bias=bias,
                alibi_slopes=alibi_slopes,
                deterministic=True if deterministic is None else deterministic,
                return_lse=return_lse,
                return_attn_probs=return_attn_probs,
                how_v3_bf16_cvt=how_v3_bf16_cvt,
                num_splits=num_splits,
                sink_ptr=sink_ptr,
            )
            return _restore_attention_output(output, q.shape[0] * q.shape[1])

        original_q_ndim = q.ndim
        q_default_max = q.shape[1] if original_q_ndim == 4 else None
        k_default_max = k.shape[1] if original_q_ndim == 4 else None
        q, k, v = self._flatten_qkv(q, k, v)
        if max_seqlen_q is None:
            max_seqlen_q = q_default_max or q.shape[0]
        else:
            max_seqlen_q = int(max_seqlen_q)
        if max_seqlen_kv is None:
            max_seqlen_kv = k_default_max or k.shape[0]
        else:
            max_seqlen_kv = int(max_seqlen_kv)
        cu_seqlens_q = self._normalize_cu_seqlens(cu_seqlens_q, q.device, q.shape[0], "cu_seqlens_q")
        cu_seqlens_kv = self._normalize_cu_seqlens(cu_seqlens_kv, k.device, k.shape[0], "cu_seqlens_kv")
        cu_seqlens_q_padded = kwargs.get("cu_seqlens_q_padded")
        cu_seqlens_kv_padded = kwargs.get("cu_seqlens_kv_padded")
        if cu_seqlens_q_padded is not None:
            cu_seqlens_q_padded = self._normalize_cu_seqlens(
                cu_seqlens_q_padded,
                q.device,
                q.shape[0],
                "cu_seqlens_q_padded",
                validate_end=False,
            )
        if cu_seqlens_kv_padded is not None:
            cu_seqlens_kv_padded = self._normalize_cu_seqlens(
                cu_seqlens_kv_padded,
                k.device,
                k.shape[0],
                "cu_seqlens_kv_padded",
                validate_end=False,
            )
        _ROUTE_COUNTS["aiter_varlen_triton_fallback"] += 1
        output = aiter_flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_kv,
            max_seqlen_q,
            max_seqlen_kv,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            logits_soft_cap=kwargs.get("logits_soft_cap", 0.0),
            causal=causal,
            window_size=window_size,
            bias=bias,
            alibi_slopes=alibi_slopes,
            deterministic=False if deterministic is None else deterministic,
            return_lse=return_lse,
            return_attn_probs=return_attn_probs,
            how_v3_bf16_cvt=how_v3_bf16_cvt,
            block_table=block_table,
            out=output_buffer,
            cu_seqlens_q_padded=cu_seqlens_q_padded,
            cu_seqlens_k_padded=cu_seqlens_kv_padded,
            sink_ptr=sink_ptr,
        )
        return _restore_attention_output(output, q.shape[0])


class _AiterBF16FlashAttnWeight(AiterAttnWeight):
    route_name = "Aiter BF16 Flash Attention"

    def _validate_bf16(self, q, k, v):
        if not (q.dtype == k.dtype == v.dtype == torch.bfloat16):
            raise RuntimeError(
                f"{self.route_name} requires BF16 Q/K/V, got "
                f"{q.dtype}, {k.dtype}, and {v.dtype}"
            )


@PLATFORM_ATTN_WEIGHT_REGISTER("aiter_flydsl_bf16_flash_attn")
class AiterFlyDSLBF16FlashAttnWeight(_AiterBF16FlashAttnWeight):
    """gfx1201 dense BF16 self-attention through Aiter FlyDSL."""

    route_name = "Aiter FlyDSL BF16 Flash Attention"

    def apply(
        self,
        q,
        k,
        v,
        cu_seqlens_q=None,
        cu_seqlens_kv=None,
        max_seqlen_q=None,
        max_seqlen_kv=None,
        **kwargs,
    ):
        self._validate_bf16(q, k, v)
        dense_single_sequence = _single_sequence_shape(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_kv,
            max_seqlen_q,
            max_seqlen_kv,
        )
        if not dense_single_sequence or not _dense_flydsl_eligible(q, k, v, kwargs):
            raise RuntimeError(
                f"{self.route_name} received a shape or option that would fall back "
                "to Aiter Triton attention"
            )
        from aiter.ops.flydsl.utils import is_flydsl_available

        if not is_flydsl_available():
            raise RuntimeError(f"{self.route_name} requires the FlyDSL runtime")
        return super().apply(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=cu_seqlens_kv,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_kv=max_seqlen_kv,
            **kwargs,
        )


@PLATFORM_ATTN_WEIGHT_REGISTER("aiter_triton_bf16_flash_attn")
class AiterTritonBF16FlashAttnWeight(_AiterBF16FlashAttnWeight):
    """gfx1201 BF16 cross/varlen attention through Aiter Triton."""

    route_name = "Aiter Triton BF16 Flash Attention"

    def apply(
        self,
        q,
        k,
        v,
        cu_seqlens_q=None,
        cu_seqlens_kv=None,
        max_seqlen_q=None,
        max_seqlen_kv=None,
        **kwargs,
    ):
        self._validate_bf16(q, k, v)
        dense_single_sequence = _single_sequence_shape(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_kv,
            max_seqlen_q,
            max_seqlen_kv,
        )
        if dense_single_sequence and _dense_flydsl_eligible(q, k, v, kwargs):
            raise RuntimeError(
                f"{self.route_name} received an input that would route to Aiter FlyDSL"
            )
        return super().apply(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=cu_seqlens_kv,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_kv=max_seqlen_kv,
            **kwargs,
        )

    def apply_with_lse(self, q, k, v, softmax_scale=None):
        """Apply one dense Triton attention block for Ring SP."""
        result = self.apply(
            q,
            k,
            v,
            softmax_scale=softmax_scale,
            return_lse=True,
        )
        if not isinstance(result, tuple) or len(result) < 2:
            raise RuntimeError(
                f"{self.route_name} expected Aiter to return (output, lse) "
                "when return_lse=True"
            )

        output, lse = result[:2]
        batch_size = q.shape[0] if q.ndim == 4 else 1
        token_count = q.shape[1] if q.ndim == 4 else q.shape[0]
        head_count = q.shape[-2]
        if lse.shape == (batch_size, head_count, token_count):
            lse = lse.transpose(1, 2).reshape(batch_size * token_count, head_count)
        elif batch_size == 1 and lse.shape == (head_count, token_count):
            lse = lse.transpose(0, 1).contiguous()
        elif lse.shape != (batch_size * token_count, head_count):
            raise RuntimeError(
                f"{self.route_name} returned unsupported LSE shape {tuple(lse.shape)}; "
                f"expected ({batch_size}, {head_count}, {token_count})"
            )
        return output, lse
