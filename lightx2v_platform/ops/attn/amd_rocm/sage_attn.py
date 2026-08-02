"""AMD dense self-attention adapter backed by Aiter SageAttention2."""

import torch
from loguru import logger

from lightx2v_platform.base.amd_rocm import AITER_INSTALL_CMD
from lightx2v_platform.ops.attn.template import AttnWeightTemplate
from lightx2v_platform.registry_factory import PLATFORM_ATTN_WEIGHT_REGISTER


IS_AMD_ROCM = hasattr(torch.version, "hip") and torch.version.hip is not None

aiter_fav3_sage_func = None
AITER_FAV3_SAGE_AVAILABLE = False
AITER_FAV3_SAGE_IMPORT_ERROR = None

try:
    from aiter.ops.triton.attention.fav3_sage import (
        fav3_sage_wrapper_func as aiter_fav3_sage_func,
    )

    AITER_FAV3_SAGE_AVAILABLE = True
    logger.info("Aiter FAv3 Sage Attention API found")
except (ImportError, RuntimeError) as e:
    AITER_FAV3_SAGE_IMPORT_ERROR = str(e)
    if IS_AMD_ROCM:
        logger.warning(
            "Aiter FAv3 Sage Attention is unavailable on AMD ROCm. Install aiter:\n{}",
            AITER_INSTALL_CMD,
        )
    else:
        logger.debug("Aiter FAv3 Sage Attention is only available on AMD ROCm")


def _sequence_length(tensor):
    return tensor.shape[1] if tensor.ndim == 4 else tensor.shape[0]


def _batch_size(tensor):
    return tensor.shape[0] if tensor.ndim == 4 else 1


def _validate_dense_cu_seqlens(values, batch_size, seqlen, name):
    if values is None:
        return
    if not isinstance(values, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(values).__name__}")
    expected_entries = batch_size + 1
    if values.ndim != 1 or values.numel() != expected_entries:
        raise ValueError(
            f"Aiter FAv3 Sage only supports dense fixed-length batches; {name} "
            f"must have {expected_entries} entries for sequence length {seqlen}"
        )


def _normalize_sage_window(window_size):
    if window_size is None:
        return (-1, -1)
    window_size = tuple(window_size)
    if len(window_size) == 3:
        if window_size[2] != 0:
            raise NotImplementedError(
                "Aiter FAv3 Sage does not support a non-zero window sink value"
            )
        window_size = window_size[:2]
    if len(window_size) != 2:
        raise ValueError(
            f"window_size must contain two values, got {window_size!r}"
        )
    return window_size


@PLATFORM_ATTN_WEIGHT_REGISTER("aiter_fav3_sage_bf16_attn")
class AiterFAv3SageBF16AttnWeight(AttnWeightTemplate):
    """BF16 Q/K/V to native gfx1201 INT8-QK/FP8-PV SageAttention2."""

    route_name = "Aiter gfx1201 SageAttention2"

    def __init__(self):
        self.config = {}
        if not IS_AMD_ROCM:
            raise RuntimeError(
                f"{self.route_name} is only available on AMD ROCm "
                "(torch.version.hip is not set)."
            )
        if not AITER_FAV3_SAGE_AVAILABLE:
            raise ImportError(
                "Aiter FAv3 Sage Attention is not installed on AMD ROCm. "
                f"Import error: {AITER_FAV3_SAGE_IMPORT_ERROR}\n{AITER_INSTALL_CMD}"
            )

    def _validate_inputs(
        self,
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_kv,
        max_seqlen_q,
        max_seqlen_kv,
    ):
        if q.ndim not in (3, 4) or k.ndim != q.ndim or v.ndim != q.ndim:
            raise ValueError(
                f"{self.route_name} expects matching 3D or 4D Q/K/V, got "
                f"{q.ndim}D, {k.ndim}D, and {v.ndim}D"
            )
        if not (q.dtype == k.dtype == v.dtype == torch.bfloat16):
            raise RuntimeError(
                f"{self.route_name} requires BF16 Q/K/V, got "
                f"{q.dtype}, {k.dtype}, and {v.dtype}"
            )
        if q.device != k.device or q.device != v.device:
            raise ValueError(f"{self.route_name} requires Q/K/V on the same device")
        if q.requires_grad or k.requires_grad or v.requires_grad:
            raise RuntimeError(f"{self.route_name} does not support backward")

        batch_size = _batch_size(q)
        if _batch_size(k) != batch_size or _batch_size(v) != batch_size:
            raise ValueError(f"{self.route_name} requires matching Q/K/V batch sizes")
        q_len = _sequence_length(q)
        k_len = _sequence_length(k)
        v_len = _sequence_length(v)
        if q_len != k_len or q_len != v_len:
            raise ValueError(
                f"{self.route_name} only supports self-attention with matching "
                f"Q/K/V sequence lengths, got {q_len}, {k_len}, and {v_len}"
            )
        if q.shape[-2] != k.shape[-2] or q.shape[-2] != v.shape[-2]:
            raise ValueError(
                f"{self.route_name} does not support GQA/MQA; Q/K/V head "
                f"counts must match, got {q.shape[-2]}, {k.shape[-2]}, "
                f"and {v.shape[-2]}"
            )
        if q.shape[-1] != 128 or k.shape[-1] != 128 or v.shape[-1] != 128:
            raise ValueError(
                f"{self.route_name} requires Q/K/V head dimension 128, got "
                f"{q.shape[-1]}, {k.shape[-1]}, and {v.shape[-1]}"
            )

        if max_seqlen_q is not None and int(max_seqlen_q) != q_len:
            raise ValueError(
                f"max_seqlen_q must equal the dense Q length {q_len}, "
                f"got {max_seqlen_q}"
            )
        if max_seqlen_kv is not None and int(max_seqlen_kv) != k_len:
            raise ValueError(
                f"max_seqlen_kv must equal the dense K/V length {k_len}, "
                f"got {max_seqlen_kv}"
            )
        _validate_dense_cu_seqlens(
            cu_seqlens_q, batch_size, q_len, "cu_seqlens_q"
        )
        _validate_dense_cu_seqlens(
            cu_seqlens_kv, batch_size, k_len, "cu_seqlens_kv"
        )

    def _validate_options(self, kwargs):
        if kwargs.get("dropout_p", kwargs.get("drop_rate", 0.0)) != 0.0:
            raise NotImplementedError(f"{self.route_name} does not support dropout")
        if kwargs.get("bias") is not None or kwargs.get("attn_bias") is not None:
            raise NotImplementedError(f"{self.route_name} does not support attention bias")
        if kwargs.get("alibi_slopes") is not None:
            raise NotImplementedError(f"{self.route_name} does not support ALiBi")
        if kwargs.get("attn_mask") is not None or kwargs.get("attention_mask") is not None:
            raise NotImplementedError(f"{self.route_name} only supports dense attention")
        if kwargs.get("return_attn_probs", False):
            raise NotImplementedError(
                f"{self.route_name} does not return attention probabilities"
            )
        if kwargs.get("return_lse", False):
            raise NotImplementedError(f"{self.route_name} does not return LSE")
        if kwargs.get("causal", kwargs.get("is_causal", False)):
            raise NotImplementedError(
                f"{self.route_name} only supports non-causal attention"
            )
        window_size = _normalize_sage_window(
            kwargs.get("window_size", (-1, -1))
        )
        if window_size != (-1, -1):
            raise NotImplementedError(
                f"{self.route_name} does not support sliding-window attention"
            )
        if kwargs.get("attention_chunk", 0) not in (0, 1):
            raise NotImplementedError(
                f"{self.route_name} does not support attention chunking"
            )
        if kwargs.get("softcap", kwargs.get("logits_soft_cap", 0.0)) != 0.0:
            raise NotImplementedError(f"{self.route_name} does not support softcap")
        if kwargs.get("sm_margin", 0) != 0:
            raise NotImplementedError(f"{self.route_name} does not support sm_margin")
        if kwargs.get("block_lut") is not None or kwargs.get("block_table") is not None:
            raise NotImplementedError(
                f"{self.route_name} LightX2V route only supports dense attention"
            )
        if kwargs.get("sink_ptr", kwargs.get("sink")) is not None:
            raise NotImplementedError(f"{self.route_name} does not support attention sinks")
        if kwargs.get("num_splits", 0) not in (0, 1):
            raise NotImplementedError(f"{self.route_name} does not support split-KV")
        if kwargs.get("out") is not None:
            raise NotImplementedError(f"{self.route_name} does not support output buffers")

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
        self._validate_inputs(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_kv,
            max_seqlen_q,
            max_seqlen_kv,
        )
        self._validate_options(kwargs)

        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        if q.ndim == 3:
            q, k, v = q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0)

        softmax_scale = kwargs.get("softmax_scale", kwargs.get("sm_scale"))
        kernel_config = kwargs.get("sage_config")
        if kernel_config is None:
            kernel_config = self.config.get("aiter_fav3_sage_config")
        kernel_config = dict(kernel_config or {})
        backend = kernel_config.get("backend", "flydsl_v2")
        if backend != "flydsl_v2":
            raise ValueError(
                f"{self.route_name} requires sage_config backend='flydsl_v2', "
                f"got {backend!r}"
            )
        kernel_config["backend"] = "flydsl_v2"

        result = aiter_fav3_sage_func(
            q,
            k,
            v,
            softmax_scale=softmax_scale,
            causal=False,
            window_size=(-1, -1),
            attention_chunk=kwargs.get("attention_chunk", 0),
            softcap=kwargs.get("softcap", kwargs.get("logits_soft_cap", 0.0)),
            deterministic=kwargs.get("deterministic", False),
            sm_margin=kwargs.get("sm_margin", 0),
            return_lse=False,
            layout="bshd",
            config=kernel_config,
            smooth_k=kwargs.get("smooth_k", True),
        )

        token_count = q.shape[0] * q.shape[1]
        return result.reshape(token_count, -1)

    def apply_with_lse(self, q, k, v, softmax_scale=None):
        del q, k, v, softmax_scale
        raise NotImplementedError(
            f"{self.route_name} does not return LSE and cannot be used by Ring SP"
        )
