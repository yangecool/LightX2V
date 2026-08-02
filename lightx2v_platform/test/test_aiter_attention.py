"""Contract tests for dense and packed Aiter attention routing."""

import pytest

torch = pytest.importorskip("torch")

from lightx2v_platform.ops.attn.amd_rocm import flash_attn as adapter
from lightx2v_platform.ops.attn.amd_rocm import sage_attn as sage_adapter


def _has_gfx1201():
    if torch.version.hip is None or not torch.cuda.is_available():
        return False
    try:
        return torch.cuda.get_device_properties(0).gcnArchName.lower().startswith(
            "gfx1201"
        )
    except Exception:
        return False


requires_gfx1201 = pytest.mark.skipif(
    not _has_gfx1201(), reason="requires a PyTorch ROCm runtime on gfx1201"
)


def _attention_reference(q, k, v, softmax_scale=None):
    scale = softmax_scale if softmax_scale is not None else q.shape[-1] ** -0.5
    scores = torch.einsum("qhd,khd->hqk", q.float(), k.float()) * scale
    probs = torch.softmax(scores, dim=-1)
    return torch.einsum("hqk,khd->qhd", probs, v.float())


def _assert_attention_close(actual, reference):
    assert torch.isfinite(actual).all()
    cosine = torch.nn.functional.cosine_similarity(
        actual.float().flatten(), reference.float().flatten(), dim=0
    )
    assert float(cosine) >= 0.999
    torch.testing.assert_close(
        actual.float(), reference.float(), atol=3e-2, rtol=3e-2
    )


def _assert_sage_attention_close(actual, reference):
    assert torch.isfinite(actual).all()
    cosine = torch.nn.functional.cosine_similarity(
        actual.float().flatten(), reference.float().flatten(), dim=0
    )
    assert float(cosine) >= 0.99
    assert float((actual.float() - reference.float()).abs().mean()) <= 5e-2


def _mock_sage_available(monkeypatch, func):
    monkeypatch.setattr(sage_adapter, "IS_AMD_ROCM", True)
    monkeypatch.setattr(sage_adapter, "AITER_FAV3_SAGE_AVAILABLE", True)
    monkeypatch.setattr(sage_adapter, "aiter_fav3_sage_func", func)


def test_sage_dense_contract_forces_native_v2(monkeypatch):
    calls = []

    def sage(q, k, v, **kwargs):
        calls.append((q, k, v, kwargs))
        return q

    _mock_sage_available(monkeypatch, sage)
    weight = sage_adapter.AiterFAv3SageBF16AttnWeight()
    q = torch.zeros((5, 2, 128), dtype=torch.bfloat16)

    output = weight.apply(
        q,
        q,
        q,
        cu_seqlens_q=torch.tensor([0, 5]),
        cu_seqlens_kv=torch.tensor([0, 5]),
        max_seqlen_q=5,
        max_seqlen_kv=5,
    )

    assert output.shape == (5, 256)
    assert calls[-1][0].shape == (1, 5, 2, 128)
    assert calls[-1][3]["layout"] == "bshd"
    assert calls[-1][3]["config"] == {"backend": "flydsl_v2"}
    assert calls[-1][3]["causal"] is False
    assert calls[-1][3]["return_lse"] is False
    assert calls[-1][3]["smooth_k"] is True


def test_sage_preserves_tuning_config_while_forcing_v2(monkeypatch):
    calls = []

    def sage(q, k, v, **kwargs):
        calls.append(kwargs)
        return q

    _mock_sage_available(monkeypatch, sage)
    weight = sage_adapter.AiterFAv3SageBF16AttnWeight()
    q = torch.zeros((5, 2, 128), dtype=torch.bfloat16)
    kernel_config = {"BLOCK_M": 64, "BLOCK_N": 32}

    weight.apply(q, q, q, sage_config=kernel_config)

    assert calls[-1]["config"] == {
        "BLOCK_M": 64,
        "BLOCK_N": 32,
        "backend": "flydsl_v2",
    }
    assert kernel_config == {"BLOCK_M": 64, "BLOCK_N": 32}


def test_sage_apply_with_lse_rejects_ring_sp(monkeypatch):
    _mock_sage_available(
        monkeypatch,
        lambda *args, **kwargs: pytest.fail("Sage kernel must not run"),
    )
    weight = sage_adapter.AiterFAv3SageBF16AttnWeight()
    q = torch.zeros((2, 3, 2, 128), dtype=torch.bfloat16)

    with pytest.raises(NotImplementedError, match="Ring SP"):
        weight.apply_with_lse(q, q, q, softmax_scale=0.25)


@pytest.mark.parametrize(
    ("q", "kwargs", "error", "message"),
    [
        (
            torch.zeros((5, 2, 128), dtype=torch.float32),
            {},
            RuntimeError,
            "requires BF16",
        ),
        (
            torch.zeros((5, 2, 128), dtype=torch.bfloat16),
            {"dropout_p": 0.1},
            NotImplementedError,
            "dropout",
        ),
        (
            torch.zeros((5, 2, 128), dtype=torch.bfloat16),
            {"cu_seqlens_q": torch.tensor([0, 2, 5])},
            ValueError,
            "dense fixed-length",
        ),
        (
            torch.zeros((5, 2, 128), dtype=torch.bfloat16),
            {"causal": True},
            NotImplementedError,
            "non-causal",
        ),
        (
            torch.zeros((5, 2, 128), dtype=torch.bfloat16),
            {"window_size": (64, 64)},
            NotImplementedError,
            "sliding-window",
        ),
        (
            torch.zeros((5, 2, 128), dtype=torch.bfloat16),
            {"return_lse": True},
            NotImplementedError,
            "does not return LSE",
        ),
    ],
)
def test_sage_rejects_unsupported_contracts(monkeypatch, q, kwargs, error, message):
    _mock_sage_available(
        monkeypatch,
        lambda *args, **kwargs: pytest.fail("Sage kernel must not run"),
    )
    weight = sage_adapter.AiterFAv3SageBF16AttnWeight()

    with pytest.raises(error, match=message):
        weight.apply(q, q, q, **kwargs)


@pytest.mark.parametrize(
    ("q_shape", "k_shape", "v_shape", "message"),
    [
        ((5, 2, 128), (4, 2, 128), (4, 2, 128), "self-attention"),
        ((5, 4, 128), (5, 2, 128), (5, 2, 128), "GQA/MQA"),
        ((5, 2, 64), (5, 2, 64), (5, 2, 64), "head dimension 128"),
    ],
)
def test_sage_rejects_non_native_shapes(
    monkeypatch, q_shape, k_shape, v_shape, message
):
    _mock_sage_available(
        monkeypatch,
        lambda *args, **kwargs: pytest.fail("Sage kernel must not run"),
    )
    weight = sage_adapter.AiterFAv3SageBF16AttnWeight()
    q = torch.zeros(q_shape, dtype=torch.bfloat16)
    k = torch.zeros(k_shape, dtype=torch.bfloat16)
    v = torch.zeros(v_shape, dtype=torch.bfloat16)

    with pytest.raises(ValueError, match=message):
        weight.apply(q, k, v)


def test_sage_rejects_non_v2_backend(monkeypatch):
    _mock_sage_available(
        monkeypatch,
        lambda *args, **kwargs: pytest.fail("Sage kernel must not run"),
    )
    weight = sage_adapter.AiterFAv3SageBF16AttnWeight()
    q = torch.zeros((5, 2, 128), dtype=torch.bfloat16)

    with pytest.raises(ValueError, match="backend='flydsl_v2'"):
        weight.apply(q, q, q, sage_config={"backend": "triton"})


def test_dense_and_varlen_metadata_contract(monkeypatch):
    adapter.get_aiter_attention_route_counts(reset=True)
    dense_calls = []
    varlen_calls = []

    def dense(q, k, v, **kwargs):
        dense_calls.append((q, k, v, kwargs))
        return q

    def varlen(q, k, v, cu_q, cu_k, max_q, max_k, **kwargs):
        varlen_calls.append((q, k, v, cu_q, cu_k, max_q, max_k, kwargs))
        return q

    monkeypatch.setattr(adapter, "IS_AMD_ROCM", True)
    monkeypatch.setattr(adapter, "AITER_AVAILABLE", True)
    monkeypatch.setattr(adapter, "aiter_flash_attn_func", dense)
    monkeypatch.setattr(adapter, "aiter_flash_attn_varlen_func", varlen)
    weight = adapter.AiterAttnWeight()

    q = torch.zeros((4, 2, 8), dtype=torch.float32)
    cu = torch.tensor([0, 4], dtype=torch.int64)
    dense_out = weight.apply(q, q, q, cu_seqlens_q=cu, cu_seqlens_kv=cu)
    assert dense_out.shape == (4, 2 * 8)
    assert dense_calls[-1][0].shape == (1, 4, 2, 8)
    assert dense_calls[-1][3]["window_size"] == (-1, -1, 0)

    q = torch.zeros((5, 2, 8), dtype=torch.float32)
    k = torch.zeros((4, 2, 8), dtype=torch.float32)
    dense_cross_out = weight.apply(
        q,
        k,
        k,
        cu_seqlens_q=torch.tensor([0, 5]),
        cu_seqlens_kv=torch.tensor([0, 4]),
    )
    assert dense_cross_out.shape == (5, 2 * 8)
    assert dense_calls[-1][0].shape == (1, 5, 2, 8)
    assert dense_calls[-1][1].shape == (1, 4, 2, 8)

    q = torch.zeros((5, 2, 8), dtype=torch.float32)
    k = torch.zeros((4, 2, 8), dtype=torch.float32)
    cu_q = torch.tensor([0, 3, 5], dtype=torch.int64)
    cu_k = torch.tensor([0, 2, 4], dtype=torch.int64)
    varlen_out = weight.apply(
        q,
        k,
        k,
        cu_seqlens_q=cu_q,
        cu_seqlens_kv=cu_k,
        max_seqlen_q=3,
        max_seqlen_kv=2,
    )
    assert varlen_out.shape == (5, 2 * 8)
    assert varlen_calls[-1][3].dtype == torch.int32
    assert varlen_calls[-1][4].dtype == torch.int32
    assert varlen_calls[-1][3].device == q.device
    assert varlen_calls[-1][4].device == k.device
    assert varlen_calls[-1][3].is_contiguous()
    assert varlen_calls[-1][4].is_contiguous()

    counts = adapter.get_aiter_attention_route_counts(reset=True)
    assert counts["aiter_dense_triton_fallback"] == 2
    assert counts["aiter_varlen_triton_fallback"] == 1


@pytest.mark.parametrize(
    ("cu_q", "message"),
    [
        (torch.tensor([1, 4]), "start at 0"),
        (torch.tensor([0, 5]), "end at 4"),
        (torch.tensor([0, 3, 2, 4]), "monotonically non-decreasing"),
    ],
)
def test_invalid_cu_seqlens_fail_before_kernel(monkeypatch, cu_q, message):
    monkeypatch.setattr(adapter, "IS_AMD_ROCM", True)
    monkeypatch.setattr(adapter, "AITER_AVAILABLE", True)
    monkeypatch.setattr(
        adapter,
        "aiter_flash_attn_func",
        lambda *args, **kwargs: pytest.fail("dense kernel must not run"),
    )
    monkeypatch.setattr(
        adapter,
        "aiter_flash_attn_varlen_func",
        lambda *args, **kwargs: pytest.fail("varlen kernel must not run"),
    )
    weight = adapter.AiterAttnWeight()
    q = torch.zeros((4, 2, 8), dtype=torch.float32)

    with pytest.raises(ValueError, match=message):
        weight.apply(
            q,
            q,
            q,
            cu_seqlens_q=cu_q,
            cu_seqlens_kv=torch.tensor([0, 4]),
            max_seqlen_q=4 if cu_q.numel() == 2 else 2,
            max_seqlen_kv=4,
        )


def test_triton_apply_with_lse_normalizes_ring_contract(monkeypatch):
    calls = []

    def dense(q, k, v, **kwargs):
        calls.append(kwargs)
        batch, tokens, heads, _ = q.shape
        lse = torch.arange(
            batch * heads * tokens,
            dtype=torch.float32,
        ).reshape(batch, heads, tokens)
        return q, lse

    monkeypatch.setattr(adapter, "IS_AMD_ROCM", True)
    monkeypatch.setattr(adapter, "AITER_AVAILABLE", True)
    monkeypatch.setattr(adapter, "aiter_flash_attn_func", dense)
    monkeypatch.setattr(adapter, "_dense_flydsl_eligible", lambda *args, **kwargs: False)
    weight = adapter.AiterTritonBF16FlashAttnWeight()

    q = torch.zeros((5, 2, 8), dtype=torch.bfloat16)
    output, lse = weight.apply_with_lse(q, q, q, softmax_scale=0.25)

    assert calls[-1]["return_lse"] is True
    assert calls[-1]["softmax_scale"] == 0.25
    assert output.shape == (5, 16)
    assert lse.shape == (5, 2)
    expected_lse = (
        torch.arange(10, dtype=torch.float32)
        .reshape(1, 2, 5)
        .transpose(1, 2)
        .reshape(5, 2)
    )
    torch.testing.assert_close(lse, expected_lse)


@pytest.mark.requires_rocm
@pytest.mark.requires_gfx1201
@requires_gfx1201
def test_gfx1201_dense_self_attention_matches_reference():
    adapter.get_aiter_attention_route_counts(reset=True)
    torch.manual_seed(17)
    q = torch.randn((256, 40, 128), device="cuda", dtype=torch.bfloat16) * 0.25
    cu = torch.tensor([0, q.shape[0]], device=q.device, dtype=torch.int32)
    weight = adapter.AiterFlyDSLBF16FlashAttnWeight()

    actual = weight.apply(
        q,
        q,
        q,
        cu_seqlens_q=cu,
        cu_seqlens_kv=cu,
        max_seqlen_q=q.shape[0],
        max_seqlen_kv=q.shape[0],
    ).reshape_as(q)
    reference = _attention_reference(q, q, q)

    counts = adapter.get_aiter_attention_route_counts(reset=True)
    assert counts["aiter_dense_flydsl_candidate"] == 1
    _assert_attention_close(actual, reference)


@pytest.mark.requires_rocm
@pytest.mark.requires_gfx1201
@requires_gfx1201
def test_gfx1201_dense_cross_attention_matches_reference():
    adapter.get_aiter_attention_route_counts(reset=True)
    torch.manual_seed(19)
    q = torch.randn((129, 8, 128), device="cuda", dtype=torch.bfloat16) * 0.25
    k = torch.randn((77, 8, 128), device="cuda", dtype=torch.bfloat16) * 0.25
    v = torch.randn((77, 8, 128), device="cuda", dtype=torch.bfloat16) * 0.25
    weight = adapter.AiterTritonBF16FlashAttnWeight()

    actual = weight.apply(
        q,
        k,
        v,
        cu_seqlens_q=torch.tensor([0, q.shape[0]], device=q.device),
        cu_seqlens_kv=torch.tensor([0, k.shape[0]], device=k.device),
    ).reshape_as(q)
    reference = _attention_reference(q, k, v)

    counts = adapter.get_aiter_attention_route_counts(reset=True)
    assert counts["aiter_dense_triton_fallback"] == 1
    _assert_attention_close(actual, reference)


@pytest.mark.requires_rocm
@pytest.mark.requires_gfx1201
@requires_gfx1201
def test_gfx1201_dense_attention_with_lse_matches_reference():
    torch.manual_seed(29)
    q = torch.randn((65, 4, 128), device="cuda", dtype=torch.bfloat16) * 0.25
    k = torch.randn((47, 4, 128), device="cuda", dtype=torch.bfloat16) * 0.25
    v = torch.randn((47, 4, 128), device="cuda", dtype=torch.bfloat16) * 0.25
    softmax_scale = q.shape[-1] ** -0.5
    weight = adapter.AiterTritonBF16FlashAttnWeight()

    actual, actual_lse = weight.apply_with_lse(
        q,
        k,
        v,
        softmax_scale=softmax_scale,
    )
    scores = torch.einsum("qhd,khd->hqk", q.float(), k.float()) * softmax_scale
    reference_lse = torch.logsumexp(scores, dim=-1).transpose(0, 1)

    _assert_attention_close(actual.reshape_as(q), _attention_reference(q, k, v))
    torch.testing.assert_close(actual_lse, reference_lse, atol=3e-2, rtol=3e-2)


@pytest.mark.requires_rocm
@pytest.mark.requires_gfx1201
@requires_gfx1201
def test_gfx1201_native_sage_self_attention_matches_reference():
    torch.manual_seed(37)
    q = torch.randn((65, 4, 128), device="cuda", dtype=torch.bfloat16) * 0.25
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    softmax_scale = q.shape[-1] ** -0.5
    weight = sage_adapter.AiterFAv3SageBF16AttnWeight()

    actual = weight.apply(
        q,
        k,
        v,
        softmax_scale=softmax_scale,
    )
    reference = _attention_reference(q, k, v, softmax_scale)

    _assert_sage_attention_close(actual.reshape_as(q), reference)


@pytest.mark.requires_rocm
@pytest.mark.requires_gfx1201
@requires_gfx1201
def test_gfx1201_packed_attention_matches_reference():
    adapter.get_aiter_attention_route_counts(reset=True)
    torch.manual_seed(23)
    q = torch.randn((96, 8, 128), device="cuda", dtype=torch.bfloat16) * 0.25
    k = torch.randn((80, 8, 128), device="cuda", dtype=torch.bfloat16) * 0.25
    v = torch.randn((80, 8, 128), device="cuda", dtype=torch.bfloat16) * 0.25
    # Deliberately provide CPU int64 metadata to exercise defensive
    # device/dtype normalization at the adapter boundary.
    cu_q = torch.tensor([0, 64, 96], dtype=torch.int64)
    cu_k = torch.tensor([0, 48, 80], dtype=torch.int64)
    weight = adapter.AiterTritonBF16FlashAttnWeight()

    actual = weight.apply(
        q,
        k,
        v,
        cu_seqlens_q=cu_q,
        cu_seqlens_kv=cu_k,
        max_seqlen_q=64,
        max_seqlen_kv=48,
    ).reshape_as(q)
    references = [
        _attention_reference(q[q0:q1], k[k0:k1], v[k0:k1])
        for q0, q1, k0, k1 in ((0, 64, 0, 48), (64, 96, 48, 80))
    ]
    reference = torch.cat(references, dim=0)

    counts = adapter.get_aiter_attention_route_counts(reset=True)
    assert counts["aiter_varlen_triton_fallback"] == 1
    _assert_attention_close(actual, reference)
