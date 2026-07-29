"""Contract tests for dense and packed Aiter attention routing."""

import pytest

torch = pytest.importorskip("torch")

from lightx2v_platform.ops.attn.amd_rocm import flash_attn as adapter


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


@pytest.mark.requires_rocm
@pytest.mark.requires_gfx1201
@requires_gfx1201
def test_gfx1201_dense_self_attention_matches_reference():
    adapter.get_aiter_attention_route_counts(reset=True)
    torch.manual_seed(17)
    q = torch.randn((256, 40, 128), device="cuda", dtype=torch.bfloat16) * 0.25
    cu = torch.tensor([0, q.shape[0]], device=q.device, dtype=torch.int32)
    weight = adapter.AiterAttnWeight()

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
    weight = adapter.AiterAttnWeight()

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
    weight = adapter.AiterAttnWeight()

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
