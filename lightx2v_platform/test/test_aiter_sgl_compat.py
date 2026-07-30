"""Contract tests for the AMD SGL/Aiter boundary.

These tests intentionally mock the kernels. Real kernel correctness and
performance require a gfx1201 ROCm host and are covered by the bring-up runbook.
"""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from lightx2v_platform.base import amd_rocm


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


def _fake_aiter():
    calls = []

    def public_gemm(*args):
        calls.append(("public", args))
        x, w = args[:2]
        return torch.zeros((x.shape[0], w.shape[0]), dtype=args[5], device=x.device)

    def ck_gemm(*args):
        calls.append(("ck", args))
        x, w = args[:2]
        return torch.zeros((x.shape[0], w.shape[0]), dtype=args[5], device=x.device)

    def quant(x, quant_dtype):
        calls.append(("quant", quant_dtype))
        return x.to(quant_dtype), torch.ones((x.shape[0], 1), dtype=torch.float32)

    module = SimpleNamespace(
        gemm_a8w8=public_gemm,
        gemm_a8w8_CK=ck_gemm,
        pertoken_quant=quant,
        dtypes=SimpleNamespace(fp8=torch.float8_e4m3fn, i8=torch.int8),
        rmsnorm2d_fwd=lambda x, weight, eps, mode: x,
        rms_norm=lambda x, weight, eps: x,
    )
    return module, calls


def test_gfx1201_rowwise_dispatch_preserves_fp8_layout_and_bias(monkeypatch):
    module, calls = _fake_aiter()
    monkeypatch.setattr(amd_rocm, "_runtime_gfx", lambda: "gfx1201")
    compat = amd_rocm.AiterSglKernelCompat(module)

    x = torch.zeros((3, 5), dtype=torch.int8)
    # SGL passes [K, N] as a transposed view of the checkpoint [N, K].
    checkpoint_weight = torch.zeros((7, 5), dtype=torch.int8)
    sgl_weight = checkpoint_weight.t()
    output = compat.fp8_scaled_mm(
        x,
        sgl_weight,
        torch.ones((3, 1), dtype=torch.float32),
        torch.ones((7, 1), dtype=torch.float32),
        torch.bfloat16,
        bias=torch.ones(7, dtype=torch.bfloat16),
    )

    assert output.shape == (3, 7)
    kind, args = calls[-1]
    _, restored_weight, _, restored_scale, restored_bias, restored_dtype = args
    assert kind == "public"
    assert tuple(restored_weight.shape) == (7, 5)
    assert restored_weight.is_contiguous()
    assert tuple(restored_scale.shape) == (7, 1)
    assert restored_bias.shape == (7,)
    assert restored_dtype == torch.bfloat16
    assert torch.all(output == 0)


def test_quant_uses_e4m3fn(monkeypatch):
    module, calls = _fake_aiter()
    monkeypatch.setattr(amd_rocm, "_runtime_gfx", lambda: "gfx1201")
    compat = amd_rocm.AiterSglKernelCompat(module)
    out = torch.empty((2, 4), dtype=torch.float8_e4m3fn)
    scale = torch.empty((2, 1), dtype=torch.float32)
    compat.sgl_per_token_quant_fp8(torch.ones((2, 4)), out, scale)
    assert calls[-1] == ("quant", torch.float8_e4m3fn)


def test_gfx1201_rowwise_dispatch_vector_scales_are_zero_copy(monkeypatch):
    module, _ = _fake_aiter()
    monkeypatch.setattr(amd_rocm, "_runtime_gfx", lambda: "gfx1201")
    compat = amd_rocm.AiterSglKernelCompat(module)
    observed = {}

    def public_gemm(x, w, x_scale, w_scale, bias, dtype):
        observed.update(x_scale=x_scale, w_scale=w_scale)
        return torch.zeros((x.shape[0], w.shape[0]), dtype=dtype)

    compat._gemm_a8w8 = public_gemm
    input_scale = torch.ones(3, dtype=torch.float32)
    weight_scale = torch.ones(7, dtype=torch.float32)
    checkpoint_weight = torch.zeros((7, 5), dtype=torch.int8)
    compat.fp8_scaled_mm(
        torch.zeros((3, 5), dtype=torch.int8),
        checkpoint_weight.t(),
        input_scale,
        weight_scale,
        torch.bfloat16,
    )

    assert observed["x_scale"].shape == (3,)
    assert observed["w_scale"].shape == (7,)
    assert observed["x_scale"].data_ptr() == input_scale.data_ptr()
    assert observed["w_scale"].data_ptr() == weight_scale.data_ptr()


def test_rmsnorm_accepts_enable_pdl_and_selects_explicit_backend(monkeypatch):
    module, _ = _fake_aiter()
    rms_calls = []
    module.rmsnorm2d_fwd = lambda x, weight, eps, mode: rms_calls.append(
        ("aiter", mode)
    ) or x
    module.rms_norm = lambda x, weight, eps: rms_calls.append(("ck", None)) or x
    monkeypatch.setattr(amd_rocm, "_runtime_gfx", lambda: "gfx1201")

    compat = amd_rocm.AiterSglKernelCompat(module)
    x = torch.ones((2, 4))
    weight = torch.ones(4)
    assert compat.rmsnorm(x, weight, 1e-6, enable_pdl=True) is x
    assert rms_calls[-1] == ("aiter", 0)

    compat.rmsnorm_backend = "ck"
    assert compat.rmsnorm(x, weight, 1e-6, enable_pdl=True) is x
    assert rms_calls[-1] == ("ck", None)


def test_non_gfx1201_uses_public_dispatcher(monkeypatch):
    module, calls = _fake_aiter()
    monkeypatch.setattr(amd_rocm, "_runtime_gfx", lambda: "gfx942")
    compat = amd_rocm.AiterSglKernelCompat(module)
    x = torch.zeros((2, 3), dtype=torch.int8)
    weight = torch.zeros((3, 4), dtype=torch.int8).t()
    compat.fp8_scaled_mm(
        x,
        weight,
        torch.ones((2, 1)),
        torch.ones((4, 1)),
        torch.bfloat16,
    )
    assert calls[-1][0] == "public"


@pytest.mark.requires_rocm
@pytest.mark.requires_gfx1201
@requires_gfx1201
def test_gfx1201_fp8_gemm_matches_dequantized_reference():
    import aiter

    compat = amd_rocm.AiterSglKernelCompat(aiter)
    assert compat._gemm_backend == "aiter-rowwise-dispatch"
    device = torch.device("cuda")
    torch.manual_seed(7)
    m, n, k = 37, 53, 128
    x = torch.randn((m, k), device=device, dtype=torch.bfloat16) * 0.25
    weight = torch.randn((n, k), device=device, dtype=torch.bfloat16) * 0.25
    x_quant, x_scale = aiter.pertoken_quant(
        x, quant_dtype=torch.float8_e4m3fn
    )
    weight_quant, weight_scale = aiter.pertoken_quant(
        weight, quant_dtype=torch.float8_e4m3fn
    )
    bias = torch.randn(n, device=device, dtype=torch.bfloat16) * 0.1

    actual = compat.fp8_scaled_mm(
        x_quant,
        weight_quant.t(),
        x_scale,
        weight_scale,
        torch.bfloat16,
        bias=bias,
    )
    reference = (
        (x_quant.float() * x_scale.float())
        @ (weight_quant.float() * weight_scale.float()).t()
        + bias.float()
    )

    assert actual.shape == (m, n)
    assert actual.dtype == torch.bfloat16
    assert torch.isfinite(actual).all()
    cosine = torch.nn.functional.cosine_similarity(
        actual.float().flatten(), reference.flatten(), dim=0
    )
    assert float(cosine) >= 0.999
    torch.testing.assert_close(actual.float(), reference, atol=3e-2, rtol=3e-2)


@pytest.mark.requires_rocm
@pytest.mark.requires_gfx1201
@requires_gfx1201
def test_gfx1201_t5_fp8_rowwise_gemm_shape():
    import aiter

    compat = amd_rocm.AiterSglKernelCompat(aiter)
    device = torch.device("cuda")
    m, n, k = 512, 4096, 4096
    x = torch.randn((m, k), device=device, dtype=torch.bfloat16) * 0.125
    weight = torch.randn((n, k), device=device, dtype=torch.bfloat16) * 0.125
    x_quant, x_scale = aiter.pertoken_quant(
        x, quant_dtype=torch.float8_e4m3fn
    )
    weight_quant, weight_scale = aiter.pertoken_quant(
        weight, quant_dtype=torch.float8_e4m3fn
    )

    actual = compat.fp8_scaled_mm(
        x_quant,
        weight_quant.t(),
        x_scale,
        weight_scale,
        torch.bfloat16,
    )

    assert actual.shape == (m, n)
    assert actual.dtype == torch.bfloat16
    assert torch.isfinite(actual).all()


@pytest.mark.requires_rocm
@pytest.mark.requires_gfx1201
@requires_gfx1201
def test_gfx1201_rmsnorm_hidden_5120_matches_torch():
    import aiter

    compat = amd_rocm.AiterSglKernelCompat(aiter)
    compat.rmsnorm_backend = "aiter"
    device = torch.device("cuda")
    torch.manual_seed(11)
    x = torch.randn((7, 5120), device=device, dtype=torch.bfloat16)
    weight = torch.randn(5120, device=device, dtype=torch.bfloat16) * 0.1 + 1
    eps = 1e-6

    actual = compat.rmsnorm(x, weight, eps, enable_pdl=True)
    reference = (
        x.float()
        * torch.rsqrt(x.float().square().mean(dim=-1, keepdim=True) + eps)
        * weight.float()
    )

    assert actual.shape == x.shape
    assert actual.dtype == x.dtype
    assert torch.isfinite(actual).all()
    cosine = torch.nn.functional.cosine_similarity(
        actual.float().flatten(), reference.flatten(), dim=0
    )
    assert float(cosine) >= 0.9999
    torch.testing.assert_close(actual.float(), reference, atol=3e-2, rtol=3e-2)
