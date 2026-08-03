"""Unit tests for Wan runtime-tuning aggregation."""

import pytest

pytest.importorskip("torch")
pytest.importorskip("loguru")

from lightx2v.utils.wan_runtune import WanRunTuner


def _category(total_ms, max_ms):
    detail = {
        "m": 74592,
        "n": 5120,
        "k": 5120,
        "dtype": "torch.float8_e4m3fn",
    }
    return {
        "calls": 2,
        "total_ms": total_ms,
        "mean_ms": total_ms / 2,
        "max_ms": max_ms,
        "details": [
            {
                "detail": detail,
                "calls": 2,
                "total_ms": total_ms,
                "mean_ms": total_ms / 2,
                "max_ms": max_ms,
            }
        ],
    }


def test_merge_step_categories_preserves_shape_details():
    steps = [
        {"gpu": {"fp8.gemm": _category(10.0, 6.0)}},
        {"gpu": {"fp8.gemm": _category(14.0, 8.0)}},
    ]

    merged = WanRunTuner._merge_step_categories(steps, "gpu")["fp8.gemm"]

    assert merged["calls"] == 4
    assert merged["total_ms"] == 24.0
    assert merged["mean_ms"] == 6.0
    assert merged["max_ms"] == 8.0
    assert merged["details"] == [
        {
            "detail": {
                "m": 74592,
                "n": 5120,
                "k": 5120,
                "dtype": "torch.float8_e4m3fn",
            },
            "calls": 4,
            "total_ms": 24.0,
            "max_ms": 8.0,
            "mean_ms": 6.0,
        }
    ]
