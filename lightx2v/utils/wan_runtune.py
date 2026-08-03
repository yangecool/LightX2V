"""Low-overhead runtime timing for full Wan inference runs."""

from __future__ import annotations

import json
import os
import tempfile
import time
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import dataclass
from functools import wraps
from typing import Any

import torch
from loguru import logger

from lightx2v_platform.base import global_var


def _device_module():
    device_api = global_var.AI_DEVICE
    if not device_api:
        raise RuntimeError("LightX2V AI device is not initialized")
    return getattr(torch, device_api)


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value, got {value!r}")


def attention_detail(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> dict[str, Any]:
    """Return stable shape metadata for dense BSHD/SHD attention calls."""

    def shape(tensor):
        if tensor.ndim == 4:
            batch, sequence, heads, head_dim = tensor.shape
        elif tensor.ndim == 3:
            batch = 1
            sequence, heads, head_dim = tensor.shape
        else:
            return {"shape": list(tensor.shape)}
        return {
            "batch": int(batch),
            "sequence": int(sequence),
            "heads": int(heads),
            "head_dim": int(head_dim),
        }

    return {
        "q": shape(q),
        "k": shape(k),
        "v": shape(v),
        "dtype": str(q.dtype),
    }


def gemm_detail(m: int, n: int, k: int, dtype: torch.dtype) -> dict[str, Any]:
    return {"m": int(m), "n": int(n), "k": int(k), "dtype": str(dtype)}


@dataclass
class _GpuSample:
    category: str
    detail: dict[str, Any] | None
    start: Any
    end: Any


class _GpuRegion:
    def __init__(self, tuner, category, detail):
        self.tuner = tuner
        self.category = category
        self.detail = detail
        self.start = None

    def __enter__(self):
        device_module = _device_module()
        self.start = device_module.Event(enable_timing=True)
        self.start.record()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        end = _device_module().Event(enable_timing=True)
        end.record()
        self.tuner._gpu_samples.append(
            _GpuSample(self.category, self.detail, self.start, end)
        )
        return False


def _stats(values: list[float]) -> dict[str, float | int]:
    total = sum(values)
    count = len(values)
    return {
        "calls": count,
        "total_ms": total,
        "mean_ms": total / count if count else 0.0,
        "max_ms": max(values, default=0.0),
    }


def gpu_timed(category: str):
    """Time all GPU work launched by a function while a runtune step is active."""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with get_wan_runtuner().gpu_region(category):
                return func(*args, **kwargs)

        return wrapper

    return decorator


class WanRunTuner:
    """Collect nested GPU event timings and explicit CPU wait timings by step."""

    def __init__(self, enabled: bool, output_path: str = "", backend: str = "unknown"):
        self.enabled = enabled
        self.output_path = output_path
        self.backend = backend
        self._step = None
        self._gpu_samples: list[_GpuSample] = []
        self._cpu_samples: list[tuple[str, dict[str, Any] | None, float]] = []
        self.report = {
            "schema_version": 1,
            "backend": backend,
            "environment": self._environment(),
            "steps": [],
            "summary": {},
        }
        if self.enabled:
            if not self.output_path:
                raise ValueError(
                    "LIGHTX2V_WAN_RUNTUNE_OUTPUT is required when runtune is enabled"
                )
            logger.info(
                "[WanRunTune] enabled: backend={}, output={}",
                self.backend,
                self.output_path,
            )

    @classmethod
    def from_env(cls):
        return cls(
            enabled=_env_flag("LIGHTX2V_WAN_RUNTUNE", False),
            output_path=os.getenv("LIGHTX2V_WAN_RUNTUNE_OUTPUT", ""),
            backend=os.getenv("LIGHTX2V_WAN_RUNTUNE_BACKEND", "unknown"),
        )

    @staticmethod
    def _environment() -> dict[str, Any]:
        device_api = global_var.AI_DEVICE
        environment = {
            "torch_version": torch.__version__,
            "hip_version": getattr(torch.version, "hip", None),
            "device_api": device_api,
        }
        if not device_api:
            return environment
        device_module = _device_module()
        if device_module.is_available():
            index = device_module.current_device()
            properties = device_module.get_device_properties(index)
            environment.update(
                {
                    "device_index": index,
                    "device_name": device_module.get_device_name(index),
                    "device_arch": getattr(properties, "gcnArchName", ""),
                    "total_memory_bytes": int(properties.total_memory),
                    "multi_processor_count": int(
                        getattr(properties, "multi_processor_count", 0)
                    ),
                }
            )
        return environment

    @property
    def active(self) -> bool:
        return self.enabled and self._step is not None

    def begin_step(self, step_index: int, total_steps: int, segment_index: int = 0):
        if not self.enabled:
            return
        if self._step is not None:
            raise RuntimeError("WanRunTune step is already active")
        self._gpu_samples = []
        self._cpu_samples = []
        self._step = {
            "step_index": int(step_index),
            "total_steps": int(total_steps),
            "segment_index": int(segment_index),
            "wall_start": time.perf_counter(),
        }

    def gpu_region(self, category: str, detail: dict[str, Any] | None = None):
        if not self.active:
            return nullcontext()
        return _GpuRegion(self, category, detail)

    def record_cpu_ms(
        self,
        category: str,
        elapsed_ms: float,
        detail: dict[str, Any] | None = None,
    ):
        if self.active:
            self._cpu_samples.append((category, detail, float(elapsed_ms)))

    @staticmethod
    def _aggregate(samples):
        by_category = defaultdict(list)
        by_detail = defaultdict(lambda: defaultdict(list))
        details = {}
        for category, detail, elapsed_ms in samples:
            by_category[category].append(elapsed_ms)
            if detail is not None:
                detail_key = json.dumps(detail, sort_keys=True, separators=(",", ":"))
                by_detail[category][detail_key].append(elapsed_ms)
                details[detail_key] = detail

        aggregated = {}
        for category, values in sorted(by_category.items()):
            category_stats = _stats(values)
            category_stats["details"] = [
                {"detail": details[key], **_stats(detail_values)}
                for key, detail_values in sorted(by_detail[category].items())
            ]
            aggregated[category] = category_stats
        return aggregated

    def end_step(self, *, success: bool = True, error: str | None = None):
        if not self.enabled:
            return
        if self._step is None:
            raise RuntimeError("WanRunTune has no active step")

        gpu_samples = []
        if success:
            _device_module().synchronize()
            for sample in self._gpu_samples:
                gpu_samples.append(
                    (
                        sample.category,
                        sample.detail,
                        float(sample.start.elapsed_time(sample.end)),
                    )
                )
        wall_ms = (time.perf_counter() - self._step["wall_start"]) * 1000.0

        step_report = {
            "step_index": self._step["step_index"],
            "total_steps": self._step["total_steps"],
            "segment_index": self._step["segment_index"],
            "success": bool(success),
            "wall_ms": wall_ms,
            "gpu": self._aggregate(gpu_samples),
            "cpu": self._aggregate(self._cpu_samples),
        }
        if error:
            step_report["error"] = error
        self.report["steps"].append(step_report)
        self._step = None
        self._gpu_samples = []
        self._cpu_samples = []
        self._refresh_summary()
        self._write_report()

        gpu = step_report["gpu"]
        logger.info(
            "[WanRunTune] step={}/{} wall={:.3f}ms self={:.3f}ms "
            "cross={:.3f}ms ffn={:.3f}ms fp8_gemm={:.3f}ms",
            step_report["step_index"] + 1,
            step_report["total_steps"],
            wall_ms,
            gpu.get("wan.self_phase", {}).get("total_ms", 0.0),
            gpu.get("wan.cross_phase", {}).get("total_ms", 0.0),
            gpu.get("wan.ffn_phase", {}).get("total_ms", 0.0),
            gpu.get("fp8.gemm", {}).get("total_ms", 0.0),
        )

    def _refresh_summary(self):
        successful = [step for step in self.report["steps"] if step["success"]]
        wall_values = [step["wall_ms"] for step in successful]
        self.report["summary"] = {
            "successful_steps": len(successful),
            "failed_steps": len(self.report["steps"]) - len(successful),
            "step_wall": _stats(wall_values),
            "gpu": self._merge_step_categories(successful, "gpu"),
            "cpu": self._merge_step_categories(successful, "cpu"),
        }

    @staticmethod
    def _merge_step_categories(steps, sample_kind):
        merged = {}
        category_names = sorted(
            {
                category
                for step in steps
                for category in step[sample_kind]
            }
        )
        for category in category_names:
            entries = [
                step[sample_kind][category]
                for step in steps
                if category in step[sample_kind]
            ]
            calls = sum(entry["calls"] for entry in entries)
            total_ms = sum(entry["total_ms"] for entry in entries)
            details = {}
            for entry in entries:
                for detail_entry in entry.get("details", []):
                    detail = detail_entry["detail"]
                    detail_key = json.dumps(
                        detail, sort_keys=True, separators=(",", ":")
                    )
                    aggregate = details.setdefault(
                        detail_key,
                        {
                            "detail": detail,
                            "calls": 0,
                            "total_ms": 0.0,
                            "max_ms": 0.0,
                        },
                    )
                    aggregate["calls"] += detail_entry["calls"]
                    aggregate["total_ms"] += detail_entry["total_ms"]
                    aggregate["max_ms"] = max(
                        aggregate["max_ms"], detail_entry["max_ms"]
                    )

            merged_details = []
            for detail_key in sorted(details):
                aggregate = details[detail_key]
                detail_calls = aggregate["calls"]
                merged_details.append(
                    {
                        **aggregate,
                        "mean_ms": (
                            aggregate["total_ms"] / detail_calls
                            if detail_calls
                            else 0.0
                        ),
                    }
                )
            merged[category] = {
                "calls": calls,
                "total_ms": total_ms,
                "mean_ms": total_ms / calls if calls else 0.0,
                "max_ms": max((entry["max_ms"] for entry in entries), default=0.0),
                "details": merged_details,
            }
        return merged

    def _write_report(self):
        output_path = os.path.abspath(self.output_path)
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)
        fd, temporary_path = tempfile.mkstemp(
            prefix=".wan-runtune-", suffix=".json", dir=output_dir
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as report_file:
                json.dump(self.report, report_file, indent=2, ensure_ascii=True)
                report_file.write("\n")
            os.replace(temporary_path, output_path)
        finally:
            if os.path.exists(temporary_path):
                os.unlink(temporary_path)


_WAN_RUNTUNER = None


def get_wan_runtuner() -> WanRunTuner:
    global _WAN_RUNTUNER
    if _WAN_RUNTUNER is None:
        _WAN_RUNTUNER = WanRunTuner.from_env()
    return _WAN_RUNTUNER
