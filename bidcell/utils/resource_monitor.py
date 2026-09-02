"""Reusable, CPU-safe stage resource monitoring."""

import json
import os
import platform
import resource
import time
from pathlib import Path
from typing import Any

import torch


def trainable_parameter_count(model: torch.nn.Module | None) -> int | None:
    if model is None:
        return None
    return sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )


def _peak_rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports KiB; macOS reports bytes.
    return int(value if platform.system() == "Darwin" else value * 1024)


class ResourceMonitor:
    """Measure one independently timed pipeline stage."""

    def __init__(self, stage: str, model: torch.nn.Module | None = None):
        self.stage = stage
        self.model = model
        self.started_at: float | None = None
        self.metrics: dict[str, Any] | None = None
        self.steps: int | None = None
        self._cuda_available = bool(torch.cuda.is_available())

    def start(self):
        if self.started_at is not None:
            raise RuntimeError("resource monitor is already running")
        if self._cuda_available:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        self.started_at = time.perf_counter()
        return self

    def stop(self, *, steps: int | None = None) -> dict[str, Any]:
        if self.started_at is None:
            raise RuntimeError("resource monitor has not been started")
        if self._cuda_available:
            torch.cuda.synchronize()
        wall_seconds = time.perf_counter() - self.started_at
        completed_steps = self.steps if steps is None else steps
        self.metrics = {
            "stage": self.stage,
            "wall_time_seconds": float(wall_seconds),
            "peak_cpu_rss_bytes": _peak_rss_bytes(),
            "peak_cuda_memory_allocated_bytes": (
                int(torch.cuda.max_memory_allocated()) if self._cuda_available else None
            ),
            "peak_cuda_memory_reserved_bytes": (
                int(torch.cuda.max_memory_reserved()) if self._cuda_available else None
            ),
            "cuda_available": self._cuda_available,
            "trainable_parameter_count": trainable_parameter_count(self.model),
            "steps": int(completed_steps) if completed_steps is not None else None,
            "steps_per_second": (
                float(completed_steps / wall_seconds)
                if completed_steps is not None and wall_seconds > 0
                else None
            ),
        }
        self.started_at = None
        return self.metrics

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop(steps=self.steps)
        return False


def update_resource_summary(
    path: str | Path,
    stage_metrics: dict[str, Any],
    *,
    model: torch.nn.Module | None = None,
) -> dict[str, Any]:
    """Merge one stage into a machine-readable resource summary."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        with destination.open() as handle:
            payload = json.load(handle)
    else:
        payload = {"schema_version": 1, "model_metadata": {}}
    stage = str(stage_metrics["stage"])
    payload[stage] = stage_metrics
    if model is not None:
        payload["model_metadata"] = {
            "model_class": model.__class__.__name__,
            "trainable_parameter_count": trainable_parameter_count(model),
        }
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    os.replace(temporary, destination)
    return payload
