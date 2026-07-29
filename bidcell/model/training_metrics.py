"""Scalar-only instrumentation for candidate-overlap V1 training."""

from __future__ import annotations

import csv
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


TRAINING_METRIC_FIELDS = [
    "global_step",
    "epoch",
    "batch_index",
    "n_candidates",
    "learning_rate",
    "total_loss",
    "loss_ne",
    "loss_os",
    "loss_cc",
    "loss_ov",
    "loss_pos",
    "loss_neg",
    "candidate_overlap_pixel_count",
    "candidate_overlap_fraction",
    "boundary_weight",
    "step_time_seconds",
    "cumulative_training_time_seconds",
    "cuda_memory_allocated_mb",
    "cuda_memory_reserved_mb",
    "cuda_max_memory_allocated_mb",
    "cuda_max_memory_reserved_mb",
]


def build_candidate_overlap(batch_sa: torch.Tensor, dtype=None) -> torch.Tensor:
    """Return pixels covered by at least two candidate search areas."""
    if batch_sa.ndim != 4 or batch_sa.shape[1] != 1:
        raise ValueError("batch_sa must have shape [n_candidates, 1, H, W]")
    output_dtype = dtype
    if output_dtype is None:
        output_dtype = batch_sa.dtype if batch_sa.is_floating_point() else torch.float32
    return (batch_sa[:, 0, :, :].sum(dim=0) >= 2).to(dtype=output_dtype)


def build_candidate_overlap_weight(
    batch_sa: torch.Tensor,
    boundary_weight: float,
    dtype=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the V1 overlap indicator and its ``1 + weight`` spatial weight."""
    if boundary_weight < 0:
        raise ValueError("boundary_weight must be greater than or equal to zero")
    candidate_overlap = build_candidate_overlap(batch_sa, dtype=dtype)
    spatial_weight = 1.0 + float(boundary_weight) * candidate_overlap
    return candidate_overlap, spatial_weight


def scalar(value: Any) -> int | float | str | None:
    """Detach a scalar tensor and convert it to a CSV/JSON-safe Python value."""
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError("Training metrics may only contain scalar tensors")
        value = value.detach().cpu().item()
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        value = value.item()
    if value is None or isinstance(value, (int, float, str)):
        return value
    raise TypeError(f"Unsupported training metric type: {type(value).__name__}")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_commit(repo_dir: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_dir,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


class TrainingMetricsLogger:
    """Append one detached scalar row at a time and flush regularly."""

    def __init__(
        self,
        output_dir: str | Path,
        *,
        config_path: str | None = None,
        flush_interval: int = 10,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.output_dir / "v1_training_metrics.csv"
        self.summary_path = self.output_dir / "training_summary.json"
        self.flush_interval = max(1, int(flush_interval))
        self.rows_written = 0
        self.started_at = _utc_now()
        self.start_monotonic = time.perf_counter()
        self.config_path = config_path
        self.git_commit = _git_commit(Path(__file__).resolve().parents[2])

        file_exists = self.csv_path.exists() and self.csv_path.stat().st_size > 0
        if file_exists:
            with self.csv_path.open(newline="", encoding="utf-8") as existing:
                header = next(csv.reader(existing), [])
            if header != TRAINING_METRIC_FIELDS:
                raise ValueError(
                    f"Existing training metrics header does not match: {self.csv_path}"
                )
        self._handle = self.csv_path.open("a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(
            self._handle,
            fieldnames=TRAINING_METRIC_FIELDS,
            extrasaction="raise",
        )
        if not file_exists:
            self._writer.writeheader()
            self._handle.flush()

    @property
    def elapsed_seconds(self) -> float:
        return time.perf_counter() - self.start_monotonic

    def write(self, values: dict[str, Any]) -> None:
        row = {field: scalar(values.get(field)) for field in TRAINING_METRIC_FIELDS}
        self._writer.writerow(row)
        self.rows_written += 1
        if self.rows_written % self.flush_interval == 0:
            self._handle.flush()

    def close(self, *, status: str = "completed", error: str | None = None) -> None:
        if self._handle.closed:
            return
        self._handle.flush()
        self._handle.close()
        summary = {
            "status": status,
            "error": error,
            "started_at": self.started_at,
            "ended_at": _utc_now(),
            "training_wall_clock_seconds": self.elapsed_seconds,
            "rows_written": self.rows_written,
            "metrics_csv": str(self.csv_path),
            "config_path": self.config_path,
            "git_commit": self.git_commit,
            "method": "candidate-overlap boundary-weighted BIDCell V1",
        }
        with self.summary_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)
            handle.write("\n")

    def __enter__(self) -> "TrainingMetricsLogger":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close(
            status="failed" if exc is not None else "completed",
            error=str(exc) if exc is not None else None,
        )
