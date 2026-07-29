"""Shared, dependency-light helpers for BIDCell benchmark tiers."""

from __future__ import annotations

import csv
import json
import math
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage


MANIFEST_FIELDS = [
    "metric_id",
    "tier",
    "metric_name",
    "status",
    "value",
    "unit",
    "method",
    "required_inputs",
    "used_inputs",
    "notes",
]
VALID_STATUSES = {
    "computed",
    "skipped_missing_input",
    "not_applicable_v1",
    "not_implemented",
    "failed",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(json_safe(value), handle, indent=2, sort_keys=True)
        handle.write("\n")


def git_commit(repo_dir: Path) -> str | None:
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


@dataclass
class Manifest:
    rows: list[dict[str, Any]] = field(default_factory=list)

    def add(
        self,
        metric_id: str,
        tier: int,
        metric_name: str,
        status: str,
        *,
        value: Any = None,
        unit: str = "",
        method: str = "",
        required_inputs: Iterable[str] = (),
        used_inputs: Iterable[str] = (),
        notes: str = "",
    ) -> None:
        if status not in VALID_STATUSES:
            raise ValueError(f"Unsupported benchmark status: {status}")
        self.rows.append(
            {
                "metric_id": metric_id,
                "tier": tier,
                "metric_name": metric_name,
                "status": status,
                "value": json_safe(value),
                "unit": unit,
                "method": method,
                "required_inputs": ";".join(required_inputs),
                "used_inputs": ";".join(used_inputs),
                "notes": notes,
            }
        )

    def missing(
        self,
        metric_id: str,
        tier: int,
        metric_name: str,
        required_inputs: Iterable[str],
        notes: str = "",
    ) -> None:
        self.add(
            metric_id,
            tier,
            metric_name,
            "skipped_missing_input",
            required_inputs=required_inputs,
            notes=notes,
        )

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
            writer.writeheader()
            writer.writerows(self.rows)


def load_mask(path: str | Path) -> np.ndarray:
    path = Path(path)
    if path.suffix.lower() == ".npy":
        array = np.load(path)
    elif path.suffix.lower() == ".npz":
        archive = np.load(path)
        if len(archive.files) != 1:
            raise ValueError(f"NPZ mask must contain exactly one array: {path}")
        array = archive[archive.files[0]]
    else:
        array = tifffile.imread(path)
    array = np.asarray(array)
    if array.ndim != 2:
        raise ValueError(f"Mask must be two dimensional: {path}")
    return array.astype(np.int64, copy=False)


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    return pd.read_csv(path)


def infer_column(frame: pd.DataFrame, names: Iterable[str]) -> str | None:
    lower = {str(column).lower(): str(column) for column in frame.columns}
    for name in names:
        if name.lower() in lower:
            return lower[name.lower()]
    return None


def transcript_assignments(
    transcripts: pd.DataFrame,
    mask: np.ndarray,
    pixel_size_um: float,
) -> pd.DataFrame:
    x_col = infer_column(
        transcripts, ("x_location", "x", "global_x", "cell_centroid_x")
    )
    y_col = infer_column(
        transcripts, ("y_location", "y", "global_y", "cell_centroid_y")
    )
    if x_col is None or y_col is None:
        raise ValueError("Transcript table requires x/y coordinate columns")
    x = np.rint(transcripts[x_col].to_numpy(dtype=float) / pixel_size_um).astype(int)
    y = np.rint(transcripts[y_col].to_numpy(dtype=float) / pixel_size_um).astype(int)
    in_bounds = (x >= 0) & (x < mask.shape[1]) & (y >= 0) & (y < mask.shape[0])
    labels = np.zeros(len(transcripts), dtype=np.int64)
    labels[in_bounds] = mask[y[in_bounds], x[in_bounds]]
    result = transcripts.copy()
    result["_pixel_x"] = x
    result["_pixel_y"] = y
    result["_in_bounds"] = in_bounds
    result["_assigned_cell_id"] = labels
    return result


def distribution(values: Iterable[float]) -> dict[str, float | int | None]:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "iqr": None,
            "p5": None,
            "p95": None,
            "p99": None,
        }
    q25, q75 = np.percentile(array, [25, 75])
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "iqr": float(q75 - q25),
        "p5": float(np.percentile(array, 5)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
    }


def label_boundary(mask: np.ndarray) -> np.ndarray:
    boundary = np.zeros(mask.shape, dtype=bool)
    vertical = mask[1:, :] != mask[:-1, :]
    horizontal = mask[:, 1:] != mask[:, :-1]
    boundary[1:, :] |= vertical
    boundary[:-1, :] |= vertical
    boundary[:, 1:] |= horizontal
    boundary[:, :-1] |= horizontal
    return boundary


def boundary_distance_um(mask: np.ndarray, pixel_size_um: float) -> np.ndarray:
    boundary = label_boundary(mask)
    if not boundary.any():
        return np.full(mask.shape, np.inf, dtype=float)
    return ndimage.distance_transform_edt(~boundary) * pixel_size_um


def directory_size_bytes(path: str | Path | None) -> int | None:
    if path is None or not Path(path).exists():
        return None
    root = Path(path)
    if root.is_file():
        return root.stat().st_size
    return sum(item.stat().st_size for item in root.rglob("*") if item.is_file())
