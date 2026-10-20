#!/usr/bin/env python3
"""
BIDCell official step-4000 baseline benchmark
==============================================

Purpose
-------
Run every benchmark metric that can be computed from the already-finished
BIDCell step-4000 baseline without retraining. Metrics that need unavailable
information are not silently approximated: they are written to
`metric_status.csv` as SKIPPED with the exact missing input.

Default remote paths are taken from the frozen baseline context:

    RUN  = /home/zwan0726/work/baseline/latest_bidcell_official_run
    REPO = /home/zwan0726/work/projects/BIDCell
    DATA = /home/zwan0726/work/data/Xenium_FFPE_Human_Breast_Cancer_Rep1/outs

The script is read-only with respect to RUN. Outputs go to:

    /home/zwan0726/work/benchmarks/bidcell_step4000_baseline

Implemented now
---------------
Tier 1: BIDCell / CellSPA-style baseline evaluation
    - cell number, foreground fraction
    - assigned transcript fraction (chunked transcript scan)
    - transcripts/cell, genes/cell, cell area, transcript density
    - fraction of cells expressing each gene
    - nucleus-to-cell transcript/gene information gain
    - raster shape metrics and nucleus/cell shape correlations
    - CellSPA-compatible positive/negative marker F1, precision, recall
    - marker expression percentages
    - cell-type average expression correlation to BIDCell sc reference
    - spatial entropy/diversity and CV correlations
    - distance-stratified neighbouring negative-marker contamination

Tier 2: boundary / structural evaluation available from final outputs
    - all-instance and cell-cell boundary masks
    - transcript fractions and assignment rates at 1/2/3/5 micrometre bands
    - nucleus containment, missed/split nuclei, zero/multi-nucleus cells
    - disconnected cell fraction (8-neighbour), unless skipped by CLI
    - cell area outlier fraction

Tier 3: efficiency information recoverable after the run
    - total wall-clock time from timestamped log lines
    - flattened status JSON inventory
    - key artifact sizes and total run disk footprint
    - checkpoint tensor parameter count when torch can load the checkpoint

Tier 4: optional pseudo-reference evaluation
    - binary foreground Dice/IoU and boundary F1/ASSD/HD95 ONLY when an
      already-aligned reference TIFF is passed with --vendor-mask.

Tier 5: statistical confidence
    - cell-level bootstrap confidence intervals for selected baseline metrics

Not currently possible without additional information
------------------------------------------------------
These items are registered in metric_status.csv and marked with TODO comments:

    - candidate-overlap area: needs per-candidate search_areas saved from the
      BIDCell dataset loader or an exact patch-wise reconstruction.
    - top-1/top-2 low-margin area: needs candidate-level logits from inference.
    - assignment-change and marker-consistent reassignment: needs V1/V2/V3.
    - V2/V3 assignment entropy and segmentation consistency: needs soft scores.
    - live peak VRAM/RAM/GPU utilisation/power: cannot be recovered after run.
    - exact FLOPs/MACs: needs an instantiated model and profiler.
    - vendor AJI+/PQ/AP: needs an aligned vendor instance mask; the current
      context does not confirm that such a mask has been converted.
    - Rep1/Rep2 replicability: needs Rep2 outputs.
    - random-seed stability and ablations: need new training runs.
    - non-zero expression concordance: needs raw single-cell reference data,
      not only the averaged BIDCell sc_breast.csv profiles.

Important interpretation notes
------------------------------
1. Shape metrics are raster-image equivalents. CellSPA's R implementation uses
   alpha-hull polygons for several shape definitions. The script keeps explicit
   metric names so these are not misrepresented as bit-for-bit CellSPA R output.
2. The marker F1 implementation follows CellSPA's published R logic:
       actual = genes in the marker list
       predicted = genes above the per-cell quantile 1 - marker_fraction
3. The combined `purity_f1_proxy` is clearly labelled as a proxy. Official
   CellSPA reports positive and negative marker metrics separately; the proxy is
   the harmonic mean of positive-marker F1 and (1 - negative-marker F1).
4. A vendor segmentation is a pseudo-reference, not manual ground truth.

Example
-------

    conda activate bidcell_local
    export PY=/home/zwan0726/miniforge3/envs/bidcell_local/bin/python

    $PY benchmark_bidcell_step4000_baseline.py

Faster first pass without connectivity analysis:

    $PY benchmark_bidcell_step4000_baseline.py --skip-connectivity

Optional, only after creating a vendor mask already aligned to the BIDCell grid:

    $PY benchmark_bidcell_step4000_baseline.py \
        --vendor-mask /path/to/vendor_mask_aligned_5478x7526.tif
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import logging
import math
import os
import platform
import re
import shutil
import sys
import time
import traceback
from collections import defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import h5py
import numpy as np
import pandas as pd
import tifffile
import yaml
from scipy import ndimage, stats
from scipy.spatial import cKDTree
from skimage import measure

try:
    import cv2
except Exception:  # pragma: no cover - fallback is implemented
    cv2 = None

try:
    import torch
except Exception:  # pragma: no cover - only parameter count is skipped
    torch = None


DEFAULT_RUN = Path("/home/zwan0726/work/baseline/latest_bidcell_official_run")
DEFAULT_REPO = Path("/home/zwan0726/work/projects/BIDCell")
DEFAULT_DATA = Path(
    "/home/zwan0726/work/data/Xenium_FFPE_Human_Breast_Cancer_Rep1/outs"
)
DEFAULT_OUTPUT = Path(
    "/home/zwan0726/work/benchmarks/bidcell_step4000_baseline"
)
DEFAULT_EXPERIMENT_ID = "2026_07_23_04_18_24"


@dataclass
class MetricStatus:
    tier: str
    metric: str
    status: str
    reason: str = ""
    output_file: str = ""
    notes: str = ""


class StatusRegistry:
    def __init__(self) -> None:
        self.rows: list[MetricStatus] = []

    def add(
        self,
        tier: str,
        metric: str,
        status: str,
        reason: str = "",
        output_file: str | Path = "",
        notes: str = "",
    ) -> None:
        self.rows.append(
            MetricStatus(
                tier=tier,
                metric=metric,
                status=status,
                reason=reason,
                output_file=str(output_file),
                notes=notes,
            )
        )

    def completed(
        self,
        tier: str,
        metric: str,
        output_file: str | Path = "",
        notes: str = "",
    ) -> None:
        self.add(tier, metric, "COMPLETED", output_file=output_file, notes=notes)

    def skipped(self, tier: str, metric: str, reason: str, notes: str = "") -> None:
        self.add(tier, metric, "SKIPPED", reason=reason, notes=notes)

    def failed(self, tier: str, metric: str, reason: str) -> None:
        self.add(tier, metric, "FAILED", reason=reason)

    def save(self, path: Path) -> None:
        pd.DataFrame([asdict(x) for x in self.rows]).to_csv(path, index=False)


@dataclass
class Paths:
    run_dir: Path
    repo_dir: Path
    data_dir: Path
    output_dir: Path
    config: Path
    connected_mask: Path
    raw_prediction: Path | None
    nuclei_mask: Path
    transcripts: Path
    final_expr: Path
    nuclei_expr: Path | None
    nuclei_types: Path | None
    gene_names: Path | None
    checkpoint: Path | None
    pipeline_log: Path | None
    status_dir: Path | None
    positive_markers: Path | None
    negative_markers: Path | None
    sc_reference: Path | None


# ---------------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------------


def configure_logging(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("bidcell_benchmark")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    fh = logging.FileHandler(output_dir / "benchmark.log", mode="w")
    fh.setFormatter(formatter)
    logger.addHandler(sh)
    logger.addHandler(fh)
    return logger


def safe_json_dump(obj: Any, path: Path) -> None:
    def convert(value: Any) -> Any:
        if isinstance(value, dict):
            return {str(k): convert(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [convert(v) for v in value]
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value

    with path.open("w", encoding="utf-8") as f:
        json.dump(convert(obj), f, indent=2, ensure_ascii=False)


def file_sha256(path: Path, block_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(block_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def package_versions() -> dict[str, str]:
    names = [
        "numpy",
        "pandas",
        "scipy",
        "scikit-image",
        "tifffile",
        "h5py",
        "opencv-python",
        "torch",
        "pyyaml",
    ]
    out: dict[str, str] = {}
    for name in names:
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = "not-installed"
    return out


def first_existing(candidates: Iterable[Path | None]) -> Path | None:
    for path in candidates:
        if path is not None and Path(path).exists():
            return Path(path).resolve()
    return None


def newest_glob(root: Path, pattern: str) -> Path | None:
    matches = list(root.glob(pattern))
    if not matches:
        return None
    return max(matches, key=lambda p: p.stat().st_mtime).resolve()


def resolve_from_config(value: str | None, config_path: Path, repo_dir: Path) -> Path | None:
    if not value:
        return None
    p = Path(value).expanduser()
    if p.is_absolute() and p.exists():
        return p.resolve()
    candidates = [
        config_path.parent / p,
        repo_dir / p,
        repo_dir.parent / p,
        Path.cwd() / p,
    ]
    return first_existing(candidates)


def recursive_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            fp = Path(root) / name
            try:
                total += fp.stat().st_size
            except FileNotFoundError:
                pass
    return total


def descriptive_stats(values: Sequence[float] | np.ndarray) -> dict[str, float | int]:
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"n": 0}
    return {
        "n": int(x.size),
        "mean": float(np.mean(x)),
        "std": float(np.std(x, ddof=1)) if x.size > 1 else 0.0,
        "min": float(np.min(x)),
        "p0_5": float(np.percentile(x, 0.5)),
        "p5": float(np.percentile(x, 5)),
        "p25": float(np.percentile(x, 25)),
        "median": float(np.median(x)),
        "p75": float(np.percentile(x, 75)),
        "p95": float(np.percentile(x, 95)),
        "p99": float(np.percentile(x, 99)),
        "p99_5": float(np.percentile(x, 99.5)),
        "max": float(np.max(x)),
    }


def coefficient_of_variation(values: pd.Series | np.ndarray) -> float:
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return np.nan
    mean = np.mean(x)
    if mean == 0:
        return np.nan
    return float(np.std(x, ddof=1) / mean)


def pearson_safe(x: Sequence[float], y: Sequence[float]) -> tuple[float, float, int]:
    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    keep = np.isfinite(xa) & np.isfinite(ya)
    xa = xa[keep]
    ya = ya[keep]
    if xa.size < 3 or np.std(xa) == 0 or np.std(ya) == 0:
        return np.nan, np.nan, int(xa.size)
    r, p = stats.pearsonr(xa, ya)
    return float(r), float(p), int(xa.size)


# ---------------------------------------------------------------------------
# Path/config discovery
# ---------------------------------------------------------------------------


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"YAML is not a mapping: {path}")
    return data


def discover_paths(args: argparse.Namespace) -> tuple[Paths, dict[str, Any]]:
    run_dir = args.run_dir.expanduser().resolve()
    repo_dir = args.repo_dir.expanduser().resolve()
    data_dir = args.data_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    config = args.config
    if config is None:
        config = newest_glob(run_dir / "configs", "*step4000*.yaml")
        if config is None:
            config = newest_glob(run_dir / "configs", "*.yaml")
    if config is None or not config.exists():
        raise FileNotFoundError("Could not locate the baseline YAML config.")
    config = config.expanduser().resolve()
    cfg = load_yaml(config)

    exp_id = args.experiment_id
    test_dir = run_dir / "model_outputs" / exp_id / "test_output"
    model_dir = run_dir / "model_outputs" / exp_id / "models"

    connected = args.connected_mask or first_existing(
        [
            test_dir / "epoch_1_step_4000_connected.tif",
            newest_glob(run_dir, "model_outputs/*/test_output/epoch_1_step_4000_connected.tif"),
        ]
    )
    if connected is None or not Path(connected).exists():
        raise FileNotFoundError("Could not locate epoch_1_step_4000_connected.tif")
    connected = Path(connected).expanduser().resolve()

    raw_prediction = first_existing(
        [
            test_dir / "epoch_1_step_4000.tif",
            newest_glob(run_dir, "model_outputs/*/test_output/epoch_1_step_4000.tif"),
        ]
    )
    nuclei = first_existing([run_dir / "nuclei.tif"])
    if nuclei is None:
        raise FileNotFoundError(f"Missing nuclei.tif under {run_dir}")

    files_cfg = cfg.get("files", {})
    transcripts = first_existing(
        [
            run_dir / str(files_cfg.get("fp_transcripts_processed", "transcripts_processed.csv")),
            run_dir / "transcripts_processed.csv",
        ]
    )
    if transcripts is None:
        raise FileNotFoundError("Missing transcripts_processed.csv")

    final_expr = first_existing(
        [
            run_dir / "cell_gene_matrices" / exp_id / "expr_mat.csv",
            newest_glob(run_dir, "cell_gene_matrices/20*/expr_mat.csv"),
        ]
    )
    if final_expr is None:
        raise FileNotFoundError("Missing final cell-level expr_mat.csv")

    nuclei_expr = first_existing(
        [run_dir / "cell_gene_matrices" / "nuclei" / "expr_mat.csv"]
    )
    nuclei_types = first_existing([run_dir / "nuclei_cell_type.h5"])
    gene_names = first_existing([run_dir / "all_gene_names.txt"])
    checkpoint = first_existing(
        [
            model_dir / "epoch_1_step_4000.pth",
            newest_glob(run_dir, "model_outputs/*/models/epoch_1_step_4000.pth"),
        ]
    )
    pipeline_log = first_existing(
        [
            run_dir / "logs" / "080_bidcell_official_step4000_full_pipeline.log",
            newest_glob(run_dir / "logs", "*.log"),
        ]
    )
    status_dir = run_dir / "status" if (run_dir / "status").exists() else None

    pos = resolve_from_config(files_cfg.get("fp_pos_markers"), config, repo_dir)
    neg = resolve_from_config(files_cfg.get("fp_neg_markers"), config, repo_dir)
    ref = resolve_from_config(files_cfg.get("fp_ref"), config, repo_dir)

    paths = Paths(
        run_dir=run_dir,
        repo_dir=repo_dir,
        data_dir=data_dir,
        output_dir=output_dir,
        config=config,
        connected_mask=connected,
        raw_prediction=raw_prediction,
        nuclei_mask=nuclei,
        transcripts=transcripts,
        final_expr=final_expr,
        nuclei_expr=nuclei_expr,
        nuclei_types=nuclei_types,
        gene_names=gene_names,
        checkpoint=checkpoint,
        pipeline_log=pipeline_log,
        status_dir=status_dir,
        positive_markers=pos,
        negative_markers=neg,
        sc_reference=ref,
    )
    return paths, cfg


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_tiff(path: Path, logger: logging.Logger) -> np.ndarray:
    logger.info("Loading TIFF: %s", path)
    try:
        arr = tifffile.memmap(path)
        logger.info("TIFF memory-mapped: shape=%s dtype=%s", arr.shape, arr.dtype)
        return arr
    except Exception:
        arr = tifffile.imread(path)
        logger.info("TIFF loaded: shape=%s dtype=%s", arr.shape, arr.dtype)
        return arr


def read_expression_matrix(path: Path, logger: logging.Logger) -> tuple[np.ndarray, list[str], np.ndarray]:
    logger.info("Reading expression matrix: %s", path)
    df = pd.read_csv(path, index_col=0, low_memory=False)
    if "cell_id" in df.columns:
        cell_ids = pd.to_numeric(df.pop("cell_id"), errors="raise").to_numpy(np.int64)
    else:
        # Fallback for matrices where the DataFrame index itself is the cell ID.
        cell_ids = pd.to_numeric(df.index, errors="raise").to_numpy(np.int64)
    genes = [str(c) for c in df.columns]
    matrix = df.to_numpy(dtype=np.float32, copy=True)
    del df
    logger.info("Expression matrix: cells=%d genes=%d", matrix.shape[0], matrix.shape[1])
    return cell_ids, genes, matrix


def load_cell_types(
    h5_path: Path | None,
    ref_path: Path | None,
    cell_ids: np.ndarray,
    logger: logging.Logger,
) -> tuple[np.ndarray, np.ndarray]:
    type_idx = np.full(cell_ids.shape[0], -1, dtype=np.int32)
    type_name = np.full(cell_ids.shape[0], "Unknown", dtype=object)
    if h5_path is None or not h5_path.exists():
        logger.warning("nuclei_cell_type.h5 missing; cell-type metrics will be skipped")
        return type_idx, type_name

    with h5py.File(h5_path, "r") as h5:
        ids = np.asarray(h5["ids"][:], dtype=np.int64)
        data = np.asarray(h5["data"][:], dtype=np.int32)
    max_id = int(max(ids.max(initial=0), cell_ids.max(initial=0)))
    lookup = np.full(max_id + 1, -1, dtype=np.int32)
    valid = (ids >= 0) & (ids <= max_id)
    lookup[ids[valid]] = data[valid]
    in_range = (cell_ids >= 0) & (cell_ids <= max_id)
    type_idx[in_range] = lookup[cell_ids[in_range]]

    idx_to_name: dict[int, str] = {}
    if ref_path is not None and ref_path.exists():
        ref = pd.read_csv(ref_path, index_col=0, low_memory=False)
        if {"ct_idx", "cell_type"}.issubset(ref.columns):
            for idx, group in ref.groupby("ct_idx", sort=False):
                names = group["cell_type"].dropna().astype(str)
                if not names.empty:
                    idx_to_name[int(idx)] = names.iloc[0]
    for i, idx in enumerate(type_idx):
        if idx >= 0:
            type_name[i] = idx_to_name.get(int(idx), f"ct_{int(idx)}")
    logger.info("Mapped cell types for %d/%d cells", int(np.sum(type_idx >= 0)), len(cell_ids))
    return type_idx, type_name


# ---------------------------------------------------------------------------
# Tier 1A: baseline and shape metrics
# ---------------------------------------------------------------------------


def compute_fast_shape_metrics(labels: np.ndarray, target_pix_um: float, logger: logging.Logger) -> pd.DataFrame:
    logger.info("Computing raster morphology with skimage.regionprops_table")
    requested = (
        "label",
        "area",
        "centroid",
        "perimeter",
        "area_convex",
        "axis_major_length",
        "axis_minor_length",
        "eccentricity",
        "solidity",
    )
    try:
        table = measure.regionprops_table(labels, properties=requested)
    except (KeyError, AttributeError):
        # Compatibility with older scikit-image property names.
        table = measure.regionprops_table(
            labels,
            properties=(
                "label",
                "area",
                "centroid",
                "perimeter",
                "convex_area",
                "major_axis_length",
                "minor_axis_length",
                "eccentricity",
                "solidity",
            ),
        )
    df = pd.DataFrame(table)
    rename = {
        "label": "cell_id",
        "centroid-0": "centroid_y_px",
        "centroid-1": "centroid_x_px",
        "area_convex": "convex_area_px",
        "convex_area": "convex_area_px",
        "axis_major_length": "major_axis_px",
        "major_axis_length": "major_axis_px",
        "axis_minor_length": "minor_axis_px",
        "minor_axis_length": "minor_axis_px",
        "eccentricity": "eccentricity_skimage",
        "area": "area_px",
        "perimeter": "perimeter_px",
    }
    df = df.rename(columns=rename)
    df["cell_id"] = df["cell_id"].astype(np.int64)
    df["area_um2"] = df["area_px"] * target_pix_um**2
    df["centroid_x_um"] = df["centroid_x_px"] * target_pix_um
    df["centroid_y_um"] = df["centroid_y_px"] * target_pix_um

    major = df["major_axis_px"].to_numpy(float)
    minor = df["minor_axis_px"].to_numpy(float)
    perimeter = df["perimeter_px"].to_numpy(float)
    area = df["area_px"].to_numpy(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        df["elongation_major_minor"] = np.where(minor > 0, major / minor, np.nan)
        df["axis_ratio_minor_major"] = np.where(major > 0, minor / major, np.nan)
        df["compactness_raster"] = np.where(
            perimeter > 0, 4.0 * np.pi * area / (perimeter**2), np.nan
        )
        # CellSPA circularity uses convex-hull perimeter. This raster equivalent
        # uses the object perimeter and is intentionally named accordingly.
        df["circularity_raster"] = df["compactness_raster"]
        df["convexity_area_proxy"] = np.where(
            df["convex_area_px"] > 0,
            df["area_px"] / df["convex_area_px"],
            np.nan,
        )
    return df


def attach_expression_metrics(
    shape_df: pd.DataFrame,
    expr_cell_ids: np.ndarray,
    genes: list[str],
    expr: np.ndarray,
    type_idx: np.ndarray,
    type_name: np.ndarray,
    logger: logging.Logger,
) -> pd.DataFrame:
    total_transcripts = np.sum(expr, axis=1, dtype=np.float64)
    total_genes = np.count_nonzero(expr > 0, axis=1)
    expr_meta = pd.DataFrame(
        {
            "cell_id": expr_cell_ids,
            "total_transcripts": total_transcripts,
            "total_genes": total_genes,
            "cell_type_idx": type_idx,
            "cell_type": type_name,
            "expr_row": np.arange(len(expr_cell_ids), dtype=np.int64),
        }
    )
    cells = shape_df.merge(expr_meta, on="cell_id", how="outer", validate="one_to_one")
    cells["transcript_density_per_um2"] = cells["total_transcripts"] / cells["area_um2"]
    logger.info("Cell table rows after mask/expression merge: %d", len(cells))
    return cells


def gene_detection_fraction(genes: list[str], expr: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "gene": genes,
            "cells_expressing": np.count_nonzero(expr > 0, axis=0),
            "fraction_cells_expressing": np.mean(expr > 0, axis=0),
            "total_transcripts": np.sum(expr, axis=0, dtype=np.float64),
        }
    )


def nucleus_to_cell_information_gain(
    cell_ids: np.ndarray,
    genes: list[str],
    expr: np.ndarray,
    nuclei_expr_path: Path | None,
    logger: logging.Logger,
) -> pd.DataFrame | None:
    if nuclei_expr_path is None or not nuclei_expr_path.exists():
        return None
    n_ids, n_genes, n_expr = read_expression_matrix(nuclei_expr_path, logger)
    common = [g for g in genes if g in set(n_genes)]
    if not common:
        logger.warning("No genes shared between cell and nucleus matrices")
        return None
    cell_gene_idx = np.array([genes.index(g) for g in common], dtype=np.int64)
    nuc_index = {g: i for i, g in enumerate(n_genes)}
    nuc_gene_idx = np.array([nuc_index[g] for g in common], dtype=np.int64)

    cell_summary = pd.DataFrame(
        {
            "cell_id": cell_ids,
            "cell_transcripts": np.sum(expr[:, cell_gene_idx], axis=1),
            "cell_genes": np.count_nonzero(expr[:, cell_gene_idx] > 0, axis=1),
        }
    )
    nuc_summary = pd.DataFrame(
        {
            "cell_id": n_ids,
            "nucleus_transcripts": np.sum(n_expr[:, nuc_gene_idx], axis=1),
            "nucleus_genes": np.count_nonzero(n_expr[:, nuc_gene_idx] > 0, axis=1),
        }
    )
    out = cell_summary.merge(nuc_summary, on="cell_id", how="inner")
    out["delta_transcripts_cell_minus_nucleus"] = (
        out["cell_transcripts"] - out["nucleus_transcripts"]
    )
    out["delta_genes_cell_minus_nucleus"] = out["cell_genes"] - out["nucleus_genes"]
    return out


# ---------------------------------------------------------------------------
# Tier 1B: CellSPA-compatible marker metrics
# ---------------------------------------------------------------------------


def read_marker_matrix(path: Path | None, genes: list[str], logger: logging.Logger) -> pd.DataFrame | None:
    if path is None or not path.exists():
        return None
    df = pd.read_csv(path, index_col=0, low_memory=False)
    common = [g for g in genes if g in df.columns]
    if not common:
        logger.warning("No marker genes overlap expression genes: %s", path)
        return None
    return df.loc[:, common].apply(pd.to_numeric, errors="coerce").fillna(0)


def cellspa_marker_metrics(
    expr: np.ndarray,
    genes: list[str],
    cell_ids: np.ndarray,
    cell_types: np.ndarray,
    marker_df: pd.DataFrame,
    prefix: str,
    logger: logging.Logger,
) -> pd.DataFrame:
    """Python equivalent of CellSPA .calculate_marker_purity and marker pct."""
    gene_to_idx = {g: i for i, g in enumerate(genes)}
    out = pd.DataFrame(
        {
            "cell_id": cell_ids,
            "cell_type": cell_types,
            f"{prefix}_marker_f1": np.zeros(len(cell_ids), dtype=np.float32),
            f"{prefix}_marker_precision": np.zeros(len(cell_ids), dtype=np.float32),
            f"{prefix}_marker_recall": np.zeros(len(cell_ids), dtype=np.float32),
            f"{prefix}_marker_expr_pct": np.zeros(len(cell_ids), dtype=np.float32),
            f"{prefix}_marker_count": np.zeros(len(cell_ids), dtype=np.int32),
        }
    )
    for ct in pd.unique(cell_types):
        if ct == "Unknown" or ct not in marker_df.index:
            continue
        marker_genes = [
            g
            for g in marker_df.columns
            if float(marker_df.loc[ct, g]) > 0 and g in gene_to_idx
        ]
        if not marker_genes:
            continue
        rows = np.where(cell_types == ct)[0]
        marker_idx = np.array([gene_to_idx[g] for g in marker_genes], dtype=np.int64)
        actual = np.zeros(len(genes), dtype=bool)
        actual[marker_idx] = True
        marker_fraction = float(np.mean(actual))
        # CellSPA: predicted = x > quantile(x, 1 - mean(gene_idx))
        x = expr[rows, :]
        q = np.quantile(x, 1.0 - marker_fraction, axis=1)
        predicted = x > q[:, None]
        tp = np.count_nonzero(predicted[:, marker_idx], axis=1).astype(np.float64)
        pred_n = np.count_nonzero(predicted, axis=1).astype(np.float64)
        actual_n = float(len(marker_idx))
        precision = np.divide(tp, pred_n, out=np.zeros_like(tp), where=pred_n > 0)
        recall = tp / actual_n
        denom = precision + recall
        f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(tp), where=denom > 0)
        expr_pct = np.mean(x[:, marker_idx] > 0, axis=1)
        out.loc[rows, f"{prefix}_marker_f1"] = f1.astype(np.float32)
        out.loc[rows, f"{prefix}_marker_precision"] = precision.astype(np.float32)
        out.loc[rows, f"{prefix}_marker_recall"] = recall.astype(np.float32)
        out.loc[rows, f"{prefix}_marker_expr_pct"] = expr_pct.astype(np.float32)
        out.loc[rows, f"{prefix}_marker_count"] = len(marker_idx)
    logger.info("Computed %s marker metrics", prefix)
    return out


def combine_marker_metrics(pos: pd.DataFrame, neg: pd.DataFrame) -> pd.DataFrame:
    out = pos.merge(neg.drop(columns=["cell_type"]), on="cell_id", how="outer")
    pos_f1 = out["positive_marker_f1"].to_numpy(float)
    neg_exclusion = 1.0 - out["negative_marker_f1"].to_numpy(float)
    denom = pos_f1 + neg_exclusion
    out["negative_marker_exclusion_f1"] = neg_exclusion
    out["purity_f1_proxy"] = np.divide(
        2 * pos_f1 * neg_exclusion,
        denom,
        out=np.zeros_like(pos_f1),
        where=denom > 0,
    )
    return out


# ---------------------------------------------------------------------------
# Tier 1B: reference expression concordance
# ---------------------------------------------------------------------------


def row_normalise_log1p(x: np.ndarray) -> np.ndarray:
    sums = np.sum(x, axis=1, keepdims=True, dtype=np.float64)
    norm = np.divide(x, sums, out=np.zeros_like(x, dtype=np.float32), where=sums > 0)
    return np.log1p(norm).astype(np.float32, copy=False)


def expression_concordance(
    expr: np.ndarray,
    genes: list[str],
    cell_types: np.ndarray,
    ref_path: Path,
    logger: logging.Logger,
) -> pd.DataFrame:
    ref = pd.read_csv(ref_path, index_col=0, low_memory=False)
    if "cell_type" not in ref.columns:
        raise ValueError("Reference CSV does not contain a cell_type column")
    metadata = {"ct_idx", "cell_type", "atlas"}
    common = [g for g in genes if g in ref.columns and g not in metadata]
    if len(common) < 3:
        raise ValueError("Fewer than three common genes with sc reference")
    expr_idx = np.array([genes.index(g) for g in common], dtype=np.int64)
    ref_x = ref[common].to_numpy(dtype=np.float32)
    ref_x = row_normalise_log1p(ref_x)
    ref_norm = pd.DataFrame(ref_x, columns=common)
    ref_norm["cell_type"] = ref["cell_type"].astype(str).to_numpy()
    ref_mean = ref_norm.groupby("cell_type", sort=False)[common].mean()

    rows: list[dict[str, Any]] = []
    for ct in sorted(set(map(str, cell_types))):
        if ct == "Unknown" or ct not in ref_mean.index:
            continue
        idx = np.where(cell_types.astype(str) == ct)[0]
        if idx.size == 0:
            continue
        # Chunk normalisation prevents creating another full 108k x 313 array.
        accum = np.zeros(len(common), dtype=np.float64)
        n = 0
        for start in range(0, idx.size, 5000):
            batch_rows = idx[start : start + 5000]
            norm = row_normalise_log1p(expr[batch_rows][:, expr_idx])
            accum += np.sum(norm, axis=0, dtype=np.float64)
            n += norm.shape[0]
        seg_mean = accum / max(n, 1)
        ref_profile = ref_mean.loc[ct, common].to_numpy(float)
        r, p, n_gene = pearson_safe(seg_mean, ref_profile)
        rows.append(
            {
                "cell_type": ct,
                "n_segmented_cells": int(idx.size),
                "n_common_genes": n_gene,
                "pearson_r": r,
                "pearson_p": p,
            }
        )
    logger.info("Expression concordance computed for %d cell types", len(rows))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Tier 1C: spatial diversity
# ---------------------------------------------------------------------------


def shannon_entropy(values: pd.Series) -> float:
    counts = values.value_counts(dropna=True).to_numpy(dtype=np.float64)
    if counts.size == 0 or counts.sum() == 0:
        return np.nan
    p = counts / counts.sum()
    return float(-np.sum(p * np.log(p)))


def spatial_diversity(
    cells: pd.DataFrame,
    bin_width_um: float,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = cells.dropna(subset=["centroid_x_um", "centroid_y_um"]).copy()
    df = df[df["cell_type"].fillna("Unknown") != "Unknown"]
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()
    df["x_bin"] = np.floor(df["centroid_x_um"] / bin_width_um).astype(int)
    df["y_bin"] = np.floor(df["centroid_y_um"] / bin_width_um).astype(int)
    df["bin"] = df["x_bin"].astype(str) + "|" + df["y_bin"].astype(str)

    metrics = [
        "total_transcripts",
        "total_genes",
        "area_um2",
        "transcript_density_per_um2",
        "elongation_major_minor",
        "compactness_raster",
        "eccentricity_skimage",
        "solidity",
    ]
    rows: list[dict[str, Any]] = []
    for bin_name, group in df.groupby("bin", sort=False):
        row: dict[str, Any] = {
            "bin": bin_name,
            "x_bin": int(group["x_bin"].iloc[0]),
            "y_bin": int(group["y_bin"].iloc[0]),
            "n_cells": int(len(group)),
            "cell_type_entropy": shannon_entropy(group["cell_type"]),
        }
        proportions = group["cell_type"].value_counts(normalize=True)
        for ct, value in proportions.items():
            row[f"cell_type_prop__{ct}"] = float(value)
        for metric in metrics:
            if metric in group.columns:
                row[f"cv__{metric}"] = coefficient_of_variation(group[metric])
        rows.append(row)
    bins = pd.DataFrame(rows)

    corr_rows: list[dict[str, Any]] = []
    for col in bins.columns:
        if not col.startswith("cv__"):
            continue
        r, p, n = pearson_safe(bins["cell_type_entropy"], bins[col])
        corr_rows.append(
            {
                "comparison": f"cell_type_entropy_vs_{col}",
                "pearson_r": r,
                "pearson_p": p,
                "n_bins": n,
            }
        )
    fibro_cols = [c for c in bins.columns if c.startswith("cell_type_prop__") and "fibro" in c.lower()]
    if fibro_cols and "cv__elongation_major_minor" in bins.columns:
        fibro_prop = bins[fibro_cols].fillna(0).sum(axis=1)
        r, p, n = pearson_safe(fibro_prop, bins["cv__elongation_major_minor"])
        corr_rows.append(
            {
                "comparison": "fibroblast_proportion_vs_elongation_cv",
                "pearson_r": r,
                "pearson_p": p,
                "n_bins": n,
            }
        )
    logger.info("Spatial diversity: %d bins", len(bins))
    return bins, pd.DataFrame(corr_rows)


# ---------------------------------------------------------------------------
# Tier 1D: neighbouring contamination
# ---------------------------------------------------------------------------


def marker_lists(marker_df: pd.DataFrame | None) -> dict[str, set[str]]:
    if marker_df is None:
        return {}
    out: dict[str, set[str]] = {}
    for ct in marker_df.index.astype(str):
        out[ct] = {g for g in marker_df.columns if float(marker_df.loc[ct, g]) > 0}
    return out


def neighbouring_contamination(
    cells: pd.DataFrame,
    expr: np.ndarray,
    genes: list[str],
    positive_markers: pd.DataFrame | None,
    negative_markers: pd.DataFrame,
    distance_breaks_um: Sequence[float],
    min_cells_per_type: int,
    logger: logging.Logger,
) -> pd.DataFrame:
    """
    CellSPA-style distance-stratified negative marker expression.

    For ordered pair source A -> neighbour B, the preferred marker set is:
        negative_markers[A] intersection positive_markers[B]
    If this intersection is empty, negative_markers[A] is used.
    """
    gene_to_idx = {g: i for i, g in enumerate(genes)}
    pos = marker_lists(positive_markers)
    neg = marker_lists(negative_markers)
    valid = cells.dropna(subset=["centroid_x_um", "centroid_y_um", "expr_row"]).copy()
    valid = valid[valid["cell_type"] != "Unknown"]
    counts = valid["cell_type"].value_counts()
    types = [ct for ct, n in counts.items() if n >= min_cells_per_type and ct in neg]
    rows: list[dict[str, Any]] = []
    breaks = np.asarray(distance_breaks_um, dtype=float)
    if not np.isinf(breaks[-1]):
        breaks = np.append(breaks, np.inf)

    type_groups = {ct: valid[valid["cell_type"] == ct] for ct in types}
    trees = {
        ct: cKDTree(group[["centroid_x_um", "centroid_y_um"]].to_numpy(float))
        for ct, group in type_groups.items()
    }

    for source in types:
        source_group = type_groups[source]
        source_coords = source_group[["centroid_x_um", "centroid_y_um"]].to_numpy(float)
        source_expr_rows = source_group["expr_row"].astype(int).to_numpy()
        for neighbour in types:
            if source == neighbour:
                continue
            marker_set = neg[source]
            if neighbour in pos:
                intersected = marker_set.intersection(pos[neighbour])
                if intersected:
                    marker_set = intersected
            marker_set = {g for g in marker_set if g in gene_to_idx}
            if not marker_set:
                continue
            marker_idx = np.array([gene_to_idx[g] for g in sorted(marker_set)], dtype=np.int64)
            distances, _ = trees[neighbour].query(source_coords, k=1)
            bin_idx = np.digitize(distances, breaks, right=False) - 1
            marker_positive = expr[source_expr_rows][:, marker_idx] > 0
            for b in range(len(breaks) - 1):
                keep = bin_idx == b
                if not np.any(keep):
                    continue
                per_marker = np.mean(marker_positive[keep], axis=0)
                rows.append(
                    {
                        "source_cell_type": source,
                        "neighbour_cell_type": neighbour,
                        "distance_low_um": float(breaks[b]),
                        "distance_high_um": float(breaks[b + 1]),
                        "n_source_cells": int(np.sum(keep)),
                        "n_negative_markers": int(len(marker_idx)),
                        "mean_negative_marker_detection_fraction": float(np.mean(per_marker)),
                        "median_negative_marker_detection_fraction": float(np.median(per_marker)),
                        "mean_distance_um": float(np.mean(distances[keep])),
                    }
                )
    logger.info("Neighbour contamination output rows: %d", len(rows))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Tier 2: boundary and transcript assignment
# ---------------------------------------------------------------------------


def boundary_masks(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return all instance edges and cell-cell edges using 4-neighbour changes."""
    all_b = np.zeros(labels.shape, dtype=bool)
    cc_b = np.zeros(labels.shape, dtype=bool)

    diff = labels[1:, :] != labels[:-1, :]
    all_b[1:, :] |= diff & (labels[1:, :] > 0)
    all_b[:-1, :] |= diff & (labels[:-1, :] > 0)
    both = diff & (labels[1:, :] > 0) & (labels[:-1, :] > 0)
    cc_b[1:, :] |= both
    cc_b[:-1, :] |= both

    diff = labels[:, 1:] != labels[:, :-1]
    all_b[:, 1:] |= diff & (labels[:, 1:] > 0)
    all_b[:, :-1] |= diff & (labels[:, :-1] > 0)
    both = diff & (labels[:, 1:] > 0) & (labels[:, :-1] > 0)
    cc_b[:, 1:] |= both
    cc_b[:, :-1] |= both
    return all_b, cc_b


def distance_to_boundary(boundary: np.ndarray) -> np.ndarray:
    if not np.any(boundary):
        return np.full(boundary.shape, np.inf, dtype=np.float32)
    inverse = (~boundary).astype(np.uint8)
    if cv2 is not None:
        return cv2.distanceTransform(inverse, cv2.DIST_L2, 3).astype(np.float32)
    return ndimage.distance_transform_edt(inverse).astype(np.float32)


def transcript_coordinate_settings(cfg: dict[str, Any]) -> dict[str, Any]:
    trans = cfg.get("transcripts", {})
    affine = cfg.get("affine", {})
    target = float(affine.get("target_pix_um", 1.0))
    base_ts_x = float(affine.get("base_ts_x", 1.0))
    base_ts_y = float(affine.get("base_ts_y", 1.0))
    return {
        "x_col": trans.get("x_col", "x_location"),
        "y_col": trans.get("y_col", "y_location"),
        "gene_col": trans.get("gene_col", "feature_name"),
        "scale_x_to_target": base_ts_x / target,
        "scale_y_to_target": base_ts_y / target,
        "shift_x": int(affine.get("global_shift_x", 0)),
        "shift_y": int(affine.get("global_shift_y", 0)),
        "target_pix_um": target,
    }


def scan_transcripts(
    transcript_path: Path,
    labels: np.ndarray,
    dist_all: np.ndarray,
    dist_cellcell: np.ndarray,
    widths_um: Sequence[float],
    cfg: dict[str, Any],
    chunksize: int,
    logger: logging.Logger,
) -> tuple[dict[str, Any], pd.DataFrame]:
    settings = transcript_coordinate_settings(cfg)
    x_col = settings["x_col"]
    y_col = settings["y_col"]
    gene_col = settings["gene_col"]
    usecols = [x_col, y_col]
    header = pd.read_csv(transcript_path, nrows=0).columns.tolist()
    if gene_col in header:
        usecols.append(gene_col)

    widths_px = np.asarray(widths_um, dtype=float) / settings["target_pix_um"]
    totals = defaultdict(int)
    gene_counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    boundary_counts = {
        float(w): {
            "all_band_total": 0,
            "all_band_assigned": 0,
            "cellcell_band_total": 0,
            "cellcell_band_assigned": 0,
        }
        for w in widths_um
    }

    for chunk_i, chunk in enumerate(
        pd.read_csv(transcript_path, usecols=usecols, chunksize=chunksize, low_memory=False)
    ):
        x = pd.to_numeric(chunk[x_col], errors="coerce").to_numpy(float)
        y = pd.to_numeric(chunk[y_col], errors="coerce").to_numpy(float)
        finite = np.isfinite(x) & np.isfinite(y)
        x_safe = np.where(finite, x, 0.0)
        y_safe = np.where(finite, y, 0.0)
        xi = np.rint(x_safe * settings["scale_x_to_target"] - settings["shift_x"]).astype(np.int64)
        yi = np.rint(y_safe * settings["scale_y_to_target"] - settings["shift_y"]).astype(np.int64)
        in_bounds = finite & (xi >= 0) & (xi < labels.shape[1]) & (yi >= 0) & (yi < labels.shape[0])
        totals["total_rows"] += len(chunk)
        totals["invalid_or_out_of_bounds"] += int(np.sum(~in_bounds))
        if not np.any(in_bounds):
            continue
        xi_b = xi[in_bounds]
        yi_b = yi[in_bounds]
        assigned = labels[yi_b, xi_b] > 0
        d_all = dist_all[yi_b, xi_b]
        d_cc = dist_cellcell[yi_b, xi_b]
        totals["in_bounds"] += int(len(xi_b))
        totals["assigned"] += int(np.sum(assigned))

        for width_um, width_px in zip(widths_um, widths_px):
            a = d_all <= width_px
            c = d_cc <= width_px
            rec = boundary_counts[float(width_um)]
            rec["all_band_total"] += int(np.sum(a))
            rec["all_band_assigned"] += int(np.sum(a & assigned))
            rec["cellcell_band_total"] += int(np.sum(c))
            rec["cellcell_band_assigned"] += int(np.sum(c & assigned))

        if gene_col in chunk.columns:
            genes_chunk = chunk.loc[in_bounds, gene_col].astype(str).to_numpy()
            # Pandas groupby avoids a Python loop over every transcript.
            tmp = pd.DataFrame({"gene": genes_chunk, "assigned": assigned.astype(np.int8)})
            grouped = tmp.groupby("gene", sort=False)["assigned"].agg(["count", "sum"])
            for gene, row in grouped.iterrows():
                gene_counts[str(gene)][0] += int(row["count"])
                gene_counts[str(gene)][1] += int(row["sum"])

        if chunk_i % 10 == 0:
            logger.info("Transcript chunks processed: %d", chunk_i + 1)

    total = totals["total_rows"]
    in_bounds = totals["in_bounds"]
    assigned = totals["assigned"]
    summary: dict[str, Any] = {
        **dict(totals),
        "in_bounds_fraction": in_bounds / total if total else np.nan,
        "assigned_fraction_all_rows": assigned / total if total else np.nan,
        "assigned_fraction_in_bounds": assigned / in_bounds if in_bounds else np.nan,
        "coordinate_settings": settings,
        "boundary_widths": boundary_counts,
    }
    gene_df = pd.DataFrame(
        [
            {
                "gene": gene,
                "in_bounds_transcripts": values[0],
                "assigned_transcripts": values[1],
                "assigned_fraction": values[1] / values[0] if values[0] else np.nan,
            }
            for gene, values in gene_counts.items()
        ]
    )
    return summary, gene_df


# ---------------------------------------------------------------------------
# Tier 2: nucleus/cell structure
# ---------------------------------------------------------------------------


def nucleus_cell_structure(
    cells: np.ndarray,
    nuclei: np.ndarray,
    logger: logging.Logger,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    if cells.shape != nuclei.shape:
        raise ValueError(f"Cell and nucleus masks differ in shape: {cells.shape} vs {nuclei.shape}")
    nuc_fg = nuclei > 0
    nuc_ids = np.unique(nuclei[nuc_fg]).astype(np.int64)
    cell_ids = np.unique(cells[cells > 0]).astype(np.int64)
    max_nuc = int(nuc_ids.max(initial=0))
    max_cell = int(cell_ids.max(initial=0))

    nuc_counts = np.bincount(nuclei[nuc_fg].astype(np.int64), minlength=max_nuc + 1)
    same = nuc_fg & (cells == nuclei)
    same_counts = np.bincount(nuclei[same].astype(np.int64), minlength=max_nuc + 1)

    both = nuc_fg & (cells > 0)
    nvals = nuclei[both].astype(np.int64)
    cvals = cells[both].astype(np.int64)
    base = max_cell + 1
    pair_codes, pair_pixel_counts = np.unique(nvals * base + cvals, return_counts=True)
    pair_nuc = pair_codes // base
    pair_cell = pair_codes % base
    pair_df = pd.DataFrame(
        {
            "nucleus_id": pair_nuc.astype(np.int64),
            "cell_id": pair_cell.astype(np.int64),
            "overlap_pixels": pair_pixel_counts.astype(np.int64),
        }
    )

    nuc_distinct_cells = pair_df.groupby("nucleus_id")["cell_id"].nunique()
    cell_distinct_nuclei = pair_df.groupby("cell_id")["nucleus_id"].nunique()
    dominant = pair_df.loc[pair_df.groupby("nucleus_id")["overlap_pixels"].idxmax()].copy()
    dominant = dominant.rename(
        columns={"cell_id": "dominant_cell_id", "overlap_pixels": "dominant_overlap_pixels"}
    )

    nuc_table = pd.DataFrame({"nucleus_id": nuc_ids})
    nuc_table["nucleus_pixels"] = nuc_counts[nuc_ids]
    nuc_table["same_id_cell_pixels"] = same_counts[nuc_ids]
    nuc_table["same_id_containment_fraction"] = np.divide(
        nuc_table["same_id_cell_pixels"], nuc_table["nucleus_pixels"]
    )
    nuc_table = nuc_table.merge(dominant, on="nucleus_id", how="left")
    nuc_table["dominant_overlap_fraction"] = (
        nuc_table["dominant_overlap_pixels"].fillna(0) / nuc_table["nucleus_pixels"]
    )
    nuc_table["n_overlapping_cells"] = (
        nuc_table["nucleus_id"].map(nuc_distinct_cells).fillna(0).astype(int)
    )
    nuc_table["missed"] = nuc_table["n_overlapping_cells"] == 0
    nuc_table["split"] = nuc_table["n_overlapping_cells"] > 1
    nuc_table["dominant_cell_preserves_id"] = (
        nuc_table["dominant_cell_id"].fillna(-1).astype(np.int64) == nuc_table["nucleus_id"]
    )

    cell_table = pd.DataFrame({"cell_id": cell_ids})
    cell_table["n_overlapping_nuclei"] = (
        cell_table["cell_id"].map(cell_distinct_nuclei).fillna(0).astype(int)
    )
    cell_table["zero_nucleus"] = cell_table["n_overlapping_nuclei"] == 0
    cell_table["multi_nucleus"] = cell_table["n_overlapping_nuclei"] > 1

    summary = {
        "n_nuclei": int(len(nuc_ids)),
        "n_cells": int(len(cell_ids)),
        "exact_same_id_nucleus_containment_fraction": float(
            np.mean(nuc_table["same_id_containment_fraction"] == 1.0)
        ),
        "well_contained_same_id_fraction_ge_0_9": float(
            np.mean(nuc_table["same_id_containment_fraction"] >= 0.9)
        ),
        "dominant_overlap_fraction_ge_0_9": float(
            np.mean(nuc_table["dominant_overlap_fraction"] >= 0.9)
        ),
        "dominant_cell_id_preserved_fraction": float(
            np.mean(nuc_table["dominant_cell_preserves_id"])
        ),
        "missed_nucleus_fraction": float(np.mean(nuc_table["missed"])),
        "split_nucleus_fraction": float(np.mean(nuc_table["split"])),
        "zero_nucleus_cell_fraction": float(np.mean(cell_table["zero_nucleus"])),
        "multi_nucleus_cell_fraction": float(np.mean(cell_table["multi_nucleus"])),
    }
    logger.info("Computed nucleus/cell structural metrics")
    return summary, nuc_table, cell_table


def disconnected_cells(labels: np.ndarray, logger: logging.Logger) -> tuple[dict[str, Any], pd.DataFrame]:
    logger.info("Computing 8-neighbour connected components for each label")
    component_image = measure.label(labels, background=0, connectivity=2)
    fg = labels > 0
    original = labels[fg].astype(np.int64)
    component = component_image[fg].astype(np.int64)
    base = int(component.max(initial=0)) + 1
    pair_codes = np.unique(original * base + component)
    pair_label = pair_codes // base
    counts = np.bincount(pair_label)
    cell_ids = np.unique(original)
    n_components = counts[cell_ids]
    out = pd.DataFrame({"cell_id": cell_ids, "n_components_8": n_components})
    summary = {
        "n_cells_checked": int(len(out)),
        "disconnected_cell_count_8": int(np.sum(out["n_components_8"] > 1)),
        "disconnected_cell_fraction_8": float(np.mean(out["n_components_8"] > 1)),
        "max_components_8": int(out["n_components_8"].max()),
    }
    del component_image, original, component, pair_codes
    return summary, out


# ---------------------------------------------------------------------------
# Tier 3: runtime/resources recoverable after the run
# ---------------------------------------------------------------------------


def flatten_json(value: Any, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    if isinstance(value, dict):
        for key, val in value.items():
            new_prefix = f"{prefix}.{key}" if prefix else str(key)
            out.update(flatten_json(val, new_prefix))
    elif isinstance(value, list):
        out[prefix] = json.dumps(value, ensure_ascii=False)
    else:
        out[prefix] = value
    return out


def parse_status_jsons(status_dir: Path | None) -> pd.DataFrame:
    if status_dir is None:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for fp in sorted(status_dir.rglob("*.json")):
        try:
            data = json.loads(fp.read_text(encoding="utf-8"))
            row = {"file": str(fp), **flatten_json(data)}
        except Exception as exc:
            row = {"file": str(fp), "parse_error": str(exc)}
        rows.append(row)
    return pd.DataFrame(rows)


TIMESTAMP_RE = re.compile(r"(20\d{2}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?)")


def parse_log_runtime(log_path: Path | None) -> tuple[dict[str, Any], pd.DataFrame]:
    if log_path is None or not log_path.exists():
        return {}, pd.DataFrame()
    events: list[dict[str, Any]] = []
    first_ts: datetime | None = None
    last_ts: datetime | None = None
    keywords = (
        "segment",
        "expression",
        "patch",
        "matrix",
        "annot",
        "training",
        "prediction",
        "postprocess",
        "finished",
        "complete",
        "model saved",
    )
    with log_path.open("r", encoding="utf-8", errors="replace") as f:
        for line_no, line in enumerate(f, start=1):
            m = TIMESTAMP_RE.search(line)
            if not m:
                continue
            ts_text = m.group(1).replace(",", ".")
            try:
                ts = datetime.fromisoformat(ts_text)
            except ValueError:
                continue
            if first_ts is None:
                first_ts = ts
            last_ts = ts
            if any(k in line.lower() for k in keywords):
                events.append(
                    {
                        "line": line_no,
                        "timestamp": ts.isoformat(),
                        "message": line.strip()[:1000],
                    }
                )
    summary: dict[str, Any] = {}
    if first_ts is not None and last_ts is not None:
        duration = (last_ts - first_ts).total_seconds()
        summary = {
            "first_timestamp": first_ts.isoformat(),
            "last_timestamp": last_ts.isoformat(),
            "log_span_seconds": duration,
            "log_span_minutes": duration / 60,
            "log_span_hours": duration / 3600,
        }
    return summary, pd.DataFrame(events)


def checkpoint_parameter_count(checkpoint: Path | None, logger: logging.Logger) -> dict[str, Any]:
    result: dict[str, Any] = {}
    if checkpoint is None or not checkpoint.exists():
        return result
    result["checkpoint_bytes"] = checkpoint.stat().st_size
    if torch is None:
        result["parameter_count_status"] = "torch_not_installed"
        return result
    try:
        obj = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if isinstance(obj, dict):
            candidates = [
                obj.get("state_dict"),
                obj.get("model_state_dict"),
                obj,
            ]
        else:
            candidates = [obj]
        state = next(
            (
                c
                for c in candidates
                if isinstance(c, dict) and any(torch.is_tensor(v) for v in c.values())
            ),
            None,
        )
        if state is None:
            result["parameter_count_status"] = "state_dict_not_found"
            return result
        tensors = {k: v for k, v in state.items() if torch.is_tensor(v)}
        result["state_tensor_count"] = len(tensors)
        result["state_tensor_numel"] = int(sum(v.numel() for v in tensors.values()))
        result["state_tensor_bytes"] = int(
            sum(v.numel() * v.element_size() for v in tensors.values())
        )
        result["parameter_count_status"] = "counted_checkpoint_tensors"
    except Exception as exc:
        logger.warning("Could not load checkpoint for parameter count: %s", exc)
        result["parameter_count_status"] = f"failed: {exc}"
    return result


# ---------------------------------------------------------------------------
# Tier 4: optional aligned pseudo-reference
# ---------------------------------------------------------------------------


def binary_and_boundary_reference_metrics(
    pred: np.ndarray,
    ref: np.ndarray,
    tolerances_px: Sequence[float],
) -> tuple[dict[str, Any], pd.DataFrame]:
    if pred.shape != ref.shape:
        raise ValueError(f"Vendor mask shape {ref.shape} != prediction shape {pred.shape}")
    p = pred > 0
    r = ref > 0
    inter = int(np.sum(p & r))
    p_sum = int(np.sum(p))
    r_sum = int(np.sum(r))
    union = int(np.sum(p | r))
    summary = {
        "foreground_dice": 2 * inter / (p_sum + r_sum) if (p_sum + r_sum) else np.nan,
        "foreground_iou": inter / union if union else np.nan,
        "foreground_precision": inter / p_sum if p_sum else np.nan,
        "foreground_recall": inter / r_sum if r_sum else np.nan,
    }
    pb, _ = boundary_masks(pred)
    rb, _ = boundary_masks(ref)
    d_to_r = distance_to_boundary(rb)
    d_to_p = distance_to_boundary(pb)
    rows: list[dict[str, Any]] = []
    p_dist = d_to_r[pb]
    r_dist = d_to_p[rb]
    if p_dist.size and r_dist.size:
        all_surface = np.concatenate([p_dist, r_dist])
        summary["assd_px"] = float((np.mean(p_dist) + np.mean(r_dist)) / 2)
        summary["hd95_px"] = float(np.percentile(all_surface, 95))
        summary["hausdorff_px"] = float(np.max(all_surface))
    for tol in tolerances_px:
        precision = float(np.mean(p_dist <= tol)) if p_dist.size else np.nan
        recall = float(np.mean(r_dist <= tol)) if r_dist.size else np.nan
        f1 = (
            2 * precision * recall / (precision + recall)
            if np.isfinite(precision + recall) and precision + recall > 0
            else np.nan
        )
        rows.append(
            {
                "tolerance_px": float(tol),
                "boundary_precision": precision,
                "boundary_recall": recall,
                "boundary_f1": f1,
            }
        )
    return summary, pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Tier 5: bootstrap
# ---------------------------------------------------------------------------


def bootstrap_ci(
    values: np.ndarray,
    metric_name: str,
    iterations: int,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return []
    means = np.empty(iterations, dtype=np.float64)
    medians = np.empty(iterations, dtype=np.float64)
    for i in range(iterations):
        sample = x[rng.integers(0, x.size, size=x.size)]
        means[i] = np.mean(sample)
        medians[i] = np.median(sample)
    rows = []
    for statistic, observed, samples in (
        ("mean", np.mean(x), means),
        ("median", np.median(x), medians),
    ):
        rows.append(
            {
                "metric": metric_name,
                "statistic": statistic,
                "observed": float(observed),
                "bootstrap_iterations": iterations,
                "ci_2_5": float(np.percentile(samples, 2.5)),
                "ci_97_5": float(np.percentile(samples, 97.5)),
                "n": int(x.size),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def write_summary_markdown(
    path: Path,
    manifest: dict[str, Any],
    tier1_summary: dict[str, Any],
    tier2_summary: dict[str, Any],
    tier3_summary: dict[str, Any],
    statuses: StatusRegistry,
) -> None:
    completed = sum(s.status == "COMPLETED" for s in statuses.rows)
    skipped = sum(s.status == "SKIPPED" for s in statuses.rows)
    failed = sum(s.status == "FAILED" for s in statuses.rows)
    lines = [
        "# BIDCell step-4000 baseline benchmark",
        "",
        f"- Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"- Run directory: `{manifest['paths']['run_dir']}`",
        f"- Connected mask: `{manifest['paths']['connected_mask']}`",
        f"- Completed metric groups: **{completed}**",
        f"- Skipped metric groups: **{skipped}**",
        f"- Failed metric groups: **{failed}**",
        "",
        "## Tier 1 headline metrics",
        "",
    ]
    for key, value in tier1_summary.items():
        if isinstance(value, (dict, list)):
            continue
        lines.append(f"- **{key}:** {value}")
    lines += ["", "## Tier 2 structural and boundary headline metrics", ""]
    for key, value in tier2_summary.items():
        if isinstance(value, (dict, list)):
            continue
        lines.append(f"- **{key}:** {value}")
    lines += ["", "## Tier 3 recoverable efficiency metrics", ""]
    for key, value in tier3_summary.items():
        if isinstance(value, (dict, list)):
            continue
        lines.append(f"- **{key}:** {value}")
    lines += [
        "",
        "## Interpretation limits",
        "",
        "- No manual cell-boundary ground truth is available.",
        "- Vendor comparison is only produced when an aligned pseudo-reference TIFF is supplied.",
        "- Candidate overlap and low-margin ambiguity require candidate-level intermediate outputs.",
        "- Historical peak VRAM/RAM/GPU utilisation cannot be reconstructed from final files.",
        "- See `metric_status.csv` for every skipped metric and its required information.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main benchmark workflow
# ---------------------------------------------------------------------------


def run(args: argparse.Namespace) -> int:
    paths, cfg = discover_paths(args)
    logger = configure_logging(paths.output_dir)
    registry = StatusRegistry()
    start = time.time()

    logger.info("Starting BIDCell step-4000 baseline benchmark")
    logger.info("Output directory: %s", paths.output_dir)

    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": package_versions(),
        "paths": {k: str(v) if v is not None else None for k, v in asdict(paths).items()},
        "config": cfg,
    }
    for key_path in [
        paths.config,
        paths.connected_mask,
        paths.nuclei_mask,
        paths.transcripts,
        paths.final_expr,
        paths.checkpoint,
    ]:
        if key_path is not None and key_path.exists():
            manifest.setdefault("files", {})[str(key_path)] = {
                "bytes": key_path.stat().st_size,
                "sha256": file_sha256(key_path) if key_path.stat().st_size < 500_000_000 else "skipped_large_file",
            }
    safe_json_dump(manifest, paths.output_dir / "benchmark_manifest.json")

    target_pix_um = float(cfg.get("affine", {}).get("target_pix_um", 1.0))
    cells_mask = load_tiff(paths.connected_mask, logger)
    nuclei_mask = load_tiff(paths.nuclei_mask, logger)
    if cells_mask.shape != nuclei_mask.shape:
        raise ValueError("Connected mask and nuclei mask are not aligned")

    # ---------------- Tier 1A ----------------
    shape_df = compute_fast_shape_metrics(cells_mask, target_pix_um, logger)
    shape_path = paths.output_dir / "tier1_cell_shape_metrics.csv.gz"
    shape_df.to_csv(shape_path, index=False, compression="gzip")
    registry.completed(
        "Tier 1",
        "Cell area and raster morphology",
        shape_path,
        "Fast raster equivalents; exact alpha-hull CellSPA geometry is not reproduced.",
    )
    registry.skipped(
        "Tier 1",
        "Exact CellSPA alpha-hull convexity and sphericity",
        "Requires running the official CellSPA R polygon/alpha-hull implementation or a separately validated Python port.",
    )

    expr_ids, genes, expr = read_expression_matrix(paths.final_expr, logger)
    type_idx, type_name = load_cell_types(paths.nuclei_types, paths.sc_reference, expr_ids, logger)
    cells = attach_expression_metrics(shape_df, expr_ids, genes, expr, type_idx, type_name, logger)

    gene_df = gene_detection_fraction(genes, expr)
    gene_path = paths.output_dir / "tier1_gene_detection_fraction.csv"
    gene_df.to_csv(gene_path, index=False)
    registry.completed("Tier 1", "Fraction of cells expressing each gene", gene_path)

    info_gain = nucleus_to_cell_information_gain(expr_ids, genes, expr, paths.nuclei_expr, logger)
    if info_gain is not None:
        info_gain_path = paths.output_dir / "tier1_nucleus_to_cell_information_gain.csv.gz"
        info_gain.to_csv(info_gain_path, index=False, compression="gzip")
        cells = cells.merge(
            info_gain[
                [
                    "cell_id",
                    "nucleus_transcripts",
                    "nucleus_genes",
                    "delta_transcripts_cell_minus_nucleus",
                    "delta_genes_cell_minus_nucleus",
                ]
            ],
            on="cell_id",
            how="left",
        )
        registry.completed("Tier 1", "Nucleus-to-cell information gain", info_gain_path)
    else:
        registry.skipped(
            "Tier 1",
            "Nucleus-to-cell information gain",
            "Missing nucleus-level expr_mat.csv or no common genes.",
        )

    cell_table_path = paths.output_dir / "tier1_cell_metrics.csv.gz"
    cells.to_csv(cell_table_path, index=False, compression="gzip")
    registry.completed("Tier 1", "Cell-level baseline characteristics", cell_table_path)

    # ---------------- Tier 1B marker metrics ----------------
    pos_df = read_marker_matrix(paths.positive_markers, genes, logger)
    neg_df = read_marker_matrix(paths.negative_markers, genes, logger)
    marker_result: pd.DataFrame | None = None
    if pos_df is not None and neg_df is not None and np.any(type_idx >= 0):
        pos_metric = cellspa_marker_metrics(
            expr, genes, expr_ids, type_name, pos_df, "positive", logger
        )
        neg_metric = cellspa_marker_metrics(
            expr, genes, expr_ids, type_name, neg_df, "negative", logger
        )
        marker_result = combine_marker_metrics(pos_metric, neg_metric)
        marker_path = paths.output_dir / "tier1_marker_purity_per_cell.csv.gz"
        marker_result.to_csv(marker_path, index=False, compression="gzip")
        summary_cols = [c for c in marker_result.columns if c not in {"cell_id", "cell_type"}]
        marker_summary = (
            marker_result.groupby("cell_type", dropna=False)[summary_cols]
            .agg(["mean", "median", "std", "count"])
        )
        marker_summary.columns = ["__".join(map(str, c)) for c in marker_summary.columns]
        marker_summary = marker_summary.reset_index()
        marker_summary_path = paths.output_dir / "tier1_marker_purity_summary.csv"
        marker_summary.to_csv(marker_summary_path, index=False)
        cells = cells.merge(marker_result, on=["cell_id", "cell_type"], how="left")
        registry.completed(
            "Tier 1",
            "Positive/negative marker F1, precision, recall and expression percentage",
            marker_path,
            "Marker F1 follows CellSPA R quantile logic. purity_f1_proxy is explicitly non-official.",
        )
    else:
        registry.skipped(
            "Tier 1",
            "Marker purity",
            "Positive/negative marker matrices or valid cell-type mapping were not found.",
        )

    if paths.sc_reference is not None and paths.sc_reference.exists() and np.any(type_idx >= 0):
        try:
            corr = expression_concordance(expr, genes, type_name, paths.sc_reference, logger)
            corr_path = paths.output_dir / "tier1_expression_reference_correlation.csv"
            corr.to_csv(corr_path, index=False)
            registry.completed("Tier 1", "Cell-type average expression concordance", corr_path)
        except Exception as exc:
            registry.failed("Tier 1", "Cell-type average expression concordance", str(exc))
    else:
        registry.skipped(
            "Tier 1",
            "Cell-type average expression concordance",
            "Missing BIDCell sc reference or valid cell-type mapping.",
        )

    # TODO-TIER1: averaged sc_breast.csv does not contain per-cell detection rates.
    registry.skipped(
        "Tier 1",
        "Non-zero expression proportion concordance to scRNA-seq",
        "Needs raw or cell-level scRNA-seq reference with per-cell zero/non-zero observations; the BIDCell averaged reference is insufficient.",
    )
    registry.skipped(
        "Tier 1",
        "Cell-type proportion concordance to matched Chromium sample",
        "Needs a matched-sample cell-type composition reference; an unrelated multi-atlas reference is not a valid composition ground truth.",
    )

    # ---------------- Tier 1C spatial diversity ----------------
    if np.any(type_idx >= 0):
        bins, spatial_corr = spatial_diversity(cells, args.spatial_bin_um, logger)
        bins_path = paths.output_dir / "tier1_spatial_bins.csv"
        corr_path = paths.output_dir / "tier1_spatial_correlations.csv"
        bins.to_csv(bins_path, index=False)
        spatial_corr.to_csv(corr_path, index=False)
        registry.completed("Tier 1", "Spatial entropy and morphology/expression diversity", bins_path)
    else:
        registry.skipped(
            "Tier 1",
            "Spatial entropy and diversity",
            "Valid cell-type labels are required.",
        )

    # ---------------- Tier 1D neighbour contamination ----------------
    if neg_df is not None and np.any(type_idx >= 0):
        try:
            contamination = neighbouring_contamination(
                cells,
                expr,
                genes,
                pos_df,
                neg_df,
                args.distance_breaks_um,
                args.min_cells_per_type,
                logger,
            )
            contam_path = paths.output_dir / "tier1_neighbour_contamination.csv"
            contamination.to_csv(contam_path, index=False)
            registry.completed(
                "Tier 1",
                "Distance-stratified neighbouring negative-marker contamination",
                contam_path,
            )
        except Exception as exc:
            registry.failed("Tier 1", "Neighbour contamination", str(exc))
    else:
        registry.skipped(
            "Tier 1",
            "Neighbour contamination",
            "Negative marker matrix and valid cell types are required.",
        )

    registry.skipped(
        "Tier 1",
        "Rep1/Rep2 replicability",
        "Needs a completed Rep2 segmentation, cell-gene matrix and matched metric table.",
    )

    # ---------------- Tier 2 boundaries and transcripts ----------------
    logger.info("Computing final-mask boundaries and distance maps")
    all_boundary, cellcell_boundary = boundary_masks(cells_mask)
    dist_all = distance_to_boundary(all_boundary)
    dist_cellcell = distance_to_boundary(cellcell_boundary)
    transcript_summary, transcript_gene = scan_transcripts(
        paths.transcripts,
        cells_mask,
        dist_all,
        dist_cellcell,
        args.boundary_widths_um,
        cfg,
        args.transcript_chunksize,
        logger,
    )
    transcript_summary_path = paths.output_dir / "tier2_transcript_assignment_summary.json"
    safe_json_dump(transcript_summary, transcript_summary_path)
    transcript_gene_path = paths.output_dir / "tier2_transcript_assignment_by_gene.csv"
    transcript_gene.to_csv(transcript_gene_path, index=False)
    registry.completed(
        "Tier 2",
        "Assigned transcript and boundary-zone transcript metrics",
        transcript_summary_path,
    )

    boundary_area_rows: list[dict[str, Any]] = []
    foreground = cells_mask > 0
    for width_um in args.boundary_widths_um:
        width_px = width_um / target_pix_um
        for name, dist in (("all_instance_edges", dist_all), ("cell_cell_edges", dist_cellcell)):
            band = dist <= width_px
            boundary_area_rows.append(
                {
                    "boundary_type": name,
                    "width_um": width_um,
                    "band_pixels": int(np.sum(band)),
                    "band_fraction_of_image": float(np.mean(band)),
                    "band_fraction_of_foreground": float(np.sum(band & foreground) / np.sum(foreground)),
                }
            )
    boundary_area_path = paths.output_dir / "tier2_boundary_area.csv"
    pd.DataFrame(boundary_area_rows).to_csv(boundary_area_path, index=False)
    registry.completed("Tier 2", "Boundary band area", boundary_area_path)

    structure_summary, nucleus_table, cell_nucleus_table = nucleus_cell_structure(
        cells_mask, nuclei_mask, logger
    )
    nucleus_path = paths.output_dir / "tier2_nucleus_structure.csv.gz"
    cell_nucleus_path = paths.output_dir / "tier2_cell_nucleus_structure.csv.gz"
    nucleus_table.to_csv(nucleus_path, index=False, compression="gzip")
    cell_nucleus_table.to_csv(cell_nucleus_path, index=False, compression="gzip")
    registry.completed("Tier 2", "Nucleus containment and merge/split proxies", nucleus_path)

    if not args.skip_connectivity:
        try:
            connectivity_summary, connectivity_table = disconnected_cells(cells_mask, logger)
            structure_summary.update(connectivity_summary)
            connectivity_path = paths.output_dir / "tier2_cell_connectivity.csv.gz"
            connectivity_table.to_csv(connectivity_path, index=False, compression="gzip")
            registry.completed("Tier 2", "8-neighbour disconnected cell fraction", connectivity_path)
        except Exception as exc:
            registry.failed("Tier 2", "8-neighbour connectivity", str(exc))
    else:
        registry.skipped(
            "Tier 2",
            "8-neighbour disconnected cell fraction",
            "Explicitly disabled with --skip-connectivity.",
        )

    area_values = cells["area_px"].dropna().to_numpy(float)
    p_low, p_high = np.percentile(area_values, [0.5, 99.5])
    structure_summary.update(
        {
            "foreground_fraction": float(np.mean(foreground)),
            "all_boundary_pixel_fraction": float(np.mean(all_boundary)),
            "cellcell_boundary_pixel_fraction": float(np.mean(cellcell_boundary)),
            "area_p0_5_px": float(p_low),
            "area_p99_5_px": float(p_high),
            "extreme_area_outlier_fraction": float(np.mean((area_values < p_low) | (area_values > p_high))),
        }
    )
    structure_summary_path = paths.output_dir / "tier2_structure_summary.json"
    safe_json_dump(structure_summary, structure_summary_path)

    # TODO-TIER2: these require intermediates not contained in final mask files.
    registry.skipped(
        "Tier 2",
        "Candidate-overlap area ratio",
        "Needs per-candidate search_areas from dataset_input.py saved per patch, or an exact patch-wise reconstruction preserving elongated-cell kernels and clipping.",
    )
    registry.skipped(
        "Tier 2",
        "Low top-1/top-2 margin ambiguous area",
        "Needs candidate-level foreground probabilities/logits from a replay of step-4000 inference with extra outputs.",
    )
    registry.skipped(
        "Tier 2",
        "Assignment-change, neighbour-only change and marker-consistent reassignment",
        "Needs a second method output (V1/V2/V3) and a common per-transcript assignment table.",
    )
    registry.skipped(
        "Tier 2",
        "Assignment entropy and segmentation-assignment consistency",
        "Needs V2/V3 soft candidate assignment scores; original BIDCell only provides hard point-in-mask assignment.",
    )

    # ---------------- Tier 3 runtime/resources ----------------
    status_df = parse_status_jsons(paths.status_dir)
    status_path = paths.output_dir / "tier3_status_json_inventory.csv"
    status_df.to_csv(status_path, index=False)
    runtime_summary, runtime_events = parse_log_runtime(paths.pipeline_log)
    runtime_events_path = paths.output_dir / "tier3_runtime_events.csv"
    runtime_events.to_csv(runtime_events_path, index=False)
    if runtime_summary:
        registry.completed("Tier 3", "Total runtime recoverable from pipeline log", runtime_events_path)
    else:
        registry.skipped(
            "Tier 3",
            "Total runtime from log",
            "No parseable timestamps found in the pipeline log.",
        )

    artifacts: list[dict[str, Any]] = []
    for name, fp in {
        "connected_mask": paths.connected_mask,
        "raw_prediction": paths.raw_prediction,
        "final_expr": paths.final_expr,
        "nuclei_expr": paths.nuclei_expr,
        "checkpoint": paths.checkpoint,
        "pipeline_log": paths.pipeline_log,
    }.items():
        if fp is not None and fp.exists():
            artifacts.append({"artifact": name, "path": str(fp), "bytes": fp.stat().st_size})
    artifact_path = paths.output_dir / "tier3_artifact_sizes.csv"
    pd.DataFrame(artifacts).to_csv(artifact_path, index=False)
    runtime_summary["run_directory_bytes"] = recursive_size(paths.run_dir)
    runtime_summary.update(checkpoint_parameter_count(paths.checkpoint, logger))
    tier3_path = paths.output_dir / "tier3_efficiency_recoverable.json"
    safe_json_dump(runtime_summary, tier3_path)
    registry.completed("Tier 3", "Artifact sizes and checkpoint tensor count", tier3_path)

    # TODO-TIER3: historical utilisation cannot be inferred after execution.
    for metric, required in [
        ("Peak GPU memory allocated/reserved", "Live torch.cuda memory reset and sampling during training/inference."),
        ("Mean/P95 GPU utilisation and peak VRAM", "nvidia-smi sampling during the run."),
        ("Peak CPU RSS and system RAM", "psutil or /usr/bin/time -v sampling during the run."),
        ("GPU power and energy", "nvidia-smi power sampling integrated over time."),
        ("Exact FLOPs/MACs per patch", "Instantiated BIDCell model plus a profiler such as fvcore/thop with representative candidate count."),
        ("V3 sparse refinement efficiency", "V3 ambiguity map, routed pixels/transcripts/candidates, pair counts and module timers."),
    ]:
        registry.skipped("Tier 3", metric, required)

    # ---------------- Tier 4 optional pseudo-reference ----------------
    if args.vendor_mask is not None:
        vendor = load_tiff(args.vendor_mask.expanduser().resolve(), logger)
        try:
            ref_summary, ref_boundary = binary_and_boundary_reference_metrics(
                cells_mask,
                vendor,
                [w / target_pix_um for w in args.boundary_widths_um],
            )
            ref_summary_path = paths.output_dir / "tier4_vendor_binary_summary.json"
            ref_boundary_path = paths.output_dir / "tier4_vendor_boundary_metrics.csv"
            safe_json_dump(ref_summary, ref_summary_path)
            ref_boundary.to_csv(ref_boundary_path, index=False)
            registry.completed(
                "Tier 4",
                "Vendor pseudo-reference foreground and boundary metrics",
                ref_summary_path,
                "Pseudo-reference only; not manual ground truth.",
            )
            registry.skipped(
                "Tier 4",
                "AJI+, PQ, detection F1, AP50 and AP75",
                "Instance-level contingency/matching is intentionally deferred until the vendor conversion and ID semantics are validated.",
            )
        except Exception as exc:
            registry.failed("Tier 4", "Vendor pseudo-reference metrics", str(exc))
    else:
        # TODO-TIER4: cells.zarr.zip/cell_boundaries.parquet must first be
        # transformed and rasterised into the exact 5478x7526 BIDCell grid.
        registry.skipped(
            "Tier 4",
            "Vendor pseudo-reference Dice/IoU/boundary/instance metrics",
            "No --vendor-mask supplied. Needs a validated Xenium vendor instance TIFF aligned to the BIDCell 1-um grid; cells.zarr.zip or polygons must first be transformed with nearest-neighbour label preservation.",
        )

    # ---------------- Tier 5 bootstrap ----------------
    bootstrap_columns = [
        "total_transcripts",
        "total_genes",
        "area_um2",
        "transcript_density_per_um2",
        "elongation_major_minor",
        "compactness_raster",
        "eccentricity_skimage",
        "solidity",
    ]
    if marker_result is not None:
        bootstrap_columns += [
            "positive_marker_f1",
            "negative_marker_exclusion_f1",
            "purity_f1_proxy",
        ]
    rng = np.random.default_rng(args.seed)
    bootstrap_rows: list[dict[str, Any]] = []
    for col in bootstrap_columns:
        if col in cells.columns:
            bootstrap_rows.extend(
                bootstrap_ci(cells[col].to_numpy(float), col, args.bootstrap_iterations, rng)
            )
    bootstrap_path = paths.output_dir / "tier5_bootstrap_confidence_intervals.csv"
    pd.DataFrame(bootstrap_rows).to_csv(bootstrap_path, index=False)
    registry.completed("Tier 5", "Cell-level bootstrap confidence intervals", bootstrap_path)

    registry.skipped(
        "Tier 5",
        "Random-seed training stability",
        "Needs at least two additional independently trained step-4000 checkpoints with different seeds.",
    )
    registry.skipped(
        "Tier 5",
        "V1/V2/V3 ablation and paired method statistics",
        "Needs completed outputs from each method under identical preprocessing, split, seed and postprocessing settings.",
    )
    registry.skipped(
        "Tier 5",
        "Cross-replicate statistical validation",
        "Needs Rep2 or another independent tissue section.",
    )

    # ---------------- Headline summaries ----------------
    matrix_assigned = float(np.sum(expr, dtype=np.float64))
    tier1_summary: dict[str, Any] = {
        "segmented_cell_count": int(len(shape_df)),
        "max_cell_label": int(np.max(cells_mask)),
        "foreground_fraction": float(np.mean(cells_mask > 0)),
        "matrix_assigned_transcripts": matrix_assigned,
        "coordinate_assigned_transcripts": transcript_summary.get("assigned"),
        "coordinate_assigned_fraction_all_rows": transcript_summary.get(
            "assigned_fraction_all_rows"
        ),
        "median_transcripts_per_cell": float(np.nanmedian(cells["total_transcripts"])),
        "median_genes_per_cell": float(np.nanmedian(cells["total_genes"])),
        "median_cell_area_um2": float(np.nanmedian(cells["area_um2"])),
        "median_transcript_density_per_um2": float(
            np.nanmedian(cells["transcript_density_per_um2"])
        ),
    }
    if marker_result is not None:
        tier1_summary.update(
            {
                "mean_positive_marker_f1": float(marker_result["positive_marker_f1"].mean()),
                "mean_negative_marker_exclusion_f1": float(
                    marker_result["negative_marker_exclusion_f1"].mean()
                ),
                "mean_purity_f1_proxy": float(marker_result["purity_f1_proxy"].mean()),
            }
        )
    tier1_summary["cell_metric_distributions"] = {
        col: descriptive_stats(cells[col])
        for col in bootstrap_columns
        if col in cells.columns
    }
    tier1_summary_path = paths.output_dir / "tier1_summary.json"
    safe_json_dump(tier1_summary, tier1_summary_path)

    tier2_summary = {
        **structure_summary,
        "transcript_assignment": transcript_summary,
    }
    tier2_summary_path = paths.output_dir / "tier2_summary.json"
    safe_json_dump(tier2_summary, tier2_summary_path)

    registry_path = paths.output_dir / "metric_status.csv"
    registry.save(registry_path)
    elapsed = time.time() - start
    runtime_summary["benchmark_script_elapsed_seconds"] = elapsed
    safe_json_dump(runtime_summary, tier3_path)

    write_summary_markdown(
        paths.output_dir / "benchmark_summary.md",
        manifest,
        tier1_summary,
        structure_summary,
        runtime_summary,
        registry,
    )
    logger.info("Benchmark completed in %.2f minutes", elapsed / 60)
    logger.info("Summary: %s", paths.output_dir / "benchmark_summary.md")
    logger.info("Metric status: %s", registry_path)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark the completed BIDCell step-4000 baseline without retraining.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--repo-dir", type=Path, default=DEFAULT_REPO)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--connected-mask", type=Path, default=None)
    parser.add_argument("--experiment-id", default=DEFAULT_EXPERIMENT_ID)
    parser.add_argument(
        "--boundary-widths-um",
        type=float,
        nargs="+",
        default=[1.0, 2.0, 3.0, 5.0],
    )
    parser.add_argument("--spatial-bin-um", type=float, default=500.0)
    parser.add_argument(
        "--distance-breaks-um",
        type=float,
        nargs="+",
        default=[0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 100.0],
    )
    parser.add_argument("--min-cells-per-type", type=int, default=100)
    parser.add_argument("--transcript-chunksize", type=int, default=500_000)
    parser.add_argument("--bootstrap-iterations", type=int, default=200)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--skip-connectivity",
        action="store_true",
        help="Skip the full-image 8-neighbour connected-component analysis.",
    )
    parser.add_argument(
        "--vendor-mask",
        type=Path,
        default=None,
        help="Optional vendor instance TIFF already aligned to the connected-mask grid.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return run(args)
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
