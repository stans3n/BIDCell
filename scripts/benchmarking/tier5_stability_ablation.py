"""Tier 5 standardized cross-run stability and ablation aggregation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .common import Manifest, write_json


def _manifest_path(run_dir: str | Path) -> Path | None:
    root = Path(run_dir)
    candidates = [
        root / "metrics" / "benchmark" / "benchmark_manifest.csv",
        root / "benchmark" / "benchmark_manifest.csv",
        root / "benchmark_manifest.csv",
    ]
    return next((path for path in candidates if path.exists()), None)


def _numeric_metrics(frame: pd.DataFrame) -> dict[str, float]:
    computed = frame[frame["status"] == "computed"].copy()
    computed["numeric_value"] = pd.to_numeric(computed["value"], errors="coerce")
    computed = computed.dropna(subset=["numeric_value"])
    return dict(zip(computed["metric_id"], computed["numeric_value"].astype(float)))


def _metadata(run_dir: str | Path) -> dict:
    root = Path(run_dir)
    for candidate in (
        root / "experiment_summary.json",
        root / "all_tiers_summary.json",
        root / "metrics" / "benchmark" / "all_tiers_summary.json",
    ):
        if candidate.exists():
            with candidate.open(encoding="utf-8") as handle:
                return json.load(handle)
    return {"run_dir": str(root)}


def run(args, manifest: Manifest) -> dict:
    output_dir = Path(args.output_dir)
    replicate_dirs = list(args.replicate_run_dir or [])
    run_records: list[tuple[str, dict[str, float]]] = []
    current_frame = pd.DataFrame(manifest.rows)
    run_records.append(
        (str(args.run_dir or "current"), _numeric_metrics(current_frame))
    )

    missing_replicates = []
    for run_dir in replicate_dirs:
        path = _manifest_path(run_dir)
        if path is None:
            missing_replicates.append(str(run_dir))
            continue
        run_records.append((str(run_dir), _numeric_metrics(pd.read_csv(path))))

    summary: dict = {
        "tier": 5,
        "run_count": len(run_records),
        "runs": [
            {"run_dir": run_dir, "metadata": _metadata(run_dir)}
            for run_dir, _ in run_records
            if run_dir != "current"
        ],
        "missing_replicate_results": missing_replicates,
    }
    ablation_rows = []
    for run_dir, _ in run_records:
        metadata = _metadata(run_dir) if run_dir != "current" else {}
        ablation_rows.append(
            {
                "run_dir": run_dir,
                "seed": metadata.get(
                    "seed", args.seed if run_dir == str(args.run_dir) else None
                ),
                "boundary_weight": metadata.get("boundary_weight"),
                "boundary_weight_overlap": metadata.get("boundary_weight_overlap"),
                "boundary_weight_positive_marker": metadata.get(
                    "boundary_weight_positive_marker"
                ),
                "boundary_weight_negative_marker": metadata.get(
                    "boundary_weight_negative_marker"
                ),
                "boundary_weight_cell_calling": metadata.get(
                    "boundary_weight_cell_calling"
                ),
            }
        )
    pd.DataFrame(ablation_rows).to_csv(
        output_dir / "tier5_ablation_table.csv", index=False
    )

    if len(run_records) == 1:
        summary["stability"] = None
        manifest.missing(
            "t5.stability",
            5,
            "cross-seed mean and standard deviation",
            ["replicate_run_dir"],
            notes="One run is metadata only; no standard deviation was fabricated.",
        )
    else:
        metric_ids = sorted(
            set.intersection(*(set(metrics.keys()) for _, metrics in run_records))
            if run_records
            else set()
        )
        stability = {}
        for metric_id in metric_ids:
            values = np.asarray(
                [metrics[metric_id] for _, metrics in run_records], dtype=float
            )
            stability[metric_id] = {
                "n": int(len(values)),
                "mean": float(values.mean()),
                "standard_deviation": (
                    float(values.std(ddof=1)) if len(values) > 1 else None
                ),
                "ci95_normal": (
                    [
                        float(
                            values.mean()
                            - 1.96 * values.std(ddof=1) / np.sqrt(len(values))
                        ),
                        float(
                            values.mean()
                            + 1.96 * values.std(ddof=1) / np.sqrt(len(values))
                        ),
                    ]
                    if len(values) > 1
                    else None
                ),
            }
        summary["stability"] = stability
        manifest.add(
            "t5.stability",
            5,
            "cross-seed mean and standard deviation",
            "computed",
            value=len(stability),
            unit="metrics aggregated",
            method="intersection of computed numeric metric IDs across standardized manifests",
            required_inputs=["two or more standardized run manifests"],
            used_inputs=[run_dir for run_dir, _ in run_records],
        )

    baseline_path = (
        _manifest_path(args.baseline_run_dir) if args.baseline_run_dir else None
    )
    if baseline_path is not None:
        baseline_metrics = _numeric_metrics(pd.read_csv(baseline_path))
        current_metrics = run_records[0][1]
        comparison = {}
        for metric_id in sorted(set(current_metrics) & set(baseline_metrics)):
            baseline_value = baseline_metrics[metric_id]
            current_value = current_metrics[metric_id]
            comparison[metric_id] = {
                "baseline": baseline_value,
                "v1": current_value,
                "absolute_difference": current_value - baseline_value,
                "percentage_change": (
                    (current_value - baseline_value) / baseline_value * 100.0
                    if baseline_value != 0
                    else None
                ),
            }
        summary["baseline_comparison"] = comparison
        manifest.add(
            "t5.baseline_comparison",
            5,
            "baseline versus V1 absolute and percentage change",
            "computed",
            value=len(comparison),
            unit="metrics compared",
            method="metric-ID matched standardized manifests",
            required_inputs=["baseline_run_dir benchmark manifest"],
            used_inputs=[str(baseline_path)],
        )
    else:
        manifest.missing(
            "t5.baseline_comparison",
            5,
            "baseline versus V1 absolute and percentage change",
            ["baseline_run_dir benchmark manifest"],
        )

    manifest.add(
        "t5.paired_statistics",
        5,
        "paired Wilcoxon or bootstrap",
        "not_implemented",
        required_inputs=["validated paired ROI or cell summaries"],
        notes="Run-level scalar summaries do not establish a valid pairing relationship.",
    )
    write_json(output_dir / "tier5_stability_summary.json", summary)
    return summary
