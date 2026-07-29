"""Tier 3 runtime, throughput, memory, and footprint summaries."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .common import Manifest, directory_size_bytes, write_json


def _first_existing(paths):
    return next((path for path in paths if path.exists()), None)


def _read_json(path: Path | None) -> dict:
    if path is None:
        return {}
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _run_resources(
    run_dir: Path,
) -> tuple[Path | None, Path | None, Path | None, Path | None]:
    training = _first_existing(
        [
            run_dir / "metrics" / "training" / "v1_training_metrics.csv",
            run_dir / "training" / "v1_training_metrics.csv",
        ]
    )
    training_summary = _first_existing(
        [
            run_dir / "metrics" / "training" / "training_summary.json",
            run_dir / "training" / "training_summary.json",
        ]
    )
    resource = _first_existing(
        [
            run_dir / "metrics" / "resources" / "resource_summary.json",
            run_dir / "resources" / "resource_summary.json",
        ]
    )
    stages = _first_existing(
        [
            run_dir / "experiment_summary.json",
            run_dir / "metrics" / "experiment_summary.json",
        ]
    )
    return training, training_summary, resource, stages


def run(args, manifest: Manifest) -> dict:
    output_dir = Path(args.output_dir)
    summary: dict = {"tier": 3}
    if not args.run_dir:
        for metric_id, name in (
            ("t3.training_wall_clock", "training wall-clock"),
            ("t3.step_throughput", "steps per second"),
            ("t3.pytorch_vram", "PyTorch peak allocated/reserved VRAM"),
            ("t3.external_resources", "nvidia-smi CPU and RAM summary"),
        ):
            manifest.missing(metric_id, 3, name, ["run_dir"])
        write_json(output_dir / "tier3_resource_summary.json", summary)
        return summary

    run_dir = Path(args.run_dir)
    training_path, training_summary_path, resource_path, stage_path = _run_resources(
        run_dir
    )
    if training_path:
        training = pd.read_csv(training_path)
        training_summary = _read_json(training_summary_path)
        wall = training_summary.get("training_wall_clock_seconds")
        if wall is None and len(training):
            wall = float(training["cumulative_training_time_seconds"].max())
        steps = int(len(training))
        candidates = (
            float(training["n_candidates"].sum())
            if "n_candidates" in training
            else None
        )
        summary["training"] = {
            "logged_steps": steps,
            "wall_clock_seconds": wall,
            "steps_per_second": steps / wall if wall else None,
            "effective_candidates_per_second": (
                candidates / wall if wall and candidates is not None else None
            ),
            "pytorch_peak_allocated_mb": float(
                training["cuda_max_memory_allocated_mb"].max()
            ),
            "pytorch_peak_reserved_mb": float(
                training["cuda_max_memory_reserved_mb"].max()
            ),
        }
        manifest.add(
            "t3.training_wall_clock",
            3,
            "training wall-clock",
            "computed",
            value=wall,
            unit="seconds",
            method="maximum cumulative time in training CSV",
            required_inputs=["run_dir training metrics"],
            used_inputs=[str(training_path)],
        )
        manifest.add(
            "t3.step_throughput",
            3,
            "steps per second",
            "computed",
            value=summary["training"]["steps_per_second"],
            unit="steps/second",
            method="logged steps divided by training wall-clock",
            required_inputs=["run_dir training metrics"],
            used_inputs=[str(training_path)],
        )
        manifest.add(
            "t3.pytorch_vram",
            3,
            "PyTorch peak allocated/reserved VRAM",
            "computed",
            value=summary["training"]["pytorch_peak_allocated_mb"],
            unit="MiB allocated",
            method="PyTorch CUDA peak-memory counters",
            required_inputs=["run_dir training metrics"],
            used_inputs=[str(training_path)],
        )
    else:
        for metric_id, name in (
            ("t3.training_wall_clock", "training wall-clock"),
            ("t3.step_throughput", "steps per second"),
            ("t3.pytorch_vram", "PyTorch peak allocated/reserved VRAM"),
        ):
            manifest.missing(metric_id, 3, name, ["run_dir training metrics"])

    resource = _read_json(resource_path)
    if resource:
        summary["resources"] = resource
        manifest.add(
            "t3.external_resources",
            3,
            "nvidia-smi GPU CPU and RAM summary",
            "computed",
            value=resource.get("gpu_utilization_percent", {}).get("mean"),
            unit="percent mean GPU utilization",
            method="external resource monitor sampling",
            required_inputs=["resource_summary.json"],
            used_inputs=[str(resource_path)],
            notes="Device-level GPU utilization can include other processes sharing the GPU.",
        )
    else:
        manifest.missing(
            "t3.external_resources",
            3,
            "nvidia-smi GPU CPU and RAM summary",
            ["resource_summary.json"],
        )

    stages = _read_json(stage_path)
    if stages:
        summary["stages"] = stages.get("stages", stages)
        manifest.add(
            "t3.stage_wall_clock",
            3,
            "stage-specific wall-clock",
            "computed",
            value=len(summary["stages"]),
            unit="stages",
            method="experiment launcher stage records",
            required_inputs=["experiment_summary.json"],
            used_inputs=[str(stage_path)],
        )
    else:
        manifest.missing(
            "t3.stage_wall_clock",
            3,
            "stage-specific wall-clock",
            ["experiment_summary.json"],
        )

    checkpoints = list(run_dir.rglob("*.pth"))
    summary["checkpoint_size_bytes"] = int(
        sum(path.stat().st_size for path in checkpoints)
    )
    summary["run_disk_footprint_bytes"] = directory_size_bytes(run_dir)
    manifest.add(
        "t3.disk_footprint",
        3,
        "checkpoint and run disk footprint",
        "computed",
        value=summary["run_disk_footprint_bytes"],
        unit="bytes",
        method="recursive file-size sum",
        required_inputs=["run_dir"],
        used_inputs=[str(run_dir)],
    )
    manifest.add(
        "t3.flops_macs",
        3,
        "FLOPs/MACs",
        "not_implemented",
        notes="No validated dependency-free counter is available for the configured model.",
    )
    manifest.add(
        "t3.trainable_parameter_count",
        3,
        "trainable parameter count",
        "not_implemented",
        required_inputs=["validated model construction metadata"],
        notes="Checkpoint tensors do not reliably distinguish trainable parameters from buffers.",
    )

    if args.baseline_run_dir:
        baseline_training, baseline_summary_path, _, _ = _run_resources(
            Path(args.baseline_run_dir)
        )
        if training_path and baseline_training:
            baseline = pd.read_csv(baseline_training)
            current_wall = summary["training"]["wall_clock_seconds"]
            baseline_summary = _read_json(baseline_summary_path)
            baseline_wall = baseline_summary.get("training_wall_clock_seconds")
            if baseline_wall is None and len(baseline):
                baseline_wall = float(
                    baseline["cumulative_training_time_seconds"].max()
                )
            if current_wall and baseline_wall:
                summary["baseline_training_overhead"] = {
                    "absolute_seconds": current_wall - baseline_wall,
                    "percentage": (current_wall - baseline_wall)
                    / baseline_wall
                    * 100.0,
                }
                manifest.add(
                    "t3.baseline_overhead",
                    3,
                    "V1 training overhead versus official baseline",
                    "computed",
                    value=summary["baseline_training_overhead"]["percentage"],
                    unit="percent",
                    method="same-script wall-clock comparison",
                    required_inputs=["run_dir", "baseline_run_dir"],
                    used_inputs=[str(training_path), str(baseline_training)],
                )
        else:
            manifest.missing(
                "t3.baseline_overhead",
                3,
                "V1 overhead versus official baseline",
                ["run_dir training metrics", "baseline_run_dir training metrics"],
            )
    else:
        manifest.missing(
            "t3.baseline_overhead",
            3,
            "V1 overhead versus official baseline",
            ["baseline_run_dir"],
        )

    write_json(output_dir / "tier3_resource_summary.json", summary)
    return summary
