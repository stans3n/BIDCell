"""Tier 2 candidate-overlap and post-hoc boundary/instance diagnostics."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage

from .common import (
    Manifest,
    boundary_distance_um,
    distribution,
    load_mask,
    read_table,
    transcript_assignments,
    write_json,
)


def _training_metrics_path(run_dir: str | None) -> Path | None:
    if not run_dir:
        return None
    candidates = [
        Path(run_dir) / "metrics" / "training" / "v1_training_metrics.csv",
        Path(run_dir) / "training" / "v1_training_metrics.csv",
        Path(run_dir) / "v1_training_metrics.csv",
    ]
    return next((path for path in candidates if path.exists()), None)


def _instance_diagnostics(mask: np.ndarray, nuclei: np.ndarray) -> dict:
    cell_ids = np.unique(mask)
    cell_ids = cell_ids[cell_ids > 0]
    nucleus_ids = np.unique(nuclei)
    nucleus_ids = nucleus_ids[nucleus_ids > 0]

    nuclei_per_cell = {int(cell_id): set() for cell_id in cell_ids}
    contained = 0
    missed = 0
    split = 0
    for nucleus_id in nucleus_ids:
        nucleus_pixels = nuclei == nucleus_id
        overlapping = np.unique(mask[nucleus_pixels])
        positive = overlapping[overlapping > 0]
        if len(overlapping) == 1 and len(positive) == 1:
            contained += 1
        if len(positive) == 0:
            missed += 1
        elif len(positive) > 1:
            split += 1
        for cell_id in positive:
            nuclei_per_cell[int(cell_id)].add(int(nucleus_id))

    component_counts = {}
    structure = np.ones((3, 3), dtype=int)
    areas = []
    for cell_id in cell_ids:
        cell = mask == cell_id
        _, n_components = ndimage.label(cell, structure=structure)
        component_counts[int(cell_id)] = int(n_components)
        areas.append(int(cell.sum()))

    no_nucleus = sum(len(ids) == 0 for ids in nuclei_per_cell.values())
    multi_nucleus = sum(len(ids) > 1 for ids in nuclei_per_cell.values())
    disconnected = sum(count > 1 for count in component_counts.values())
    areas_array = np.asarray(areas, dtype=float)
    if areas_array.size:
        q1, q3 = np.percentile(areas_array, [25, 75])
        iqr = q3 - q1
        lower = max(0.0, q1 - 3.0 * iqr)
        upper = q3 + 3.0 * iqr
        extreme = int(((areas_array < lower) | (areas_array > upper)).sum())
    else:
        extreme = 0

    n_cells = len(cell_ids)
    n_nuclei = len(nucleus_ids)
    return {
        "nucleus_count": int(n_nuclei),
        "nucleus_containment_rate": contained / n_nuclei if n_nuclei else None,
        "missed_nucleus_proxy_fraction": missed / n_nuclei if n_nuclei else None,
        "split_nucleus_proxy_fraction": split / n_nuclei if n_nuclei else None,
        "zero_nucleus_cell_fraction": no_nucleus / n_cells if n_cells else None,
        "multi_nucleus_cell_fraction": multi_nucleus / n_cells if n_cells else None,
        "disconnected_cell_fraction_8_neighbor": (
            disconnected / n_cells if n_cells else None
        ),
        "merge_proxy_fraction": multi_nucleus / n_cells if n_cells else None,
        "fragmentation_proxy_fraction": disconnected / n_cells if n_cells else None,
        "extreme_area_outlier_fraction": extreme / n_cells if n_cells else None,
        "cell_area_px2": distribution(areas),
    }


def run(args, manifest: Manifest) -> dict:
    output_dir = Path(args.output_dir)
    summary: dict = {"tier": 2}

    training_path = _training_metrics_path(args.run_dir)
    if training_path:
        training = pd.read_csv(training_path)
        overlap_summary = {
            "pixel_count": distribution(training["candidate_overlap_pixel_count"]),
            "fraction": distribution(training["candidate_overlap_fraction"]),
            "steps": int(len(training)),
        }
        summary["candidate_overlap"] = overlap_summary
        manifest.add(
            "t2.candidate_overlap",
            2,
            "candidate-overlap per-step and aggregate summary",
            "computed",
            value=overlap_summary["fraction"]["mean"],
            unit="mean fraction",
            method="aggregate of detached training CSV scalars",
            required_inputs=["run_dir training metrics"],
            used_inputs=[str(training_path)],
        )
    else:
        manifest.missing(
            "t2.candidate_overlap",
            2,
            "candidate-overlap per-step and aggregate summary",
            ["run_dir training metrics"],
        )

    mask = load_mask(args.connected_mask) if args.connected_mask else None
    assignments = None
    if mask is not None and args.transcripts:
        transcripts = read_table(args.transcripts)
        assignments = transcript_assignments(
            transcripts, mask, pixel_size_um=args.pixel_size_um
        )
        distance = boundary_distance_um(mask, args.pixel_size_um)
        valid = assignments["_in_bounds"].to_numpy()
        x = assignments["_pixel_x"].to_numpy()
        y = assignments["_pixel_y"].to_numpy()
        transcript_distance = np.full(len(assignments), np.inf, dtype=float)
        transcript_distance[valid] = distance[y[valid], x[valid]]
        boundary_rows = []
        for width_um in (1.0, 2.0, 3.0, 5.0):
            in_band = valid & (transcript_distance <= width_um)
            band_count = int(in_band.sum())
            assigned_count = int(
                (in_band & (assignments["_assigned_cell_id"].to_numpy() > 0)).sum()
            )
            row = {
                "width_um": width_um,
                "transcript_count": band_count,
                "transcript_fraction": (
                    band_count / int(valid.sum()) if valid.any() else None
                ),
                "assigned_count": assigned_count,
                "assigned_rate": assigned_count / band_count if band_count else None,
            }
            boundary_rows.append(row)
            manifest.add(
                f"t2.boundary_band_{int(width_um)}um",
                2,
                f"{int(width_um)} um boundary-zone transcript fraction and assignment rate",
                "computed",
                value=row["transcript_fraction"],
                unit="fraction",
                method="distance to final connected-mask label discontinuity; evaluation only",
                required_inputs=["connected_mask", "transcripts", "pixel_size_um"],
                used_inputs=[args.connected_mask, args.transcripts],
                notes="Final connected-mask boundaries are post-hoc evaluation data, never a training target.",
            )
        summary["boundary_bands"] = boundary_rows
    else:
        for width_um in (1, 2, 3, 5):
            manifest.missing(
                f"t2.boundary_band_{width_um}um",
                2,
                f"{width_um} um boundary-zone transcript fraction and assignment rate",
                ["connected_mask", "transcripts"],
            )

    if mask is not None and args.nuclei_mask:
        nuclei = load_mask(args.nuclei_mask)
        if nuclei.shape != mask.shape:
            raise ValueError("Connected and nuclei masks must have identical shapes")
        instance_summary = _instance_diagnostics(mask, nuclei)
        summary["instances"] = instance_summary
        for metric_name in (
            "nucleus_containment_rate",
            "zero_nucleus_cell_fraction",
            "multi_nucleus_cell_fraction",
            "missed_nucleus_proxy_fraction",
            "split_nucleus_proxy_fraction",
            "disconnected_cell_fraction_8_neighbor",
            "merge_proxy_fraction",
            "fragmentation_proxy_fraction",
            "extreme_area_outlier_fraction",
        ):
            manifest.add(
                f"t2.{metric_name}",
                2,
                metric_name.replace("_", " "),
                "computed",
                value=instance_summary[metric_name],
                unit="fraction",
                method="instance overlap and 8-neighbour connected-components diagnostics",
                required_inputs=["connected_mask", "nuclei_mask"],
                used_inputs=[args.connected_mask, args.nuclei_mask],
            )
    else:
        for metric_name in (
            "nucleus_containment_rate",
            "zero_nucleus_cell_fraction",
            "multi_nucleus_cell_fraction",
            "disconnected_cell_fraction_8_neighbor",
            "extreme_area_outlier_fraction",
        ):
            manifest.missing(
                f"t2.{metric_name}",
                2,
                metric_name.replace("_", " "),
                ["connected_mask", "nuclei_mask"],
            )

    assignment_changes = pd.DataFrame()
    baseline_mask_path = None
    if args.baseline_run_dir:
        candidates = sorted(Path(args.baseline_run_dir).rglob("*_connected.tif"))
        baseline_mask_path = candidates[-1] if candidates else None
    if mask is not None and assignments is not None and baseline_mask_path is not None:
        baseline_mask = load_mask(baseline_mask_path)
        if baseline_mask.shape != mask.shape:
            raise ValueError("Baseline and V1 masks must have identical shapes")
        baseline = transcript_assignments(
            read_table(args.transcripts), baseline_mask, args.pixel_size_um
        )
        valid = assignments["_in_bounds"] & baseline["_in_bounds"]
        changed = (
            assignments["_assigned_cell_id"] != baseline["_assigned_cell_id"]
        ) & valid
        neighbour_only = (
            changed
            & (assignments["_assigned_cell_id"] > 0)
            & (baseline["_assigned_cell_id"] > 0)
        )
        assignment_changes = pd.DataFrame(
            {
                "transcript_index": np.arange(len(assignments)),
                "baseline_cell_id": baseline["_assigned_cell_id"],
                "v1_cell_id": assignments["_assigned_cell_id"],
                "changed": changed,
                "neighbour_only_changed": neighbour_only,
            }
        )
        summary["assignment_changes"] = {
            "change_rate": float(changed.sum() / valid.sum()) if valid.any() else None,
            "neighbour_only_change_rate": (
                float(neighbour_only.sum() / valid.sum()) if valid.any() else None
            ),
        }
        manifest.add(
            "t2.assignment_change_rate",
            2,
            "baseline versus V1 transcript assignment-change rate",
            "computed",
            value=summary["assignment_changes"]["change_rate"],
            unit="fraction",
            method="label comparison at transcript coordinates",
            required_inputs=["connected_mask", "baseline_run_dir", "transcripts"],
            used_inputs=[
                args.connected_mask,
                str(baseline_mask_path),
                args.transcripts,
            ],
            notes="Assumes comparable instance labels between baseline and V1.",
        )
        manifest.add(
            "t2.neighbour_only_assignment_change_rate",
            2,
            "neighbour-only assignment-change rate",
            "computed",
            value=summary["assignment_changes"]["neighbour_only_change_rate"],
            unit="fraction",
            method="changed assignments where both masks assign a positive instance label",
            required_inputs=["connected_mask", "baseline_run_dir", "transcripts"],
            used_inputs=[
                args.connected_mask,
                str(baseline_mask_path),
                args.transcripts,
            ],
            notes="Assumes comparable instance labels between baseline and V1.",
        )
    else:
        manifest.missing(
            "t2.assignment_change_rate",
            2,
            "baseline versus V1 transcript assignment-change rate",
            ["connected_mask", "baseline_run_dir connected mask", "transcripts"],
        )
        manifest.missing(
            "t2.neighbour_only_assignment_change_rate",
            2,
            "neighbour-only assignment-change rate",
            ["connected_mask", "baseline_run_dir connected mask", "transcripts"],
        )
    assignment_changes.to_csv(output_dir / "tier2_assignment_changes.csv", index=False)

    for metric_id, name in (
        ("t2.assignment_entropy", "assignment entropy"),
        ("t2.top1_top2_margin", "top1-top2 assignment margin"),
        (
            "t2.segmentation_assignment_consistency",
            "segmentation-assignment consistency",
        ),
    ):
        manifest.add(
            metric_id,
            2,
            name,
            "not_applicable_v1",
            notes="Requires a V2/V3 assignment-confidence mechanism absent from V1.",
        )
    write_json(output_dir / "tier2_boundary_summary.json", summary)
    return summary
