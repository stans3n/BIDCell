"""Measure where a V1 connected mask differs from a baseline mask."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.optimize import linear_sum_assignment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-mask", required=True)
    parser.add_argument("--v1-mask", required=True)
    parser.add_argument("--baseline-config", required=True)
    parser.add_argument("--v1-config", required=True)
    parser.add_argument("--bands-um", type=float, nargs="+", default=[1, 2, 3, 5])
    parser.add_argument("--output-dir", required=True)
    return parser


def match_instances(baseline: np.ndarray, comparison: np.ndarray):
    """Match non-background instances by maximum total mask IoU."""

    baseline = np.asarray(baseline)
    comparison = np.asarray(comparison)
    if baseline.shape != comparison.shape or baseline.ndim != 2:
        raise ValueError("masks must be two-dimensional and have the same shape")
    base_ids, base_inverse = np.unique(baseline, return_inverse=True)
    comp_ids, comp_inverse = np.unique(comparison, return_inverse=True)
    contingency = np.bincount(
        base_inverse * len(comp_ids) + comp_inverse,
        minlength=len(base_ids) * len(comp_ids),
    ).reshape(len(base_ids), len(comp_ids))
    base_areas = contingency.sum(axis=1)
    comp_areas = contingency.sum(axis=0)
    unions = base_areas[:, None] + comp_areas[None, :] - contingency
    iou = np.divide(
        contingency,
        unions,
        out=np.zeros_like(contingency, dtype=float),
        where=unions > 0,
    )
    base_indices = np.flatnonzero(base_ids != 0)
    comp_indices = np.flatnonzero(comp_ids != 0)
    matches = []
    if len(base_indices) and len(comp_indices):
        rows, columns = linear_sum_assignment(-iou[np.ix_(base_indices, comp_indices)])
        for row, column in zip(rows, columns):
            base_index = base_indices[row]
            comp_index = comp_indices[column]
            if contingency[base_index, comp_index] > 0:
                matches.append(
                    {
                        "baseline_id": int(base_ids[base_index]),
                        "v1_id": int(comp_ids[comp_index]),
                        "intersection": int(contingency[base_index, comp_index]),
                        "iou": float(iou[base_index, comp_index]),
                    }
                )
    return matches, base_ids[base_ids != 0], comp_ids[comp_ids != 0]


def align_instance_ids(
    baseline: np.ndarray, comparison: np.ndarray, matches: list[dict]
) -> np.ndarray:
    """Relabel matched V1 instances to baseline IDs for change analysis."""

    aligned = np.zeros(comparison.shape, dtype=np.int64)
    mapping = {item["v1_id"]: item["baseline_id"] for item in matches}
    next_id = int(np.max(baseline, initial=0)) + 1
    for comparison_id in np.unique(comparison):
        if comparison_id == 0:
            continue
        target = mapping.get(int(comparison_id))
        if target is None:
            target = next_id
            next_id += 1
        aligned[comparison == comparison_id] = target
    return aligned


def _label_boundaries(labels: np.ndarray) -> np.ndarray:
    boundaries = np.zeros(labels.shape, dtype=bool)
    horizontal = labels[:, 1:] != labels[:, :-1]
    vertical = labels[1:, :] != labels[:-1, :]
    boundaries[:, 1:] |= horizontal
    boundaries[:, :-1] |= horizontal
    boundaries[1:, :] |= vertical
    boundaries[:-1, :] |= vertical
    return boundaries & (labels > 0)


def locality_metrics(
    baseline: np.ndarray,
    comparison: np.ndarray,
    *,
    pixel_size_um: float = 1.0,
    bands_um=(1.0, 2.0, 3.0, 5.0),
):
    if pixel_size_um <= 0:
        raise ValueError("pixel_size_um must be positive")
    matches, base_ids, comp_ids = match_instances(baseline, comparison)
    aligned = align_instance_ids(baseline, comparison, matches)
    changed = np.asarray(baseline) != aligned
    matched_base = {item["baseline_id"] for item in matches}
    matched_comp = {item["v1_id"] for item in matches}
    changed_matches = sum(item["iou"] < 1.0 - 1.0e-12 for item in matches)
    changed_cells = (
        changed_matches
        + len(set(base_ids.tolist()) - matched_base)
        + len(set(comp_ids.tolist()) - matched_comp)
    )
    instance_union_count = (
        len(matches)
        + len(set(base_ids.tolist()) - matched_base)
        + len(set(comp_ids.tolist()) - matched_comp)
    )
    boundaries = _label_boundaries(np.asarray(baseline))
    distance_um = distance_transform_edt(~boundaries) * pixel_size_um
    summary = {
        "changed_pixel_fraction": float(changed.mean()),
        "changed_cell_fraction": (
            float(changed_cells / instance_union_count) if instance_union_count else 0.0
        ),
        "baseline_cell_count": int(len(base_ids)),
        "v1_cell_count": int(len(comp_ids)),
        "matched_cell_count": len(matches),
        "changed_cell_definition": (
            "matched pair with IoU below 1, plus every unmatched instance, "
            "divided by matched pairs plus unmatched instances"
        ),
    }
    for band in bands_um:
        domain = distance_um <= float(band)
        summary[f"boundary_band_{float(band):g}um_change_fraction"] = (
            float(changed[domain].mean()) if domain.any() else 0.0
        )
    max_band = max((float(value) for value in bands_um), default=0.0)
    interior = (np.asarray(baseline) > 0) & (distance_um > max_band)
    summary["non_boundary_interior_preservation"] = (
        float((~changed[interior]).mean()) if interior.any() else 1.0
    )
    return summary, matches, changed, aligned


def _pixel_size(path: str) -> float:
    import yaml

    with Path(path).open() as handle:
        raw = yaml.safe_load(handle)
    return float(raw.get("affine", {}).get("target_pix_um", 1.0))


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    import tifffile

    baseline = tifffile.imread(args.baseline_mask)
    comparison = tifffile.imread(args.v1_mask)
    baseline_pixel_size = _pixel_size(args.baseline_config)
    v1_pixel_size = _pixel_size(args.v1_config)
    if not np.isclose(baseline_pixel_size, v1_pixel_size):
        raise ValueError("baseline and V1 configs use different target pixel sizes")
    summary, matches, changed, aligned = locality_metrics(
        baseline,
        comparison,
        pixel_size_um=baseline_pixel_size,
        bands_um=args.bands_um,
    )
    summary["pixel_size_um"] = baseline_pixel_size
    summary["matching_method"] = (
        "Hungarian maximum-total-IoU, positive intersections only"
    )
    summary["interpretation"] = "Inter-version IoU measures change, not accuracy."
    matched_base = {item["baseline_id"] for item in matches}
    matched_v1 = {item["v1_id"] for item in matches}
    per_cell_rows = list(matches)
    per_cell_rows.extend(
        {
            "baseline_id": int(instance_id),
            "v1_id": "",
            "intersection": 0,
            "iou": 0.0,
        }
        for instance_id in np.unique(baseline)
        if instance_id != 0 and int(instance_id) not in matched_base
    )
    per_cell_rows.extend(
        {
            "baseline_id": "",
            "v1_id": int(instance_id),
            "intersection": 0,
            "iou": 0.0,
        }
        for instance_id in np.unique(comparison)
        if instance_id != 0 and int(instance_id) not in matched_v1
    )
    destination = Path(args.output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / "locality_summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    with (destination / "per_cell_interversion_iou.csv").open(
        "w", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["baseline_id", "v1_id", "intersection", "iou"]
        )
        writer.writeheader()
        writer.writerows(per_cell_rows)
    tifffile.imwrite(destination / "changed_pixels.tif", changed.astype(np.uint8))
    tifffile.imwrite(destination / "v1_ids_aligned_to_baseline.tif", aligned)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
