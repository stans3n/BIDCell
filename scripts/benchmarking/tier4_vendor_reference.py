"""Tier 4 comparison against an explicitly supplied vendor pseudo-reference."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.optimize import linear_sum_assignment

from .common import Manifest, label_boundary, load_mask, write_json


REFERENCE_NOTE = "Reference is vendor/pseudo-reference, not manual ground truth."


def foreground_scores(prediction: np.ndarray, reference: np.ndarray) -> dict:
    pred = prediction > 0
    ref = reference > 0
    intersection = int((pred & ref).sum())
    pred_count = int(pred.sum())
    ref_count = int(ref.sum())
    union = int((pred | ref).sum())
    return {
        "dice": (
            2.0 * intersection / (pred_count + ref_count)
            if pred_count + ref_count
            else 1.0
        ),
        "iou": intersection / union if union else 1.0,
    }


def boundary_scores(
    prediction: np.ndarray,
    reference: np.ndarray,
    tolerance_um: float,
    pixel_size_um: float,
) -> dict:
    pred_boundary = label_boundary(prediction) & (prediction > 0)
    ref_boundary = label_boundary(reference) & (reference > 0)
    if not pred_boundary.any() and not ref_boundary.any():
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not pred_boundary.any() or not ref_boundary.any():
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    distance_to_ref = ndimage.distance_transform_edt(~ref_boundary) * pixel_size_um
    distance_to_pred = ndimage.distance_transform_edt(~pred_boundary) * pixel_size_um
    precision = float((distance_to_ref[pred_boundary] <= tolerance_um).mean())
    recall = float((distance_to_pred[ref_boundary] <= tolerance_um).mean())
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def _surface_distances(
    prediction: np.ndarray, reference: np.ndarray, pixel_size_um: float
) -> tuple[np.ndarray, np.ndarray]:
    pred_boundary = label_boundary(prediction) & (prediction > 0)
    ref_boundary = label_boundary(reference) & (reference > 0)
    if not pred_boundary.any() or not ref_boundary.any():
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    to_ref = ndimage.distance_transform_edt(~ref_boundary) * pixel_size_um
    to_pred = ndimage.distance_transform_edt(~pred_boundary) * pixel_size_um
    return to_ref[pred_boundary], to_pred[ref_boundary]


def _instance_matching(prediction: np.ndarray, reference: np.ndarray) -> dict:
    pred_ids = np.unique(prediction)
    pred_ids = pred_ids[pred_ids > 0]
    ref_ids = np.unique(reference)
    ref_ids = ref_ids[ref_ids > 0]
    if not len(pred_ids) or not len(ref_ids):
        return {
            "matched_instances": 0,
            "mean_matched_iou": None,
            "mean_matched_dice": None,
        }
    pred_index = {int(label): i for i, label in enumerate(pred_ids)}
    ref_index = {int(label): i for i, label in enumerate(ref_ids)}
    intersections = np.zeros((len(pred_ids), len(ref_ids)), dtype=np.int64)
    both = (prediction > 0) & (reference > 0)
    for pred_label, ref_label in zip(prediction[both], reference[both]):
        intersections[pred_index[int(pred_label)], ref_index[int(ref_label)]] += 1
    pred_area = np.asarray([(prediction == label).sum() for label in pred_ids])
    ref_area = np.asarray([(reference == label).sum() for label in ref_ids])
    union = pred_area[:, None] + ref_area[None, :] - intersections
    iou = np.divide(
        intersections,
        union,
        out=np.zeros_like(intersections, dtype=float),
        where=union > 0,
    )
    pred_rows, ref_cols = linear_sum_assignment(-iou)
    positive = iou[pred_rows, ref_cols] > 0
    matched_iou = iou[pred_rows[positive], ref_cols[positive]]
    matched_intersections = intersections[pred_rows[positive], ref_cols[positive]]
    matched_denominator = pred_area[pred_rows[positive]] + ref_area[ref_cols[positive]]
    matched_dice = np.divide(
        2.0 * matched_intersections,
        matched_denominator,
        out=np.zeros_like(matched_intersections, dtype=float),
        where=matched_denominator > 0,
    )
    return {
        "matched_instances": int(positive.sum()),
        "mean_matched_iou": float(matched_iou.mean()) if matched_iou.size else None,
        "mean_matched_dice": float(matched_dice.mean()) if matched_dice.size else None,
    }


def run(args, manifest: Manifest) -> dict:
    output_dir = Path(args.output_dir)
    summary: dict = {"tier": 4, "reference_note": REFERENCE_NOTE}
    if not args.vendor_mask or not args.connected_mask:
        for metric_id, name in (
            ("t4.foreground_dice", "foreground Dice"),
            ("t4.foreground_iou", "foreground IoU"),
            ("t4.boundary_f1", "boundary precision recall F1"),
            ("t4.assd", "average symmetric surface distance"),
            ("t4.hd95", "95th-percentile Hausdorff distance"),
            ("t4.instance_matching", "instance-matched IoU and Dice"),
        ):
            manifest.missing(metric_id, 4, name, ["connected_mask", "vendor_mask"])
        write_json(output_dir / "tier4_vendor_reference_summary.json", summary)
        return summary

    prediction = load_mask(args.connected_mask)
    reference = load_mask(args.vendor_mask)
    if prediction.shape != reference.shape:
        raise ValueError("Prediction and vendor masks must have identical shapes")

    scores = foreground_scores(prediction, reference)
    summary["foreground"] = scores
    for metric in ("dice", "iou"):
        manifest.add(
            f"t4.foreground_{metric}",
            4,
            f"foreground {metric.upper()}",
            "computed",
            value=scores[metric],
            unit="fraction",
            method="binary foreground overlap",
            required_inputs=["connected_mask", "vendor_mask"],
            used_inputs=[args.connected_mask, args.vendor_mask],
            notes=REFERENCE_NOTE,
        )

    summary["boundary"] = {}
    for tolerance_um in (1.0, 2.0, 3.0):
        scores_at_tolerance = boundary_scores(
            prediction, reference, tolerance_um, args.pixel_size_um
        )
        summary["boundary"][f"{int(tolerance_um)}um"] = scores_at_tolerance
        manifest.add(
            f"t4.boundary_f1_{int(tolerance_um)}um",
            4,
            f"boundary F1 at {int(tolerance_um)} um tolerance",
            "computed",
            value=scores_at_tolerance["f1"],
            unit="fraction",
            method="bidirectional boundary matching within physical tolerance",
            required_inputs=["connected_mask", "vendor_mask", "pixel_size_um"],
            used_inputs=[args.connected_mask, args.vendor_mask],
            notes=REFERENCE_NOTE,
        )

    pred_to_ref, ref_to_pred = _surface_distances(
        prediction, reference, args.pixel_size_um
    )
    if pred_to_ref.size and ref_to_pred.size:
        all_distances = np.concatenate([pred_to_ref, ref_to_pred])
        summary["assd_um"] = float((pred_to_ref.mean() + ref_to_pred.mean()) / 2.0)
        summary["hd95_um"] = float(np.percentile(all_distances, 95))
        manifest.add(
            "t4.assd",
            4,
            "average symmetric surface distance",
            "computed",
            value=summary["assd_um"],
            unit="um",
            method="mean bidirectional foreground-boundary distance",
            required_inputs=["connected_mask", "vendor_mask", "pixel_size_um"],
            used_inputs=[args.connected_mask, args.vendor_mask],
            notes=REFERENCE_NOTE,
        )
        manifest.add(
            "t4.hd95",
            4,
            "95th-percentile Hausdorff distance",
            "computed",
            value=summary["hd95_um"],
            unit="um",
            method="P95 pooled bidirectional foreground-boundary distance",
            required_inputs=["connected_mask", "vendor_mask", "pixel_size_um"],
            used_inputs=[args.connected_mask, args.vendor_mask],
            notes=REFERENCE_NOTE,
        )

    summary["instance_matching"] = _instance_matching(prediction, reference)
    manifest.add(
        "t4.instance_matching",
        4,
        "instance-matched IoU and Dice",
        "computed",
        value=summary["instance_matching"]["mean_matched_iou"],
        unit="mean matched IoU",
        method="Hungarian matching on pairwise instance IoU",
        required_inputs=["connected_mask", "vendor_mask"],
        used_inputs=[args.connected_mask, args.vendor_mask],
        notes=REFERENCE_NOTE,
    )
    for metric_id, name in (
        ("t4.aji_plus", "AJI+"),
        ("t4.pq", "panoptic quality"),
        ("t4.detection_f1", "instance detection F1"),
    ):
        manifest.add(
            metric_id,
            4,
            name,
            "not_implemented",
            required_inputs=["validated instance matching"],
            notes=f"{REFERENCE_NOTE} Requires validated benchmark-compatible matching.",
        )
    write_json(output_dir / "tier4_vendor_reference_summary.json", summary)
    return summary
