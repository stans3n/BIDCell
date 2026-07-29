import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

from scripts.benchmarking.run_all_tiers import main as run_all_main
from scripts.benchmarking.run_with_resource_monitor import main as monitor_main
from scripts.benchmarking.tier4_vendor_reference import (
    boundary_scores,
    foreground_scores,
)


def _synthetic_inputs(tmp_path: Path):
    mask = np.zeros((8, 8), dtype=np.uint16)
    mask[1:4, 1:4] = 1
    mask[5, 1] = 1  # disconnected component for cell 1
    mask[2:6, 5:7] = 2
    nuclei = np.zeros_like(mask)
    nuclei[2, 2] = 1
    nuclei[3, 5] = 2
    vendor = mask.copy()
    mask_path = tmp_path / "connected.tif"
    nuclei_path = tmp_path / "nuclei.tif"
    vendor_path = tmp_path / "vendor.tif"
    tifffile.imwrite(mask_path, mask)
    tifffile.imwrite(nuclei_path, nuclei)
    tifffile.imwrite(vendor_path, vendor)
    transcripts = pd.DataFrame(
        {
            "x_location": [2, 5, 0, 20],
            "y_location": [2, 3, 0, 20],
            "feature_name": ["A", "B", "A", "C"],
        }
    )
    transcripts_path = tmp_path / "transcripts.csv"
    transcripts.to_csv(transcripts_path, index=False)
    return mask_path, nuclei_path, vendor_path, transcripts_path


def test_synthetic_dice_iou_and_boundary_f1():
    mask = np.zeros((5, 5), dtype=int)
    mask[1:4, 1:4] = 1
    scores = foreground_scores(mask, mask)
    assert scores == {"dice": 1.0, "iou": 1.0}
    boundary = boundary_scores(mask, mask, tolerance_um=1.0, pixel_size_um=1.0)
    assert boundary == {"precision": 1.0, "recall": 1.0, "f1": 1.0}


def test_all_tiers_synthetic_smoke_and_skip_semantics(tmp_path):
    mask, nuclei, vendor, transcripts = _synthetic_inputs(tmp_path)
    output = tmp_path / "benchmark"
    result = run_all_main(
        [
            "--connected-mask",
            str(mask),
            "--nuclei-mask",
            str(nuclei),
            "--transcripts",
            str(transcripts),
            "--vendor-mask",
            str(vendor),
            "--output-dir",
            str(output),
            "--pixel-size-um",
            "1.0",
        ]
    )
    assert result == 0
    manifest = pd.read_csv(output / "benchmark_manifest.csv")
    assert (
        manifest.loc[manifest.metric_id == "t1.cell_count", "value"].iloc[0] == "2"
        or float(manifest.loc[manifest.metric_id == "t1.cell_count", "value"].iloc[0])
        == 2
    )
    assert (
        manifest.loc[
            manifest.metric_id == "t2.disconnected_cell_fraction_8_neighbor",
            "status",
        ].iloc[0]
        == "computed"
    )
    assert (
        manifest.loc[manifest.metric_id == "t4.foreground_dice", "status"].iloc[0]
        == "computed"
    )
    assert (
        manifest.loc[manifest.metric_id == "t2.assignment_entropy", "status"].iloc[0]
        == "not_applicable_v1"
    )
    assert (
        manifest.loc[manifest.metric_id == "t5.stability", "status"].iloc[0]
        == "skipped_missing_input"
    )
    with (output / "tier2_boundary_summary.json").open() as handle:
        tier2 = json.load(handle)
    assert tier2["instances"]["disconnected_cell_fraction_8_neighbor"] == 0.5


def test_resource_monitor_degrades_and_preserves_nonzero_exit(tmp_path):
    output = tmp_path / "resources"
    result = monitor_main(
        [
            "--output-dir",
            str(output),
            "--interval",
            "0.01",
            "--",
            sys.executable,
            "-c",
            "raise SystemExit(3)",
        ]
    )
    assert result == 3
    assert (output / "resource_samples.csv").exists()
    with (output / "resource_summary.json").open() as handle:
        summary = json.load(handle)
    assert summary["exit_code"] == 3
    assert summary["sample_count"] >= 1


def test_tier5_multiple_standardized_runs_aggregate_mean_and_sd(tmp_path):
    mask, nuclei, _, transcripts = _synthetic_inputs(tmp_path)
    replicate = tmp_path / "replicate" / "benchmark"
    first_result = run_all_main(
        [
            "--connected-mask",
            str(mask),
            "--nuclei-mask",
            str(nuclei),
            "--transcripts",
            str(transcripts),
            "--output-dir",
            str(replicate),
        ]
    )
    assert first_result == 0

    output = tmp_path / "current"
    second_result = run_all_main(
        [
            "--connected-mask",
            str(mask),
            "--nuclei-mask",
            str(nuclei),
            "--transcripts",
            str(transcripts),
            "--replicate-run-dir",
            str(replicate),
            "--output-dir",
            str(output),
        ]
    )
    assert second_result == 0
    with (output / "tier5_stability_summary.json").open() as handle:
        stability = json.load(handle)
    assert stability["run_count"] == 2
    assert stability["stability"]["t1.cell_count"]["mean"] == 2.0
    assert stability["stability"]["t1.cell_count"]["standard_deviation"] == 0.0
