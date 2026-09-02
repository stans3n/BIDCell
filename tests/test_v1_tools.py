import json

import numpy as np
import yaml

from scripts.v1.locality_eval import locality_metrics
from scripts.v1.make_dev_configs import generate_configs
from scripts.v1.threshold_sweep import threshold_summary


def test_locality_unchanged_masks_and_label_permutation():
    baseline = np.zeros((10, 10), dtype=np.int32)
    baseline[1:4, 1:4] = 1
    baseline[6:9, 6:9] = 2
    permuted = np.where(baseline == 1, 9, np.where(baseline == 2, 4, 0))
    summary, matches, changed, _ = locality_metrics(baseline, permuted)
    assert summary["changed_pixel_fraction"] == 0
    assert summary["changed_cell_fraction"] == 0
    assert not changed.any()
    assert sorted(item["iou"] for item in matches) == [1.0, 1.0]


def test_locality_distinguishes_boundary_and_interior_changes():
    baseline = np.zeros((15, 15), dtype=np.int32)
    baseline[2:13, 2:13] = 1
    boundary_change = baseline.copy()
    boundary_change[2, 7] = 0
    boundary_summary, _, _, _ = locality_metrics(
        baseline, boundary_change, bands_um=(1,)
    )
    assert boundary_summary["boundary_band_1um_change_fraction"] > 0
    assert boundary_summary["non_boundary_interior_preservation"] == 1.0

    interior_change = baseline.copy()
    interior_change[7, 7] = 0
    interior_summary, _, _, _ = locality_metrics(
        baseline, interior_change, bands_um=(1,)
    )
    assert interior_summary["non_boundary_interior_preservation"] < 1.0


def test_threshold_summary_uses_transcript_counts_and_unique_pairs():
    records = {
        "margin": [0.05, 0.2, 0.1],
        "transcript_count": [2, 1, 3],
        "candidate_count": [2, 2, 3],
        "principal_a_id": [1, 1, 2],
        "principal_b_id": [2, 2, 3],
        "all_transcript_count": 10,
        "all_transcript_pixel_count": 5,
    }
    summary = threshold_summary(records, 0.15)
    assert summary["ambiguous_transcript_count"] == 5
    assert summary["ambiguous_transcript_ratio"] == 0.5
    assert summary["ambiguous_pixel_ratio"] == 0.4
    assert summary["candidate_pair_count"] == 2


def test_dev_configs_cover_e3_e4_e5_and_use_distinct_outputs(tmp_path):
    baseline = {
        "training_params": {"total_steps": 4000},
        "testing_params": {"test_step": 4000},
        "experiment_dirs": {},
    }
    output_dir = tmp_path / "configs"
    run_root = tmp_path / "runs"
    generated = generate_configs(
        baseline,
        output_dir,
        steps=1000,
        experiment_output_root=run_root,
        patch_manifest="/externally/supplied/patch_manifest.json",
    )
    assert len(generated) == 11
    assert len({item["experiment_output"] for item in generated}) == 11
    names = {item["name"] for item in generated}
    assert {"e3_overlap_only", "e3_overlap_geometry"}.issubset(names)
    assert "e4_cellcalling_overlap_marker_weighted" in names
    assert {
        "e5_lambda_0p5",
        "e5_lambda_1p0",
        "e5_lambda_2p0",
        "e5_lambda_5p0",
    }.issubset(names)
    loaded = yaml.safe_load((output_dir / "e4_official_bidcell.yaml").read_text())
    assert loaded["training_params"]["total_steps"] == 1000
    assert loaded["v1"]["enabled"] is False
    assert loaded["v1"]["patch_manifest"].endswith("patch_manifest.json")
    assert loaded["experiment_dirs"]["output_root"] == str(run_root)


def test_dev_config_manifest_content_is_json_serializable(tmp_path):
    payload = {"configs": [{"name": "example", "config": "a.yaml"}]}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload))
    assert json.loads(path.read_text()) == payload
