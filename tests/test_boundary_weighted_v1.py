import csv

import pytest
import torch
from pydantic import ValidationError

from bidcell.config import Config, TrainingParams
from bidcell.model.model.losses import CellCallingLoss, OverlapLoss, PosNegMarkerLoss
from bidcell.model.training_metrics import (
    TRAINING_METRIC_FIELDS,
    TrainingMetricsLogger,
    build_candidate_overlap_weight,
)


def _inputs(n_candidates=2):
    torch.manual_seed(3)
    logits = torch.randn(n_candidates, 2, 4, 4, requires_grad=True)
    nuclei = torch.zeros(n_candidates, 1, 4, 4, dtype=torch.long)
    positive = torch.zeros_like(nuclei)
    positive[:, :, 1:3, 1:3] = 1
    negative = torch.zeros_like(nuclei)
    negative[:, :, 0:2, 0:2] = 1
    search = torch.zeros_like(nuclei)
    search[0, :, :3, :3] = 1
    if n_candidates > 1:
        search[1, :, 1:, 1:] = 1
    if n_candidates > 2:
        search[2:, :, 2:, :2] = 1
    return logits, nuclei, search, positive, negative


def test_training_params_defaults_and_negative_weight():
    old = TrainingParams()
    assert old.boundary_weighting_enabled is False
    assert old.boundary_weight == 0.0
    assert old.boundary_weight_cell_calling is False
    v1 = TrainingParams(boundary_weighting_enabled=True, boundary_weight=1.0)
    assert v1.boundary_weight_overlap is True
    with pytest.raises(ValidationError):
        TrainingParams(boundary_weight=-0.1)


def test_old_and_v1_full_config_models_load(tmp_path):
    required_files = {}
    for name in ("data", "dapi", "transcripts", "reference", "positive", "negative"):
        path = tmp_path / name
        if name == "data":
            path.mkdir()
        else:
            path.touch()
        required_files[name] = str(path)
    payload = {
        "files": {
            "data_dir": required_files["data"],
            "fp_dapi": required_files["dapi"],
            "fp_transcripts": required_files["transcripts"],
            "fp_ref": required_files["reference"],
            "fp_pos_markers": required_files["positive"],
            "fp_neg_markers": required_files["negative"],
        },
        "nuclei_fovs": {"stitch_nuclei_fovs": False},
        "nuclei": {},
        "transcripts": {"transcripts_to_filter": []},
        "affine": {
            "base_pix_x": 1.0,
            "base_pix_y": 1.0,
            "base_ts_x": 1.0,
            "base_ts_y": 1.0,
        },
        "model_params": {"patch_size": 8, "elongated": []},
        "training_params": {},
        "testing_params": {},
        "cpus": 1,
    }
    old = Config(**payload)
    assert old.training_params.boundary_weighting_enabled is False
    payload["training_params"] = {
        "boundary_weighting_enabled": True,
        "boundary_weight": 1.0,
    }
    v1 = Config(**payload)
    assert v1.training_params.boundary_weight == 1.0


@pytest.mark.parametrize("n_candidates", [1, 2, 4])
def test_candidate_overlap_shape_values_and_candidate_counts(n_candidates):
    _, _, search, _, _ = _inputs(n_candidates)
    overlap, weight = build_candidate_overlap_weight(search, 2.5)
    assert overlap.shape == (4, 4)
    assert weight.shape == (4, 4)
    assert torch.all(weight[overlap == 0] == 1.0)
    assert torch.all(weight[overlap == 1] == 3.5)
    if n_candidates == 1:
        assert overlap.sum() == 0


def test_zero_weight_and_no_overlap_are_backward_compatible():
    logits, nuclei, search, positive, negative = _inputs()
    overlap_loss = OverlapLoss(1.0, torch.device("cpu"))
    marker_loss = PosNegMarkerLoss(1.0, 1.0, torch.device("cpu"))
    cell_calling_loss = CellCallingLoss(1.0, torch.device("cpu"))

    _, zero_weight = build_candidate_overlap_weight(search, 0.0)
    assert torch.allclose(
        overlap_loss(logits, nuclei),
        overlap_loss(logits, nuclei, spatial_weight=zero_weight),
    )
    assert torch.allclose(
        marker_loss(logits, positive, negative),
        marker_loss(logits, positive, negative, spatial_weight=zero_weight),
    )
    assert torch.allclose(
        cell_calling_loss(logits, search),
        cell_calling_loss(logits, search, spatial_weight=zero_weight),
    )

    no_overlap = torch.zeros_like(search)
    no_overlap[0, :, :, :2] = 1
    no_overlap[1, :, :, 2:] = 1
    overlap, unit_weight = build_candidate_overlap_weight(no_overlap, 5.0)
    assert overlap.sum() == 0
    assert torch.allclose(
        marker_loss(logits, positive, negative),
        marker_loss(logits, positive, negative, spatial_weight=unit_weight),
    )


def test_synthetic_overlap_increases_selected_losses_and_gradients_are_finite():
    logits, nuclei, search, positive, negative = _inputs()
    _, spatial_weight = build_candidate_overlap_weight(search, 2.0)
    overlap_loss = OverlapLoss(1.0, torch.device("cpu"))
    marker_loss = PosNegMarkerLoss(1.0, 1.0, torch.device("cpu"))
    cell_calling_loss = CellCallingLoss(1.0, torch.device("cpu"))

    baseline_overlap = overlap_loss(logits, nuclei)
    weighted_overlap = overlap_loss(logits, nuclei, spatial_weight=spatial_weight)
    assert weighted_overlap > baseline_overlap

    baseline_marker, baseline_diag = marker_loss(
        logits, positive, negative, return_diagnostics=True
    )
    weighted_marker, weighted_diag = marker_loss(
        logits,
        positive,
        negative,
        spatial_weight=spatial_weight,
        return_diagnostics=True,
    )
    assert weighted_marker > baseline_marker
    assert weighted_diag["loss_pos"] > baseline_diag["loss_pos"]
    assert weighted_diag["loss_neg"] > baseline_diag["loss_neg"]

    positive_only, positive_only_diag = marker_loss(
        logits,
        positive,
        negative,
        spatial_weight=spatial_weight,
        weight_positive=True,
        weight_negative=False,
        return_diagnostics=True,
    )
    assert positive_only_diag["loss_pos"] > baseline_diag["loss_pos"]
    assert torch.allclose(positive_only_diag["loss_neg"], baseline_diag["loss_neg"])

    total = (
        weighted_overlap
        + weighted_marker
        + cell_calling_loss(logits, search, spatial_weight=spatial_weight)
    )
    total.backward()
    assert torch.isfinite(total)
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_training_metrics_header_and_scalar_only_rows(tmp_path):
    logger = TrainingMetricsLogger(tmp_path, flush_interval=1)
    logger.write(
        {
            "global_step": torch.tensor(0),
            "epoch": 1,
            "batch_index": 0,
            "n_candidates": 2,
            "total_loss": torch.tensor(1.25, requires_grad=True),
        }
    )
    logger.close()
    with (tmp_path / "v1_training_metrics.csv").open(newline="") as handle:
        reader = csv.reader(handle)
        assert next(reader) == TRAINING_METRIC_FIELDS
        assert len(next(reader)) == len(TRAINING_METRIC_FIELDS)

    logger = TrainingMetricsLogger(tmp_path)
    with pytest.raises(ValueError):
        logger.write({"total_loss": torch.ones(2)})
    logger.close(status="failed", error="expected test error")
