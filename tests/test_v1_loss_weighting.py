import pytest
import torch
import yaml

from bidcell.config import V1Params, load_config
from bidcell.model.model.losses import (
    CellCallingLoss,
    OverlapLoss,
    PosNegMarkerLoss,
    build_ambiguity_weights,
)


@pytest.fixture
def loss_inputs():
    torch.manual_seed(11)
    prediction = torch.randn(3, 2, 5, 6)
    nuclei = torch.zeros(3, 1, 5, 6, dtype=torch.long)
    nuclei[0, 0, 1:3, 1] = 1
    nuclei[1, 0, 1:3, 4] = 1
    nuclei[2, 0, 3, 2:4] = 1
    search = torch.zeros_like(nuclei)
    search[:, :, 1:4, 1:5] = 1
    positive = torch.zeros_like(nuclei)
    positive[:, :, 2, 2] = 1
    negative = torch.zeros_like(nuclei)
    negative[:, :, 3, 3] = 1
    ambiguity_candidate = torch.zeros(3, 5, 6, dtype=torch.bool)
    ambiguity_candidate[:, 2, 2] = True
    ambiguity_global = ambiguity_candidate.any(dim=0)
    return (
        prediction,
        nuclei,
        search,
        positive,
        negative,
        ambiguity_candidate,
        ambiguity_global,
    )


def test_default_v1_is_disabled_and_cellcalling_not_weighted():
    settings = V1Params()
    assert not settings.enabled
    assert not settings.weight_cellcalling


def test_unweighted_losses_match_baseline_formulas(loss_inputs):
    prediction, nuclei, search, positive, negative, _, _ = loss_inputs
    device = torch.device("cpu")

    batch_n = nuclei[:, 0]
    probabilities = torch.softmax(prediction, dim=1)
    all_nuclei = batch_n.sum(dim=0)
    all_not_nuclei = torch.ones(batch_n.shape) - all_nuclei
    cytoplasm_probabilities = probabilities[:, 1] * all_not_nuclei
    predicted_cytoplasm = torch.sigmoid(cytoplasm_probabilities - 0.5)
    overlap_penalty = torch.relu(predicted_cytoplasm.sum(dim=0) - all_not_nuclei)
    expected_overlap = overlap_penalty.sum() / (
        prediction.shape[0] * prediction.shape[2] * prediction.shape[3]
    )
    actual_overlap = OverlapLoss(1.0, device)(prediction, nuclei)
    assert torch.equal(actual_overlap, expected_overlap)

    expected_positive = torch.nn.functional.cross_entropy(
        prediction, positive[:, 0], reduction="sum"
    )
    predicted_cells = torch.sigmoid(probabilities[:, 1] - 0.5)
    expected_negative = (predicted_cells * negative[:, 0]).sum()
    expected_marker = (1.3 * expected_positive + 0.7 * expected_negative) / len(
        prediction
    )
    actual_marker = PosNegMarkerLoss(1.3, 0.7, device)(prediction, positive, negative)
    assert torch.equal(actual_marker, expected_marker)

    expected_calling = torch.nn.functional.cross_entropy(
        prediction, search[:, 0], reduction="none"
    ).sum() / len(prediction)
    actual_calling = CellCallingLoss(1.0, device)(prediction, search)
    assert torch.equal(actual_calling, expected_calling)


def test_lambda_zero_and_zero_ambiguity_preserve_overlap(loss_inputs):
    prediction, nuclei, _, _, _, _, ambiguity_global = loss_inputs
    criterion = OverlapLoss(1.0, torch.device("cpu"))
    original = criterion(prediction, nuclei)
    lambda_zero = criterion(prediction, nuclei, ambiguity_global, lambda_boundary=0.0)
    all_zero = criterion(
        prediction,
        nuclei,
        torch.zeros_like(ambiguity_global),
        lambda_boundary=2.0,
    )
    assert torch.equal(original, lambda_zero)
    assert torch.equal(original, all_zero)


def test_lambda_zero_and_zero_ambiguity_preserve_marker_loss(loss_inputs):
    prediction, _, _, positive, negative, ambiguity_candidate, _ = loss_inputs
    criterion = PosNegMarkerLoss(1.3, 0.7, torch.device("cpu"))
    original = criterion(prediction, positive, negative)
    lambda_zero = criterion(
        prediction,
        positive,
        negative,
        ambiguity_candidate,
        lambda_boundary=0.0,
    )
    all_zero = criterion(
        prediction,
        positive,
        negative,
        torch.zeros_like(ambiguity_candidate),
        lambda_boundary=2.0,
    )
    assert torch.equal(original, lambda_zero)
    assert torch.equal(original, all_zero)


def test_normalized_weights_have_unit_mean():
    ambiguity = torch.tensor([[False, True], [True, False]])
    weights = build_ambiguity_weights(ambiguity, 2.0, mean_normalize=True)
    assert weights.mean().item() == pytest.approx(1.0)


def test_cellcalling_is_unchanged_without_explicit_mask(loss_inputs):
    prediction, _, search, _, _, ambiguity_candidate, _ = loss_inputs
    criterion = CellCallingLoss(1.0, torch.device("cpu"))
    original = criterion(prediction, search)
    disabled = criterion(prediction, search, ambiguity_mask=None, lambda_boundary=5.0)
    assert torch.equal(original, disabled)
    weighted = criterion(prediction, search, ambiguity_candidate, lambda_boundary=1.0)
    assert torch.isfinite(weighted)


def test_empty_ambiguity_and_marker_support_do_not_create_nan(loss_inputs):
    prediction, _, _, positive, negative, ambiguity_candidate, _ = loss_inputs
    criterion = PosNegMarkerLoss(1.0, 1.0, torch.device("cpu"))
    loss = criterion(
        prediction,
        torch.zeros_like(positive),
        torch.zeros_like(negative),
        torch.zeros_like(ambiguity_candidate),
        lambda_boundary=5.0,
    )
    assert torch.isfinite(loss)


def test_old_yaml_without_v1_loads_with_disabled_defaults(tmp_path):
    existing = tmp_path / "exists"
    existing.write_text("placeholder")
    raw = {
        "files": {
            "data_dir": str(tmp_path),
            "fp_transcripts": str(existing),
            "fp_ref": str(existing),
            "fp_pos_markers": str(existing),
            "fp_neg_markers": str(existing),
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
        "model_params": {"patch_size": 48, "elongated": []},
        "training_params": {},
        "testing_params": {},
        "cpus": 1,
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw))
    config = load_config(str(path))
    assert config.v1 == V1Params()
