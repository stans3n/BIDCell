import json

import numpy as np
import pytest

from bidcell.model.utils.geometry_ambiguity import (
    apply_spatial_transform,
    compute_geometry_ambiguity,
    compute_nucleus_areas,
)
from bidcell.model.utils.patch_manifest import (
    deterministic_patch_records,
    filter_patch_paths,
    save_patch_manifest,
)


def _two_nuclei_geometry(transcript_column):
    labels = np.zeros((7, 9), dtype=np.int32)
    labels[3, 1] = 1
    labels[3, 7] = 2
    nuclei = np.stack((labels == 1, labels == 2), axis=-1)
    search = np.zeros_like(nuclei)
    search[3, transcript_column, :] = True
    transcripts = np.zeros(labels.shape, dtype=np.float32)
    transcripts[3, transcript_column] = 1
    return compute_geometry_ambiguity(
        nuclei,
        search,
        transcripts,
        np.array([1, 2]),
        compute_nucleus_areas(labels),
        tau=0.15,
    )


def test_symmetric_nuclei_have_zero_margin_and_are_ambiguous():
    result = _two_nuclei_geometry(4)
    assert result.margin[3, 4] == pytest.approx(0.0)
    assert result.ambiguity_global[3, 4]


def test_transcript_closer_to_a_has_larger_margin():
    symmetric = _two_nuclei_geometry(4)
    closer = _two_nuclei_geometry(2)
    assert closer.margin[3, 2] > symmetric.margin[3, 4]
    assert not closer.ambiguity_global[3, 2]


def test_single_candidate_and_no_overlap_are_not_ambiguous():
    result = _two_nuclei_geometry(4)
    search = np.zeros((7, 9, 2), dtype=bool)
    search[3, 4, 0] = True
    transcripts = np.zeros((7, 9), dtype=np.float32)
    transcripts[3, 4] = 1
    nuclei = np.zeros_like(search)
    nuclei[3, 1, 0] = True
    nuclei[3, 7, 1] = True
    no_overlap = compute_geometry_ambiguity(
        nuclei, search, transcripts, np.array([1, 2]), np.array([0, 1, 1])
    )
    assert no_overlap.candidate_count[3, 4] == 1
    assert not no_overlap.ambiguity_global[3, 4]
    assert not result.ambiguity_global[0, 0]


def test_equivalent_radius_uses_full_nucleus_area():
    labels = np.zeros((9, 11), dtype=np.int32)
    labels[4, 1] = 1
    labels[3:5, 8:10] = 2
    nuclei = np.stack((labels == 1, labels == 2), axis=-1)
    search = np.zeros_like(nuclei)
    search[4, 5, :] = True
    transcripts = np.zeros(labels.shape)
    transcripts[4, 5] = 1
    result = compute_geometry_ambiguity(
        nuclei,
        search,
        transcripts,
        np.array([1, 2]),
        compute_nucleus_areas(labels),
    )
    raw = result.raw_distances[4, 5]
    normalized = result.normalized_distances[4, 5]
    assert normalized[0] == pytest.approx(raw[0] / np.sqrt(1 / np.pi), rel=1e-5)
    assert normalized[1] == pytest.approx(raw[1] / np.sqrt(4 / np.pi), rel=1e-5)


def test_supplied_full_area_controls_normalization_for_cropped_nucleus():
    nuclei = np.zeros((5, 5, 2), dtype=bool)
    nuclei[2, 0, 0] = True
    nuclei[2, 4, 1] = True
    search = np.zeros_like(nuclei)
    search[2, 2, :] = True
    transcripts = np.zeros((5, 5))
    transcripts[2, 2] = 1
    result = compute_geometry_ambiguity(
        nuclei,
        search,
        transcripts,
        np.array([1, 2]),
        np.array([0, 9, 1]),
    )
    assert result.normalized_distances[2, 2, 0] == pytest.approx(
        result.raw_distances[2, 2, 0] / np.sqrt(9 / np.pi), rel=1e-5
    )


def test_three_candidates_choose_two_closest_deterministically():
    labels = np.zeros((7, 7), dtype=np.int32)
    labels[3, 1] = 10
    labels[3, 5] = 20
    labels[0, 3] = 30
    nuclei = np.stack([labels == value for value in (10, 20, 30)], axis=-1)
    search = np.zeros_like(nuclei)
    search[3, 3, :] = True
    transcripts = np.zeros(labels.shape)
    transcripts[3, 3] = 1
    result = compute_geometry_ambiguity(
        nuclei,
        search,
        transcripts,
        np.array([10, 20, 30]),
        compute_nucleus_areas(labels),
    )
    assert result.candidate_count[3, 3] == 3
    assert result.principal_ids[3, 3].tolist() == [10, 20]


def test_shared_spatial_transform_preserves_channel_alignment():
    stacked = np.zeros((5, 5, 3), dtype=np.int32)
    stacked[1, 3, :] = [2, 5, 9]
    transformed = apply_spatial_transform(stacked, flip_up_down=True, quarter_turns=3)
    locations = [np.argwhere(transformed[:, :, channel] > 0) for channel in range(3)]
    assert np.array_equal(locations[0], locations[1])
    assert np.array_equal(locations[1], locations[2])


def test_actual_dataset_augmentation_keeps_channels_aligned(monkeypatch):
    pytest.importorskip("cv2")
    pytest.importorskip("imgaug")
    from bidcell.model.dataio.dataset_input import DataProcessing

    values = iter((np.array([1]), np.array([3])))
    monkeypatch.setattr(np.random, "randint", lambda *args, **kwargs: next(values))
    stacked = np.zeros((5, 5, 2), dtype=np.int32)
    stacked[1, 3, :] = 1
    augmented, _, _ = DataProcessing.__new__(DataProcessing).augment_data(stacked)
    assert np.array_equal(augmented[0, :, :, 0], augmented[0, :, :, 1])


def test_manifest_selection_is_deterministic_and_serializable(tmp_path):
    paths = [f"/patches/{row}_{column}.hdf5" for row in range(3) for column in range(2)]
    first = deterministic_patch_records(paths, seed=7, count=3)
    second = deterministic_patch_records(reversed(paths), seed=7, count=3)
    assert first == second
    manifest = tmp_path / "patches.json"
    save_patch_manifest(manifest, first, seed=7)
    payload = json.loads(manifest.read_text())
    assert payload["seed"] == 7
    assert len(filter_patch_paths(paths, manifest)) == 3
