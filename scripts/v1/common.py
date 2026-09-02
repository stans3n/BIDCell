"""Shared helpers for V1 diagnostic CLIs."""

from pathlib import Path
import sys

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


def as_numpy(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def deterministic_indices(length: int, count: int, seed: int = 1234) -> list[int]:
    if length <= 0 or count <= 0:
        return []
    generator = np.random.default_rng(seed)
    selected = generator.choice(length, size=min(length, count), replace=False)
    return sorted(int(value) for value in selected)


def geometry_from_dataset_sample(
    sample, full_nucleus_areas, config, *, tau=None, dataset=None
):
    """Reconstruct patch geometry from the stable non-training tuple."""

    from bidcell.model.utils.geometry_ambiguity import compute_geometry_ambiguity

    nucleus_masks = as_numpy(sample[1]).astype(bool)
    search_areas = as_numpy(sample[2]).astype(bool)
    nuclei_labels = as_numpy(sample[7])
    transcript_counts = as_numpy(sample[8])
    cell_ids = np.unique(nuclei_labels).astype(np.int64)
    cell_ids = cell_ids[cell_ids != 0]
    raw_distances = None
    if dataset is not None:
        raw_distances = dataset.full_nucleus_distance_maps(
            int(sample[5]),
            int(sample[6]),
            cell_ids,
            transcript_counts,
        )
    geometry = compute_geometry_ambiguity(
        nucleus_masks,
        search_areas,
        transcript_counts,
        cell_ids,
        full_nucleus_areas,
        tau=config.v1.tau if tau is None else tau,
        eps=config.v1.eps,
        mode=config.v1.ambiguity_mode,
        raw_distances=raw_distances,
    )
    return geometry, cell_ids, transcript_counts
