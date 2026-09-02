"""Geometry ambiguity utilities for the V1 BIDCell experiment.

All coordinates in this module are pixel coordinates.  Candidate membership
comes exclusively from BIDCell search-area channels; this module never replaces
that definition with globally nearest nuclei.
"""

from dataclasses import dataclass
from typing import Mapping

import numpy as np
from scipy.ndimage import distance_transform_edt


@dataclass(frozen=True)
class GeometryAmbiguityResult:
    """Patch-level geometry products.

    Spatial arrays use ``[height, width]``.  Candidate arrays use
    ``[height, width, candidates]`` and principal arrays end in an A/B axis.
    Margins and principal IDs are meaningful only where ``overlap_mask`` is
    true; elsewhere they are NaN and -1 respectively.
    """

    candidate_count: np.ndarray
    overlap_mask: np.ndarray
    raw_distances: np.ndarray
    normalized_distances: np.ndarray
    principal_indices: np.ndarray
    principal_ids: np.ndarray
    margin: np.ndarray
    ambiguity_global: np.ndarray
    ambiguity_candidate: np.ndarray


def apply_spatial_transform(
    array: np.ndarray, *, flip_up_down: bool, quarter_turns: int
) -> np.ndarray:
    """Apply one shared spatial transform to every channel in an array."""

    transformed = np.asarray(array)
    if transformed.ndim < 2:
        raise ValueError("array must have at least two spatial dimensions")
    if flip_up_down:
        transformed = np.flip(transformed, axis=0)
    transformed = np.rot90(transformed, k=int(quarter_turns) % 4, axes=(0, 1))
    return np.ascontiguousarray(transformed)


def compute_nucleus_areas(label_mask: np.ndarray) -> np.ndarray:
    """Return an ID-indexed array of full-nucleus pixel areas."""

    labels = np.asarray(label_mask)
    if labels.ndim != 2:
        raise ValueError("label_mask must be a two-dimensional labelled image")
    if np.any(labels < 0):
        raise ValueError("nucleus labels must be non-negative")
    if labels.size == 0:
        return np.zeros(1, dtype=np.int64)
    return np.bincount(labels.astype(np.int64, copy=False).ravel())


def equivalent_radius(area: np.ndarray | float, eps: float = 1.0e-6):
    """Compute ``sqrt(area / pi)`` with a positive numerical floor."""

    values = np.asarray(area, dtype=np.float64)
    return np.sqrt(np.maximum(values, eps) / np.pi)


def _areas_for_ids(
    cell_ids: np.ndarray,
    full_nucleus_areas: np.ndarray | Mapping[int, int],
) -> np.ndarray:
    if isinstance(full_nucleus_areas, Mapping):
        return np.asarray(
            [full_nucleus_areas.get(int(cell_id), 0) for cell_id in cell_ids],
            dtype=np.float64,
        )
    areas = np.asarray(full_nucleus_areas)
    result = np.zeros(len(cell_ids), dtype=np.float64)
    valid = (cell_ids >= 0) & (cell_ids < len(areas))
    result[valid] = areas[cell_ids[valid].astype(np.int64)]
    return result


def compute_geometry_ambiguity(
    nucleus_masks: np.ndarray,
    search_areas: np.ndarray,
    transcript_counts: np.ndarray,
    cell_ids: np.ndarray,
    full_nucleus_areas: np.ndarray | Mapping[int, int],
    *,
    tau: float = 0.15,
    eps: float = 1.0e-6,
    mode: str = "overlap_geometry",
    raw_distances: np.ndarray | None = None,
) -> GeometryAmbiguityResult:
    """Compute geometry ambiguity for one BIDCell patch.

    The nearest-nucleus distance is Euclidean distance to a nucleus mask pixel,
    never a centroid distance. When ``raw_distances`` is omitted it is computed
    from the supplied masks; BIDCell's dataset supplies distances queried from
    the full nuclei image. Radius normalization uses the supplied full-image
    nucleus areas. Candidates A/B are the two smallest normalized distances,
    with channel order breaking ties deterministically; callers should therefore
    supply deterministically sorted ``cell_ids`` (the dataset uses
    ``numpy.unique`` order).
    """

    nuclei = np.asarray(nucleus_masks, dtype=bool)
    candidates = np.asarray(search_areas, dtype=bool)
    transcripts = np.asarray(transcript_counts)
    ids = np.asarray(cell_ids, dtype=np.int64)

    if nuclei.ndim != 3 or candidates.shape != nuclei.shape:
        raise ValueError("nucleus_masks and search_areas must share [H, W, C]")
    if transcripts.shape != nuclei.shape[:2]:
        raise ValueError("transcript_counts must have shape [H, W]")
    if len(ids) != nuclei.shape[2]:
        raise ValueError("cell_ids length must match the candidate axis")
    if mode not in {"overlap_only", "overlap_geometry"}:
        raise ValueError("mode must be overlap_only or overlap_geometry")
    if not 0 <= tau <= 1:
        raise ValueError("tau must be between 0 and 1")
    if eps <= 0:
        raise ValueError("eps must be positive")

    height, width, n_candidates = nuclei.shape
    candidate_count = candidates.sum(axis=2, dtype=np.int32)
    overlap_mask = (transcripts > 0) & (candidate_count >= 2)
    if raw_distances is None:
        raw_distance_values = np.full(
            (height, width, n_candidates), np.inf, dtype=np.float32
        )
    else:
        raw_distance_values = np.asarray(raw_distances, dtype=np.float32)
        if raw_distance_values.shape != nuclei.shape:
            raise ValueError("raw_distances must have shape [H, W, C]")
        raw_distance_values = raw_distance_values.copy()
    normalized_distances = np.full_like(raw_distance_values, np.inf)

    full_areas = _areas_for_ids(ids, full_nucleus_areas)
    patch_areas = nuclei.sum(axis=(0, 1), dtype=np.int64)
    # A missing area lookup is recoverable for generic callers.  BIDCell's data
    # path supplies full-image areas, so this fallback is mainly test-friendly.
    full_areas = np.where(full_areas > 0, full_areas, patch_areas)
    radii = equivalent_radius(full_areas, eps=eps)

    for candidate_index in range(n_candidates):
        if raw_distances is None:
            if not nuclei[:, :, candidate_index].any():
                continue
            distance = distance_transform_edt(~nuclei[:, :, candidate_index])
            raw_distance_values[:, :, candidate_index] = distance.astype(np.float32)
        normalized_distances[:, :, candidate_index] = (
            raw_distance_values[:, :, candidate_index] / (radii[candidate_index] + eps)
        ).astype(np.float32)

    eligible_distances = np.where(candidates, normalized_distances, np.inf)
    principal_indices = np.full((height, width, 2), -1, dtype=np.int64)
    principal_ids = np.full((height, width, 2), -1, dtype=np.int64)
    margin = np.full((height, width), np.nan, dtype=np.float32)

    if n_candidates >= 2 and overlap_mask.any():
        # Stable sorting preserves candidate-channel order for exact ties.
        order = np.argsort(eligible_distances, axis=2, kind="stable")[:, :, :2]
        principal_indices[overlap_mask] = order[overlap_mask]
        principal_ids[overlap_mask] = ids[order[overlap_mask]]
        first = np.take_along_axis(eligible_distances, order[:, :, :1], axis=2)[:, :, 0]
        second = np.take_along_axis(eligible_distances, order[:, :, 1:2], axis=2)[
            :, :, 0
        ]
        first_overlap = first[overlap_mask]
        second_overlap = second[overlap_mask]
        values = np.abs(first_overlap - second_overlap) / (
            first_overlap + second_overlap + eps
        )
        margin[overlap_mask] = values.astype(np.float32)

    if mode == "overlap_only":
        ambiguity_global = overlap_mask
    else:
        ambiguity_global = overlap_mask & (margin < tau)
    ambiguity_candidate = candidates & ambiguity_global[:, :, np.newaxis]

    return GeometryAmbiguityResult(
        candidate_count=candidate_count,
        overlap_mask=overlap_mask,
        raw_distances=raw_distance_values,
        normalized_distances=normalized_distances,
        principal_indices=principal_indices,
        principal_ids=principal_ids,
        margin=margin,
        ambiguity_global=ambiguity_global,
        ambiguity_candidate=ambiguity_candidate,
    )


def build_ambiguity_weights(
    ambiguity_mask: np.ndarray,
    lambda_boundary: float,
    *,
    mean_normalize: bool = True,
    support: np.ndarray | None = None,
    eps: float = 1.0e-6,
) -> np.ndarray:
    """Build ``1 + lambda * ambiguity`` weights with safe normalization."""

    ambiguity = np.asarray(ambiguity_mask, dtype=np.float32)
    weights = 1.0 + float(lambda_boundary) * ambiguity
    domain = np.ones_like(ambiguity, dtype=bool)
    if support is not None:
        domain = np.asarray(support, dtype=bool)
        if domain.shape != ambiguity.shape:
            raise ValueError("support and ambiguity_mask must have equal shapes")
    if mean_normalize and domain.any():
        mean_weight = float(weights[domain].mean())
        if np.isfinite(mean_weight) and mean_weight > eps:
            weights = weights.copy()
            weights[domain] = weights[domain] / mean_weight
    return weights.astype(np.float32, copy=False)


def geometry_statistics(
    result: GeometryAmbiguityResult, transcript_counts: np.ndarray
) -> dict[str, float | int]:
    """Return JSON-friendly patch diagnostics."""

    counts = np.asarray(transcript_counts, dtype=np.float64)
    transcript_pixels = counts > 0
    ambiguous = result.ambiguity_global
    pairs = result.principal_ids[ambiguous]
    unique_pairs = set()
    for pair in pairs:
        unique_pairs.add(tuple(sorted((int(pair[0]), int(pair[1])))))
    total_transcripts = float(counts[transcript_pixels].sum())
    ambiguous_transcripts = float(counts[ambiguous].sum())
    return {
        "transcript_count": total_transcripts,
        "ambiguous_transcript_count": ambiguous_transcripts,
        "ambiguous_transcript_fraction": (
            ambiguous_transcripts / total_transcripts if total_transcripts else 0.0
        ),
        "transcript_pixel_count": int(transcript_pixels.sum()),
        "ambiguous_pixel_count": int(ambiguous.sum()),
        "ambiguous_pixel_fraction": (
            float(ambiguous.sum() / transcript_pixels.sum())
            if transcript_pixels.any()
            else 0.0
        ),
        "candidate_overlap_pixel_count": int(result.overlap_mask.sum()),
        "selected_candidate_pair_count": len(unique_pairs),
    }
