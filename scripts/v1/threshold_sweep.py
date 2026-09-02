"""Evaluate V1 ambiguity thresholds on user-supplied BIDCell data."""

import argparse
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sweep geometry-margin thresholds without training BIDCell."
    )
    parser.add_argument("--config", required=True, help="BIDCell YAML configuration")
    parser.add_argument("--taus", required=True, type=float, nargs="+")
    parser.add_argument("--output-dir", required=True)
    return parser


def threshold_summary(records, tau: float) -> dict[str, float | int]:
    """Summarize flattened overlap-geometry records for one threshold."""

    import numpy as np

    margins = np.asarray(records["margin"], dtype=float)
    counts = np.asarray(records["transcript_count"], dtype=float)
    candidate_counts = np.asarray(records["candidate_count"], dtype=float)
    selected = margins < tau
    total_transcripts = float(records["all_transcript_count"])
    overlap_transcripts = float(counts.sum())
    ambiguous_count = float(counts[selected].sum())
    pairs = {
        tuple(sorted((int(a), int(b))))
        for a, b, keep in zip(
            records["principal_a_id"], records["principal_b_id"], selected
        )
        if keep
    }
    return {
        "tau": float(tau),
        "ambiguous_transcript_count": ambiguous_count,
        "ambiguous_transcript_ratio": (
            ambiguous_count / total_transcripts if total_transcripts else 0.0
        ),
        "ambiguous_pixel_ratio": (
            float(selected.sum() / records["all_transcript_pixel_count"])
            if records["all_transcript_pixel_count"]
            else 0.0
        ),
        "candidate_pair_count": len(pairs),
        "mean_candidates_per_selected_transcript": (
            float(np.average(candidate_counts[selected], weights=counts[selected]))
            if selected.any() and counts[selected].sum() > 0
            else 0.0
        ),
        "overlap_region_coverage": (
            ambiguous_count / overlap_transcripts if overlap_transcripts else 0.0
        ),
    }


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    from bidcell.config import load_config
    from bidcell.model.dataio.dataset_input import DataProcessing
    from bidcell.model.utils.geometry_ambiguity import compute_nucleus_areas
    from scripts.v1.common import geometry_from_dataset_sample

    if any(tau < 0 or tau > 1 for tau in args.taus):
        raise ValueError("all tau values must be between 0 and 1")
    config = load_config(args.config)
    destination = Path(args.output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    dataset = DataProcessing(config, isTraining=False, all_patches=True)
    full_areas = compute_nucleus_areas(dataset.nuclei)

    flattened = {
        "margin": [],
        "transcript_count": [],
        "candidate_count": [],
        "principal_a_id": [],
        "principal_b_id": [],
        "principal_a_distance": [],
        "principal_b_distance": [],
        "principal_a_normalized_distance": [],
        "principal_b_normalized_distance": [],
        "all_transcript_count": 0.0,
        "all_transcript_pixel_count": 0,
    }
    transcript_by_coordinate = {}
    for index in range(len(dataset)):
        sample = dataset[index]
        geometry, _, transcript_counts = geometry_from_dataset_sample(
            sample, full_areas, config, dataset=dataset
        )
        row_start, column_start = int(sample[5]), int(sample[6])
        transcript_rows, transcript_columns = np.where(transcript_counts > 0)
        for row, column in zip(transcript_rows, transcript_columns):
            coordinate = (row_start + int(row), column_start + int(column))
            border_clearance = min(
                int(row),
                int(column),
                transcript_counts.shape[0] - 1 - int(row),
                transcript_counts.shape[1] - 1 - int(column),
            )
            values = {
                "patch_index": index,
                "y": coordinate[0],
                "x": coordinate[1],
                "patch_border_clearance": border_clearance,
                "margin": None,
                "transcript_count": float(transcript_counts[row, column]),
                "candidate_count": int(geometry.candidate_count[row, column]),
                "principal_a_id": None,
                "principal_b_id": None,
                "principal_a_distance": None,
                "principal_b_distance": None,
                "principal_a_normalized_distance": None,
                "principal_b_normalized_distance": None,
            }
            if geometry.overlap_mask[row, column]:
                principal = geometry.principal_indices[row, column]
                values.update(
                    {
                        "margin": float(geometry.margin[row, column]),
                        "principal_a_id": int(geometry.principal_ids[row, column, 0]),
                        "principal_b_id": int(geometry.principal_ids[row, column, 1]),
                        "principal_a_distance": float(
                            geometry.raw_distances[row, column, principal[0]]
                        ),
                        "principal_b_distance": float(
                            geometry.raw_distances[row, column, principal[1]]
                        ),
                        "principal_a_normalized_distance": float(
                            geometry.normalized_distances[row, column, principal[0]]
                        ),
                        "principal_b_normalized_distance": float(
                            geometry.normalized_distances[row, column, principal[1]]
                        ),
                    }
                )
            previous = transcript_by_coordinate.get(coordinate)
            if (
                previous is None
                or border_clearance > previous["patch_border_clearance"]
            ):
                transcript_by_coordinate[coordinate] = values

    flattened["all_transcript_count"] = float(
        sum(values["transcript_count"] for values in transcript_by_coordinate.values())
    )
    flattened["all_transcript_pixel_count"] = len(transcript_by_coordinate)
    raw_rows = [
        transcript_by_coordinate[key]
        for key in sorted(transcript_by_coordinate)
        if transcript_by_coordinate[key]["candidate_count"] >= 2
    ]
    for values in raw_rows:
        for key in (
            "margin",
            "transcript_count",
            "candidate_count",
            "principal_a_id",
            "principal_b_id",
            "principal_a_distance",
            "principal_b_distance",
            "principal_a_normalized_distance",
            "principal_b_normalized_distance",
        ):
            flattened[key].append(values[key])

    summaries = [threshold_summary(flattened, tau) for tau in args.taus]
    pd.DataFrame(summaries).to_csv(destination / "threshold_summary.csv", index=False)
    pd.DataFrame(raw_rows).to_csv(
        destination / "geometry_distributions.csv", index=False
    )
    with (destination / "threshold_metadata.json").open("w") as handle:
        json.dump(
            {
                "all_transcript_count": flattened["all_transcript_count"],
                "all_transcript_pixel_count": flattened["all_transcript_pixel_count"],
                "overlap_pixel_count": len(flattened["margin"]),
                "duplicate_patch_policy": (
                    "deduplicate global transcript pixels; retain the occurrence "
                    "farthest from a patch border"
                ),
            },
            handle,
            indent=2,
        )

    figure, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].hist(flattened["margin"], bins=40)
    axes[0].set_title("M_t distribution")
    axes[0].set_xlabel("M_t")
    axes[1].hist(
        flattened["principal_a_distance"] + flattened["principal_b_distance"],
        bins=40,
    )
    axes[1].set_title("Principal nucleus distances")
    axes[1].set_xlabel("pixels")
    axes[2].plot(
        [row["tau"] for row in summaries],
        [row["ambiguous_transcript_ratio"] for row in summaries],
        marker="o",
    )
    axes[2].set_title("Tau sensitivity")
    axes[2].set_xlabel("tau")
    axes[2].set_ylabel("ambiguous transcript ratio")
    figure.tight_layout()
    figure.savefig(destination / "threshold_sweep.png", dpi=180)
    plt.close(figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
