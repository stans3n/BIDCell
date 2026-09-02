"""Inspect V1 geometry on user-supplied BIDCell data."""

import argparse
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Save geometry-ambiguity records and ROI visualizations."
    )
    parser.add_argument("--config", required=True, help="BIDCell YAML configuration")
    parser.add_argument(
        "--output-dir", required=True, help="diagnostic output directory"
    )
    parser.add_argument("--num-rois", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1234)
    return parser


def _transcript_lookup(config):
    import pandas as pd

    path = Path(config.files.data_dir) / config.files.fp_transcripts_processed
    if not path.exists():
        return {}
    frame = pd.read_csv(path)
    x_col = config.transcripts.x_col
    y_col = config.transcripts.y_col
    gene_col = config.transcripts.gene_col
    if not {x_col, y_col}.issubset(frame.columns):
        return {}
    lookup = {}
    for _, row in frame.iterrows():
        key = (int(round(row[y_col])), int(round(row[x_col])))
        lookup.setdefault(key, []).append(row.get(gene_col))
    return lookup


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import tifffile

    from bidcell.config import load_config
    from bidcell.model.dataio.dataset_input import DataProcessing
    from bidcell.model.utils.geometry_ambiguity import (
        compute_nucleus_areas,
        geometry_statistics,
    )
    from scripts.v1.common import (
        as_numpy,
        deterministic_indices,
        geometry_from_dataset_sample,
    )

    config = load_config(args.config)
    destination = Path(args.output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    dataset = DataProcessing(config, isTraining=False, all_patches=True)
    full_areas = compute_nucleus_areas(dataset.nuclei)
    indices = deterministic_indices(len(dataset), args.num_rois, args.seed)
    genes_by_coordinate = _transcript_lookup(config)

    dapi = None
    resized_dapi = Path(config.files.data_dir) / config.files.fp_rdapi
    if resized_dapi.exists():
        dapi = tifffile.imread(resized_dapi)
    elif config.files.fp_dapi and Path(config.files.fp_dapi).exists():
        candidate_dapi = tifffile.imread(config.files.fp_dapi)
        if candidate_dapi.shape[-2:] == dataset.nuclei.shape:
            dapi = candidate_dapi

    records = []
    summaries = []
    for roi_number, index in enumerate(indices):
        sample = dataset[index]
        geometry, cell_ids, transcript_counts = geometry_from_dataset_sample(
            sample, full_areas, config, dataset=dataset
        )
        row_start, column_start = int(sample[5]), int(sample[6])
        search_areas = as_numpy(sample[2]).astype(bool)
        nuclei_labels = as_numpy(sample[7])
        summary = geometry_statistics(geometry, transcript_counts)
        summary.update(
            {
                "roi": roi_number,
                "dataset_index": index,
                "row_start": row_start,
                "column_start": column_start,
            }
        )
        summaries.append(summary)

        overlap_rows, overlap_columns = np.where(geometry.overlap_mask)
        for row, column in zip(overlap_rows, overlap_columns):
            candidate_indices = np.flatnonzero(search_areas[row, column])
            candidate_ids = cell_ids[candidate_indices]
            global_key = (row_start + int(row), column_start + int(column))
            genes = genes_by_coordinate.get(global_key)
            repeats = (
                genes if genes else [None] * max(1, int(transcript_counts[row, column]))
            )
            principal_indices = geometry.principal_indices[row, column]
            for gene in repeats:
                records.append(
                    {
                        "roi": roi_number,
                        "x": global_key[1],
                        "y": global_key[0],
                        "gene": gene,
                        "candidate_ids": json.dumps(candidate_ids.tolist()),
                        "candidate_count": int(geometry.candidate_count[row, column]),
                        "raw_distances": json.dumps(
                            geometry.raw_distances[
                                row, column, candidate_indices
                            ].tolist()
                        ),
                        "normalized_distances": json.dumps(
                            geometry.normalized_distances[
                                row, column, candidate_indices
                            ].tolist()
                        ),
                        "principal_a_id": int(geometry.principal_ids[row, column, 0]),
                        "principal_b_id": int(geometry.principal_ids[row, column, 1]),
                        "principal_a_distance": float(
                            geometry.raw_distances[row, column, principal_indices[0]]
                        ),
                        "principal_b_distance": float(
                            geometry.raw_distances[row, column, principal_indices[1]]
                        ),
                        "margin": float(geometry.margin[row, column]),
                        "ambiguous": bool(geometry.ambiguity_global[row, column]),
                    }
                )

        np.savez_compressed(
            destination / f"roi_{roi_number:03d}_geometry.npz",
            candidate_count=geometry.candidate_count,
            overlap_mask=geometry.overlap_mask,
            raw_distances=geometry.raw_distances,
            normalized_distances=geometry.normalized_distances,
            principal_ids=geometry.principal_ids,
            margin=geometry.margin,
            ambiguity_global=geometry.ambiguity_global,
            ambiguity_candidate=geometry.ambiguity_candidate,
            cell_ids=cell_ids,
        )

        panels = 5 if dapi is not None else 4
        figure, axes = plt.subplots(1, panels, figsize=(4 * panels, 4))
        panel = 0
        if dapi is not None:
            patch_size = nuclei_labels.shape[0]
            axes[panel].imshow(
                dapi[
                    row_start : row_start + patch_size,
                    column_start : column_start + patch_size,
                ],
                cmap="gray",
            )
            axes[panel].set_title("DAPI")
            panel += 1
        axes[panel].imshow(nuclei_labels, cmap="nipy_spectral")
        axes[panel].set_title("Nuclei")
        axes[panel + 1].imshow(search_areas.sum(axis=2), cmap="viridis")
        axes[panel + 1].set_title("Candidate search areas")
        axes[panel + 2].imshow(geometry.overlap_mask, cmap="gray")
        axes[panel + 2].set_title("Transcript candidate overlap")
        margin_image = np.ma.masked_invalid(geometry.margin)
        axes[panel + 3].imshow(margin_image, cmap="magma", vmin=0, vmax=1)
        transcript_rows, transcript_columns = np.where(transcript_counts > 0)
        axes[panel + 3].scatter(
            transcript_columns,
            transcript_rows,
            s=5,
            facecolors="none",
            edgecolors="cyan",
        )
        axes[panel + 3].set_title("Geometry margin M_t")
        for axis in axes:
            axis.set_axis_off()
        figure.tight_layout()
        figure.savefig(destination / f"roi_{roi_number:03d}.png", dpi=180)
        plt.close(figure)

    pd.DataFrame(records).to_csv(destination / "geometry_records.csv", index=False)
    with (destination / "geometry_summary.json").open("w") as handle:
        json.dump({"seed": args.seed, "rois": summaries}, handle, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
