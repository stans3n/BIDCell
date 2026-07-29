"""Tier 1 biological and morphology summaries at BIDCell/CellSPA scope."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from skimage.measure import perimeter, regionprops

from .common import (
    Manifest,
    distribution,
    infer_column,
    load_mask,
    read_table,
    transcript_assignments,
    write_json,
)


def _cell_morphology(mask: np.ndarray, pixel_size_um: float) -> pd.DataFrame:
    rows = []
    for region in regionprops(mask):
        area_px = float(region.area)
        perimeter_px = float(region.perimeter)
        convex_perimeter_px = float(perimeter(region.convex_image))
        circularity = (
            4.0 * math.pi * area_px / (perimeter_px**2) if perimeter_px > 0 else None
        )
        major = float(region.major_axis_length)
        minor = float(region.minor_axis_length)
        rows.append(
            {
                "cell_id": int(region.label),
                "centroid_y_px": float(region.centroid[0]),
                "centroid_x_px": float(region.centroid[1]),
                "area_px2": area_px,
                "area_um2": area_px * pixel_size_um**2,
                "elongation": major / minor if minor > 0 else None,
                "circularity": circularity,
                "compactness": 1.0 / circularity if circularity else None,
                "convexity": (
                    convex_perimeter_px / perimeter_px if perimeter_px > 0 else None
                ),
                "eccentricity": float(region.eccentricity),
                "solidity": float(region.solidity),
            }
        )
    return pd.DataFrame(rows)


def run(args, manifest: Manifest) -> dict:
    output_dir = Path(args.output_dir)
    summary: dict = {"tier": 1}
    if not args.connected_mask:
        manifest.missing("t1.cell_count", 1, "segmented cell count", ["connected_mask"])
        manifest.missing(
            "t1.basic_morphology",
            1,
            "basic morphology distributions",
            ["connected_mask"],
        )
        write_json(output_dir / "tier1_cellspa_summary.json", summary)
        return summary

    mask = load_mask(args.connected_mask)
    cell_level = _cell_morphology(mask, args.pixel_size_um)
    n_cells = int(len(cell_level))
    summary["segmented_cell_count"] = n_cells
    manifest.add(
        "t1.cell_count",
        1,
        "segmented cell count",
        "computed",
        value=n_cells,
        unit="cells",
        method="number of unique positive connected-mask labels",
        required_inputs=["connected_mask"],
        used_inputs=[args.connected_mask],
    )

    morphology_columns = [
        "area_um2",
        "elongation",
        "circularity",
        "compactness",
        "convexity",
        "eccentricity",
        "solidity",
    ]
    summary["morphology"] = {
        column: distribution(cell_level[column].dropna())
        for column in morphology_columns
    }
    manifest.add(
        "t1.basic_morphology",
        1,
        "basic morphology distributions",
        "computed",
        value=n_cells,
        unit="cells summarized",
        method="skimage region properties on connected-mask instances",
        required_inputs=["connected_mask"],
        used_inputs=[args.connected_mask],
    )

    assigned = None
    gene_col = None
    gene_expression_proportion = None
    if args.transcripts:
        transcripts = read_table(args.transcripts)
        assigned = transcript_assignments(transcripts, mask, args.pixel_size_um)
        valid = assigned["_in_bounds"]
        is_assigned = assigned["_assigned_cell_id"] > 0
        total = int(valid.sum())
        assigned_count = int((valid & is_assigned).sum())
        unassigned_count = total - assigned_count
        summary["transcripts"] = {
            "in_bounds_count": total,
            "assigned_count": assigned_count,
            "unassigned_count": unassigned_count,
            "assigned_fraction": assigned_count / total if total else None,
            "unassigned_fraction": unassigned_count / total if total else None,
        }
        manifest.add(
            "t1.assigned_transcript_fraction",
            1,
            "assigned transcript fraction",
            "computed",
            value=summary["transcripts"]["assigned_fraction"],
            unit="fraction",
            method="connected-mask label sampled at transcript coordinates",
            required_inputs=["connected_mask", "transcripts"],
            used_inputs=[args.connected_mask, args.transcripts],
        )

        counts = (
            assigned.loc[is_assigned]
            .groupby("_assigned_cell_id")
            .size()
            .rename("transcripts")
        )
        gene_col = infer_column(
            transcripts, ("feature_name", "gene", "gene_name", "target")
        )
        if gene_col:
            genes = (
                assigned.loc[is_assigned]
                .groupby("_assigned_cell_id")[gene_col]
                .nunique()
                .rename("genes")
            )
            expressing = (
                assigned.loc[is_assigned, ["_assigned_cell_id", gene_col]]
                .drop_duplicates()
                .groupby(gene_col)
                .size()
            )
            gene_expression_proportion = {
                str(gene): float(count / n_cells) if n_cells else None
                for gene, count in expressing.items()
            }
        else:
            genes = pd.Series(dtype=float, name="genes")
        cell_level = cell_level.merge(
            counts, how="left", left_on="cell_id", right_index=True
        ).merge(genes, how="left", left_on="cell_id", right_index=True)

    if args.cell_gene_matrix:
        matrix = read_table(args.cell_gene_matrix)
        cell_id_col = infer_column(matrix, ("cell_id", "label", "id"))
        if cell_id_col is None:
            cell_id_col = str(matrix.columns[0])
        gene_columns = [
            column
            for column in matrix.columns
            if column != cell_id_col and pd.api.types.is_numeric_dtype(matrix[column])
        ]
        matrix_stats = pd.DataFrame(
            {
                "cell_id": matrix[cell_id_col].astype(int),
                "transcripts": matrix[gene_columns].sum(axis=1),
                "genes": (matrix[gene_columns] > 0).sum(axis=1),
            }
        )
        cell_level = cell_level.drop(
            columns=["transcripts", "genes"], errors="ignore"
        ).merge(matrix_stats, on="cell_id", how="left")
        gene_expression_proportion = {
            str(column): float((matrix[column] > 0).mean()) for column in gene_columns
        }

    if "transcripts" in cell_level:
        cell_level["transcripts"] = cell_level["transcripts"].fillna(0)
        cell_level["genes"] = cell_level["genes"].fillna(0)
        cell_level["transcript_density_per_um2"] = (
            cell_level["transcripts"] / cell_level["area_um2"]
        )
        summary["cell_level"] = {
            column: distribution(cell_level[column])
            for column in (
                "transcripts",
                "genes",
                "area_um2",
                "transcript_density_per_um2",
            )
        }
        manifest.add(
            "t1.cell_level_expression",
            1,
            "cell transcripts genes area and density summaries",
            "computed",
            value=len(cell_level),
            unit="cells summarized",
            method="per-instance aggregation",
            required_inputs=["connected_mask", "transcripts or cell_gene_matrix"],
            used_inputs=[
                value
                for value in (
                    args.connected_mask,
                    args.transcripts,
                    args.cell_gene_matrix,
                )
                if value
            ],
        )
    else:
        manifest.missing(
            "t1.cell_level_expression",
            1,
            "cell transcripts genes area and density summaries",
            ["transcripts or cell_gene_matrix"],
        )

    if gene_expression_proportion is not None:
        summary["proportion_cells_expressing_each_gene"] = gene_expression_proportion
        manifest.add(
            "t1.gene_expression_proportion",
            1,
            "proportion of cells expressing each gene",
            "computed",
            value=len(gene_expression_proportion),
            unit="genes summarized",
            method="fraction of segmented cells with at least one assigned transcript",
            required_inputs=["connected_mask", "transcripts or cell_gene_matrix"],
            used_inputs=[
                value
                for value in (
                    args.connected_mask,
                    args.transcripts,
                    args.cell_gene_matrix,
                )
                if value
            ],
        )
    else:
        manifest.missing(
            "t1.gene_expression_proportion",
            1,
            "proportion of cells expressing each gene",
            ["transcripts with gene column or cell_gene_matrix"],
        )

    cell_level.to_csv(output_dir / "tier1_cell_level.csv", index=False)
    if args.cell_annotation:
        manifest.add(
            "t1.cell_type_morphology",
            1,
            "cell-type grouped morphology",
            "not_implemented",
            required_inputs=["cell_annotation"],
            used_inputs=[args.cell_annotation],
            notes="Input is retained as an extension point; annotation schema requires validation.",
        )
    else:
        manifest.missing(
            "t1.cell_type_morphology",
            1,
            "cell-type grouped morphology",
            ["cell_annotation"],
        )
    manifest.missing(
        "t1.nucleus_cell_information_gain",
        1,
        "nucleus-to-cell transcript and gene information gain",
        ["nucleus_level_matrix", "cell_gene_matrix"],
    )
    if args.cell_annotation:
        manifest.add(
            "t1.regional_cell_type_summary",
            1,
            "regional cell-type entropy and feature CV",
            "not_implemented",
            required_inputs=["cell_annotation", "validated cell centroids/grid"],
            used_inputs=[args.cell_annotation],
            notes="Requires validation of the supplied annotation and centroid schema.",
        )
    else:
        manifest.missing(
            "t1.regional_cell_type_summary",
            1,
            "regional cell-type entropy and feature CV",
            ["cell_annotation", "cell centroids"],
        )

    for metric_id, name, required, provided in (
        (
            "t1.marker_purity",
            "positive/negative marker precision recall F1 and purity F1",
            ["positive_markers", "negative_markers", "cell_annotation"],
            bool(
                args.positive_markers and args.negative_markers and args.cell_annotation
            ),
        ),
        (
            "t1.scrna_concordance",
            "matched scRNA-seq concordance",
            ["scrna_reference", "cell_annotation"],
            bool(args.scrna_reference and args.cell_annotation),
        ),
        (
            "t1.replicability",
            "replicate concordance",
            ["replicate_run_dir"],
            bool(args.replicate_run_dir),
        ),
    ):
        if provided:
            manifest.add(
                metric_id,
                1,
                name,
                "not_implemented",
                required_inputs=required,
                notes="Inputs are present, but this extension requires a validated schema/matching method.",
            )
        else:
            manifest.missing(metric_id, 1, name, required)
    write_json(output_dir / "tier1_cellspa_summary.json", summary)
    return summary
