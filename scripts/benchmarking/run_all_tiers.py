#!/usr/bin/env python3
"""Run all available BIDCell V1 benchmark tiers with explicit skip semantics."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.benchmarking.common import Manifest, git_commit, utc_now, write_json
from scripts.benchmarking import (
    tier1_cellspa,
    tier2_boundary_competition,
    tier3_resources,
    tier4_vendor_reference,
    tier5_stability_ablation,
)


OUTPUT_FILES = (
    "benchmark_manifest.csv",
    "all_tiers_summary.json",
    "tier1_cellspa_summary.json",
    "tier1_cell_level.csv",
    "tier2_boundary_summary.json",
    "tier2_assignment_changes.csv",
    "tier3_resource_summary.json",
    "tier4_vendor_reference_summary.json",
    "tier5_stability_summary.json",
    "tier5_ablation_table.csv",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir")
    parser.add_argument("--connected-mask")
    parser.add_argument("--raw-prediction")
    parser.add_argument("--nuclei-mask")
    parser.add_argument("--transcripts")
    parser.add_argument("--cell-gene-matrix")
    parser.add_argument("--cell-annotation")
    parser.add_argument("--positive-markers")
    parser.add_argument("--negative-markers")
    parser.add_argument("--baseline-run-dir")
    parser.add_argument("--vendor-mask")
    parser.add_argument("--scrna-reference")
    parser.add_argument("--replicate-run-dir", action="append", default=[])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--pixel-size-um", type=float, default=1.0)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--config-path")
    parser.add_argument(
        "--method-label",
        default="candidate-overlap boundary-weighted BIDCell V1",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _validate_inputs(args) -> None:
    if args.pixel_size_um <= 0:
        raise ValueError("--pixel-size-um must be greater than zero")
    for attribute in (
        "run_dir",
        "connected_mask",
        "raw_prediction",
        "nuclei_mask",
        "transcripts",
        "cell_gene_matrix",
        "cell_annotation",
        "positive_markers",
        "negative_markers",
        "baseline_run_dir",
        "vendor_mask",
        "scrna_reference",
        "config_path",
    ):
        value = getattr(args, attribute)
        if value and not Path(value).exists():
            raise FileNotFoundError(
                f"--{attribute.replace('_', '-')} does not exist: {value}"
            )
    for path in args.replicate_run_dir:
        if not Path(path).exists():
            raise FileNotFoundError(f"--replicate-run-dir does not exist: {path}")


def run_all(args) -> int:
    _validate_inputs(args)
    output_dir = Path(args.output_dir)
    existing = [
        output_dir / name for name in OUTPUT_FILES if (output_dir / name).exists()
    ]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Benchmark outputs already exist; use --overwrite explicitly: "
            + ", ".join(str(path) for path in existing)
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = Manifest()
    started_at = utc_now()
    summaries = {}
    failures = []
    for tier_number, module in (
        (1, tier1_cellspa),
        (2, tier2_boundary_competition),
        (3, tier3_resources),
        (4, tier4_vendor_reference),
        (5, tier5_stability_ablation),
    ):
        try:
            summaries[f"tier{tier_number}"] = module.run(args, manifest)
        except Exception as exc:
            failures.append({"tier": tier_number, "error": str(exc)})
            manifest.add(
                f"tier{tier_number}.execution",
                tier_number,
                f"Tier {tier_number} execution",
                "failed",
                notes=f"{type(exc).__name__}: {exc}",
            )

    repo_dir = Path(__file__).resolve().parents[2]
    summary = {
        "status": "failed" if failures else "completed",
        "failures": failures,
        "started_at": started_at,
        "ended_at": utc_now(),
        "method_label": args.method_label,
        "git_commit": git_commit(repo_dir),
        "config_path": args.config_path,
        "run_dir": args.run_dir,
        "seed": args.seed,
        "pixel_size_um": args.pixel_size_um,
        "tiers": summaries,
    }
    manifest.write(output_dir / "benchmark_manifest.csv")
    write_json(output_dir / "all_tiers_summary.json", summary)
    return 1 if failures else 0


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run_all(args)
    except Exception as exc:
        print(f"benchmark failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
