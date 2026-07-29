#!/usr/bin/env python3
"""Run training through available Tier 1-5 V1 benchmarks with stage recovery."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bidcell.config import load_config
from scripts.benchmarking.common import git_commit, write_json


STAGES = (
    "training",
    "inference",
    "prediction_combination",
    "morphological_postprocessing",
    "final_cell_gene_matrix",
    "benchmark",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run_id() -> str:
    return datetime.now().strftime("v1_%Y_%m_%d_%H_%M_%S")


def _internal_stage(stage: str, config_path: str, run_id: str) -> int:
    config = load_config(config_path)
    config.experiment_dirs.dir_id = run_id
    if stage == "training":
        from bidcell.model.train import train

        train(config, config_path=config_path, experiment_id=run_id)
    elif stage == "inference":
        from bidcell.model.predict import predict

        predict(config)
    elif stage == "prediction_combination":
        from bidcell.model.predict import fill_grid

        fill_grid(config, run_id)
    elif stage == "morphological_postprocessing":
        from bidcell.model.postprocess_predictions import postprocess_predictions

        postprocess_predictions(config, run_id)
    elif stage == "final_cell_gene_matrix":
        from bidcell.processing.cell_gene_matrix import make_cell_gene_mat

        make_cell_gene_mat(config, is_cell=True, timestamp=run_id)
    else:
        raise ValueError(f"Unknown internal stage: {stage}")
    return 0


def _expected_outputs(config, run_dir: Path, run_id: str) -> dict[str, list[Path]]:
    epoch = config.testing_params.test_epoch
    step = config.testing_params.test_step
    shift = int(config.model_params.patch_size / 2)
    test_dir = run_dir / config.experiment_dirs.test_output_dir
    data_dir = Path(config.files.data_dir)
    return {
        "training": [
            run_dir
            / config.experiment_dirs.model_dir
            / f"epoch_{epoch}_step_{step}.pth"
        ],
        "inference": [
            test_dir / f"epoch_{epoch}_step_{step}_seg_shift0.tif",
            test_dir / f"epoch_{epoch}_step_{step}_seg_shift{shift}.tif",
        ],
        "prediction_combination": [test_dir / f"epoch_{epoch}_step_{step}.tif"],
        "morphological_postprocessing": [
            test_dir / f"epoch_{epoch}_step_{step}_connected.tif"
        ],
        "final_cell_gene_matrix": [
            data_dir / config.files.dir_cgm / run_id / config.files.fp_expr
        ],
        "benchmark": [
            run_dir / "metrics" / "benchmark" / "benchmark_manifest.csv",
            run_dir / "metrics" / "benchmark" / "all_tiers_summary.json",
        ],
    }


def _outputs_ready(stage: str, expected: list[Path]) -> bool:
    return bool(expected) and all(path.exists() for path in expected)


def _stage_command(
    script: Path, stage: str, config_path: Path, run_id: str
) -> list[str]:
    return [
        sys.executable,
        str(script),
        "--internal-stage",
        stage,
        "--config",
        str(config_path),
        "--run-id",
        run_id,
    ]


def _run_process(command: list[str], stdout_path: Path, stderr_path: Path) -> int:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    with (
        stdout_path.open("a", encoding="utf-8") as stdout,
        stderr_path.open("a", encoding="utf-8") as stderr,
    ):
        process = subprocess.Popen(command, stdout=stdout, stderr=stderr)
        try:
            return process.wait()
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-stage", action="append", choices=STAGES, default=[])
    parser.add_argument("--resource-interval", type=float, default=2.0)
    parser.add_argument("--baseline-run-dir")
    parser.add_argument("--vendor-mask")
    parser.add_argument("--scrna-reference")
    parser.add_argument("--replicate-run-dir", action="append", default=[])
    parser.add_argument("--cell-annotation")
    parser.add_argument("--positive-markers")
    parser.add_argument("--negative-markers")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--internal-stage", choices=STAGES[:-1], help=argparse.SUPPRESS)
    return parser


def _validate_optional_paths(args) -> None:
    for attribute in (
        "baseline_run_dir",
        "vendor_mask",
        "scrna_reference",
        "cell_annotation",
        "positive_markers",
        "negative_markers",
    ):
        value = getattr(args, attribute)
        if value and not Path(value).exists():
            raise FileNotFoundError(f"{attribute} does not exist: {value}")
    for path in args.replicate_run_dir:
        if not Path(path).exists():
            raise FileNotFoundError(f"replicate_run_dir does not exist: {path}")


def run_pipeline(args) -> int:
    _validate_optional_paths(args)
    config_path = Path(args.config).resolve()
    config = load_config(str(config_path))
    run_id = args.run_id or _run_id()
    if Path(run_id).name != run_id or run_id in {".", ".."}:
        raise ValueError("--run-id must be one safe directory name")

    data_dir = Path(config.files.data_dir)
    run_dir = data_dir / "model_outputs" / run_id
    baseline_dir = (
        Path(args.baseline_run_dir).resolve() if args.baseline_run_dir else None
    )
    if baseline_dir is not None and run_dir.resolve() == baseline_dir:
        raise ValueError("V1 run directory must not be the official baseline directory")
    if run_dir.exists() and not args.resume:
        raise FileExistsError(f"Run directory already exists; use --resume: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = run_dir / "experiment_manifest.json"
    summary_path = run_dir / "experiment_summary.json"
    script = Path(__file__).resolve()
    repo_dir = script.parents[2]
    expected = _expected_outputs(config, run_dir, run_id)
    manifest = {
        "method": "candidate-overlap boundary-weighted BIDCell V1",
        "run_id": run_id,
        "run_dir": str(run_dir),
        "config_path": str(config_path),
        "git_commit": git_commit(repo_dir),
        "seed": args.seed,
        "boundary_weighting_enabled": (
            config.training_params.boundary_weighting_enabled
        ),
        "boundary_weight": config.training_params.boundary_weight,
        "boundary_weight_overlap": config.training_params.boundary_weight_overlap,
        "boundary_weight_positive_marker": (
            config.training_params.boundary_weight_positive_marker
        ),
        "boundary_weight_negative_marker": (
            config.training_params.boundary_weight_negative_marker
        ),
        "boundary_weight_cell_calling": (
            config.training_params.boundary_weight_cell_calling
        ),
        "created_at": _utc_now(),
        "stages": [],
    }
    write_json(manifest_path, manifest)

    benchmark_dir = run_dir / "metrics" / "benchmark"
    resource_dir = run_dir / "metrics" / "resources"
    logs_dir = run_dir / "logs"
    connected_mask = expected["morphological_postprocessing"][0]
    raw_prediction = expected["prediction_combination"][0]
    cell_gene_matrix = expected["final_cell_gene_matrix"][0]
    nuclei_mask = data_dir / config.files.fp_nuclei
    processed_transcripts = data_dir / config.files.fp_transcripts_processed
    transcripts = (
        processed_transcripts
        if processed_transcripts.exists()
        else Path(config.files.fp_transcripts)
    )

    failed = False
    for stage in STAGES:
        stage_record = {
            "stage": stage,
            "status": None,
            "started_at": None,
            "ended_at": None,
            "duration_seconds": None,
            "return_code": None,
            "expected_outputs": [str(path) for path in expected[stage]],
        }
        if stage in args.skip_stage:
            stage_record["status"] = "skipped_user"
            manifest["stages"].append(stage_record)
            write_json(manifest_path, manifest)
            continue
        if args.resume and _outputs_ready(stage, expected[stage]):
            stage_record["status"] = "skipped_existing"
            manifest["stages"].append(stage_record)
            write_json(manifest_path, manifest)
            continue

        if stage != "training":
            previous = STAGES[STAGES.index(stage) - 1]
            if not _outputs_ready(previous, expected[previous]):
                stage_record["status"] = "failed_missing_previous_output"
                stage_record["return_code"] = 1
                manifest["stages"].append(stage_record)
                write_json(manifest_path, manifest)
                failed = True
                break

        if stage == "benchmark":
            command = [
                sys.executable,
                str(repo_dir / "scripts" / "benchmarking" / "run_all_tiers.py"),
                "--run-dir",
                str(run_dir),
                "--connected-mask",
                str(connected_mask),
                "--raw-prediction",
                str(raw_prediction),
                "--nuclei-mask",
                str(nuclei_mask),
                "--transcripts",
                str(transcripts),
                "--cell-gene-matrix",
                str(cell_gene_matrix),
                "--output-dir",
                str(benchmark_dir),
                "--pixel-size-um",
                str(config.affine.target_pix_um),
                "--config-path",
                str(config_path),
            ]
            for option, value in (
                ("--baseline-run-dir", args.baseline_run_dir),
                ("--vendor-mask", args.vendor_mask),
                ("--scrna-reference", args.scrna_reference),
                ("--cell-annotation", args.cell_annotation),
                ("--positive-markers", args.positive_markers),
                ("--negative-markers", args.negative_markers),
            ):
                if value:
                    command.extend([option, str(value)])
            for replicate in args.replicate_run_dir:
                command.extend(["--replicate-run-dir", str(replicate)])
            if args.seed is not None:
                command.extend(["--seed", str(args.seed)])
            if args.resume:
                command.append("--overwrite")
        else:
            command = _stage_command(script, stage, config_path, run_id)
            if stage == "training":
                training_command = command
                command = [
                    sys.executable,
                    str(
                        repo_dir
                        / "scripts"
                        / "benchmarking"
                        / "run_with_resource_monitor.py"
                    ),
                    "--output-dir",
                    str(resource_dir),
                    "--interval",
                    str(args.resource_interval),
                    "--config-path",
                    str(config_path),
                    "--stage-name",
                    "training",
                ]
                if args.resume:
                    command.append("--overwrite")
                command.extend(["--", *training_command])

        stage_record["command"] = command
        stage_record["started_at"] = _utc_now()
        started = time.perf_counter()
        try:
            return_code = _run_process(
                command,
                logs_dir / f"{stage}.stdout.log",
                logs_dir / f"{stage}.stderr.log",
            )
        except Exception as exc:
            return_code = 1
            stage_record["error"] = f"{type(exc).__name__}: {exc}"
        stage_record["duration_seconds"] = time.perf_counter() - started
        stage_record["ended_at"] = _utc_now()
        stage_record["return_code"] = return_code
        if return_code != 0:
            stage_record["status"] = "failed"
            failed = True
        elif not _outputs_ready(stage, expected[stage]):
            stage_record["status"] = "failed_missing_expected_output"
            failed = True
        else:
            stage_record["status"] = "completed"
        manifest["stages"].append(stage_record)
        write_json(manifest_path, manifest)
        if failed:
            break

    summary = {
        **manifest,
        "status": "failed" if failed else "completed",
        "ended_at": _utc_now(),
        "available_benchmark_outputs": [
            str(path) for path in expected["benchmark"] if path.exists()
        ],
    }
    write_json(summary_path, summary)
    return 1 if failed else 0


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.internal_stage:
            return _internal_stage(args.internal_stage, args.config, args.run_id)
        return run_pipeline(args)
    except Exception as exc:
        print(f"V1 pipeline failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
