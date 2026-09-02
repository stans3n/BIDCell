"""Generate deterministic V1 E3/E4/E5 development configurations."""

import argparse
import copy
import glob
import json
from pathlib import Path
import sys

import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument(
        "--experiment-output-root",
        help="run root; defaults to <output-dir>/runs",
    )
    parser.add_argument(
        "--patch-manifest",
        help="optional fixed patch manifest shared by every generated config",
    )
    parser.add_argument(
        "--num-patches",
        type=int,
        help="optionally select this many existing shift-0 patches into a manifest",
    )
    parser.add_argument("--manifest-seed", type=int, default=1234)
    return parser


def _v1_settings(**overrides):
    settings = {
        "enabled": True,
        "ambiguity_mode": "overlap_geometry",
        "tau": 0.15,
        "lambda_boundary": 1.0,
        "eps": 1.0e-6,
        "mean_normalize_weights": True,
        "weight_overlap": True,
        "weight_positive_marker": True,
        "weight_negative_marker": True,
        "weight_cellcalling": False,
        "save_geometry_diagnostics": True,
        "save_loss_diagnostics": True,
        "manifest_seed": 1234,
    }
    settings.update(overrides)
    return settings


def development_variants():
    """Return named V1 variants for E3, E4, and E5."""

    variants = {
        "e3_overlap_only": _v1_settings(ambiguity_mode="overlap_only"),
        "e3_overlap_geometry": _v1_settings(ambiguity_mode="overlap_geometry"),
        "e4_official_bidcell": _v1_settings(
            enabled=False,
            weight_overlap=False,
            weight_positive_marker=False,
            weight_negative_marker=False,
        ),
        "e4_overlap_weighted": _v1_settings(
            weight_positive_marker=False, weight_negative_marker=False
        ),
        "e4_marker_weighted": _v1_settings(weight_overlap=False),
        "e4_overlap_marker_weighted": _v1_settings(),
        "e4_cellcalling_overlap_marker_weighted": _v1_settings(weight_cellcalling=True),
    }
    for value in (0.5, 1.0, 2.0, 5.0):
        suffix = str(value).replace(".", "p")
        variants[f"e5_lambda_{suffix}"] = _v1_settings(lambda_boundary=value)
    return variants


def generate_configs(
    baseline: dict,
    output_dir: Path,
    *,
    steps: int,
    experiment_output_root: Path,
    patch_manifest: str | None = None,
    manifest_seed: int = 1234,
) -> list[dict[str, str]]:
    if steps <= 0:
        raise ValueError("steps must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    generated = []
    for name, v1 in development_variants().items():
        config = copy.deepcopy(baseline)
        config.setdefault("training_params", {})["total_steps"] = steps
        config.setdefault("testing_params", {})["test_step"] = steps
        config.setdefault("experiment_dirs", {})["output_root"] = str(
            experiment_output_root
        )
        config["experiment_dirs"]["dir_id"] = name
        if patch_manifest is not None:
            v1 = {
                **v1,
                "patch_manifest": patch_manifest,
                "manifest_seed": manifest_seed,
            }
        config["v1"] = v1
        destination = output_dir / f"{name}.yaml"
        with destination.open("w") as handle:
            yaml.safe_dump(config, handle, sort_keys=False)
        generated.append(
            {
                "name": name,
                "config": str(destination),
                "experiment_output": str(experiment_output_root / name),
            }
        )
    return generated


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    baseline_path = Path(args.baseline_config)
    with baseline_path.open() as handle:
        baseline = yaml.safe_load(handle)
    if not isinstance(baseline, dict):
        raise ValueError("baseline config must contain a YAML mapping")
    output_dir = Path(args.output_dir).resolve()
    output_root = (
        Path(args.experiment_output_root).resolve()
        if args.experiment_output_root
        else output_dir / "runs"
    )
    patch_manifest = args.patch_manifest
    if patch_manifest is None and args.num_patches is not None:
        from bidcell.model.utils.patch_manifest import (
            deterministic_patch_records,
            save_patch_manifest,
        )

        data_dir = Path(baseline["files"]["data_dir"])
        if not data_dir.is_absolute():
            data_dir = (baseline_path.parent / data_dir).resolve()
        files = baseline["files"]
        patch_size = int(baseline["model_params"]["patch_size"])
        patch_dir = (
            data_dir
            / files.get("dir_out_maps", "expr_maps")
            / f"{files.get('dir_patches', 'expr_maps_input_patches_')}"
            f"{patch_size}x{patch_size}_shift_0"
        )
        patch_paths = glob.glob(str(patch_dir / "*.hdf5"))
        if not patch_paths:
            raise FileNotFoundError(f"No shift-0 patches found under {patch_dir}")
        records = deterministic_patch_records(
            patch_paths,
            seed=args.manifest_seed,
            count=args.num_patches,
            shift=0,
        )
        manifest_path = output_dir / "v1_dev_patch_manifest.json"
        save_patch_manifest(manifest_path, records, seed=args.manifest_seed)
        patch_manifest = str(manifest_path)
    generated = generate_configs(
        baseline,
        output_dir,
        steps=args.steps,
        experiment_output_root=output_root,
        patch_manifest=patch_manifest,
        manifest_seed=args.manifest_seed,
    )
    with (output_dir / "v1_dev_config_manifest.json").open("w") as handle:
        json.dump({"steps": args.steps, "configs": generated}, handle, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
