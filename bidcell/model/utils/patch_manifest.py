"""Deterministic development patch manifest support."""

import json
import os
import random
import re
from pathlib import Path
from typing import Iterable


def patch_record(path: str, shift: int = 0) -> dict[str, int | str]:
    numbers = re.findall(r"\d+", os.path.basename(path))
    if len(numbers) < 2:
        raise ValueError(f"Cannot infer patch coordinates from {path}")
    row, column = int(numbers[0]), int(numbers[1])
    return {
        "patch_id": f"shift{shift}:{row}:{column}",
        "row": row,
        "column": column,
        "shift": int(shift),
        "filename": os.path.basename(path),
    }


def deterministic_patch_records(
    paths: Iterable[str], *, seed: int = 1234, count: int | None = None, shift: int = 0
) -> list[dict[str, int | str]]:
    records = sorted(
        (patch_record(path, shift=shift) for path in paths),
        key=lambda item: (item["shift"], item["row"], item["column"], item["filename"]),
    )
    if count is None or count >= len(records):
        return records
    if count < 0:
        raise ValueError("count must be non-negative")
    chosen = random.Random(seed).sample(records, count)
    return sorted(chosen, key=lambda item: item["patch_id"])


def save_patch_manifest(
    path: str | Path,
    records: list[dict[str, int | str]],
    *,
    seed: int = 1234,
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {"version": 1, "seed": int(seed), "patches": records}
    with destination.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def load_patch_manifest(path: str | Path) -> dict:
    with Path(path).open() as handle:
        payload = json.load(handle)
    if payload.get("version") != 1 or not isinstance(payload.get("patches"), list):
        raise ValueError("Unsupported or invalid patch manifest")
    return payload


def filter_patch_paths(
    paths: Iterable[str], manifest_path: str | Path, *, shift: int = 0
) -> list[str]:
    payload = load_patch_manifest(manifest_path)
    allowed = {
        (int(item["shift"]), int(item["row"]), int(item["column"]))
        for item in payload["patches"]
    }
    selected = []
    for path in paths:
        record = patch_record(path, shift=shift)
        key = (shift, int(record["row"]), int(record["column"]))
        if key in allowed:
            selected.append(path)
    return selected
