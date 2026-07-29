#!/usr/bin/env python3
"""Run a command while sampling device GPU and child-process CPU/RAM resources."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import signal
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


FIELDS = [
    "timestamp",
    "elapsed_seconds",
    "child_pid",
    "gpu_index",
    "gpu_utilization_percent",
    "gpu_memory_used_mb",
    "gpu_memory_total_mb",
    "gpu_power_watts",
    "process_cpu_percent",
    "process_rss_mb",
    "system_ram_percent",
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _parse_number(value: str) -> float | None:
    value = value.strip()
    if not value or value.lower() in {"n/a", "na", "[not supported]"}:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _query_gpus() -> list[dict]:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return []
    base_fields = ["index", "utilization.gpu", "memory.used", "memory.total"]
    fields = base_fields + ["power.draw"]
    for selected in (fields, base_fields):
        try:
            result = subprocess.run(
                [
                    executable,
                    f"--query-gpu={','.join(selected)}",
                    "--format=csv,noheader,nounits",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            )
        except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
            continue
        rows = []
        for line in result.stdout.splitlines():
            values = [value.strip() for value in line.split(",")]
            row = dict(zip(selected, values))
            rows.append(
                {
                    "gpu_index": row.get("index", ""),
                    "gpu_utilization_percent": _parse_number(
                        row.get("utilization.gpu", "")
                    ),
                    "gpu_memory_used_mb": _parse_number(row.get("memory.used", "")),
                    "gpu_memory_total_mb": _parse_number(row.get("memory.total", "")),
                    "gpu_power_watts": _parse_number(row.get("power.draw", "")),
                }
            )
        return rows
    return []


def _stats(values: list[float]) -> dict:
    if not values:
        return {"mean": None, "median": None, "p95": None, "peak": None}
    ordered = sorted(values)
    rank = max(0, min(len(ordered) - 1, int(0.95 * (len(ordered) - 1))))
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p95": ordered[rank],
        "peak": max(values),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--interval", type=float, default=2.0)
    parser.add_argument("--config-path")
    parser.add_argument("--stage-name", default="command")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser


def run(args) -> int:
    if args.interval <= 0:
        raise ValueError("--interval must be greater than zero")
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise ValueError("A command is required after --")

    output_dir = Path(args.output_dir)
    samples_path = output_dir / "resource_samples.csv"
    summary_path = output_dir / "resource_summary.json"
    if not args.overwrite and (samples_path.exists() or summary_path.exists()):
        raise FileExistsError(
            f"Resource outputs already exist in {output_dir}; use --overwrite"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import psutil
    except ImportError:
        psutil = None
        print(
            "psutil is not installed; CPU/RSS/system RAM fields will be NA",
            file=sys.stderr,
        )

    started_at = _utc_now()
    started = time.perf_counter()
    process = subprocess.Popen(command)
    child = psutil.Process(process.pid) if psutil is not None else None
    if child is not None:
        child.cpu_percent(interval=None)
    forwarded_signal = None

    def handle_signal(signum, _frame):
        nonlocal forwarded_signal
        forwarded_signal = signum
        if process.poll() is None:
            process.send_signal(signum)

    previous_handlers = {}
    for signum in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[signum] = signal.signal(signum, handle_signal)

    samples = []
    try:
        with samples_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            handle.flush()
            while True:
                elapsed = time.perf_counter() - started
                cpu_percent = None
                rss_mb = None
                system_ram_percent = None
                if child is not None:
                    try:
                        descendants = child.children(recursive=True)
                        cpu_percent = child.cpu_percent(interval=None) + sum(
                            descendant.cpu_percent(interval=None)
                            for descendant in descendants
                        )
                        rss_mb = (
                            child.memory_info().rss
                            + sum(
                                descendant.memory_info().rss
                                for descendant in descendants
                            )
                        ) / (1024.0 * 1024.0)
                        system_ram_percent = psutil.virtual_memory().percent
                    except (psutil.Error, OSError):
                        pass

                gpu_rows = _query_gpus() or [
                    {
                        "gpu_index": "",
                        "gpu_utilization_percent": None,
                        "gpu_memory_used_mb": None,
                        "gpu_memory_total_mb": None,
                        "gpu_power_watts": None,
                    }
                ]
                timestamp = _utc_now()
                for gpu in gpu_rows:
                    row = {
                        "timestamp": timestamp,
                        "elapsed_seconds": elapsed,
                        "child_pid": process.pid,
                        **gpu,
                        "process_cpu_percent": cpu_percent,
                        "process_rss_mb": rss_mb,
                        "system_ram_percent": system_ram_percent,
                    }
                    writer.writerow(
                        {
                            key: "" if value is None else value
                            for key, value in row.items()
                        }
                    )
                    samples.append(row)
                handle.flush()
                if process.poll() is not None:
                    break
                time.sleep(args.interval)
    finally:
        for signum, previous in previous_handlers.items():
            signal.signal(signum, previous)
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()

    return_code = process.returncode
    duration = time.perf_counter() - started
    metrics = {}
    for field in (
        "gpu_utilization_percent",
        "gpu_memory_used_mb",
        "gpu_power_watts",
        "process_cpu_percent",
        "process_rss_mb",
        "system_ram_percent",
    ):
        metrics[field] = _stats(
            [float(row[field]) for row in samples if row.get(field) is not None]
        )
    summary = {
        "started_at": started_at,
        "ended_at": _utc_now(),
        "duration_seconds": duration,
        "sample_count": len(samples),
        "child_pid": process.pid,
        "exit_code": return_code,
        "forwarded_signal": forwarded_signal,
        "command": command,
        "stage_name": args.stage_name,
        "git_commit": _git_commit(),
        "config_path": args.config_path,
        "gpu_scope_note": (
            "nvidia-smi utilization is device-level and can include other processes "
            "sharing the same GPU; unavailable values are recorded as NA, not zero."
        ),
        **metrics,
        "peak_device_vram_mb": metrics["gpu_memory_used_mb"]["peak"],
        "peak_rss_mb": metrics["process_rss_mb"]["peak"],
    }
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    if return_code is None:
        return 1
    if return_code < 0:
        return 128 + abs(return_code)
    return return_code


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run(args)
    except Exception as exc:
        print(f"resource monitor failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
