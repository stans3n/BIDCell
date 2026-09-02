import json
import time

import torch

from bidcell.utils.resource_monitor import (
    ResourceMonitor,
    trainable_parameter_count,
    update_resource_summary,
)


def test_context_records_wall_time_and_cpu_only_metrics(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with ResourceMonitor("training") as monitor:
        time.sleep(0.001)
    assert monitor.metrics["wall_time_seconds"] > 0
    assert monitor.metrics["peak_cpu_rss_bytes"] > 0
    assert monitor.metrics["peak_cuda_memory_allocated_bytes"] is None
    assert monitor.metrics["peak_cuda_memory_reserved_bytes"] is None


def test_parameter_count_and_steps_per_second():
    model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Linear(4, 2))
    expected = 3 * 4 + 4 + 4 * 2 + 2
    assert trainable_parameter_count(model) == expected
    monitor = ResourceMonitor("training", model=model).start()
    metrics = monitor.stop(steps=5)
    assert metrics["trainable_parameter_count"] == expected
    assert metrics["steps_per_second"] > 0


def test_resource_summary_is_json_serializable_and_merges_stages(tmp_path):
    path = tmp_path / "evaluation" / "tier4" / "resource_summary.json"
    update_resource_summary(
        path,
        {
            "stage": "training",
            "wall_time_seconds": 1.0,
            "peak_cpu_rss_bytes": 100,
            "peak_cuda_memory_allocated_bytes": None,
            "peak_cuda_memory_reserved_bytes": None,
            "cuda_available": False,
            "trainable_parameter_count": None,
            "steps": 4,
            "steps_per_second": 4.0,
        },
    )
    update_resource_summary(
        path,
        {
            "stage": "inference",
            "wall_time_seconds": 2.0,
            "peak_cpu_rss_bytes": 120,
            "peak_cuda_memory_allocated_bytes": None,
            "peak_cuda_memory_reserved_bytes": None,
            "cuda_available": False,
            "trainable_parameter_count": None,
            "steps": None,
            "steps_per_second": None,
        },
    )
    payload = json.loads(path.read_text())
    assert {"training", "inference", "model_metadata"}.issubset(payload)
