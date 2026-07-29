# Candidate-overlap boundary-weighted BIDCell V1

## Scope

V1 makes one attributable training change. For a training patch whose permuted
search-area tensor has shape `[n_candidates, 1, H, W]`, it marks a pixel
ambiguous when at least two candidate search areas cover it:

```python
candidate_overlap = (batch_sa[:, 0].sum(dim=0) >= 2).to(seg_pred.dtype)
spatial_weight = 1.0 + boundary_weight * candidate_overlap
```

The optional weight multiplies the existing overlap, positive-marker, and
negative-marker pixel penalties. `CellCallingLoss` supports the same API but is
off by default. The original denominators remain unchanged, so a weight above
one actually increases ambiguous-region influence.

V1 adds no boundary head, attention block, backbone, transcript classifier,
pairwise competition, prediction rule, or morphology rule. In particular, the
final connected-mask boundary is post-hoc output. Feeding that boundary, a
3-pixel post-hoc band, vendor segmentation, or a manual boundary back into
training would leak downstream information and would no longer be this V1.

## Configuration

The example is
`bidcell/example_params/boundary_weighted_v1.yaml`. Its data paths are relative
placeholders.

```yaml
training_params:
  boundary_weighting_enabled: true
  boundary_weight: 1.0
  boundary_weight_overlap: true
  boundary_weight_positive_marker: true
  boundary_weight_negative_marker: true
  boundary_weight_cell_calling: false
  log_boundary_metrics: true
  metrics_interval: 1
```

`boundary_weight` must be non-negative. Old YAML files remain valid. Setting
`boundary_weighting_enabled: false` completely disables spatial weighting;
setting `boundary_weight: 0.0` produces an all-one weight and the original
losses within floating-point tolerance. Every weighted loss keeps a
`spatial_weight=None` default, so existing Python calls remain valid.

`metrics_interval` controls scalar CSV sampling. A larger value avoids
per-step CUDA synchronization overhead; the final training summary still uses
an end-to-end monotonic wall clock.

## Training instrumentation

Only data available inside backpropagation is logged by
`bidcell/model/training_metrics.py`: detached losses, candidate count,
candidate-overlap count/fraction, learning rate, step/cumulative time, and
PyTorch allocated/reserved CUDA memory. No prediction, batch, search-area,
spatial-weight, or other graph-retaining tensor is stored.

Outputs are appended and regularly flushed at:

```text
<run-dir>/metrics/training/v1_training_metrics.csv
<run-dir>/metrics/training/training_summary.json
```

The high-level experiment launcher runs training through the external resource
monitor:

```bash
python scripts/experiments/run_v1_full_pipeline.py \
  --config bidcell/example_params/boundary_weighted_v1.yaml \
  --run-id v1_seed_1 \
  --seed 1
```

It runs training, inference, prediction combination, unchanged morphological
postprocessing, final cell-gene matrix extraction, and all available benchmark
tiers. It writes stage logs and `experiment_manifest.json` /
`experiment_summary.json` inside the dynamically resolved run directory.
Use `--resume` to skip stages whose expected outputs already exist, or
`--skip-stage STAGE` for an explicit skip.

For an independently constructed training command:

```bash
python scripts/benchmarking/run_with_resource_monitor.py \
  --output-dir RUN_DIR/metrics/resources \
  --interval 2 \
  --config-path CONFIG.yaml \
  -- python -m bidcell.model.train \
  --config_dir CONFIG.yaml \
  --experiment_id RUN_ID
```

The monitor never substitutes zero for unavailable `nvidia-smi` fields.
Device-level GPU utilization may include other processes on the same GPU.

No local full 4000-step run is part of implementation or unit testing. That
experiment should be run later on the intended training machine.
