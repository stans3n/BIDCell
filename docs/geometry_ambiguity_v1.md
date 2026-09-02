# V1 geometry-guided ambiguity weighting

V1 changes only where existing BIDCell biological losses are emphasized. It
does not modify `SegmentationModel`, add input channels to the network, add a
head, or change prediction and morphological postprocessing semantics.

## Geometry and tensor contract

For each transcript pixel, candidates are exactly the nucleus-specific BIDCell
search-area channels that contain that pixel. The data loader computes the
Euclidean distance to the nearest pixel of each full nucleus (not its centroid)
with a cached per-nucleus `cKDTree`. Equivalent radii come from a single
full-label-image `bincount` scan. For the two smallest normalized distances,

```text
M_t = abs(d_norm_A - d_norm_B) / (d_norm_A + d_norm_B + eps)
```

ties are broken by the sorted candidate channel/ID order. `overlap_geometry`
selects transcript pixels with at least two candidates and `M_t < tau`;
`overlap_only` selects every transcript pixel with at least two candidates.
`geometry_only` is intentionally unsupported because geometry outside the
existing search-area candidate construction is not scientifically defined.

For a patch of height `H`, width `W`, `G` genes, and `C` candidate cells:

- expression input before training permutation: `[H, W, G, C]`;
- nucleus, search-area, and marker tensors: `[H, W, C]`;
- global ambiguity and margin: `[H, W]`;
- candidate ambiguity: `[H, W, C]`;
- principal candidate IDs: `[H, W, 2]`;
- network input after permutation: `[C, G, H, W]`;
- loss targets and candidate ambiguity after permutation: `[C, 1, H, W]`
  and `[C, H, W]` respectively.

Expression, nucleus IDs, and full-nucleus distance channels receive one shared
flip/rotation. Search areas and ambiguity are constructed after that transform.
The prediction-only data tuple is unchanged.

Weights are `1 + lambda_boundary * A_tau`. Mean normalization is configurable;
the global overlap/positive-marker domains use their complete spatial tensors,
while negative-marker normalization uses its non-empty marker support. No V1
weights are passed when V1 is disabled. CellCalling weighting is off unless the
E4 ablation flag is explicitly enabled.

## Configuration

Add this block to a real, externally supplied BIDCell configuration:

```yaml
v1:
  enabled: true
  ambiguity_mode: overlap_geometry
  tau: 0.15                 # development default, not an optimum
  lambda_boundary: 1.0      # development default, not an optimum
  eps: 1.0e-6
  mean_normalize_weights: true
  weight_overlap: true
  weight_positive_marker: true
  weight_negative_marker: true
  weight_cellcalling: false
  save_geometry_diagnostics: true
  save_loss_diagnostics: true
  patch_manifest: null
  manifest_seed: 1234
```

Old configurations without `v1` default to disabled. `experiment_dirs` also
accepts an optional `output_root`; when supplied with a non-`last` `dir_id`, it
provides a distinct deterministic run directory without changing the input
`data_dir`.

## Local diagnostics and remote commands

The repository does not contain a validated CellSPA implementation. V1 keeps
the standard connected mask and cell-gene-matrix outputs compatible with a
later external evaluator; no approximate Purity F1 or contamination metric is
implemented here.

Run synthetic tests:

```bash
python -m pytest -q \
  tests/test_v1_geometry_ambiguity.py \
  tests/test_v1_loss_weighting.py \
  tests/test_resource_monitor.py
```

E1 geometry sanity:

```bash
python scripts/v1/geometry_sanity.py \
  --config <REMOTE_BASELINE_OR_V1_CONFIG> \
  --output-dir <REMOTE_V1_E1_OUTPUT> \
  --num-rois 5
```

E2 threshold sweep:

```bash
python scripts/v1/threshold_sweep.py \
  --config <REMOTE_CONFIG> \
  --taus 0.05 0.10 0.15 0.20 0.30 \
  --output-dir <REMOTE_V1_E2_OUTPUT>
```

Generate E3/E4/E5 configurations. Add `--num-patches <N>` to create a fixed,
seeded manifest from existing shift-0 patches, or pass `--patch-manifest`:

```bash
python scripts/v1/make_dev_configs.py \
  --baseline-config <REMOTE_BASELINE_CONFIG> \
  --output-dir <REMOTE_DEV_CONFIG_DIR> \
  --experiment-output-root <REMOTE_DEV_RUN_ROOT> \
  --steps 1000
```

The repository's training CLI uses `--config_dir`. Run each desired generated
configuration separately:

```bash
python -m bidcell.model.train --config_dir <REMOTE_DEV_CONFIG_DIR>/<E3_E4_OR_E5_CONFIG>.yaml
```

For the final 4000-step run, supply a real config that retains the official
preprocessing, nuclei, annotation, optimizer, learning rate, one epoch, patch
size 48, `total_steps: 4000`, and `test_step: 4000`, with only the V1 block as
the scientific change:

```bash
python -m bidcell.model.train --config_dir <REMOTE_FINAL_V1_CONFIG>
python - <<'PY'
from bidcell import BIDCellModel
model = BIDCellModel("<REMOTE_FINAL_V1_CONFIG>")
model.predict()
PY
```

Tier 3 locality/change analysis:

```bash
python scripts/v1/locality_eval.py \
  --baseline-mask <REMOTE_BASELINE_CONNECTED_MASK> \
  --v1-mask <REMOTE_V1_CONNECTED_MASK> \
  --baseline-config <REMOTE_BASELINE_CONFIG> \
  --v1-config <REMOTE_V1_CONFIG> \
  --bands-um 1 2 3 5 \
  --output-dir <REMOTE_TIER3_OUTPUT>
```

Training, inference, and postprocessing merge their measurements into
`<experiment>/evaluation/tier4/resource_summary.json`. CUDA fields are null on
CPU-only hosts. CPU RSS is process-level; the postprocessing value does not
aggregate the independent peak RSS of every multiprocessing child. The main
static V1 overhead is CPU work for per-patch nearest-nucleus queries and the
temporary distance channels used only before the unchanged network forward
pass. A bounded 256-nucleus tree cache prevents unbounded tree retention.
