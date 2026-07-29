# BIDCell V1 Tier 1-5 benchmarking

## Responsibility boundary

Training-time loss, overlap, timing, and PyTorch-memory hooks live in
`bidcell/`. GPU utilization, device VRAM, CPU/RAM, post-prediction biology,
morphology, boundary, pseudo-reference, and cross-run comparisons live in
`scripts/benchmarking/`. `run_all_tiers.py` is a post-training aggregator; it
does not replace `v1_training_metrics.csv`.

Every requested metric produces a manifest status:

- `computed`: required inputs were present and the implementation ran;
- `skipped_missing_input`: an optional input was absent;
- `not_applicable_v1`: the metric requires a V2/V3 mechanism;
- `not_implemented`: the required validated method is an explicit extension;
- `failed`: an exception occurred, is recorded, and makes the command non-zero.

Missing optional inputs never become empty arrays, inferred constants, random
numbers, or false zeroes.

## Tier coverage and inputs

| Tier | Currently computed | Required inputs | Explicit extensions |
|---|---|---|---|
| 1 | cell count; assigned transcript fraction; cell transcripts, genes, area and density distributions; area, elongation, circularity, compactness, convexity, eccentricity and solidity | connected mask; transcripts and/or cell-gene matrix | marker purity requires marker lists plus cell annotation; scRNA concordance requires matched scRNA plus annotation; replicability requires another replicate; grouped/regional cell-type summaries require a validated annotation schema |
| 2 | per-step/aggregate candidate overlap; 1/2/3/5 µm boundary-band transcript fraction and assigned rate; optional baseline assignment changes; nucleus containment; zero-/multi-nucleus cells; missed/split proxies; 8-neighbour disconnection; merge/fragmentation proxies; extreme-area outliers | training CSV and/or connected mask, transcripts, nuclei mask; baseline mask for changes | assignment entropy, top1–top2 margin, and segmentation-assignment consistency are `not_applicable_v1` |
| 3 | training wall-clock and throughput; effective candidate throughput; PyTorch VRAM; external GPU/CPU/RAM summary; stage timing; checkpoint/run footprint; optional baseline overhead | run directory, training/resource/stage summaries | FLOPs/MACs remain `not_implemented` without a reliable model-aware counter |
| 4 | foreground Dice/IoU; boundary precision/recall/F1 at 1/2/3 µm; ASSD; HD95; Hungarian instance-matched IoU/Dice | connected mask and explicit vendor mask | AJI+, PQ, and detection F1 require validated benchmark-compatible matching |
| 5 | standardized multi-run mean/SD/optional normal 95% CI; baseline absolute and percentage change; ablation metadata table | multiple standardized benchmark manifests and/or baseline manifest | paired Wilcoxon/bootstrap requires a validated ROI/cell pairing, not just run-level scalars |

Tier 4 always labels its reference as vendor/pseudo-reference, not manual
ground truth. Tier 2 uses final connected-mask boundaries only for post-hoc
evaluation; they never become a V1 target.

## Running the post-training benchmark

```bash
python scripts/benchmarking/run_all_tiers.py \
  --run-dir RUN_DIR \
  --connected-mask CONNECTED.tif \
  --raw-prediction RAW.tif \
  --nuclei-mask NUCLEI.tif \
  --transcripts TRANSCRIPTS.csv \
  --cell-gene-matrix CELL_GENE.csv \
  --baseline-run-dir BASELINE_RUN_DIR \
  --vendor-mask VENDOR_PSEUDO_REFERENCE.tif \
  --replicate-run-dir V1_REPLICATE_RUN_DIR \
  --pixel-size-um 1.0 \
  --seed 1 \
  --output-dir RUN_DIR/metrics/benchmark
```

Omit unavailable optional arguments. The manifest will record the corresponding
metrics as skipped. Existing outputs are protected unless `--overwrite` is
explicit.

The standard output tree is:

```text
<run-dir>/metrics/
  training/v1_training_metrics.csv
  training/training_summary.json
  resources/resource_samples.csv
  resources/resource_summary.json
  benchmark/benchmark_manifest.csv
  benchmark/all_tiers_summary.json
  benchmark/tier1_cellspa_summary.json
  benchmark/tier1_cell_level.csv
  benchmark/tier2_boundary_summary.json
  benchmark/tier2_assignment_changes.csv
  benchmark/tier3_resource_summary.json
  benchmark/tier4_vendor_reference_summary.json
  benchmark/tier5_stability_summary.json
```

Recommended full-experiment order on the training machine is:

1. resource-monitored training;
2. inference and unchanged prediction combination/postprocessing;
3. final cell-gene matrix extraction;
4. the same `run_all_tiers.py` invocation for baseline and V1.

Preprocessing stages may be timed separately, but V1 overhead comparisons
should foreground training and inference rather than hide them inside an
end-to-end total. The resource wrapper accepts `--stage-name`, so the same
wrapper can independently record nuclei segmentation, expression-map
generation, patch generation, nucleus matrix extraction, preannotation,
training, inference, prediction combination, morphology, and final matrix
commands.
