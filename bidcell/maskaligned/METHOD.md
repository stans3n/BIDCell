# MaskAligned A ROI pilot implementation

Scientific version: MA-A-PILOT-1.1. The authoritative frozen spatial selection,
training sequence and screening thresholds are in the run's `spec.json` and
`frozen.json`. Calibration uses only its first 50 training views. No dev method
result is used to choose a configuration. The final endpoint is update 500.

The original baseline source modules are unchanged. Dataset construction uses
the original nucleus-in-patch candidate list, every candidate SearchArea, the
original marker dilation/type priors and augmented-input construction. Cached
backbone logits are actual forwards of each keyed flip/rotation, never rotated
unaugmented logits. All model parameters and BN buffers remain frozen in eval.
The saved checkpoint is strict-loaded; its historical optimizer is not loaded.

## Representation and controls

The residual gate is candidate SearchArea AND SearchArea count >=2 AND outside
every native patch nucleus. It covers all eligible pixels, including pixels
with zero detected transcripts. Full global nuclear masks provide Euclidean
distance and equivalent radius; pixel spacing is the audited 1 micrometre.
Geometric competitor minimizes normalized distance among other supported
candidates; ties use canonical ID. Score competitor maximizes baseline fg-bg
among other supported candidates, independently. IDs never enter the MLP.

E04 absolute and E05 relative each use a 14→32→32→1 shared MLP, 1569 parameters.
E05 deterministically re-expresses exactly the E04 information as differences;
this tests representation, not additional candidate information.
The optional nonlinear normalized-margin coordinate from the broader proposal
is not added in this minimal pilot; this is the frozen difference-based
relative representation, not a claim to test every possible relative feature.
E02 receives
only score/self-competitor, candidate count and density; its remaining inputs
are zero. It has 1569 nominal parameters but 1249 effectively connected ones.
E03 follows the plan's simple clipped absolute-distance rule
`clip(-beta*dtilde_self+b,-1,1)`, restricted to the same gate. Train-only median
distance and logit scales define beta and b once. E00 and E01 are zero adapters;
acceptance separately exercises zero-initialized E02/E04/E05 heads.

The last layer is zero initialized. Learned delta is tanh(head), kappa=1.
Only foreground logits receive delta; background logits remain bitwise equal.
All learned methods receive the same seed, actual 500 optimizer updates,
native candidate groups, sample/augmentation sequence, Adam LR0.0003 and one
common residual penalty. Before training, the draft lambda0.1 was replaced by
a common train-only gradient-scale calibration, with no dev tuning: across
the first10 nonempty-gate views, use the median of
`0.5 * norm(dL_native/dz_fg at delta=+0.1 on G) * sqrt(number of G entries)`.
This targets10% of the native logit-gradient norm at residual0.1 and
approximately100% at residual1, preserving the original loss reductions.
The calibration overlay and all individual scales are saved; no sweep or
method-specific fitting is performed.
The actual frozen common value is lambda=69.95866394042969. The ten observed
candidate counts were3–14, gate counts520–4013 and native gradient norms
2.05–3.38; draft lambda0.1 would give only0.0003–0.0009 regularizer norm at
delta0.1. E03 freezes beta=0.8890590740563818 and b=1.2457919120788574.

## Exact original six losses

Let C denote the native candidate count and P=48×48. All six audited weights
are1. The implementation invokes original classes rather than copying or
changing their formulas. For reporting, positive and negative contributions
are isolated by setting the other weight to0 in that same original class;
training calls the original combined positive-negative function.

| Term | Actual native formula/reduction | Residual gradient interpretation |
|---|---|---|
| nuclei encapsulation | full C×P mean two-class CE against binary self-nucleus | nuclear-exterior target is background, hence nonzero gradient in G; not a nuclear-interior-only constant |
| cell calling | full-patch CE against SearchArea, sum/C | foreground pressure in candidate-supported G; source is SA, not owner truth |
| oversegmentation | ReLU(sum sigmoid(p_fg×nonselfnucleus−0.5) − sum sigmoid(p_fg×selfnucleus−0.5))/C | smooth proxy, including its original nonzero sigmoid floor; no topology derivative |
| overlap | sum ReLU(sum_c sigmoid(p_fg,c×outside_all_nuclei−0.5) − outside_all_nuclei)/(C×P) | complete candidate group and original reduction retained |
| positive marker | full-patch CE against original dilated positive-marker×SA binary map, sum/C | target0 outside positive markers still gives background gradient; zero-positive markers do not make this loss constant |
| negative marker | sum sigmoid(p_fg−0.5)×negative_marker/C | only negative-marker locations contribute, no-signal views reported |

The regularizer is mean(delta²) over candidate gate entries. No owner labels,
owner CE, CellSPA training loss, legacy ambiguity weighting, router or image
encoder is introduced. First-step hidden-layer gradients are legitimately zero
because the output layer starts at zero; after its update they can propagate.

## Native output and ROI limits

Native `get_seg_mask` compares all foreground probabilities with mean background
probability and restores nuclei. It does not gate baseline predictions by
SearchArea, so this quirk is deliberately preserved. Canonical candidate order
is the native sorted nucleus order. The global dual-shift fill mask is cropped
using the original row/column/gap selection. Native `postprocess_connect` then
`remove_islands` and final nucleus restoration operate in each 672-pixel ROI
frame. A fixed random priority from the full block candidate universe controls
cell processing order identically across methods, even if a label disappears.

This is same-scope ROI native processing, not historical full-field1024-chunk
equivalence. No whole-field E01 claim is made. Native prediction support is
bounded by patches containing each nucleus (<48 pixels beyond nuclear bbox).
The conservative additional8 pixels cover morphology. Every candidate whose
full bbox+56 intersects a core must be wholly covered by the context; acceptance
hard-fails otherwise. Complete-cell eligibility requires bbox+56 inside core.

The initial96-pixel halo failed the complete competitor-context check before
any method training or endpoint results:1482 candidate contexts exceeded the
frame, maximum34pixels. Version1.1 adds48pixels uniformly on every side
(halo144/context672), preserving all original384core coordinates, density
ranks, complete-cell IDs and500 training views. It checks every global nuclear
bbox for relevant core influence, image bounds and cross-split candidate ID
disjointness. The original specification, probe cache and failures remain
preserved; their actual costs count toward the same budget.

Capture and foreground use all384×384 core pixels/transcripts. Matrix and
cell-area/fragmentation summaries use the same fixed complete canonical cells,
including zero-count cells. Background and assignments to excluded edge cells
are reported separately. All coordinates reuse the audited global OpenCV
nearest-axis/ties-to-even transcript cache. Owners and counts are recomputed
from each method's own final mask and carry mask/coordinate hashes.

## Evaluation and interpretation

Both existing engineering annotation profiles remain separate. Official
CellSPA functions are unchanged; the ROI wrapper only skips a B-to-T curve
when no eligible B/T pair exists and records missing_domain, allowing the
second profile to execute. Unexpected evaluator errors fail their stage.
Official primary Purity F1 remains missing_definition, never replaced by
positive-marker F1, purity proxy or a favorable endpoint.

Pooled screening thresholds: capture decline ≤0.5 percentage points; absolute
relative foreground and median eligible-cell area change ≤5%; final8-connected
disconnected-cell increase ≤0.1 percentage points. Per-block values and
pre-connect changes are diagnostics. Passing these vetoes does not establish
biological improvement. Historical whole-slide backbone/preannotation exposure
means held-out adapter spatial evaluation, not an independent pipeline test.
