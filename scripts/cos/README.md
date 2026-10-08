# COS-min-v1 four-arm continuation

This implementation follows `BIDCell_COS_FourArm_Overnight_Change_Plan_20261008.md`.
The four production configurations replay the same 4,418 historical views, including
418 empty views, for exactly 4,000 new updates per arm. Parent B1/G checkpoints are
resolved through the historical evaluation registry and complete receipts. A fresh
Adam optimizer is used; trained G tensors and native BN are loaded without reset.

`model.py` adds the zero-initialized shared 320→1 foreground residual only when
`cos_enabled`. `ownership.py` implements the full-patch candidate domain, stable
background-first/global-ID tie rule, background-compatible joint scores and nuclear
anchor. The head adds 321 parameters and is a linear reparameterization; its scores
are not calibrated ownership probabilities. Original losses consume the corrected
binary logits; Unknown candidates lose only POS/NEG supervision with total real K
as the denominator. Historical annotation indices are float64 representations of
integers; exact integral values are losslessly normalized before range validation.

Frozen augmentation uses NumPy flip/clockwise rot90, proven pixel-identical to all
eight historical imgaug transformations. This avoids imgaug's lazily initialized
private RNG consuming an extra NumPy draw after process restart. Checkpoints retain
all four RNG families, model/BN/optimizer, actual update and view cursor. Validated
milestones precede latest publication; interrupted publication resumes the newest
validated milestone, while corrupted bytes, counters or dependencies fail closed.

The native postprocess tile edge remains 1024 pixels, producing 48 full-field tiles;
48×48 is the network patch size. Raw two-shift maps are captured before patch nuclear
restoration. Ordered tile merge, split cleanup and pre-global-restoration endpoints
are separately saved. Fixed probes contain base and corrected logits/probabilities,
residuals, candidate support, margins, entropy and nuclear competition diagnostics.

The detached queue runs each complete arm, then each pair, in the frozen order:
B1_CONT → B1_COS → B1 pair → G_CONT → G_COS → G pair → four-arm summary.
Three accepted evaluation profiles execute on each new arm's own final matrix.
Intermediate maps use frozen labels without repeating scClassify. Purity uses the
complete four-arm set only. Scientific interpretation remains pending independent
analysis after the user resumes the goal; negative results are legitimate outcomes.

Production commands (absolute config path required):

```
PYTHON -m scripts.cos.runner --config QUEUE --freeze
scripts/cos/launch.sh --config QUEUE
PYTHON -m scripts.cos.status --config QUEUE
scripts/cos/resume.sh --config QUEUE
PYTHON -m scripts.cos.validate_checkpoint --config ARM --update 100 --output RECEIPT
```

`launch.sh` resolves the interpreter from the frozen queue and creates a new OS
session. `resume.sh` rejects a live runner or child and retains verified stage
receipts; it does not silently reset optimizers or discard failed science stages.
There is no LLM call, network dependency or user-analysis barrier in the queue.

Tests include mathematical/Unknown contracts, real B1/G zero-init and gradients,
new-process bitwise continuation including consumed RNG, a four-arm real regional
pipeline with actual R calls, and independent subprocess publication/failure tests.
Mini outputs are isolated test fixtures and are never full-field scientific results.
