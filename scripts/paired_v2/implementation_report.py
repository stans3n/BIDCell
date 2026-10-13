"""Build the implementer's evidence index, never independent admission."""
import subprocess
from .common import *


def build(root=DEFAULT_ROOT):
    root=Path(root);requirements=[]
    def row(key,status,claim,paths,limit=None):
        requirements.append(dict(id=key,status=status,claim=claim,evidence=[identity(root/p) for p in paths],limit=limit))
    for name in ('functional','initial_equivalence_cpu','initial_equivalence_cuda','resume_BASE_MATCH','resume_V2_INTERACT','runner_resume','resources','gateway'):
        r=read(root/'tests'/f'{name}.json')
        if r['status']!='passed':raise ValueError('required evidence incomplete: '+name)
        if 'producer' in r and r['producer']!=producer('train'):raise ValueError('training evidence producer changed: '+name)
    initial=read(root/'prepared/step_zero_receipt.json')
    if initial['source']!=producer('initial'):raise ValueError('initial producer changed')
    learned=read(root/'tests/learned_interaction.json')
    if learned['required_block_checks_status']!='passed':raise ValueError('learned block evidence incomplete')
    row('P01','passed','55 frozen historical source files and all immutable prepared/dependency/raw-map/native-patch hashes verified; isolated new checkout.', ['prepared/input_audit.json'])
    row('P02','passed','Run IDs BASE_MATCH/V2_INTERACT map explicitly to model kinds BASE/V2_INTERACT; immutable input root separated from each output root.', ['BASE_MATCH/configs/run.json','V2_INTERACT/configs/run.json'])
    row('P03','passed','Both original step-zero snapshots frozen before pilots; complete shared tensors/BN, empty Adam, independent Morph20261013/interaction20261014 RNG construction, common post-dataset Python/NumPy/Torch/CUDA/imgaug RNG exact.', ['prepared/step_zero_receipt.json'])
    row('P04','passed','Full old M-V2.2 chunk_size=0 retained; interaction after Morph at h3 before every downstream consumer; only interaction final projection zero.', ['prepared/step_zero_receipt.json','tests/functional.json'])
    row('P05','passed','Frozen 3998 visits,373 empty,3625 effective updates, original candidates/coordinates/augmentation/patch hashes; pilot and resumed sequence match the external frozen sequence.', ['prepared/input_audit.json','tests/runner_resume.json'])
    row('P06','passed','All six original loss implementations/reduction/weights and Unknown=-1 native marker indexing retained; no calibrated/V1 loss; full K FP32 original BN and Adam parameters.', ['tests/functional.json','tests/initial_equivalence_cpu.json','tests/initial_equivalence_cuda.json'])
    row('P07','passed','CPU original/BASE/zero-Morph/zero-interaction logits, losses, shared gradients, BN and first Adam states exact.', ['tests/initial_equivalence_cpu.json'])
    row('P08','passed','Same-host CUDA initial logits and BN exact, six losses rtol1e-6; gradients checked against five contemporary native repeats RMS <=2*max_repeat_RMS+1e-7; independent original Adam formula.', ['tests/initial_equivalence_cuda.json'], 'CUDA backward is not promised bitwise deterministic. No tolerance was enlarged.')
    row('P09','passed','Full-nucleus centroid/radius relative geometry and all eight flip/rot90 transforms independently checked; duplicate/nonpositive/missing IDs rejected.', ['tests/functional.json'])
    row('P10','passed','Independent nonzero FP64 per-position/head legal-neighbor softmax oracle and interaction candidate permutation; no self/type filtering or key truncation, exact K1/no-neighbor, K2 single-neighbor Q/K gradients zero.', ['tests/functional.json'])
    row('P11','passed','BASE diagnostic sidecar executes the update100 boundary; parameter/BN/Adam/.grad hashes and complete RNG remain exact and external cursor unchanged.', ['tests/pilots/BASE_MATCH/diagnostics/probe_0100.json','tests/runner_resume.json'])
    row('P12','passed','E2 real first output projection gradients; updates2/32/64/100 fixed eight multi-neighbor probes have active backbone/Morph/interaction deltas and gradients including Q/K weights; residual/coverage/disabled logits/owner diagnostics.', ['tests/pilots/V2_INTERACT/first_interaction_gradients.json','tests/learned_interaction.json'])
    row('P13','passed','Learned interaction block candidate permutation <=1e-6, illegal weights exact zero, no-neighbor residual exact zero and K1 exact.', ['tests/learned_interaction.json'])
    row('P14','passed','Both arms restore model/BN/Adam/Python/NumPy/Torch/CUDA/imgaug/cursor exactly; fresh-construction next logits/BN/RNG exact, gradients within five-repeat envelope and both updates obey independent Adam formula; all header fields reject drift.', ['tests/resume_BASE_MATCH.json','tests/resume_V2_INTERACT.json'])
    row('P15','passed','Actual runner25->100 continuation preserves external sequence and removes uncommitted tail; gateway recovers a partial final JSON line while refusing any changed/truncated committed prefix; checkpoints every25 updates.', ['tests/runner_resume.json','tests/gateway.json'], 'The fixture simulates crash-window log bytes; it does not claim an OS-kill experiment.')
    row('P16','passed','Actual train maximum K40 and actual prediction maxima including K41, both native shifts, complete candidates, real hostCUDA; every measured device estimate retains >=1GiB.', ['tests/resources.json'], 'Timing uses maximum-K repeats plus15 fixed representative views per shift; full-slide timing remains an estimate.')
    row('P17','passed','New E0/E2 pilot small ROI: both native shifts, merge, original seeded postprocess, own integer matrix independent native owner/conservation validation, own QC/scClassify and required evaluation files.', ['tests/roi/complete.json'], '144x144 interface admission only; no new formal endpoint or full-slide efficacy conclusion.')
    row('P18','passed','E1 full historical train/predict/postprocess/matrix/evaluation producers and scientific core checked;36,344 referenced files hashed; complete18,170/17,248 shift coverage and native postprocess ordering.', ['tests/historical/provenance.json','reviews/historical_endpoint_reuse.json'])
    row('P19','passed','Old frozen subprocess and new subprocess strictly load the same E1 endpoint on fixed ROI: candidate order/logits and both shift/merge/pre/fill/final masks exact; tile/split order exact.', ['tests/roi/complete.json'], 'Historical full-slide reuse rests on the separate full provenance audit, not on extrapolating ROI equality.')
    row('P20','passed','E1 existing own integer matrix enters authoritative scorer; QC IDs, reused verified scClassify labels, per-cell finite/NA and six metrics and aggregate values agree atol1e-10 rtol1e-8.', ['tests/historical_scores/complete.json'])
    row('P21','passed','Immutable byte-copy of authoritative18 protocol and machine contract with source provenance; old runtime protocol scientific core/reference model/fixed population verified against that authority; real paper R environment resolved from prepared receipt.', ['prepared/authoritative_protocol.json.source.json','prepared/experiment_contract.json.source.json','tests/historical_scores/complete.json'])
    row('P22','passed','Independent callable train/predict/postprocess/matrix/evaluate stage gateway, dependency receipts and semantic validators; no automatic stage advance, own matrix can be handed off before score opening; missing formal stages rejected.', ['tests/gateway.json'], 'Full new formal stages are intentionally not run by implementation agent; parent supplies independent orchestration and final source-review enforcement.')
    row('X01','failed','Additional learned WHOLE-NETWORK FP32 reverse-candidate elementwise atol=rtol=1e-6 check fails; this threshold is retained and is separate from the plan-defined block permutation check.', ['tests/learned_interaction.json','tests/permutation_diagnostic_cpu.json','tests/permutation_diagnostic_cuda.json','tests/learned_checks_original_failed.py','logs/learned_checks.log'], 'B must independently assess the residual limitation. Fixed probe owner changes are0; do not generalize to all views.')
    row('X02','not_verified','Earlier ad hoc full-network CPU float64 exploratory rows show max~2.14 and have no validated test harness; retained for complete failure/debug disclosure.', ['tests/permutation_cpu_diagnostic.json'], 'Not a passed test, not formal FP32 execution, not evidence for whole-network FP64 equivariance; original mixed-precision Morph internally casts attention arithmetic tofloat32. Root cause of this exploratory construction has not been established.')
    row('O01','not_verified','Independent B review, final orchestration integration/review, admitted receipts, formal full-slide runs and new full evaluations remain pending.', [], 'Parent-owned next phase; no admission or formal-start claim in this implementation report.')
    commits=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    resources=read(root/'tests/resources.json');diagnostic=read(root/'tests/permutation_diagnostic_cuda.json')
    result=dict(status='implementation_complete_pending_independent_review_with_disclosed_extra_failure',independent_approval=False,formal_training_started=False,
        git_commit=commits,checkout=str(REPO),branch='feature/v2-paired-20261014',root=str(root),
        requirements=requirements,scientific_train_producer=producer('train'),requirements_source=identity('/home/zwan0726/work/docs/BIDCell_Tonight_Paired_V2_Experiment_Plan_20261014.md'),
        training_initial_source_unchanged_since_successful_pre_pilot_freeze=True,
        permutation_controls={k:dict(logits=v['logits'],native_id_owner_changes=v['native_id_owner_changes']) for k,v in diagnostic['cases'].items()},
        raw_failed_attempt_directories=['tests/prepare_attempts/lazy_imgaug_rng_failure','tests/learned_checks_original_failed.py','logs/learned_checks.log','tests/permutation_cpu_diagnostic.json'],
        resources=identity(root/'tests/resources.json'),gate_record_kind='implementation evidence only; no status=admitted produced')
    atomic_json(root/'reviews/implementation_report.json',result)
    for run in RUNS:
        gate=dict(status='implementation_checks_complete_pending_independent_review',run_id=run,model_kind=RUNS[run],
            base_only=run=='BASE_MATCH',lambda_max=0,eta=0,loss='unchanged original six loss terms only',
            train_producer=producer('train'),source_report=identity(root/'reviews/implementation_report.json'),
            initial=identity(root/'prepared'/f'{run}_step_zero.pth'),reused_old_admission=False,independent_approval=False)
        atomic_json(root/'reviews'/f'{run}_implementation_gate.json',gate)
    lines=['# Paired V2 implementation evidence','',
        'Implementation agent A; actual Astra / Max verified by the coordinator. This is an implementation report, not independent approval. No formal training has started.',
        '',f'Checkout: `{REPO}`. Branch `feature/v2-paired-20261014`; implementation commit `{commits}`.',
        '', '## Requirement-by-requirement evidence','', '| ID | Status | Verified statement and scope | Evidence |','|---|---|---|---|']
    for item in requirements:
        note=item['claim']+(' LIMIT: '+item['limit'] if item['limit'] else '')
        links='; '.join(f"[{Path(x['path']).relative_to(root)}]({x['path']})" for x in item['evidence'])
        lines.append(f"| {item['id']} | {item['status']} | {note} | {links} |")
    lines += ['', '## Recorded numerical limitation','',
        'The extra whole-network 1e-6 elementwise gate remains failed. With ordinary advanced indexing, input strides change; the first discrepancy is already in native conv1. Same-checkpoint no-interaction and no-Morph controls reproduce this, and repeat forwards are bitwise identical. Preserving original input/auxiliary strides makes Morph-only and native-only whole-network permutations bitwise identical. In the full model, same-stride h3 first differs at interaction by max2.3841858e-7 (RMS5.3427e-10); decoder propagation yields logit max5.8412552e-5 (RMS5.59115e-6). All these fixed-probe owner comparisons have0 changes. The independent FP64 interaction oracle and learned fixed-h3 interaction check pass. The backward noise envelope is not used to justify forward differences.',
        '', 'The earlier `permutation_cpu_diagnostic.json` is retained as unqualified exploratory evidence, including the unresolved float64 max~2.14 rows. It is not a passing admission check. The formal model remains FP32. No native input layout, layer, tolerance, loss, or frozen checkpoint was changed to make this extra check pass.',
        '', 'The first prepare attempt failed on lazy imgaug initialization changing NumPy state. It occurred before pilots. Its partial snapshot and failure are archived under `tests/prepare_attempts/lazy_imgaug_rng_failure/`. The successful pair was rebuilt from original untrained inputs after making initialization and the BASE diagnostic-sidecar construction RNG-pure. Successful frozen headers/snapshots have never been rewritten.',
        '', '## Stage API and receipts','',
        '`scripts.paired_v2.stages.run_stage(config_path, stage)` is the formal callable gateway. Stages: `train`, `predict`, `postprocess`, `matrix`, `evaluate`. It returns a validated bound receipt and never automatically advances. `validate_stage(config_path, stage, require_bound=True)` is read-only acceptance. `dependencies` and `native_receipt` expose the expected dependency identity and native receipt path. GPU locks and bounded retries belong to the caller; do not acquire the shared GPU lock twice.',
        '', 'Each arm writes `checkpoints/{latest,final}.pth`, `effective_sequence.jsonl`, immutable attempt logs, `diagnostics/`, `predictions/<run_id>/`, `matrix/<run_id>/`, `evaluation/<run_id>/`, and bound `stages/<stage>.json`. Each native stage has its own `complete.json` and content hashes. Prediction writes128-patch chunks for both complete native shifts and validates cached dependencies. Native postprocessing preserves seed1234/tile1024/tile seed rule and ascending128-ID split chunks. Matrix is the mature independent native lookup/conservation implementation. Evaluation runs both existing profiles and fixed-population diagnostics against the authority snapshot.',
        '', 'Required parent order: freeze original-nucleus type/support/density/grid strata before new training; execute BASE matrix and freeze BASE count quartiles before opening any new score outputs; then evaluate each own matrix. Public formal callers must use `run_stage`, which repairs only the uncommitted JSONL suffix after validating its exact committed prefix. Direct `training.train` is the internal pilot/compute function.',
        '', 'BASE has a new implementation gate record with `base_only=true, lambda_max=0, eta=0`; E2 has its own record. Neither reuses or renames old admission. Parent may create `admission/<run_id>.json` with `status=admitted` and exact `train_producer` only after independent B approval and final full-source binding.',
        '', '## Final source integration and immutable initial state','',
        'Scientific component producer maps exclude orchestration/test additions. Complete review bundles additionally include every Python/R/shell source in bidcell/scripts/tests-new_methods, including paired_v2 and paired_v2_pipeline. Frozen mature imported modules are covered by that whole-tree bundle and verified unchanged against the55-file historical source list, except the disclosed h3 insertion. Parent must verify the final complete bundle before execution; component headers alone are not a substitute for final-source review.',
        '', 'After source is committed, run `python -m scripts.paired_v2.finalize --label implementation`. It reconstructs both original initial models and compares all tensors/BN/Morph/Adam/RNG/headers with existing snapshots. It freezes a versioned source bundle and environment lock; it never rewrites step zero. After parent orchestration is added, run with a new label such as `final_integration_v1`, obtain B review bound to that exact bundle, and use `--verify /absolute/path/to/bundle.json` in preflight. Existing labels cannot be overwritten with changed content. Any scientific change that invalidates step zero must be explicitly versioned, rebuilt from ORIGINAL untrained input, and revalidated; do not edit old headers or receipts.',
        '', '## Executed and reusable commands','',
        f'Working directory: `{REPO}`. Python: `/home/zwan0726/miniforge3/envs/bidcell_local/bin/python`. Set `OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4`. CUDA commands require the actual MCV-CGB4 host, not a sandbox without the device. During isolated admission use `flock -n /home/zwan0726/work/outputs/cos_gpu.lock`; the parent runner will own that lock for formal work.',
        '', '```bash',f'cd {REPO}',
        'export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4',
        'python -m scripts.paired_v2.cli prepare',
        'python -m scripts.paired_v2.admission_tests functional',
        'python -m scripts.paired_v2.admission_tests cpu',
        'python -m scripts.paired_v2.admission_tests cuda',
        'python -m scripts.paired_v2.admission_tests pilot --run-id BASE_MATCH',
        'python -m scripts.paired_v2.admission_tests pilot --run-id V2_INTERACT',
        'python -m scripts.paired_v2.admission_tests resume --run-id BASE_MATCH',
        'python -m scripts.paired_v2.admission_tests resume --run-id V2_INTERACT',
        'python -m scripts.paired_v2.admission_tests resources',
        'python -m scripts.paired_v2.runner_resume_test',
        'python -m scripts.paired_v2.gateway_tests',
        'python -m scripts.paired_v2.learned_checks',
        'python -m scripts.paired_v2.permutation_diagnostic --device cuda',
        'python -m scripts.paired_v2.permutation_diagnostic --device cpu',
        'python -m scripts.paired_v2.reuse_tests provenance',
        'python -m scripts.paired_v2.reuse_tests roi',
        'python -m scripts.paired_v2.reuse_tests scores',
        'python -m scripts.paired_v2.reuse_tests finish',
        '```','',
        'These exact test entries have been exercised. A new finalize label is a reviewed-source snapshot operation. No `train --run-id ...` without pilot arguments was executed as a formal experiment.',
        '', '## Resource scope and remaining work','']
    for run,v in resources['arms'].items():
        t=v['train'];lines.append(f"- {run}: train K40 peak reserved {t['peak_reserved']/1024**3:.3f}GiB, estimated device headroom {t['headroom']/1024**3:.3f}GiB, {t['minutes_per_1000']:.3f}min/1000updates. Shift0/24 representative prediction rates {v['prediction']['0']['minutes_per_1000']:.3f}/{v['prediction']['24']['minutes_per_1000']:.3f}min/1000patches; actual maximum groups also measured.")
    lines += ['', 'Independent review and parent final integration remain pending. Parent owns locks/state/status/timeout policy, whole-slide fixed strata, paired/bootstrap decisions, full output manifest, independent hosting and launch. Whole-slide new endpoint acceptance remains a runtime requirement. Historical whole-slide E1 was audited and reused, not recomputed. No main table or context has been changed by the implementation agent.']
    lines=[line.replace('python -m','/home/zwan0726/miniforge3/envs/bidcell_local/bin/python -m') if line.startswith('python -m') else line for line in lines]
    (root/'reviews/implementation_report.md').write_text('\n'.join(lines)+'\n')
    return result


if __name__=='__main__':print(build()['status'])
