"""Required semantic evidence collector; independent reviewer remains a separate gate."""
import argparse,json,subprocess,sys
from pathlib import Path
from scripts.crgf.common import atomic_json,identity,sha256,verify_outputs,source_identity
from .validation import validate_stage
REQUIRED=('tests/post_math.json','tests/post_resume_v2/complete.json','tests/post_real_v2/complete.json','diagnostics/B_native_replay/complete.json',
    'prepared/coverage.json','prepared/exposure/complete.json','tests/equivalence_shared_v1.json','tests/shared_loss_v1.json','tests/cpu_equivalence_shared_v1.json',
    'diagnostics/numeric_admission_v2.json','tests/checkpoint_faults_v2.json','diagnostics/gradient_diagnostics_shared_v1.json',
    'tests/report_population_gzip_v2.json','tests/aggregate_mini_shared_v1.json','tests/runner_engineering_final.json',
    'tests/mini_v5/mini_complete.json','tests/deployment_test.json','tests/resources.json')

def read(path):return json.loads(Path(path).read_text())
def capture_method_sources(root):
    from .training import TRAIN_SOURCES
    paths=TRAIN_SOURCES+['scripts/bs4/tests/numeric.py','scripts/bs4/tests/equivalence.py','scripts/bs4/tests/cpu_equivalence.py']
    dest=Path(root)/'tests/method_execution_sources.json';value=dict(status='captured',source=source_identity(paths))
    if dest.exists() and read(dest)!=value:raise ValueError('method source changed; preserve evidence and rerun affected tests')
    if not dest.exists():atomic_json(dest,value)
    return dest

def validate_sources(receipt):
    repo=Path(__file__).resolve().parents[2]
    for field in ('execution_source','source'):
        mapping=receipt.get(field)
        if not isinstance(mapping,dict):continue
        if 'path' in mapping and 'sha256' in mapping:mapping={mapping['path']:mapping}
        for name,obj in mapping.items():
            if isinstance(obj,str):path=repo/name;expected=obj
            elif isinstance(obj,dict) and 'path' in obj and 'sha256' in obj:path=Path(obj['path']);expected=obj['sha256']
            else:raise ValueError('invalid executed-source identity '+name)
            if sha256(path)!=expected:raise ValueError('executed source differs from current '+str(path))
    return receipt


def validate_b_replay(root):
    import numpy as np,pandas as pd,tifffile
    from .postprocess import validate_tile
    root=Path(root);folder=root/'diagnostics/B_native_replay';done=read(folder/'complete.json');verify_outputs(done)
    validation=read(folder/'validation.json');postpath=folder/'predictions/B/postprocess/complete.json';post=read(postpath);verify_outputs(post);validate_sources(post['dependencies'])
    if validation.get('status')!='passed' or post['dependencies']['arm']!='B' or post['dependencies']['rule']!='B' or post['dependencies']['seed']!=20261008:raise ValueError('B native identity')
    if validation['postprocess']!=identity(postpath):raise ValueError('B postprocess receipt drift')
    for row in validation['native_masks']:
        for k in ('historical','replay'):
            if identity(row[k]['path'])!=row[k]:raise ValueError('B native endpoint identity drift')
        a=tifffile.memmap(row['historical']['path']);b=tifffile.memmap(row['replay']['path'])
        if a.dtype!=np.uint32 or b.dtype!=a.dtype or a.shape!=b.shape or not np.array_equal(a,b):raise ValueError('B native endpoint is not bitwise historical')
    manifest=read(folder/'predictions/B/postprocess/manifest.json');base=post['dependencies'];shape=tifffile.memmap(base['pre']['path']).shape;size=base['patch_size']
    coords=[(y,x) for y in list(range(0,shape[0]-size,size))+[shape[0]-size] for x in list(range(0,shape[1]-size,size))+[shape[1]-size]]
    if len(manifest['tiles'])!=len(coords) or validation['tiles_verified']!=len(coords):raise ValueError('B tile coverage incomplete')
    for tile,(y,x) in zip(manifest['tiles'],coords):
        if identity(tile['path'])!=tile:raise ValueError('B tile receipt changed')
        validate_tile(tile['path'],dict(base=base,coords=[y,x],shape=[size,size],dtype='uint32'))
    canonical=validation['new_postconnect_endpoint']
    if identity(canonical['path'])!=canonical:raise ValueError('B canonical endpoint drift')
    stage=folder/'evaluation/diagnostics/stages';rec=validate_sources(read(stage/'validation.json'))
    if not all(rec.get(k) for k in ('integer_conservation','cell_gene_gain_loss_verified','all_required_endpoints')):raise ValueError('B stage validation incomplete')
    required={'raw_shift0','raw_shift24','shift0','shift24','pre','postconnect_canonical','tile_merge','split_cleanup','before_global_nuclear_restore','final'}
    if set(rec['count_artifacts'])!=required:raise ValueError('B stage count coverage')
    for name,obj in rec['count_artifacts'].items():
        if identity(obj['path'])!=obj:raise ValueError('B stage count identity drift')
        data=np.load(obj['path']);ids=data['cell_id'];counts=data['counts']
        if len(np.unique(ids))!=len(ids) or counts.shape[0]!=len(ids) or counts.dtype.kind not in 'iu' or (counts<0).any():raise ValueError('B stage count schema')
        if name=='final':
            q=read(root/'configs/queue.json');matrix=Path(q['historical_methods']['B']['matrix']);ownids=pd.read_csv(matrix/'cell_mapping.csv').canonical_cell_id.to_numpy()
            positions=pd.Index(ids).get_indexer(ownids)
            if (positions<0).any() or not np.array_equal(counts[positions],np.load(matrix/'counts.npy',mmap_mode='r')):raise ValueError('B final native counts differ')
        data.close()
    return {str(p.relative_to(folder)):identity(p) for p in stage.rglob('*') if p.is_file()}


def validate_post_real(root,r):
    from .postprocess import validate_tile
    root=Path(root);rows=r['rows'];expected={(a,xy) for a in ('B','P1','P2') for xy in ((0,0),(1024,1024))}
    if len(rows)!=6 or {(v['arm'],tuple(v['coords'])) for v in rows}!=expected:raise ValueError('production tile benchmark coverage')
    for key in ('oracle_validation','full_B_cost'):
        if identity(r[key]['path'])!=r[key]:raise ValueError('benchmark B authority receipt drift')
    for row in rows:
        obj=row['receipt']
        if identity(obj['path'])!=obj:raise ValueError('production tile receipt drift')
        saved=read(obj['path']);actual=validate_tile(obj['path'],saved['dependencies'])
        if actual['arm']!=row['arm'] or actual['coords']!=row['coords'] or actual['shape']!=[1024,1024] or row['diagnostics']!=actual['diagnostics']:raise ValueError('benchmark row identity drift')
        if row['bytes']!=sum(v['bytes'] for v in actual['outputs'].values()):raise ValueError('benchmark output byte mismatch')
        if row['arm']=='B':
            if 'wall_seconds' in row:raise ValueError('B reuse must not fabricate a new tile wall time')
        elif row.get('wall_seconds',0)<=0 or row.get('peak_rss_bytes',0)<=0:raise ValueError('P actual production measurement missing')
    for xy in ((0,0),(1024,1024)):
        subset=[v for v in rows if tuple(v['coords'])==xy]
        if len({json.dumps(v['diagnostics']['candidate_hashes'],sort_keys=True) for v in subset})!=1 or len({v['diagnostics']['rng_enter_islands'] for v in subset})!=1:raise ValueError('P common proposal/RNG contract')


def collect(root):
    root=Path(root);q=read(root/'configs/queue.json');failures=[];evidence={}
    for rel in REQUIRED:
        path=root/rel
        try:
            r=validate_sources(read(path))
            if rel=='tests/resources.json':
                if r.get('version')!='BS4-storage-v2' or not r.get('measured') or not r.get('sufficient'):raise ValueError('actual resource measurement or capacity insufficient')
            elif r.get('status') not in ('passed','succeeded'):raise ValueError('evidence did not pass')
            if r.get('outputs'):verify_outputs(r)
            if rel=='tests/post_real_v2/complete.json':validate_post_real(root,r)
            if rel=='diagnostics/B_native_replay/complete.json':evidence['B_native_recursive_artifacts']=validate_b_replay(root)
            evidence[rel]=identity(path)
        except (OSError,ValueError,KeyError) as e:failures.append(rel+': '+str(e))
    try:
        num=validate_sources(read(root/'diagnostics/numeric_admission_v2.json'))
        from .training import TRAIN_SOURCES
        expected=source_identity(TRAIN_SOURCES+['scripts/bs4/tests/numeric.py'])
        if num.get('execution_source')!=expected:raise ValueError('numerical execution source set incomplete/drift')
        for relative in ('tests/equivalence_shared_v1.json','diagnostics/gradient_diagnostics_shared_v1.json','tests/mini_v5/mini_complete.json'):
            other=validate_sources(read(root/relative));executed=other.get('execution_source',other.get('source',{}))
            if any(executed.get(p)!=source_identity([p])[p] for p in TRAIN_SOURCES):raise ValueError('required core execution binding missing '+relative)
        validate_sources(read(root/'tests/method_execution_sources.json'))
        common=validate_sources(read(root/'tests/shared_loss_v1.json'))
        if common.get('uses_scalar_tolerance') is not False or common.get('expands_scalar_envelope') is not False or common.get('prior_failures_not_reclassified') is not True:raise ValueError('shared-loss identity evidence incomplete')
        if {c['case'] for c in common['cases']}!={'disabled','zero_coefficients','empty_H','T1','T2'} or any(c['original_calls']!=1 or not all(c['non_POS_same_tensor_objects'].values()) for c in common['cases']):raise ValueError('shared original-loss provenance')
        for obj in common['prior_failed_implementation_evidence']:
            if sha256(obj['path'])!=obj['sha256'] or read(obj['path'])['status']!='failed':raise ValueError('prior scalar failures not preserved')
        for arm in ('T1','T2'):
            r=num['arms'][arm]
            if r['updates']!=32 or not r['strict_resume_next_step_bitwise'] or not r['probe_full_state_and_next_update_bitwise']:raise ValueError(arm+' numerical requirements')
            coverage=read(root/f'prepared/coverage_{arm}.json')['formal_exposure']
            if coverage['F']<=0 or coverage['views']!=4000 or coverage['weight_reduction_sum']!=0:raise ValueError(arm+' target coverage')
            if arm=='T2' and coverage['p1_softened_pixels']<=0:raise ValueError('T2 p1 softening absent')
        mini=read(root/'tests/mini_v5/configs/queue.json')
        from .runner import stage_specs
        for spec in stage_specs(mini):
            if spec['stage']!='report':validate_stage(spec['stage'],read(spec['receipt']),mini,spec)
        aq=read(root/'tests/mini_v5/configs/aggregate_queue.json');validate_stage('report',read(root/'tests/mini_v5/comparison/complete.json'),aq,dict(method=None))
    except (OSError,ValueError,KeyError) as e:failures.append('semantic admission: '+str(e))
    frozen=root/'source/frozen.json'
    if not frozen.exists():failures.append('source not frozen')
    result=dict(version='BS4-OVERNIGHT-v2.0',run_id=q['run_id'],status='passed' if not failures else 'failed',approved_to_launch=not failures,required_failures=failures,evidence=evidence,resources=read(root/'tests/resources.json') if (root/'tests/resources.json').exists() else {},queue=identity(root/'configs/queue.json'),frozen_source=identity(frozen) if frozen.exists() else None,resolved_configs={a:read(s['config']) for a,s in q['methods'].items()},arm_frozen={a:identity(Path(s['cost_root'])/'configs/frozen.json') for a,s in q['methods'].items()},independent_review_required=True,scientific_acceptance=False)
    atomic_json(root/'tests/admission.json',result);return result

def execute_missing(root):
    """Idempotent validated preparation; unimplemented/missing evidence fails closed."""
    root=Path(root);q=read(root/'configs/queue.json');cwd=q['methods']['P1']['worktree']
    commands=[('prepared/coverage.json',['scripts.bs4.prepare','--root',str(root)]),
      ('tests/post_math.json',['scripts.bs4.tests.post_math','--output',str(root/'tests/post_math.json')]),
      ('diagnostics/B_native_replay/complete.json',['scripts.bs4.tests.b_replay','--root',str(root)]),
      ('tests/post_real_v2/complete.json',['scripts.bs4.tests.post_real','--root',str(root)]),
      ('tests/equivalence_shared_v1.json',['scripts.bs4.tests.equivalence','--root',str(root)]),
      ('tests/cpu_equivalence_shared_v1.json',['scripts.bs4.tests.cpu_equivalence','--root',str(root)]),
      ('diagnostics/numeric_admission_v2.json',['scripts.bs4.tests.numeric','--root',str(root)]),
      ('tests/checkpoint_faults_v2.json',['scripts.bs4.tests.checkpoint_faults','--root',str(root)]),
      ('diagnostics/gradient_diagnostics_shared_v1.json',['scripts.bs4.tests.gradients','--root',str(root)]),
      ('tests/report_population_gzip_v2.json',['scripts.bs4.tests.report_population','--root',str(root)]),
      ('tests/aggregate_mini_shared_v1.json',['scripts.bs4.tests.aggregate_mini','--root',str(root)]),
      ('tests/runner_engineering_final.json',['scripts.bs4.test_engineering','--root',str(root)])]
    # Freeze scientific dependencies before producing any strict numerical checkpoint.
    if not (root/'prepared/coverage.json').exists():
        subprocess.run([q['python'],'-m','scripts.bs4.prepare','--root',str(root)],cwd=cwd,check=True,timeout=14400)
    if not (root/'prepared/exposure/complete.json').exists():subprocess.run([q['python'],'-m','scripts.bs4.exposure','--root',str(root)],cwd=cwd,check=True,timeout=14400)
    capture_method_sources(root)
    from .entry import freeze_inputs
    freeze_inputs(root/'configs/queue.json')
    deferred={'tests/aggregate_mini_shared_v1.json','tests/runner_engineering_final.json'}
    commands.insert(6,('tests/shared_loss_v1.json',['scripts.bs4.tests.shared_loss','--root',str(root)]))
    for rel,command in commands:
        if rel in deferred:continue
        path=root/rel
        if path.exists():
            r=validate_sources(read(path))
            if r.get('status') not in ('passed','succeeded'):raise ValueError('existing failed evidence requires preserved explicit repair attempt '+rel)
            if r.get('outputs'):verify_outputs(r)
            continue
        subprocess.run([q['python'],'-m',*command],cwd=cwd,check=True,timeout=14400)
    mini=root/'tests/mini_v5'
    if not (mini/'fixture.json').exists():subprocess.run([q['python'],'-m','scripts.bs4.mini','--queue',str(root/'configs/queue.json'),'--output',str(mini)],cwd=cwd,check=True,timeout=3600)
    subprocess.run([q['python'],'-m','scripts.bs4.mini','--queue',str(mini/'configs/queue.json'),'--stages'],cwd=cwd,check=True,timeout=14400)
    if not (root/'tests/post_resume_v2/complete.json').exists():subprocess.run([q['python'],'-m','scripts.bs4.tests.post_resume','--root',str(root)],cwd=cwd,check=True,timeout=1800)
    for rel,command in commands:
        if rel in deferred and not (root/rel).exists():subprocess.run([q['python'],'-m',*command],cwd=cwd,check=True,timeout=14400)
    from .runner import freeze
    freeze(root/'configs/queue.json')
    subprocess.run([q['python'],'-m','scripts.bs4.tests.deployment','--root',str(root)],cwd=cwd,check=True,timeout=14400)
    from .resources import measure_final
    measure_final(root/'configs/queue.json')
    # All work above is real and remains blocked by independent review at launch.
    for rel in ('tests/deployment_test.json','tests/resources.json'):
        r=read(root/rel)
        if rel.endswith('resources.json') and (not r.get('measured') or not r.get('sufficient')):raise ValueError('resource admission missing')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--run-tests',action='store_true');a=p.parse_args()
    if a.run_tests:execute_missing(a.root)
    r=collect(a.root);print(json.dumps(r,indent=2));raise SystemExit(0 if r['status']=='passed' else 2)
