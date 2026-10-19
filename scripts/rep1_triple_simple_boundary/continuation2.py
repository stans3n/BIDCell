"""Closed two-donor continuation: immutable r01 base plus reviewed r02 delta."""
from pathlib import Path
import numpy as np
import tifffile
from . import continuation as c
from .common import (check_identity, check_receipt, dependencies, digest, identity,
                     read, within)

SCHEMA = 'rep1-triple-simple-continuation-v2'
DONOR = c.DONOR.parent/'rep1-triple-simple-boundary-20261017-r02'
DONOR_COMMIT = 'c45f0f9ce47b1f1293fd9a97fccc35b0f9862c6b'
DONOR_CONFIG = '49fc16a400f6bdbb3e9000cc094abb038e878e3ed8176c7ebbbbe49e9ce933f0'
PLAN = c.WORK/'docs/BIDCell_Rep1_Triple_Simple_Recovery02_ChangePlan_20261018.md'
PLAN_SHA = '5379990fc79def4e8e71fc7a4b2707fbdac2b6e1a24128faf3f6b0ca1216fbe0'
AUDIT = c.WORK/'docs/BIDCell_Rep1_Triple_Simple_Recovery02_ChangePlan_Audit_20261018.json'
AUDIT_SHA = '8a32df80ef600510f8a698ca9ded5dfe4c2d2fc3d3e787284adae7453ced1565'
PROOF = c.WORK/'docs/BIDCell_Rep1_Triple_Simple_Recovery02_RootDiagnosis_20261018.json'
PROOF_SHA = '73c72955ea21740da9d02e75d32b3f97567732b91460f8baf030614806fc702d'
BASELINE = c.WORK/'docs/BIDCell_Rep1_Triple_Simple_Recovery02_R02Baseline_20261018.json'
BASELINE_SHA = 'e0362398dc491a4addc7b641d99a073fe96fd2537ab6338db5d563ed9cedbe5f'
BASE_BASELINE = c.WORK/'docs/BIDCell_Rep1_Triple_Simple_Recovery01_DonorBaseline_20261017.json'
COORDS = [[0, 0], [0, 1024], [0, 2048], [0, 3072]]
FIELDS = {'schema','target_run_id','target_run_root','new_source_commit','new_source_digest',
          'base_continuation','donor2_config','donor2_source_manifest','donor2_admission',
          'donor2_validation','donor2_baseline','base_baseline','repair_plan','repair_plan_audit',
          'source_compatibility','fingerprints','stage05','partial_p2',
          'additional_reuse_credit_files','validated_external_reuse_bytes'}
_EXPANDED = {}
_FP = {}


def approved():
    for path, sha in ((PLAN, PLAN_SHA), (AUDIT, AUDIT_SHA)):
        check_identity(dict(path=str(path), sha256=sha))
    report=read(AUDIT)
    if report.get('decision')!='PASS_FOR_REPAIR_IMPLEMENTATION' or report.get('open_blockers')!=[] or report.get('plan')!=identity(PLAN):
        raise PermissionError('Recovery02 plan not approved')


def fixed(item, path, sha=None):
    if item.get('path')!=str(path) or (sha is not None and item.get('sha256')!=sha):
        raise ValueError('unapproved second-generation evidence location/identity')
    return c._bound_json(item)


def stage05(donor, proof, proof_identity):
    receipt=proof['stage05_receipt'];old=c._bound_json(receipt)
    if receipt!=identity(DONOR/'replay/SIMPLE_NATIVE/complete.json') or old['dependencies']!=dependencies(donor,'05') or old.get('scope')!='full_field' or old.get('final_pixels_equal') is not True:
        raise ValueError('second producer has no complete native replay')
    return dict(receipt=receipt,dependencies=old['dependencies'],config_digest=DONOR_CONFIG,
                source_commit=DONOR_COMMIT,source_digest=donor['source_digest'],
                config=identity(DONOR/'configs/run.json'),semantic_validation=proof_identity,
                validated_seconds=proof['stage05_actual_frozen_semantic_validation_seconds'])


def credit_files(base, native, tiles):
    """Only committed scientific inventory; source/control/logs never get credit."""
    seen={(r['dev'],r['ino']) for r in base['reuse_credit_files'].values()};result={}
    items=[r for name,r in native['outputs'].items() if name!='machine.json']
    items += [r for tile in tiles for r in c._bound_json(tile)['outputs'].values()]
    for item in items:
        p=Path(item['path']);s=p.stat();key=(s.st_dev,s.st_ino)
        if key in seen:continue
        if DONOR not in p.parents or 'progress' in p.name or p.is_symlink():
            raise ValueError('second-generation credit outside committed science')
        seen.add(key)
        result[str(p)]=dict(category='p2_replay',dev=s.st_dev,ino=s.st_ino,
                           allocated=s.st_blocks*512,identity=item)
    return result


def expand(cfg, obj):
    """No general merge: every allowed overlay field is typed and reconstructed."""
    if set(obj)!=FIELDS or obj.get('schema')!=SCHEMA:
        raise ValueError('unknown second-generation continuation fields')
    if obj['target_run_id']!=cfg['run_id'] or obj['target_run_root']!=cfg['run_root'] or obj['new_source_commit']!=cfg['source_commit'] or obj['new_source_digest']!=cfg['source_digest']:
        raise ValueError('second-generation target/source binding differs')
    donor=fixed(obj['donor2_config'],DONOR/'configs/run.json')
    if digest(donor)!=DONOR_CONFIG or donor['source_commit']!=DONOR_COMMIT or donor['run_root']!=str(DONOR):
        raise ValueError('unapproved second-generation producer')
    if obj['base_continuation']!=donor['continuation']:
        raise ValueError('second-generation base retargeted')
    base=c.manifest(donor)
    if base.get('generation2'):raise ValueError('only the two explicitly approved donors are permitted')
    proof=fixed(obj['donor2_validation'],PROOF,PROOF_SHA)
    baseline=fixed(obj['donor2_baseline'],BASELINE,BASELINE_SHA)
    fixed(obj['donor2_source_manifest'],DONOR/'source_manifest.json')
    admission=fixed(obj['donor2_admission'],DONOR/'admission/final.json')
    fixed(obj['base_baseline'],BASE_BASELINE)
    if obj['repair_plan']!=identity(PLAN) or obj['repair_plan_audit']!=identity(AUDIT):
        raise ValueError('second-generation repair approval differs')
    if proof.get('status')!='REPRODUCED_STAGE06_PRODUCER_LABEL_MISMATCH_WITH_COMPLETE05_VALIDATED' or proof['config']!=obj['donor2_config'] or proof['source_manifest']!=obj['donor2_source_manifest'] or proof['source_commit']!=DONOR_COMMIT or proof['r02_baseline']!=obj['donor2_baseline'] or proof.get('actual_execution_preflight_pass') is not True:
        raise ValueError('second-generation full semantic validation binding differs')
    if admission.get('status')!='PASS_FOR_LAUNCH' or admission.get('config_digest')!=DONOR_CONFIG or admission.get('source_digest')!=donor['source_digest'] or baseline['root']!=str(DONOR) or baseline['config_digest']!=DONOR_CONFIG:
        raise ValueError('second-generation original admission/baseline differs')
    for key in ('source_compatibility','fingerprints'):
        if obj[key]['path']!=str(Path(cfg['run_root'])/'admission'/('continuation_'+key+'.json')):
            raise ValueError('second-generation local proof location differs')
        c._bound_json(obj[key])
    cache_key=(cfg['continuation']['path'],cfg['continuation']['sha256'])
    if cache_key in _EXPANDED:return _EXPANDED[cache_key]
    expected_fp=dict(schema=SCHEMA,base=base['fingerprints'],execution_preflight=identity(DONOR/'admission/preflight.json'),
                     paths=delta_paths(donor,base,baseline))
    if c._bound_json(obj['fingerprints'])!=expected_fp:
        raise ValueError('second-generation scientific fingerprint closure differs')
    expected_compat=dict(schema=SCHEMA,first_producer=c.source_compatibility(c.DONOR/'source_snapshot',cfg['source_snapshot']),
                         second_producer=c.source_compatibility(donor['source_snapshot'],cfg['source_snapshot']))
    if c._bound_json(obj['source_compatibility'])!=expected_compat:
        raise ValueError('second-generation numerical/source compatibility proof differs')
    expected_stage=stage05(donor,proof,obj['donor2_validation'])
    if obj['stage05']!=expected_stage:raise ValueError('second-generation native producer overlay differs')
    tiles=[r['receipt'] for r in proof['partial_P2_tiles']]
    expected_partial=dict(tile_receipts=tiles,tile_base=c._bound_json(tiles[0])['dependencies']['base'],coords=COORDS)
    if obj['partial_p2']!=expected_partial or len(tiles)!=4 or [c._bound_json(r)['coords'] for r in tiles]!=COORDS:
        raise ValueError('second-generation partial P2 overlay differs')
    credit=credit_files(base,c._bound_json(expected_stage['receipt']),tiles)
    if obj['additional_reuse_credit_files']!=credit:raise ValueError('second-generation credit inventory differs')
    totals=dict(base['validated_external_reuse_bytes']);totals['p2_replay']+=sum(r['allocated'] for r in credit.values())
    if obj['validated_external_reuse_bytes']!=totals:raise ValueError('second-generation credit double-counted or altered')
    result=dict(base)
    for key in ('target_run_id','target_run_root','new_source_commit','new_source_digest','repair_plan','repair_plan_audit','source_compatibility','fingerprints','validated_external_reuse_bytes'):
        result[key]=obj[key]
    result['stages']=dict(base['stages'], **{'05':expected_stage})
    result['reuse_credit_files']=dict(base['reuse_credit_files'],**credit)
    result['generation2']=obj
    _EXPANDED[cache_key]=result
    return result


def fingerprints(manifest):
    obj=manifest['generation2'];delta=c._bound_json(obj['fingerprints'])
    base=c._bound_json(obj['base_continuation']);original=c._bound_json(base['fingerprints'])
    if set(delta)!={'schema','base','execution_preflight','paths'} or delta['schema']!=SCHEMA or delta['base']!=base['fingerprints'] or delta['execution_preflight']['path']!=str(DONOR/'admission/preflight.json'):
        raise ValueError('second-generation fingerprint base differs')
    execution=c._bound_json(delta['execution_preflight'])
    if execution.get('config_digest')!=DONOR_CONFIG or execution.get('full_content_checks') is not True:
        raise ValueError('second-generation execution proof differs')
    key=(obj['fingerprints']['path'],obj['fingerprints']['sha256'])
    if key not in _FP:
        merged=dict(original['paths'])
        for closure in (execution['fingerprints'],delta['paths']):
            for p,fp in closure.items():
                if p in merged and merged[p]!=fp:raise ValueError('second-generation fingerprint overwrites original producer')
                merged[p]=fp
        _FP[key]=merged
    return _FP[key]


def delta_paths(donor, base, baseline):
    paths={str(DONOR/p):fp for p,fp in baseline['paths'].items() if p.startswith(('source_snapshot/','replay/SIMPLE_NATIVE/','endpoints/P2_SIMPLE/predictions/postprocess/')) and Path(p).name!='progress.json'}
    for p in (DONOR/'source_snapshot',DONOR/'configs/run.json',DONOR/'source_manifest.json',DONOR/'environment.json',
              DONOR/'admission/final.json',DONOR/'admission/preflight.json',DONOR/'state.json',Path(donor['input_manifest']),
              Path(donor['continuation']['path']),Path(base['fingerprints']['path']),PLAN,AUDIT,PROOF,BASELINE,BASE_BASELINE):
        paths[str(p)]=c.fingerprint(p)
    for p in tuple(paths):
        if DONOR in Path(p).parents:
            for parent in Path(p).parents:
                if parent==DONOR or DONOR in parent.parents:paths[str(parent)]=c.fingerprint(parent)
    return paths


def evidence_items(manifest):
    obj=manifest['generation2']
    return [obj[k] for k in ('base_continuation','donor2_config','donor2_source_manifest','donor2_admission',
                            'donor2_validation','donor2_baseline','base_baseline')]


def preserve(baseline, root, field):
    paths=baseline[field]
    if {str(p.relative_to(root)) for p in root.rglob('*')}!=set(paths):
        raise ValueError('historical donor inventory changed')
    for relative,fp in paths.items():
        if c.fingerprint(root/relative)!=fp:raise ValueError('historical donor changed: '+relative)


def check_tiles(donor, partial):
    """Bind saved numeric payloads to the exact native candidates and old kernel."""
    from .p2 import verify_record, RULE_VERSION
    oldbase=partial['tile_base']
    if oldbase['run']!=dependencies(donor,'P2_SIMPLE_POSTPROCESS') or oldbase['rule']!='P2' or oldbase['seed']!=1234 or oldbase['patch_size']!=1024 or oldbase['origin']!=[0,0] or oldbase['rule_version']!=RULE_VERSION:
        raise ValueError('partial P2 original producer/scientific parameters differ')
    native_root=DONOR/'replay/SIMPLE_NATIVE/predictions/postprocess'
    native=verify_record(native_root/'complete.json')
    if oldbase['native_replay']!=identity(native_root/'complete.json'):
        raise ValueError('partial P2 native lineage differs')
    for key in ('pre','nuclei','full_nuclei','seed','patch_size','origin','rule_version'):
        if oldbase[key]!=native['dependencies'][key]:raise ValueError('partial P2 native input/RNG contract differs')
    sources={'p2.py':'scripts/rep1_triple_simple_boundary/p2.py','p2_rules.py':'scripts/boundary_three/p2_rules.py','postprocess_predictions.py':'bidcell/model/postprocess_predictions.py'}
    if oldbase['source']!={name:identity(Path(donor['source_snapshot'])/rel) for name,rel in sources.items()}:
        raise ValueError('partial P2 original source differs')
    for item,coords in zip(partial['tile_receipts'],COORDS):
        if Path(item['path'])!=DONOR/'endpoints/P2_SIMPLE/predictions/postprocess'/f'tile_{coords[0]}_{coords[1]}.json':
            raise ValueError('partial P2 receipt retargeted')
        old=verify_record(check_identity(item),dict(base=oldbase,coords=coords,shape=[1024,1024]))
        ref=verify_record(native_root/f'tile_{coords[0]}_{coords[1]}.json')
        for key in ('candidate_count','candidate_order','candidate_hashes','rng_before_shuffle','rng_enter_islands','proposal_histogram'):
            if old['diagnostics'][key]!=ref['diagnostics'][key]:raise ValueError('partial P2 candidate/RNG lineage differs')
        if old['coords']!=coords or old['shape']!=[1024,1024] or set(old['outputs'])!={'postconnect','postislands','diagnostics'}:
            raise ValueError('partial P2 geometry/output inventory differs')
        with np.load(check_identity(old['outputs']['diagnostics']),allow_pickle=False) as z:
            counts=z['proposal_count'];resolution=z['conflict_resolution']
            if counts.shape!=(1024,1024) or resolution.ndim!=2 or resolution.shape[1]!=8 or resolution.dtype!=np.int64 or len(resolution)!=old['diagnostics']['conflict_pixels']:
                raise ValueError('partial P2 numerical diagnostics contract differs')
            values,nums=np.unique(counts,return_counts=True)
            if {str(int(k)):int(v) for k,v in zip(values,nums)}!=old['diagnostics']['proposal_histogram']:
                raise ValueError('partial P2 proposal histogram differs from payload')
        for key in ('postconnect','postislands'):
            a=tifffile.imread(old['outputs'][key]['path'])
            if a.shape!=(1024,1024) or a.dtype!=np.uint32:raise ValueError('partial P2 tile payload shape/dtype differs')


def adopt_tiles(cfg, dep, out, tasks):
    from .p2 import record,verify_record
    m=c.manifest(cfg);obj=m.get('generation2');partial=obj['partial_p2'] if obj else None
    if partial is None or dep.get('run')!=dependencies(cfg,'06') or dep.get('rule')!='P2' or Path(out)!=within(cfg['run_root'],'endpoints/P2_SIMPLE/predictions/postprocess'):
        raise ValueError('P2 tile adoption requires current stage06 and explicit two-donor approval')
    oldbase=partial['tile_base']
    if {k:v for k,v in dep.items() if k not in ('run','source')}!={k:v for k,v in oldbase.items() if k not in ('run','source')}:
        raise ValueError('current P2 input/native dependency differs from approved donor')
    if len(tasks)!=48 or [[t[0],t[1]] for t in tasks[:4]]!=COORDS:
        raise ValueError('P2 partial adoption task geometry differs')
    fps=c.fingerprint_paths(m)
    for task,item in zip(tasks[:4],partial['tile_receipts']):
        y,x,size,dest,tiledep,native_pred=task
        if Path(native_pred)!=DONOR/'replay/SIMPLE_NATIVE/predictions':raise ValueError('P2 native reader uses wrong producer')
        old=verify_record(item['path'],dict(base=oldbase,coords=[y,x],shape=tiledep['shape']))
        for proof in (item,*old['outputs'].values()):
            if c.fingerprint(proof['path'])!=fps[proof['path']]:raise ValueError('adopted P2 numeric producer mutated')
        path=Path(dest).with_suffix('.json')
        if path.exists():
            rec=verify_record(path,tiledep)
            if rec.get('reused_from')!=item or rec.get('original_producer')!=old or rec.get('adopted') is not True or rec.get('recomputed') is not False:
                raise ValueError('P2 adopted tile original producer drift')
        else:
            record(path,tiledep,{k:v['path'] for k,v in old['outputs'].items()},coords=old['coords'],shape=old['shape'],
                   diagnostics=old['diagnostics'],reused_from=item,original_producer=old,adopted=True,recomputed=False,continuation=cfg['continuation'])


def build(partial, validation, baseline):
    from . import execution as ex
    approved();root=Path(partial['run_root']);donor=read(DONOR/'configs/run.json')
    if digest(donor)!=DONOR_CONFIG or donor['source_commit']!=DONOR_COMMIT:raise ValueError('second donor cfg differs')
    proof=fixed(identity(validation),PROOF,PROOF_SHA);base_line=fixed(identity(baseline),BASELINE,BASELINE_SHA)
    ex.verify_snapshot(donor,location=False);ex.verify_admission(donor)
    base=c.verify(donor)
    preserve(read(BASE_BASELINE),c.DONOR,'old_files_and_directories');preserve(base_line,DONOR,'paths')
    native=check_receipt(proof['stage05_receipt']['path'],dependencies(donor,'05'),DONOR)
    tiles=[r['receipt'] for r in proof['partial_P2_tiles']]
    partial_p2=dict(tile_receipts=tiles,tile_base=c._bound_json(tiles[0])['dependencies']['base'],coords=COORDS)
    check_tiles(donor,partial_p2)
    compat=dict(schema=SCHEMA,first_producer=c.source_compatibility(c.DONOR/'source_snapshot',partial['source_snapshot']),
                second_producer=c.source_compatibility(donor['source_snapshot'],partial['source_snapshot']))
    compat_path=root/'admission/continuation_source_compatibility.json';c._frozen_json(compat_path,compat)
    credits=credit_files(base,native,tiles);totals=dict(base['validated_external_reuse_bytes'])
    totals['p2_replay']+=sum(r['allocated'] for r in credits.values())
    paths=delta_paths(donor,base,base_line)
    fp_path=root/'admission/continuation_fingerprints.json'
    c._frozen_json(fp_path,dict(schema=SCHEMA,base=base['fingerprints'],execution_preflight=identity(DONOR/'admission/preflight.json'),paths=paths))
    obj=dict(schema=SCHEMA,target_run_id=partial['run_id'],target_run_root=partial['run_root'],
             new_source_commit=partial['source_commit'],new_source_digest=partial['source_digest'],
             base_continuation=donor['continuation'],donor2_config=identity(DONOR/'configs/run.json'),
             donor2_source_manifest=identity(DONOR/'source_manifest.json'),donor2_admission=identity(DONOR/'admission/final.json'),
             donor2_validation=identity(validation),donor2_baseline=identity(baseline),base_baseline=identity(BASE_BASELINE),
             repair_plan=identity(PLAN),repair_plan_audit=identity(AUDIT),source_compatibility=identity(compat_path),fingerprints=identity(fp_path),
             stage05=stage05(donor,proof,identity(validation)),partial_p2=partial_p2,
             additional_reuse_credit_files=credits,validated_external_reuse_bytes=totals)
    target=root/'configs/continuation.json';c._frozen_json(target,obj)
    c.verify(dict(partial,continuation=identity(target)))
    return identity(target)
