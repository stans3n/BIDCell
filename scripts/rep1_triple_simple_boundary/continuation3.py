"""Closed r03 partial-P2 overlay on its immutable, already reviewed v2 base."""
from pathlib import Path
import numpy as np
from . import continuation as c,continuation2 as c2
from .common import check_identity,dependencies,digest,identity,read,within

SCHEMA='rep1-triple-simple-continuation-v3'
DONOR=c.DONOR.parent/'rep1-triple-simple-boundary-20261017-r03'
DONOR_COMMIT='4a5537731cf65cce36ef2d063d70331a2e81582f'
DONOR_CONFIG='6230b57df049935d8326e902da0dcfff1cf774c1a7fa75333293c1158bf8203a'
DOCS=c.WORK/'docs';STEM='BIDCell_Rep1_Triple_Simple_Recovery03_'
PLAN=DOCS/(STEM+'ChangePlan_20261018.md')
AUDIT=DOCS/(STEM+'ChangePlan_Audit_20261018.json')
PROOF=DOCS/(STEM+'RootDiagnosis_20261018.json')
BASELINE=DOCS/(STEM+'R03Baseline_20261018.json')
NUMERIC=DOCS/(STEM+'PartialP2NumericProof_20261018.json')
OVERRIDE=DOCS/'BIDCell_Rep1_Triple_Simple_DiskReserve_14GiB_Override_20261018.json'
FIXED={PLAN:'79ab996f62eca21c3b2a38b230e16f8fa504a9bf49d9f12070bf96a61f3dea2f',
       AUDIT:'05da1fdd35a0e4d43d6ab01fbeec085b0953d027f7db9eb9fabc42fb4567806a',
       PROOF:'70aa6696279215863c5aa55b6ea805d5d093b2fb36e3c7d0d3398b2bd4ce9b3b',
       BASELINE:'639cd16afd6cda3ccaaa3dc3c5ca1c237c5d4193592547f3103193356eccfc9f',
       NUMERIC:'14d451be81d814bdb763d28f9797353b10711c459ca7b3a86f939a05cf11b265',
       OVERRIDE:'1a5f88107afc49248f85c285fc22d727b761739efa01c4b1091d7dec2fd90e48'}
ADMISSION_KEYS=('base_continuation','donor3_config','donor3_source_manifest','donor3_admission',
                'donor3_preflight','donor3_validation','donor3_baseline','numeric_validation','disk_reserve_override')
FIELDS=set(ADMISSION_KEYS)|{'schema','target_run_id','target_run_root','new_source_commit','new_source_digest',
    'repair_plan','repair_plan_audit','source_compatibility','fingerprints','partial_p2',
    'additional_reuse_credit_files','validated_external_reuse_bytes'}
_EXPANDED={};_FP={}


def fixed(item,path):
    if item.get('path')!=str(path) or (path in FIXED and item.get('sha256')!=FIXED[path]):
        raise ValueError('unapproved third-producer evidence')
    return c._bound_json(item)


def approved():
    for p in (PLAN,AUDIT,OVERRIDE):check_identity(dict(path=str(p),sha256=FIXED[p]))
    audit=read(AUDIT);override=read(OVERRIDE)
    if audit.get('decision')!='PASS_FOR_REPAIR_IMPLEMENTATION' or audit.get('open_blockers')!=[] or audit.get('plan')!=identity(PLAN):
        raise PermissionError('Recovery03 plan not approved')
    if override.get('requested_reserve_bytes')!=14*1024**3 or override.get('all_other_resource_values_unchanged') is not True:
        raise PermissionError('Recovery03 disk-only override differs')


def partial(proof):
    return dict(tile_receipts=proof['ordered_tile_receipts'],tile_base=proof['tile_base'],coords=proof['coords'],
        split_receipts=proof['split_receipts'],full_split_id_count=proof['full_split_id_count'],
        full_split_id_hash_encoding=proof['full_split_id_hash_encoding'],full_split_id_hash=proof['full_split_id_hash'],
        old_opaque_split_digest=proof['old_opaque_split_digest'])


def credit_files(base,part):
    seen={(r['dev'],r['ino']) for r in base['reuse_credit_files'].values()};result={}
    for item in (*part['tile_receipts'],*part['split_receipts']):
        for leaf in c._bound_json(item)['outputs'].values():
            path=Path(leaf['path']);s=path.stat();key=(s.st_dev,s.st_ino)
            if key in seen:continue
            if DONOR not in path.parents or path.is_symlink():raise ValueError('third-producer credit outside registered science')
            seen.add(key);result[str(path)]=dict(category='p2_replay',dev=s.st_dev,ino=s.st_ino,allocated=s.st_blocks*512,identity=leaf)
    if len(result)!=133 or sum(r['allocated'] for r in result.values())!=413978624:
        raise ValueError('third-producer unique scientific credit differs')
    return result


def compatibility(snapshot):
    sources={'r01':c.DONOR,'r02':c2.DONOR,'r03':DONOR}
    checks={name:c.source_compatibility(root/'source_snapshot',snapshot) for name,root in sources.items()}
    for name in ('p2.py','science.py','nucleus_validation.py','p2_diagnostics.py'):
        rel=Path('scripts/rep1_triple_simple_boundary')/name
        if identity(DONOR/'source_snapshot'/rel)['sha256']!=identity(Path(snapshot)/rel)['sha256']:
            raise ValueError('Recovery03 changed a frozen science entry: '+name)
    return dict(schema=SCHEMA,producer_compatibility=checks)


def delta_paths(donor):
    baseline=read(BASELINE)
    paths={str(DONOR/p):fp for p,fp in baseline['paths'].items()
           if p.startswith(('source_snapshot/','endpoints/P2_SIMPLE/predictions/')) and Path(p).name!='progress.json'}
    for p in (DONOR/'source_snapshot',DONOR/'configs/run.json',DONOR/'source_manifest.json',DONOR/'environment.json',
              DONOR/'admission/final.json',DONOR/'admission/preflight.json',DONOR/'state.json',
              Path(donor['input_manifest']),Path(donor['continuation']['path']),*FIXED):
        paths[str(p)]=c.fingerprint(p)
    for p in tuple(paths):
        if DONOR in Path(p).parents:
            for parent in Path(p).parents:
                if parent==DONOR or DONOR in parent.parents:paths[str(parent)]=c.fingerprint(parent)
    return paths


def expand(cfg,obj):
    if set(obj)!=FIELDS or obj.get('schema')!=SCHEMA:raise ValueError('unknown third-producer overlay fields')
    for k,v in dict(target_run_id=cfg['run_id'],target_run_root=cfg['run_root'],new_source_commit=cfg['source_commit'],new_source_digest=cfg['source_digest']).items():
        if obj[k]!=v:raise ValueError('third-producer target/source binding differs')
    if cfg['resources']['reserve_bytes']!=14*1024**3:raise ValueError('Recovery03 requires authorized14GiB disk reserve')
    donor=fixed(obj['donor3_config'],DONOR/'configs/run.json')
    if digest(donor)!=DONOR_CONFIG or donor['source_commit']!=DONOR_COMMIT or donor['run_root']!=str(DONOR):raise ValueError('unapproved third producer')
    if obj['base_continuation']!=donor['continuation'] or c._bound_json(obj['base_continuation']).get('schema')!=c2.SCHEMA:
        raise ValueError('third-producer base retargeted')
    base=c.manifest(donor)
    if not base.get('generation2') or base.get('generation3'):raise ValueError('third-producer base generation differs')
    for key,path in [('donor3_source_manifest',DONOR/'source_manifest.json'),('donor3_admission',DONOR/'admission/final.json'),
                     ('donor3_preflight',DONOR/'admission/preflight.json'),('donor3_validation',PROOF),('donor3_baseline',BASELINE),
                     ('numeric_validation',NUMERIC),('disk_reserve_override',OVERRIDE),('repair_plan',PLAN),('repair_plan_audit',AUDIT)]:
        fixed(obj[key],path)
    diagnosis=c._bound_json(obj['donor3_validation']);numeric=c._bound_json(obj['numeric_validation'])
    for proof in (diagnosis,numeric):
        if proof['config']!=obj['donor3_config'] or proof['config_digest']!=DONOR_CONFIG or proof['source_commit']!=DONOR_COMMIT or proof['source_manifest']!=obj['donor3_source_manifest'] or proof['baseline']!=obj['donor3_baseline']:
            raise ValueError('third-producer validation lineage differs')
    if diagnosis['status']!='VERIFIED_STOPPED06_WITH48_VALID_TILES_AND_COMMITTED_SPLITS' or numeric['status']!='PASS_ALL48_TILE_LINEAGE_AND_FIRST128_OF2920_SPLIT_NUMERIC' or numeric['root_diagnosis']!=obj['donor3_validation']:
        raise ValueError('third-producer scientific validation absent')
    admission=c._bound_json(obj['donor3_admission']);preflight=c._bound_json(obj['donor3_preflight']);baseline=c._bound_json(obj['donor3_baseline'])
    if admission.get('status')!='PASS_FOR_LAUNCH' or admission.get('config_digest')!=DONOR_CONFIG or admission.get('source_digest')!=donor['source_digest'] or preflight.get('config_digest')!=DONOR_CONFIG or preflight.get('full_content_checks') is not True or baseline['root']!=str(DONOR) or baseline['config_digest']!=DONOR_CONFIG:
        raise ValueError('third-producer original execution proof differs')
    for key in ('source_compatibility','fingerprints'):
        if obj[key]['path']!=str(Path(cfg['run_root'])/'admission'/('continuation_'+key+'.json')):raise ValueError('third-producer local proof retargeted')
        c._bound_json(obj[key])
    key=(cfg['continuation']['path'],cfg['continuation']['sha256'])
    if key in _EXPANDED:return _EXPANDED[key]
    expected_fp=dict(schema=SCHEMA,base_continuation=obj['base_continuation'],execution_preflight=obj['donor3_preflight'],paths=delta_paths(donor))
    if c._bound_json(obj['fingerprints'])!=expected_fp:raise ValueError('third-producer full fingerprint closure differs')
    if c._bound_json(obj['source_compatibility'])!=compatibility(cfg['source_snapshot']):raise ValueError('third-producer source compatibility differs')
    if obj['partial_p2']!=partial(numeric):raise ValueError('third-producer partial science overlay differs')
    credit=credit_files(base,obj['partial_p2']);totals=dict(base['validated_external_reuse_bytes'])
    totals['p2_replay']+=sum(r['allocated'] for r in credit.values())
    if obj['additional_reuse_credit_files']!=credit or obj['validated_external_reuse_bytes']!=totals:raise ValueError('third-producer credit omitted or double-counted')
    result=dict(base)
    for name in ('target_run_id','target_run_root','new_source_commit','new_source_digest','repair_plan','repair_plan_audit','source_compatibility','fingerprints','validated_external_reuse_bytes'):result[name]=obj[name]
    result['reuse_credit_files']=dict(base['reuse_credit_files'],**credit);result['generation3']=obj
    _EXPANDED[key]=result;return result


def fingerprints(manifest):
    obj=manifest['generation3'];delta=c._bound_json(obj['fingerprints'])
    if set(delta)!={'schema','base_continuation','execution_preflight','paths'} or delta['schema']!=SCHEMA or delta['base_continuation']!=obj['base_continuation'] or delta['execution_preflight']!=obj['donor3_preflight']:
        raise ValueError('third-producer fingerprint base differs')
    key=(obj['fingerprints']['path'],obj['fingerprints']['sha256'])
    if key not in _FP:
        base=c.manifest(c._bound_json(obj['donor3_config']));merged=dict(c.fingerprint_paths(base))
        execution=c._bound_json(delta['execution_preflight'])
        for closure in (execution['fingerprints'],delta['paths']):
            for path,fp in closure.items():
                if path in merged and merged[path]!=fp:raise ValueError('third-producer overwrites inherited fingerprint')
                merged[path]=fp
        _FP[key]=merged
    return _FP[key]


def evidence_items(manifest):return [manifest['generation3'][key] for key in ADMISSION_KEYS]


def check_records(donor,base,proof):
    from .p2 import verify_record,tile_coordinates,RULE_VERSION
    import tifffile
    part=partial(proof);oldbase=part['tile_base'];post=DONOR/'endpoints/P2_SIMPLE/predictions/postprocess'
    native=Path(base['stages']['05']['receipt']['path']).parent/'predictions/postprocess'
    if oldbase['run']!=dependencies(donor,'06') or oldbase['native_replay']!=identity(native/'complete.json') or oldbase['rule']!='P2' or oldbase['seed']!=1234 or oldbase['patch_size']!=1024 or oldbase['origin']!=[0,0] or oldbase['rule_version']!=RULE_VERSION:
        raise ValueError('third-producer P2 numerical/native lineage differs')
    sources={'p2.py':'scripts/rep1_triple_simple_boundary/p2.py','p2_rules.py':'scripts/boundary_three/p2_rules.py','postprocess_predictions.py':'bidcell/model/postprocess_predictions.py'}
    if oldbase['source']!={k:identity(Path(donor['source_snapshot'])/p) for k,p in sources.items()}:raise ValueError('third-producer numerical source differs')
    with tifffile.TiffFile(oldbase['pre']['path']) as f:shape=f.series[0].shape
    size,_,_,coords=tile_coordinates(shape,1024)
    if len(coords)!=48 or [list(x) for x in coords]!=part['coords'] or len(part['tile_receipts'])!=48:raise ValueError('third-producer ordered tile geometry differs')
    for i,(item,coords) in enumerate(zip(part['tile_receipts'],part['coords'])):
        y,x=coords
        if item['path']!=str(post/f'tile_{y}_{x}.json'):raise ValueError('third-producer tile receipt retargeted')
        old=verify_record(check_identity(item),dict(base=oldbase,coords=coords,shape=[size,size]))
        if old['coords']!=coords or old['shape']!=[size,size] or set(old['outputs'])!={'postconnect','postislands','diagnostics'}:raise ValueError('third-producer tile payload contract differs')
        ref=proof['tile_proofs'][i]
        if ref['receipt']!=item or ref['coords']!=coords or ref['candidate_and_RNG_equal_to_native'] is not True or ref['native_receipt']['path']!=str(native/f'tile_{y}_{x}.json'):raise ValueError('third-producer native proof retargeted')
        previous=c._bound_json(ref['native_receipt'])
        for k in ('candidate_count','candidate_order','candidate_hashes','rng_before_shuffle','rng_enter_islands','proposal_histogram'):
            if old['diagnostics'][k]!=previous['diagnostics'][k]:raise ValueError('third-producer candidate/RNG lineage differs')
        if i<4:
            original=base['generation2']['partial_p2']['tile_receipts'][i]
            if old.get('adopted') is not True or old.get('recomputed') is not False or old.get('reused_from')!=original or old.get('original_producer')!=c._bound_json(original) or old['outputs']!=c._bound_json(original)['outputs']:
                raise ValueError('third-producer nested adoption history differs')
        else:
            expected={'postconnect':post/f'tile_{y}_{x}_postconnect.tif','postislands':post/f'tile_{y}_{x}.tif','diagnostics':post/f'tile_{y}_{x}_diagnostics.npz'}
            if old.get('adopted') or old.get('reused_from') or {k:v['path'] for k,v in old['outputs'].items()}!={k:str(v) for k,v in expected.items()}:raise ValueError('third-producer fresh tile output retargeted')
    ids=proof['full_split_ids']
    if len(ids)!=2920 or ids!=sorted(set(ids)) or ids[0]<=0 or part['full_split_id_count']!=len(ids) or part['full_split_id_hash_encoding']!='canonical-json-integer-list-v1' or part['full_split_id_hash']!=digest(ids):raise ValueError('third-producer full split population differs')
    olddigest=digest(dict(base=oldbase,tiles=part['tile_receipts'],ids=ids))
    if olddigest!=part['old_opaque_split_digest'] or part['split_receipts']!=[identity(post/'splits/000000.json')] or proof['first_split_ids']!=ids[:128]:raise ValueError('third-producer split chain differs')
    rec=verify_record(part['split_receipts'][0]['path'],dict(base=olddigest,ids=ids[:128],run=oldbase['run']))
    if rec['outputs']!={'split':proof['split_payload']} or proof['split_payload']['path']!=str(post/'splits/000000.npz') or proof['first128_independent_split_patch_equal'] is not True or proof['merged_images_rebuilt_from_all_tiles_pixel_equal'] is not True:raise ValueError('third-producer numeric split proof differs')
    with np.load(check_identity(proof['split_payload']),allow_pickle=False) as z:
        if set(z.files)!={str(i) for i in ids[:128]} or any(z[k].dtype!=np.dtype(bool) or z[k].ndim!=2 for k in z.files):raise ValueError('third-producer split payload dtype/IDs differ')


def binding(cfg,dep,out):
    m=c.manifest(cfg);obj=m['generation3'];part=obj['partial_p2']
    if dep.get('run')!=dependencies(cfg,'06') or dep.get('rule')!='P2' or Path(out)!=within(cfg['run_root'],'endpoints/P2_SIMPLE/predictions/postprocess'):raise ValueError('third-producer adoption requires current stage06')
    if {k:v for k,v in dep.items() if k not in ('run','source')}!={k:v for k,v in part['tile_base'].items() if k not in ('run','source')}:raise ValueError('third-producer current science inputs differ')
    expected={name:identity(Path(cfg['source_snapshot'])/rel) for name,rel in {'p2.py':'scripts/rep1_triple_simple_boundary/p2.py','p2_rules.py':'scripts/boundary_three/p2_rules.py','postprocess_predictions.py':'bidcell/model/postprocess_predictions.py'}.items()}
    if dep['source']!=expected:raise ValueError('third-producer current source differs')
    return m,obj,part


def adopted(cfg,path,dep,item,old,**details):
    from .p2 import record
    return record(path,dep,{k:v['path'] for k,v in old['outputs'].items()},**details,reused_from=item,
                  original_producer=old,adopted=True,recomputed=False,continuation=cfg['continuation'])


def adopt_tiles(cfg,dep,out,tasks):
    from .p2 import verify_record
    m,obj,part=binding(cfg,dep,out);fps=c.fingerprint_paths(m)
    if len(tasks)!=len(part['tile_receipts']) or [[t[0],t[1]] for t in tasks]!=part['coords']:raise ValueError('third-producer current tile order differs')
    native=Path(m['stages']['05']['receipt']['path']).parent/'predictions'
    for task,item in zip(tasks,part['tile_receipts']):
        y,x,size,dest,tiledep,native_pred=task;path=Path(out)/f'tile_{y}_{x}.json'
        if Path(dest).with_suffix('.json')!=path or Path(native_pred)!=native or size!=1024 or tiledep!=dict(base=dep,coords=[y,x],shape=[1024,1024]):raise ValueError('third-producer current tile task differs')
        old=verify_record(item['path'],dict(base=part['tile_base'],coords=[y,x],shape=tiledep['shape']))
        for leaf in (item,*old['outputs'].values()):
            if c.fingerprint(leaf['path'])!=fps[leaf['path']]:raise ValueError('third-producer adopted tile mutated')
        adopted(cfg,path,tiledep,item,old,coords=old['coords'],shape=old['shape'],diagnostics=old['diagnostics'])


def adopt_split(cfg,dep,splitdir,splitdeps,ids):
    from .p2 import verify_record
    m,obj,part=binding(cfg,dep,Path(splitdir).parent)
    if Path(splitdir)!=within(cfg['run_root'],'endpoints/P2_SIMPLE/predictions/postprocess/splits') or len(ids)!=part['full_split_id_count'] or part['full_split_id_hash_encoding']!='canonical-json-integer-list-v1' or digest(ids)!=part['full_split_id_hash']:
        raise ValueError('third-producer current split ID set differs')
    olddigest=digest(dict(base=part['tile_base'],tiles=part['tile_receipts'],ids=ids))
    if olddigest!=part['old_opaque_split_digest']:raise ValueError('third-producer original opaque split digest differs')
    current=[]
    for coords,item in zip(part['coords'],part['tile_receipts']):
        y,x=coords;path=Path(splitdir).parent/f'tile_{y}_{x}.json';old=verify_record(item['path'])
        if not path.is_file():raise ValueError('third-producer split requires all current tile wrappers')
        adopted(cfg,path,dict(base=dep,coords=coords,shape=old['shape']),item,old,coords=coords,shape=old['shape'],diagnostics=old['diagnostics'])
        current.append(identity(path))
    if splitdeps!=digest(dict(base=dep,tiles=current,ids=ids)):raise ValueError('third-producer current opaque split digest differs')
    item=part['split_receipts'][0];old=verify_record(item['path'],dict(base=olddigest,ids=ids[:128],run=part['tile_base']['run']))
    fps=c.fingerprint_paths(m)
    for leaf in (item,*old['outputs'].values()):
        if c.fingerprint(leaf['path'])!=fps[leaf['path']]:raise ValueError('third-producer adopted split mutated')
    adopted(cfg,Path(splitdir)/'000000.json',dict(base=splitdeps,ids=ids[:128],run=dep['run']),item,old)


def build(cfg,validation,baseline):
    from . import execution as ex
    approved();root=Path(cfg['run_root']);donor=read(DONOR/'configs/run.json')
    fixed(identity(validation),PROOF);fixed(identity(baseline),BASELINE);numeric=fixed(identity(NUMERIC),NUMERIC)
    ex.verify_snapshot(donor,location=False);ex.verify_admission(donor);base=c.verify(donor)
    for path,producer,field in ((c2.BASE_BASELINE,c.DONOR,'old_files_and_directories'),(c2.BASELINE,c2.DONOR,'paths'),(BASELINE,DONOR,'paths')):
        c2.preserve(read(path),producer,field)
    check_records(donor,base,numeric);part=partial(numeric);credits=credit_files(base,part)
    totals=dict(base['validated_external_reuse_bytes']);totals['p2_replay']+=sum(r['allocated'] for r in credits.values())
    compat=root/'admission/continuation_source_compatibility.json';c._frozen_json(compat,compatibility(cfg['source_snapshot']))
    fp=root/'admission/continuation_fingerprints.json'
    c._frozen_json(fp,dict(schema=SCHEMA,base_continuation=donor['continuation'],execution_preflight=identity(DONOR/'admission/preflight.json'),paths=delta_paths(donor)))
    obj=dict(schema=SCHEMA,target_run_id=cfg['run_id'],target_run_root=cfg['run_root'],new_source_commit=cfg['source_commit'],new_source_digest=cfg['source_digest'],
        base_continuation=donor['continuation'],donor3_config=identity(DONOR/'configs/run.json'),donor3_source_manifest=identity(DONOR/'source_manifest.json'),
        donor3_admission=identity(DONOR/'admission/final.json'),donor3_preflight=identity(DONOR/'admission/preflight.json'),donor3_validation=identity(validation),
        donor3_baseline=identity(baseline),numeric_validation=identity(NUMERIC),disk_reserve_override=identity(OVERRIDE),repair_plan=identity(PLAN),repair_plan_audit=identity(AUDIT),
        source_compatibility=identity(compat),fingerprints=identity(fp),partial_p2=part,additional_reuse_credit_files=credits,validated_external_reuse_bytes=totals)
    target=root/'configs/continuation.json';c._frozen_json(target,obj);c.verify(dict(cfg,continuation=identity(target)))
    return identity(target)
