"""One audited r01 -> r02 adoption; copied checkpoints are not new computation."""
from __future__ import annotations
import ast
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from .common import (atomic, atomic_json, check_identity, check_receipt, dependencies,
                     digest, identity, read, sha256, within)
from . import execution as ex

TREES=('bt_existing6','endpoints/P2_BASE_GT/gt/domain',
       'endpoints/P2_BASE_GT/gt/scores','endpoints/P2_BASE_GT/gt/recovery')
STAGES={'BT_EXISTING6','BT_EXISTING6_POPULATIONS','PAIRED_F1_6','P2_BASE_GT_GT'}
SUPPORT=('source_inventory','source_identity_graph','source_integrity',
         'compatibility_baseline','design_review','amendment')


def independent_file(p):
    p=Path(p)
    if p.is_symlink() or not p.is_file() or p.stat().st_nlink!=1:
        raise ValueError('checkpoint must be an independent regular file: '+str(p))
    for parent in p.parents:
        if parent.is_symlink():raise ValueError('symlink in checkpoint path')
    return p


def prepare_plan(root,seed_path):
    root=Path(root);seed_path=Path(seed_path)
    independent_file(seed_path)
    if sha256(seed_path)!=ex.IMPORT_SEED_SHA:raise ValueError('unreviewed recovery seed')
    seed=read(seed_path)
    if seed['source_root']!=str(ex.OLD_ROOT) or seed['target_root']!=str(root) or seed['allowed_trees']!=list(TREES):
        raise ValueError('recovery scope drift')
    folder=root/'configs/recovery';folder.mkdir()
    shutil.copyfile(seed_path,folder/'seed.json');(folder/'seed.json').chmod(0o444)
    plan=dict(seed,seed=identity(folder/'seed.json'))
    for key in SUPPORT:
        source=check_identity(seed[key]);independent_file(source)
        target=folder/(key+source.suffix);shutil.copyfile(source,target);target.chmod(0o444)
        if sha256(target)!=seed[key]['sha256']:raise ValueError('support copy mismatch')
        plan[key]=identity(target)
    atomic_json(folder/'import_plan.json',plan);(folder/'import_plan.json').chmod(0o444)
    return identity(folder/'import_plan.json')


def verify_plan(cfg):
    item=cfg['checkpoint_import_plan'];check_identity(item)
    independent_file(item['path'])
    if Path(item['path']).stat().st_mode & 0o222:raise ValueError('recovery plan must be read-only')
    if item['path']!=str(Path(cfg['run_root'])/'configs/recovery/import_plan.json'):
        raise ValueError('wrong recovery plan location')
    plan=read(item['path']);check_identity(plan['seed'])
    independent_file(plan['seed']['path'])
    if plan['seed']['path']!=str(Path(item['path']).parent/'seed.json'):raise ValueError('wrong recovery seed location')
    if plan['seed']['sha256']!=ex.IMPORT_SEED_SHA:raise ValueError('recovery seed drift')
    seed=read(plan['seed']['path'])
    expected=dict(seed,seed=plan['seed'])
    for key in SUPPORT:
        check_identity(plan[key]);independent_file(plan[key]['path'])
        if Path(plan[key]['path']).parent!=Path(item['path']).parent or plan[key]['sha256']!=seed[key]['sha256'] or plan[key]['bytes']!=seed[key]['bytes']:
            raise ValueError('recovery support differs from approved seed')
        expected[key]=plan[key]
    if plan!=expected or plan['target_root']!=cfg['run_root'] or plan['source_root']!=str(ex.OLD_ROOT) or plan['allowed_trees']!=list(TREES):
        raise ValueError('recovery plan scope/content mismatch')
    if plan['amendment']['sha256']!=ex.AMENDMENT_SHA or cfg['input_digest']!=plan['input_digest']:
        raise ValueError('recovery contract/input mismatch')
    # Original contract and evidence remain fixed, too; copied support is not a
    # replacement for original producer identities.
    for key in ('source_config','source_manifest','source_admission','source_environment',
                'source_input_manifest','source_failed_state'):
        check_identity(plan[key])
    return plan


def module_ast(text,kind):
    tree=ast.parse(text,type_comments=True)
    if kind=='common':
        found=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name=='atomic_tiff']
        if len(found)!=1 or type(found[0]) is not ast.FunctionDef or found[0].decorator_list:
            raise ValueError('exactly one undecorated atomic_tiff helper required')
        tree.body.remove(found[0])
    elif kind=='gt':
        found=[(n,a) for n in tree.body if isinstance(n,ast.ImportFrom) and n.module=='common' and n.level==1
               for a in n.names if a.name=='atomic_tiff']
        if len(found)!=1 or found[0][1].asname is not None:raise ValueError('exact unaliased atomic_tiff import required')
        found[0][0].names.remove(found[0][1])
    else:raise ValueError('unknown AST compatibility rule')
    return hashlib.sha256(ast.dump(tree,annotate_fields=True,include_attributes=False).encode()).hexdigest()


def compatibility(cfg,plan):
    baseline=read(plan['compatibility_baseline']['path']);root=Path(cfg['source_snapshot']);fixed=[];review=[]
    for row in baseline['files']:
        old=independent_file(row['snapshot_path']);check_identity(dict(path=str(old),bytes=row['bytes'],sha256=row['sha256']))
        target=independent_file(root/row['relative']);rule=row['compatibility_rule']
        if rule=='byte_identical':
            if sha256(target)!=row['sha256']:raise ValueError('scientific byte compatibility drift: '+row['relative'])
            fixed.append(row['relative'])
        elif row['relative']=='scripts/rep1_bt_gt_fusion/common.py':
            if module_ast(target.read_text(),'common')!='d65a14b4e931a1c7c71ba676ee9f1cbe5e8947099a068a4de94fea6a10a3b6d3':
                raise ValueError('common full-module AST compatibility drift')
        elif row['relative']=='scripts/rep1_bt_gt_fusion/gt_refine.py':
            if module_ast(target.read_text(),'gt')!='abade49c08f74fdf7a3855f86f0a429c556e1196042a6e4a16e3ea4472ff2d6c':
                raise ValueError('GT full-module AST compatibility drift')
        elif rule=='manual_review_required_not_compatibility_waiver':review.append(row['relative'])
        else:raise ValueError('unknown compatibility baseline row')
    if len(fixed)!=27 or len(review)!=17:raise ValueError('compatibility coverage mismatch')
    return dict(status='PASS_SCIENTIFIC_COMPATIBILITY',byte_identical=fixed,strict_AST=['common','gt'],
                review_required=review,independent_final_audit_still_required=True,
                source_digest=cfg['source_digest'],baseline=plan['compatibility_baseline'])


def partial_gt(cfg,manifest):
    """Read-only integrity, never a substitute for final topology validation."""
    import numpy as np
    import tifffile
    from .gt_refine import REASONS
    root=Path(cfg['run_root']);gt=root/'endpoints/P2_BASE_GT/gt'
    parent=manifest['endpoints']['P2_BASE'];base=dependencies(cfg,'P2_BASE_GT_GT')
    check_receipt(gt/'domain/complete.json',dict(base,parent=manifest['files'][str(Path(parent['final']).resolve())]),root)
    check_receipt(gt/'scores/complete.json',dict(base,domain=identity(gt/'domain/complete.json')),root)
    r=check_receipt(gt/'recovery/state.json',dict(base,proposals=identity(gt/'scores/proposals.npy')),root)
    for k in ('mask','decisions'):
        if r[k]!=r['outputs'][k]:raise ValueError('recovery output identity mismatch')
    p=np.load(gt/'scores/proposals.npy',mmap_mode='r');d=np.load(r['decisions']['path'],mmap_mode='r');m=np.load(r['mask']['path'],mmap_mode='r')
    with tifffile.TiffFile(parent['final']) as t:shape=t.series[0].shape
    with tifffile.TiffFile(parent['nuclei']) as t:nshape=t.series[0].shape
    if r['cursor']!=196771 or len(p)!=196771 or len(d)!=196771 or m.shape!=shape or shape!=nshape or m.dtype!=np.dtype('uint32'):
        raise ValueError('recovery dimensions/cursor mismatch')
    y,x=p['y'],p['x']
    if not np.all((d>0)&(d<len(REASONS))) or not np.all((y>=0)&(y<shape[0])&(x>=0)&(x<shape[1])):
        raise ValueError('invalid proposal or decision')
    if len(np.unique(y.astype(np.int64)*shape[1]+x))!=len(p) or not np.array_equal(np.lexsort((p['target'],x,y,-p['margin'])),np.arange(len(p))):
        raise ValueError('proposal order/uniqueness mismatch')
    if not np.all(p['margin']>=np.log(2)) or not np.array_equal(m[y,x],np.where(d==1,p['target'],p['source'])):
        raise ValueError('proposal decisions disagree with recovered mask')
    return dict(cursor=r['cursor'],proposals_n=len(p),decision_counts={n:int(np.count_nonzero(d==i)) for i,n in enumerate(REASONS)},
                mask_shape=list(m.shape),mask_dtype=str(m.dtype),full_topology_validation_still_required=True)


class Importer:
    """Deterministic leaf-to-root rehash; existing different files fail closed."""
    def __init__(self,cfg,old,plan,rows,notify=None):
        self.cfg=cfg;self.old=old;self.plan=plan;self.root=Path(cfg['run_root']);self.origin=Path(old['run_root'])
        self.rows={r['path']:r for r in rows};self.done={};self.active=set();self.notify=notify
        if len(self.rows)!=len(rows):raise ValueError('duplicate import source')
        for row in rows:
            p=Path(row['path']);rel=p.relative_to(self.origin)
            if row['relative']!=str(rel) or not any(rel.is_relative_to(t) for t in TREES):raise ValueError('unapproved import tree')
            independent_file(p);check_identity(row)

    def target(self,p):
        return within(self.root,Path(p).relative_to(self.origin))

    def transform(self,obj):
        if isinstance(obj,dict):
            if {'path','bytes','sha256'}<=obj.keys():
                p=obj['path']
                if p in self.rows:
                    row=self.rows[p]
                    if obj['bytes']!=row['bytes'] or obj['sha256']!=row['sha256']:raise ValueError('source identity edge drift')
                    new=self.materialize(p)
                    return dict(self.transform({k:v for k,v in obj.items() if k not in ('path','bytes','sha256')}),**new)
                if isinstance(p,str) and Path(p).is_relative_to(self.origin):raise ValueError('unselected old-run identity edge')
                check_identity(obj)
                return {k:self.transform(v) for k,v in obj.items()}
            common={'schema','run_id','stage','config_digest','source_digest','input_digest','plan_sha256'}
            if common<=obj.keys():
                stage=obj['stage']
                if stage not in STAGES or any(obj[k]!=v for k,v in dependencies(self.old,stage).items()):raise ValueError('unapproved dependency rebinding')
                return dict(self.transform({k:v for k,v in obj.items() if k not in common}),**dependencies(self.cfg,stage))
            return {self.transform(k):self.transform(v) for k,v in obj.items()}
        if isinstance(obj,list):return [self.transform(v) for v in obj]
        if isinstance(obj,str) and obj.startswith(str(self.origin)+'/'):
            rel=Path(obj).relative_to(self.origin)
            if obj in self.rows or any(rel==Path(t) or rel.is_relative_to(t) for t in TREES):return str(self.target(obj))
        return obj

    def materialize(self,source):
        if source in self.done:return self.done[source]['consumer']
        if source in self.active:raise ValueError('generated identity cycle')
        self.active.add(source);row=self.rows[source];target=self.target(source)
        check_identity(row)
        if target.exists():independent_file(target)
        old_identity={k:row[k] for k in ('path','bytes','sha256')}
        if Path(source).suffix=='.json':
            original=read(source)
            if not isinstance(original,dict) or '_checkpoint_import' in original:raise ValueError('unapproved JSON checkpoint shape/provenance')
            value=self.transform(original)
            value['_checkpoint_import']=dict(schema='rep1-bt-gt-checkpoint-provenance-v1',producer=old_identity,
                producer_config=self.plan['source_config'],producer_source=self.plan['source_manifest'],
                producer_commit=self.old['source_commit'],producer_dependencies=original.get('dependencies'),
                consumer_config_digest=digest(self.cfg),consumer_source_digest=self.cfg['source_digest'],
                consumer_commit=self.cfg['source_commit'],import_plan=self.cfg['checkpoint_import_plan'],
                scientific_computation_reexecuted=False)
            encoded=json.dumps(value,indent=2,sort_keys=True,allow_nan=False).encode()
            if target.exists():
                if target.read_bytes()!=encoded:raise ValueError('existing derived JSON drift: '+str(target))
            else:atomic(target,lambda f:f.write(encoded))
        else:
            if target.exists():
                if target.stat().st_size!=row['bytes'] or sha256(target)!=row['sha256']:raise ValueError('existing binary checkpoint drift')
            else:
                def copy(stream):
                    with open(source,'rb') as original:shutil.copyfileobj(original,stream,8*1024*1024)
                atomic(target,copy)
            if sha256(target)!=row['sha256']:raise ValueError('copied checkpoint differs')
        independent_file(target);new=identity(target)
        self.active.remove(source);self.done[source]=dict(producer=old_identity,consumer=new,
            transformation='JSON dependency/provenance rebinding' if Path(source).suffix=='.json' else 'byte-identical independent copy')
        if self.notify and len(self.done)%100==0:self.notify(len(self.done),len(self.rows))
        return new

    def run(self):
        for p in sorted(self.rows):self.materialize(p)
        return [self.done[p] for p in sorted(self.done)]


def verify_import(cfg):
    plan=verify_plan(cfg);receipt=read(Path(cfg['run_root'])/'admission/checkpoint_import.json')
    if receipt.get('status')!='ADOPTED' or receipt.get('config_digest')!=digest(cfg) or receipt.get('import_plan')!=cfg['checkpoint_import_plan']:
        raise ValueError('checkpoint adoption binding mismatch')
    if receipt.get('new_computation_claim') is not False or receipt.get('manual_resume_authorized') is not False:
        raise ValueError('checkpoint adoption cannot authorize science/evaluation')
    if len(receipt['mapping'])!=plan['files_n']:raise ValueError('checkpoint adoption count mismatch')
    rows=read(plan['source_inventory']['path'])['files'];expected={r['path']:{k:r[k] for k in ('path','bytes','sha256')} for r in rows}
    seen=set()
    for pair in receipt['mapping']:
        old=pair['producer'];new=pair['consumer']
        if old['path'] in seen or expected.get(old['path'])!=old:raise ValueError('adoption producer inventory mismatch')
        seen.add(old['path'])
        target=within(cfg['run_root'],Path(old['path']).relative_to(plan['source_root']))
        if new['path']!=str(target):raise ValueError('adoption consumer path mismatch')
        independent_file(target);check_identity(new)
    for key in ('legacy_validation','compatibility','consumer_validation'):check_identity(receipt[key])
    return identity(Path(cfg['run_root'])/'admission/checkpoint_import.json')


def adopt(cfg):
    from . import runtime as rt
    from .inputs import load_inputs
    from .science import validate_stage
    root=Path(cfg['run_root']);plan=verify_plan(cfg);old=read(plan['source_config']['path'])
    with ExitStack() as stack:
        newfds=tuple(stack.enter_context(rt.lock(p)) for p in rt.paths(cfg))
        oldfds=tuple(stack.enter_context(rt.lock(p,create=False)) for p in rt.paths(old)[:2])
        state=rt.bound_state(cfg);oldstate=read(plan['source_failed_state']['path'])
        if state['status']!='PREPARED' or any(rt.alive(s.get(k)) for s in (state,oldstate) for k in ('supervisor','worker')):
            raise ValueError('adoption requires inactive PREPARED target and dead old queue')
        if oldstate['status']!='FAILED' or oldstate['child_exit_codes']!={'BT_EXISTING6':0,'P2_BASE_GT_PRODUCE':1}:
            raise ValueError('unexpected failed producer state')
        rt.check_fds(cfg,newfds);ex.verify_snapshot(cfg);ex.verify_environment(cfg,newfds);manifest=load_inputs(cfg)
        if digest(old)!=plan['source_config_digest'] or old['source_digest']!=plan['source_digest'] or old['source_commit']!=plan['source_commit']:
            raise ValueError('producer config differs from recovery contract')
        comp=compatibility(cfg,plan);atomic_json(root/'admission/recovery_compatibility.json',comp)
        # Legacy semantics execute in their own isolated old snapshot with all
        # five OFDs inherited; none of the producer's files are rewritten.
        driver=Path(cfg['source_snapshot'])/'scripts/rep1_bt_gt_fusion/recovery_legacy.py'
        result=subprocess.run([ex.PYTHON,'-I','-S','-B',str(driver),str(ex.OLD_ROOT),
            ','.join(map(str,(oldfds[0],oldfds[1],newfds[2])))],cwd=ex.OLD_ROOT/'source_snapshot',
            env=rt.child_environment(cfg),pass_fds=(*newfds,*oldfds),stdin=subprocess.DEVNULL,
            capture_output=True,text=True,timeout=300)
        if result.returncode:raise RuntimeError('legacy checkpoint validation failed: '+result.stderr[-12000:])
        legacy=json.loads(result.stdout);atomic_json(root/'admission/legacy_checkpoint_validation.json',legacy)
        if legacy['status']!='PASS_LEGACY_CHECKPOINTS':raise ValueError('legacy validation failed')
        rows=read(plan['source_inventory']['path'])['files']
        actual={str(p) for t in TREES for p in (ex.OLD_ROOT/t).rglob('*') if p.is_file() or p.is_symlink()}
        if actual!={r['path'] for r in rows} or len(rows)!=plan['files_n'] or sum(r['bytes'] for r in rows)!=plan['copy_bytes']:
            raise ValueError('old checkpoint exact inventory mismatch')
        def notify(n,total):atomic_json(root/'control/import_progress.json',dict(copied_files=n,total_files=total,
                            new_scientific_progress=False,time=rt.utc()))
        mapping=Importer(cfg,old,plan,rows,notify).run()
        target_actual={str(p) for t in TREES for p in (root/t).rglob('*')
                       if (p.is_file() or p.is_symlink()) and not any(x.startswith('.') for x in p.relative_to(root/t).parts)}
        if target_actual!={p['consumer']['path'] for p in mapping}:raise ValueError('unexpected target checkpoint inventory')
        validate_stage(cfg,'BT_EXISTING6',manifest);gt=partial_gt(cfg,manifest)
        forbidden=[root/p for p in ('endpoints/P2_BASE_GT/final.tif','endpoints/P2_BASE_GT/gt/complete.json',
                    'endpoints/P2_BASE_GT/matrix','endpoints/V2_MORPH_GT','production_complete.json','control/manual_resume.json')]
        if any(p.exists() for p in forbidden):raise ValueError('adoption manufactured later scientific outputs')
        consumer=dict(status='PASS_ADOPTED_CHECKPOINT_INTEGRITY',BT_EXISTING6=identity(root/'bt_existing6/complete.json'),
                      GT_partial=gt,scientific_production_still_required=plan['required_post_import_science'])
        atomic_json(root/'admission/consumer_checkpoint_validation.json',consumer)
        receipt=dict(schema='rep1-bt-gt-checkpoint-adoption-v1',status='ADOPTED',config_digest=digest(cfg),
            import_plan=cfg['checkpoint_import_plan'],new_computation_claim=False,manual_resume_authorized=False,
            legacy_validation=identity(root/'admission/legacy_checkpoint_validation.json'),
            compatibility=identity(root/'admission/recovery_compatibility.json'),
            consumer_validation=identity(root/'admission/consumer_checkpoint_validation.json'),mapping=mapping)
        destination=root/'admission/checkpoint_import.json'
        if destination.exists():
            if read(destination)!=receipt:raise ValueError('completed adoption receipt differs')
        else:atomic_json(destination,receipt)
        verify_import(cfg);rt.event(cfg,'CHECKPOINTS_ADOPTED',receipt=identity(destination),new_computation_claim=False)
        return dict(status='ADOPTED',files=len(mapping),producer_bytes=plan['copy_bytes'],GT_cursor=gt['cursor'],
                    receipt=identity(destination),formal_launch_still_requires_independent_admission=True)
