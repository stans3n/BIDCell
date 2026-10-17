"""Explicit, immutable adoption of the one reviewed failed producer run.

Data remains at its original producer path. No code, data links, historical
receipts, or historical config are rewritten. The cached proof is valid only
while every recorded mutation fingerprint remains exact.
"""
from __future__ import annotations
import ast
import copy
from pathlib import Path
import stat

from .common import (atomic_json, check_identity, dependencies, digest, identity,
                     read, safe, sha256, within, write_receipt)

WORK = Path('/home/zwan0726/work')
DONOR = WORK/'outputs/bidcell_rep1_triple_simple_boundary_20261017/rep1-triple-simple-boundary-20261017-r01'
DONOR_COMMIT = '6f1ee253aeb2f24563c2d01dadb5bfcd90815933'
DONOR_CONFIG = '2050f88d3e73bc09512d5498ae138f1780c973bfc6e85433a7bbf870ddb7d3f8'
REPAIR_PLAN = WORK/'docs/BIDCell_Rep1_Triple_Simple_Recovery01_ChangePlan_20261017.md'
REPAIR_PLAN_SHA = '750ed91b9be31694691ed252ec3ce3ebf078591d80584a04e3cd89bf4ff8ef9e'
REPAIR_AUDIT = WORK/'docs/BIDCell_Rep1_Triple_Simple_Recovery01_ChangePlan_Audit_20261017.json'
REPAIR_AUDIT_SHA = '4b32bbc859350604f107453c6f0a750db7c19e9454c0257e14116fff45af1443'
SCHEMA = 'rep1-triple-simple-continuation-v1'
_CACHE = {}
_DATA_CACHE = {}


def fingerprint(path):
    p = Path(path); s = p.lstat()
    if stat.S_ISLNK(s.st_mode) or not (stat.S_ISREG(s.st_mode) or stat.S_ISDIR(s.st_mode)):
        raise ValueError('continuation requires unlinked regular data/source paths: '+str(p))
    return [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns,s.st_mode,s.st_nlink]


def _frozen_json(path, value):
    if Path(path).exists():
        if read(path) != safe(value):raise ValueError('committed continuation evidence differs: '+str(path))
    else:atomic_json(path,value)


def _bound_json(item):
    key=(item['path'],item['sha256'],item.get('bytes'))
    fp=fingerprint(item['path']);cached=_CACHE.get(key)
    if cached is not None:
        if cached[0]!=fp:raise ValueError('cached continuation proof mutated: '+item['path'])
        return cached[1]
    check_identity(item);value=read(item['path']);_CACHE[key]=(fp,value)
    return value


def approved_repair(cfg=None):
    check_identity(dict(path=str(REPAIR_PLAN),sha256=REPAIR_PLAN_SHA))
    check_identity(dict(path=str(REPAIR_AUDIT),sha256=REPAIR_AUDIT_SHA))
    a=read(REPAIR_AUDIT)
    if a.get('decision')!='PASS_FOR_REPAIR_IMPLEMENTATION' or a.get('open_blockers')!=[]:
        raise PermissionError('repair plan has no root implementation approval')
    if cfg and cfg.get('continuation') and manifest(cfg).get('generation2'):
        from .continuation2 import approved
        approved()
    if cfg and cfg.get('continuation') and manifest(cfg).get('generation3'):
        from .continuation3 import approved
        approved()


def manifest(cfg):
    """Check the acyclic execution binding; do not silently infer a donor."""
    item=cfg.get('continuation')
    if not item:raise ValueError('explicit continuation manifest required')
    p=within(cfg['run_root'],'configs/continuation.json')
    if item['path']!=str(p):raise ValueError('continuation manifest location differs')
    m=_bound_json(item)
    if m.get('schema')=='rep1-triple-simple-continuation-v3':
        from .continuation3 import expand
        m=expand(cfg,m)
    elif m.get('schema')=='rep1-triple-simple-continuation-v2':
        from .continuation2 import expand
        m=expand(cfg,m)
    if m.get('schema')!=SCHEMA or m.get('target_run_id')!=cfg['run_id'] or m.get('target_run_root')!=cfg['run_root']:
        raise ValueError('continuation target identity differs')
    if m['new_source_digest']!=cfg['source_digest'] or m['new_source_commit']!=cfg['source_commit']:
        raise ValueError('continuation reviewed source differs')
    donor=_bound_json(m['donor_config'])
    if m['donor_root']!=str(DONOR) or donor['run_root']!=str(DONOR) or digest(donor)!=DONOR_CONFIG or donor['source_commit']!=DONOR_COMMIT:
        raise ValueError('unapproved continuation producer')
    if 'input_digest' in cfg and (cfg['input_digest']!=donor['input_digest'] or cfg['domain_digest']!=donor['domain_digest'] or cfg['domain_manifest']!=donor['domain_manifest']):
        raise ValueError('continuation changed frozen scientific inputs/domain')
    return m


def fingerprint_paths(m):
    if m.get('generation3'):
        from .continuation3 import fingerprints
        return fingerprints(m)
    if m.get('generation2'):
        from .continuation2 import fingerprints
        return fingerprints(m)
    return _bound_json(m['fingerprints'])['paths']


def evidence_items(m):
    result=[m[k] for k in ('donor_config','donor_source_manifest','donor_admission','donor_validation','split_validation','repair_plan','repair_plan_audit','source_compatibility','fingerprints')]
    if m.get('generation2'):
        from .continuation2 import evidence_items as additional
        result+=additional(m)
    if m.get('generation3'):
        from .continuation3 import evidence_items as third
        result+=third(m)
    return result


def verify(cfg, all_fingerprints=True):
    m=manifest(cfg)
    for key in ('donor_config','donor_source_manifest','donor_admission','donor_validation','split_validation','repair_plan','repair_plan_audit','source_compatibility','fingerprints'):
        item=m[key]
        if key in ('repair_plan','repair_plan_audit'):check_identity(item)
        else:_bound_json(item)
    if all_fingerprints:
        for path,expected in fingerprint_paths(m).items():
            if fingerprint(path)!=expected:raise ValueError('adopted producer mutation fingerprint drift: '+path)
    return m


def frozen_root(cfg):
    return Path(manifest(cfg)['domain_manifest']['path']).parent if cfg.get('continuation') else Path(cfg['run_root'])/'frozen_domain'


def initial_root(cfg):
    return Path(manifest(cfg)['initial_root']) if cfg.get('continuation') else Path(cfg['run_root'])/'analysis/initial'


def producer_cfg(cfg, name):
    if cfg.get('continuation') and name=='P2_MORPH_GT':return _bound_json(manifest(cfg)['donor_config'])
    return cfg


def endpoint_root(cfg,name):
    producer=producer_cfg(cfg,name)
    return within(producer['run_root'],'endpoints',name)


def validate_adopted_stage(cfg,stage):
    """Use already completed full semantic validation, never a filename check."""
    from .contracts import STAGES
    m=manifest(cfg)
    if stage not in m['stages']:raise ValueError('stage not authorized for adoption')
    p=within(cfg['run_root'],STAGES[stage]['required_receipt']);r=read(p)
    if r.get('schema')!=SCHEMA or r.get('status')!='succeeded' or r.get('dependencies')!=dependencies(cfg,stage) or r.get('adopted') is not True or r.get('recomputed') is not False:
        raise ValueError('adoption wrapper identity differs')
    expected=m['stages'][stage]
    if r.get('original_producer')!=expected or r.get('continuation')!=cfg['continuation'] or r.get('reused_from')!=expected['receipt']:
        raise ValueError('adoption wrapper producer differs')
    if r.get('outputs')!={'original_receipt':expected['receipt']}:raise ValueError('adoption wrapper output authorization differs')
    # ExecutionValidator checks the whole fingerprint set at each boundary.
    # These narrow checks also make direct read-only stage validation fail closed.
    fps=fingerprint_paths(m)
    for obj in (expected['receipt'],expected.get('config',m['donor_config']),expected['semantic_validation']):
        if fingerprint(obj['path'])!=fps[obj['path']]:raise ValueError('adopted stage producer drift')
    return r


def native_replay_root(cfg):
    if cfg.get('continuation') and manifest(cfg).get('generation2'):
        validate_adopted_stage(cfg,'05')
        return Path(manifest(cfg)['stages']['05']['receipt']['path']).parent
    from .common import check_receipt
    root=within(cfg['run_root'],'replay/SIMPLE_NATIVE')
    check_receipt(root/'complete.json',dependencies(cfg,'05'),cfg['run_root'])
    return root


def install_stages(cfg):
    from .contracts import STAGES
    m=verify(cfg)
    for stage,original in m['stages'].items():
        p=within(cfg['run_root'],STAGES[stage]['required_receipt'])
        _frozen_json(p,dict(schema=SCHEMA,status='succeeded',dependencies=dependencies(cfg,stage),
            outputs={'original_receipt':original['receipt']},adopted=True,recomputed=False,
            reused_from=original['receipt'],original_producer=original,continuation=cfg['continuation']))


def _partition_files(source, rec):
    source=Path(source);result={}
    for item in rec['outputs'].values():
        p=Path(item['path'])
        if source not in p.parents:raise ValueError('partition output outside original producer directory')
        relative=str(p.relative_to(source))
        if relative in result and result[relative]!=item:raise ValueError('partition conflicting output identity')
        result[relative]=item
    return result


def partition_authorization(cfg, source, target, deps, binding):
    """Separate cross-version predicate; same-config reuse stays unchanged."""
    m=manifest(cfg);source=Path(source).resolve(strict=True);target=within(cfg['run_root'],target)
    entry=m['partitions'].get(str(source))
    if entry is None or str(target.relative_to(cfg['run_root']))!=entry['consumer_relative']:
        raise ValueError('partition is not an approved completed continuation producer/consumer')
    old=_bound_json(entry['receipt']);donor=_bound_json(m['donor_config'])
    if old.get('status')!='succeeded' or old.get('dependencies')!=entry['producer_dependencies'] or old.get('reuse_binding')!=entry['scientific_binding'] or binding!=entry['scientific_binding']:
        raise ValueError('continuation partition exact scientific binding differs')
    if any(old['dependencies'].get(k)!=dependencies(donor,'unused')[k] for k in ('config_digest','run_id','source_digest','input_digest','domain_digest','plan_digest')):
        raise ValueError('continuation partition old producer dependencies differ')
    if any(deps.get(k)!=dependencies(cfg,'unused')[k] for k in ('config_digest','run_id','source_digest','input_digest','domain_digest','plan_digest')):
        raise ValueError('continuation partition current consumer dependencies differ')
    if _partition_files(source,old)!=entry['files']:raise ValueError('partition output mapping differs')
    fps=fingerprint_paths(m)
    for item in (entry['receipt'],*entry['files'].values()):
        if fingerprint(item['path'])!=fps[item['path']]:raise ValueError('continuation partition content changed')
    return entry,old


def reuse_partition(cfg,source,target,deps,binding,receipt_name):
    entry,old=partition_authorization(cfg,source,target,deps,binding)
    target=within(cfg['run_root'],target);target.mkdir(parents=True,exist_ok=True)
    ref=target/'.data-reference.json'
    _frozen_json(ref,dict(schema=SCHEMA,source=str(Path(source).resolve()),target=str(target),
        continuation=cfg['continuation'],execution_config=identity(Path(cfg['run_root'])/'configs/run.json'),
        mapping_digest=digest(entry['files']),receipt_name=receipt_name))
    fields={k:v for k,v in old.items() if k not in ('schema','status','dependencies','outputs','reused_from','producer_dependencies')}
    fields.update(reused_from=entry['receipt'],producer_dependencies=entry['producer_dependencies'],
                  adoption_kind='explicit_cross_version_readonly_reference',continuation=cfg['continuation'],
                  data_reference=identity(ref),recomputed=False)
    return write_receipt(target/receipt_name,deps,{'data_reference':identity(ref)},**fields)


def read_data(path):
    """Resolve only a fully registered scientific leaf; never a write/import API.

    Pure reused partitions contain only the consumer receipt and this mapping.
    Transfer partitions additionally write a new structure subtree. A mapped
    leaf can neither be shadowed by a new local file nor retargeted.
    """
    p=Path(path).absolute()
    for parent in p.parents:
        ref=parent/'.data-reference.json'
        if not ref.is_file():continue
        fp=fingerprint(ref);cached=_DATA_CACHE.get(str(ref))
        if cached is None or cached[0]!=fp:
            r=read(ref);cfg=_bound_json(r['execution_config']);m=manifest(cfg)
            if r['execution_config']['path']!=str(Path(cfg['run_root'])/'configs/run.json'):raise ValueError('data reference requires the actual immutable execution config path')
            if r.get('schema')!=SCHEMA or r.get('continuation')!=cfg['continuation'] or r.get('target')!=str(parent):raise ValueError('data reference execution binding drift')
            rec=read(parent/r['receipt_name'])
            if rec.get('data_reference')!=identity(ref):raise ValueError('data reference receipt binding drift')
            entry,_=partition_authorization(cfg,r['source'],parent,rec['dependencies'],rec['reuse_binding'])
            if r.get('mapping_digest')!=digest(entry['files']):raise ValueError('data reference mapping drift')
            cached=(fp,r,cfg,entry,fingerprint(parent/r['receipt_name']),None);_DATA_CACHE[str(ref)]=cached
        _,r,cfg,entry,receipt_fp,directory_fp=cached
        m=manifest(cfg)
        if fingerprint(parent/r['receipt_name'])!=receipt_fp:raise ValueError('data consumer receipt changed')
        current_directory_fp=fingerprint(parent)
        if current_directory_fp!=directory_fp:
            allowed={'.data-reference.json',r['receipt_name']}
            if r['receipt_name']=='initial_reuse.json':allowed.update(('structure','complete.json'))
            for child in parent.iterdir():
                if child.name not in allowed or child.is_symlink() or (child.name=='structure')!=child.is_dir():
                    raise ValueError('unregistered entry in read-only data reference: '+str(child))
            _DATA_CACHE[str(ref)]=(fp,r,cfg,entry,receipt_fp,current_directory_fp)
        rel=str(p.relative_to(parent))
        if rel in entry['files']:
            if p.exists() or p.is_symlink():raise ValueError('read-only referenced data shadowed by local file')
            result=Path(entry['files'][rel]['path'])
            if fingerprint(result)!=fingerprint_paths(m)[str(result)]:raise ValueError('referenced leaf changed')
            return result
        if rel in ('complete.json','initial_reuse.json','.data-reference.json') or (r['receipt_name']=='initial_reuse.json' and p.relative_to(parent).parts[0]=='structure'):
            return p
        raise ValueError('unregistered leaf in read-only data reference: '+str(p))
    return p


def reused_bytes(cfg):
    if not cfg.get('continuation'):return dict(gt=0,p2_replay=0,analysis=0,shared=0)
    return dict(manifest(cfg)['validated_external_reuse_bytes'])


def adopt_tiles(cfg,dep,out,tasks):
    from .p2 import record,verify_record
    if dep.get('rule')=='P2':
        if manifest(cfg).get('generation3'):
            from .continuation3 import adopt_tiles as adopt_p2
            return adopt_p2(cfg,dep,out,tasks)
        from .continuation2 import adopt_tiles as adopt_p2
        return adopt_p2(cfg,dep,out,tasks)
    m=manifest(cfg);partial=m['partial_replay'];oldbase=partial['tile_base']
    if dep.get('run')!=dependencies(cfg,'05') or dep.get('rule')!='B' or Path(out)!=within(cfg['run_root'],'replay/SIMPLE_NATIVE/predictions/postprocess'):
        raise ValueError('partial replay adoption requires current stage05 native producer')
    if {k:v for k,v in dep.items() if k not in ('run','source')}!={k:v for k,v in oldbase.items() if k not in ('run','source')}:
        raise ValueError('partial replay actual input/native contract differs')
    if len(tasks)!=len(partial['tile_receipts']):raise ValueError('partial replay tile geometry/coverage differs')
    oldtiles={tuple(read(item['path'])['coords']):item for item in partial['tile_receipts']}
    if len(oldtiles)!=len(tasks):raise ValueError('duplicate donor tile coordinates')
    fps=fingerprint_paths(m)
    for y,x,size,dest,tiledep,native_pred in tasks:
        item=oldtiles.get((y,x))
        if item is None or fingerprint(item['path'])!=fps[item['path']]:raise ValueError('unapproved donor tile')
        old=verify_record(item['path'],dict(base=oldbase,coords=[y,x],shape=tiledep['shape']))
        if old['coords']!=[y,x] or old['shape']!=tiledep['shape']:raise ValueError('donor tile shape/order differs')
        receipt=Path(dest).with_suffix('.json')
        if receipt.exists():
            current=verify_record(receipt,tiledep)
            if current.get('reused_from')!=item or current.get('original_producer')!=old:raise ValueError('partial tile adoption provenance drift')
        else:
            record(receipt,tiledep,{k:v['path'] for k,v in old['outputs'].items()},coords=old['coords'],shape=old['shape'],diagnostics=old['diagnostics'],
                reused_from=item,adopted=True,recomputed=False,original_producer=old,continuation=cfg['continuation'])


def adopt_split(cfg,dep,splitdir,splitdeps,ids):
    from .p2 import record,verify_record
    if dep.get('rule')=='P2':
        if manifest(cfg).get('generation3'):
            from .continuation3 import adopt_split as adopt_p2_split
            return adopt_p2_split(cfg,dep,splitdir,splitdeps,ids)
        if dep.get('run')!=dependencies(cfg,'06') or not manifest(cfg).get('generation2'):
            raise ValueError('new P2 splits require current stage06 continuation')
        return # No old P2 split exists; every P2 split is computed normally.
    m=manifest(cfg);partial=m['partial_replay']
    expected_ids=partial['split_id_digest']
    if dep.get('run')!=dependencies(cfg,'05') or len(ids)!=partial['split_id_count'] or expected_ids.get('encoding')!='canonical-json-sha256' or digest(ids)!=expected_ids['sha256']:
        raise ValueError('canonical merged native split ID set differs')
    original=verify_record(partial['old_split']['path'],partial['old_split_expected_dependencies'])
    # Reconstruct the opaque legacy dependency from the exact original ordered
    # tile receipts and newly reconstructed canonical sorted IDs.
    if digest(dict(base=partial['tile_base'],tiles=partial['tile_receipts'],ids=ids))!=original['dependencies']['base'] or ids[:128]!=original['dependencies']['ids']:
        raise ValueError('legacy split opaque digest/upstream lineage differs')
    d=dict(base=splitdeps,ids=ids[:128],run=dep['run']);receipt=Path(splitdir)/'000000.json'
    if receipt.exists():
        current=verify_record(receipt,d)
        if current.get('reused_from')!=partial['old_split'] or current.get('original_producer')!=original:raise ValueError('partial split adoption provenance drift')
    else:
        record(receipt,d,{k:v['path'] for k,v in original['outputs'].items()},reused_from=partial['old_split'],
            adopted=True,recomputed=False,original_producer=original,continuation=cfg['continuation'])


def source_compatibility(old_root,new_root):
    """Compare actual kernels, not adapter filenames or asserted source SHAs."""
    old_root,new_root=Path(old_root),Path(new_root);unchanged={}
    for relative in ('bidcell/model/postprocess_predictions.py','scripts/boundary_three/p2_rules.py',
                     'scripts/boundary_three/core_refine.py','scripts/boundary_three/core_scores.py',
                     'scripts/crgf/postprocess.py','scripts/crgf/matrix.py','scripts/rep1_triple_simple_boundary/gt.py',
                     'scripts/rep1_triple_simple_boundary/matrix.py','scripts/rep1_triple_simple_boundary/spatial_bootstrap.py',
                     'scripts/rep1_triple_simple_boundary/evaluation.py','scripts/rep1_triple_simple_boundary/paper_evaluation.R'):
        a,b=identity(old_root/relative),identity(new_root/relative)
        if a['sha256']!=b['sha256']:raise ValueError('unapproved numerical kernel change: '+relative)
        unchanged[relative]=dict(original=a,current=b)
    def nodes(root):
        p=root/'scripts/rep1_triple_simple_boundary/p2.py'
        return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(p.read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    a,b=nodes(old_root),nodes(new_root);functions={}
    for name in ('NativeWrites','native_connect','tile_task','tile_coordinates','assert_matrix_equal'):
        if a[name]!=b[name]:raise ValueError('native numerical function changed: '+name)
        functions[name]=digest(a[name])
    def expression(text):return ast.parse(text,mode='eval').body
    replacements={ast.dump(expression(before),include_attributes=False):expression(after) for before,after in (
        ("tile_record['outputs']['postislands']['path']","out/f'tile_{y}_{x}.tif'"),
        ("tile_record['outputs']['postconnect']['path']","out/f'tile_{y}_{x}_postconnect.tif'"),
        ("verify_record(receipt,d)['outputs']['split']","identity(dest)"))}
    class EnvelopeOnly(ast.NodeTransformer):
        def visit_FunctionDef(self,node):
            pairs=[(a,d) for a,d in zip(node.args.kwonlyargs,node.args.kw_defaults) if a.arg not in ('continuation_cfg','index_validation')]
            node.args.kwonlyargs=[a for a,d in pairs];node.args.kw_defaults=[d for a,d in pairs]
            return self.generic_visit(node)
        def visit_If(self,node):
            if ast.unparse(node.test)=='continuation_cfg is not None':
                allowed=("from .continuation import adopt_tiles\nadopt_tiles(continuation_cfg, dep, out, tasks)",
                         "from .continuation import adopt_split\nadopt_split(continuation_cfg, dep, splitdir, splitdeps, check)")
                if any(ast.dump(ast.Module(body=node.body,type_ignores=[]),include_attributes=False)==ast.dump(ast.parse(text),include_attributes=False) for text in allowed) and not node.orelse:return None
                raise ValueError('unreviewed continuation hook body')
            if ast.unparse(node.test)=='index_validation is not None':
                text='from .nucleus_validation import commit\ncommit(_INDEX, out, dep, index_validation, callback)'
                if ast.dump(ast.Module(body=node.body,type_ignores=[]),include_attributes=False)==ast.dump(ast.parse(text),include_attributes=False) and not node.orelse:return None
                raise ValueError('unreviewed nuclear validation hook body')
            if ast.unparse(node.test)=='dependency_context is not None' and len(node.body)==1 and isinstance(node.body[0],ast.Assign) and ast.unparse(node.body[0].targets[0])=="d['run']":return None
            return self.generic_visit(node)
        def visit_Assign(self,node):
            if len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id=='tile_record':return None
            return self.generic_visit(node)
        def visit_Subscript(self,node):
            replacement=replacements.get(ast.dump(node,include_attributes=False))
            return copy.deepcopy(replacement) if replacement is not None else self.generic_visit(node)
    def process_tree(root,normalize):
        tree=next(n for n in ast.parse((root/'scripts/rep1_triple_simple_boundary/p2.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='process')
        return ast.dump(EnvelopeOnly().visit(tree) if normalize else tree,include_attributes=False)
    old_process,new_process=process_tree(old_root,True),process_tree(new_root,True)
    if old_process!=new_process:raise ValueError('process numerical/order/cleanup kernel changed beyond explicit envelope and verified input readers')
    class DiagnosticReaderOnly(ast.NodeTransformer):
        def visit_ImportFrom(self,node):
            if ast.unparse(node) in ('from .p2 import verify_record','from .continuation import frozen_root'):return None
            return self.generic_visit(node)
        def visit_BinOp(self,node):
            if ast.unparse(node)=="frozen_root(cfg) / 'cells.csv'":
                return expression("Path(cfg['run_root'])/'frozen_domain/cells.csv'")
            return self.generic_visit(node)
        def visit_Assign(self,node):
            if ast.unparse(node)=="tile_record = verify_record(pred / 'postprocess' / f'tile_{y}_{x}.json')":return None
            return self.generic_visit(node)
        def visit_Subscript(self,node):
            if ast.unparse(node)=="tile_record['outputs']['diagnostics']['path']":
                return expression("pred/'postprocess'/f'tile_{y}_{x}_diagnostics.npz'")
            return self.generic_visit(node)
    def diagnostics_tree(root):
        tree=ast.parse((root/'scripts/rep1_triple_simple_boundary/p2_diagnostics.py').read_text())
        return ast.dump(DiagnosticReaderOnly().visit(tree),include_attributes=False)
    old_diag,new_diag=diagnostics_tree(old_root),diagnostics_tree(new_root)
    if old_diag!=new_diag:raise ValueError('P2 diagnostics changed beyond the verified tile payload reader')
    return dict(schema='rep1-triple-simple-source-compatibility-v1',unchanged_kernels=unchanged,
        exact_AST_functions=functions,process_kernel_AST_sha256=digest(old_process),
        diagnostics_kernel_AST_sha256=digest(old_diag),
        process_normalizations=['remove exact continuation-only adoption hooks','remove exact full-index numerical validation hook','replace verified tile/split output readers with original output locations','remove explicit split run envelope'],
        adapter_original=identity(old_root/'scripts/rep1_triple_simple_boundary/p2.py'),
        adapter_current=identity(new_root/'scripts/rep1_triple_simple_boundary/p2.py'),
        scope='Numerical kernels unchanged; explicit receipt/source/reader adaptations require independent full commit review')


def build(partial,donor_validation,split_validation,baseline):
    """Preparation-only assembly from separately completed root validations."""
    from . import execution as ex
    from .contracts import STAGES
    approved_repair();root=Path(partial['run_root'])
    donor=read(DONOR/'configs/run.json')
    if digest(donor)!=DONOR_CONFIG or donor['source_commit']!=DONOR_COMMIT:raise ValueError('donor cfg/commit differs')
    ex.verify_snapshot(donor,location=False)
    proof=read(donor_validation);split=read(split_validation);base=read(baseline)
    if proof.get('status')!='PASS_ORIGINAL01_TO04_AND_PARTIAL_REPLAY_CONTENT_LINEAGE':raise ValueError('root full donor validation missing')
    if proof['donor_config']!=identity(DONOR/'configs/run.json') or proof['source_commit']!=DONOR_COMMIT or proof['source_manifest']!=identity(DONOR/'source_manifest.json'):
        raise ValueError('root validation belongs to another producer')
    if proof.get('input_digest')!=donor['input_digest'] or proof.get('domain_digest')!=donor['domain_digest'] or proof.get('input_files')!=len(read(donor['input_manifest'])['files']) or proof.get('tile_count')!=len(proof['tile_receipts']) or proof.get('old_split_integrity_outputs_ids_and_parent_digest_passed') is not True:
        raise ValueError('root validation scientific closure differs')
    if split.get('status')!='PASS_FIRST128_ORIGINAL_SPLIT_NUMERIC_RECOMPUTATION' or split.get('donor_config')!=proof['donor_config'] or len(split.get('rows',[]))!=128 or not all(r.get('exact') is True for r in split['rows']) or split.get('upstream_chain_validation')!=identity(donor_validation) or split.get('split_receipt')!=proof['old_split']:
        raise ValueError('original first split full numerical validation missing')
    if split['split_kernel']!=identity(Path(donor['source_snapshot'])/'scripts/crgf/postprocess.py') or split['nuclei']!=proof['tile_base']['nuclei'] or split['canonical_merge']!=identity(DONOR/'replay/SIMPLE_NATIVE/predictions/postislands_ordered_tile_merge.tif'):
        raise ValueError('split numerical proof source/input bindings differ')
    if base['root']!=str(DONOR) or base['old_config_digest']!=DONOR_CONFIG:raise ValueError('donor baseline binding differs')
    original_fp={str(DONOR/path):fp for path,fp in base['old_files_and_directories'].items()}
    if {str(p) for p in DONOR.rglob('*')}!=set(original_fp):raise ValueError('donor file/directory inventory changed')
    for path,fp in original_fp.items():
        if fingerprint(path)!=fp:raise ValueError('donor changed since immutable baseline: '+path)
    checks={r['stage']:r for r in proof['stage_semantic_checks']}
    if set(checks)!=set(('01','02','03','04')):raise ValueError('all completed donor stages must be semantically validated')
    stages={}
    for stage,check in checks.items():
        item=identity(DONOR/STAGES[stage]['required_receipt'])
        if check['receipt']!=item:raise ValueError('donor semantic receipt binding differs')
        expected=dependencies(donor,stage)
        if stage=='03':expected['production']=identity(DONOR/'endpoints/P2_MORPH_GT/production_complete.json')
        if read(item['path'])['dependencies']!=expected:raise ValueError('donor stage identity differs')
        stages[stage]=dict(receipt=item,dependencies=read(item['path'])['dependencies'],
            config_digest=DONOR_CONFIG,source_commit=DONOR_COMMIT,source_digest=donor['source_digest'],
            semantic_validation=identity(donor_validation),validated_seconds=check['validated_seconds'])
    compatibility=source_compatibility(donor['source_snapshot'],partial['source_snapshot'])
    compat_path=root/'admission/continuation_source_compatibility.json';_frozen_json(compat_path,compatibility)
    partitions={}
    for parent in (DONOR/'analysis/initial/populations',DONOR/'analysis/initial/comparisons',DONOR/'analysis/initial/bt',DONOR/'analysis/initial/boundary/methods',DONOR/'analysis/initial/boundary/comparisons',*(DONOR/'endpoints/P2_MORPH_GT/evaluation/paired'/part for part in ('populations','comparisons','bt'))):
        for p in sorted(parent.iterdir()):
            if not p.is_dir() or not (p/'complete.json').is_file():continue
            rec=read(p/'complete.json')
            if not rec.get('reuse_binding'):raise ValueError('completed donor partition lacks scientific binding')
            relative=str(p.relative_to(DONOR/'analysis/initial')) if DONOR/'analysis/initial' in p.parents else str(p.relative_to(DONOR/'endpoints/P2_MORPH_GT/evaluation/paired'))
            partitions[str(p)]=dict(receipt=identity(p/'complete.json'),producer_dependencies=rec['dependencies'],
                scientific_binding=rec['reuse_binding'],files=_partition_files(p,rec),consumer_relative='analysis/all12/'+relative)
    # Credit only products actually adopted/reused, once by device/inode. This
    # reduces logical remaining work; it never adds to the observed free disk.
    credits=dict(gt=0,p2_replay=0,analysis=0,shared=0);seen=set();credit_files={}
    groups={'gt':read(DONOR/'endpoints/P2_MORPH_GT/production_complete.json')['outputs'],
            'shared':read(donor['domain_manifest'])['outputs']}
    groups['gt']=dict(groups['gt'],**{'eval/'+k:v for k,v in read(DONOR/'endpoints/P2_MORPH_GT/evaluation/complete.json')['outputs'].items() if '/evaluation/paired/' not in v['path']})
    # Use paths as keys: relative names repeat across different partitions.
    groups['analysis']={item['path']:item for entry in partitions.values() for item in entry['files'].values()}
    groups['p2_replay']={item['path']:item for rec in proof['tile_receipts'] for item in read(rec['path'])['outputs'].values()}
    for item in read(proof['old_split']['path'])['outputs'].values():groups['p2_replay'][item['path']]=item
    for category,items in groups.items():
        for item in items.values():
            p=Path(item['path']);s=p.stat();key=(s.st_dev,s.st_ino)
            if str(p) not in original_fp:raise ValueError('reuse credit absent from immutable donor baseline')
            if key in seen:continue
            seen.add(key);credits[category]+=s.st_blocks*512;credit_files[str(p)]=dict(category=category,dev=s.st_dev,ino=s.st_ino,allocated=s.st_blocks*512,identity=item)
    # The full 84k-path donor baseline is checked above and retained for the
    # final preservation audit. Runtime binds the complete scientific closure,
    # not unrelated historical heartbeat/progress log files.
    scientific={str(DONOR/'configs/run.json'),str(DONOR/'source_manifest.json'),
        str(DONOR/'environment.json'),str(DONOR/'admission/final.json'),str(DONOR/'admission/preflight.json'),
        str(DONOR/'state.json'),str(donor['input_manifest']),str(donor['domain_manifest'])}
    for entry in stages.values():
        scientific.add(entry['receipt']['path']);scientific.update(x['path'] for x in read(entry['receipt']['path'])['outputs'].values())
    scientific.update(x['snapshot_path'] for x in read(DONOR/'source_manifest.json')['files'])
    scientific.update(x['path'] for x in read(donor['domain_manifest'])['outputs'].values())
    for item in (*proof['tile_receipts'],proof['old_split']):
        scientific.add(item['path']);scientific.update(x['path'] for x in read(item['path'])['outputs'].values())
    for key in ('canonical_merge','nuclei','split_kernel'):scientific.add(split[key]['path'])
    for path in tuple(scientific):
        p=Path(path)
        if DONOR in p.parents:
            scientific.update(str(parent) for parent in p.parents if parent==DONOR or DONOR in parent.parents)
    paths={p:(original_fp[p] if p in original_fp else fingerprint(p)) for p in scientific}
    inputs=read(donor['input_manifest'])
    for item in inputs['files'].values():check_identity(item)
    for item in [*inputs['files'].values(),identity(donor_validation),identity(split_validation),identity(baseline),identity(REPAIR_PLAN),identity(REPAIR_AUDIT)]:paths[item['path']]=fingerprint(item['path'])
    fp_path=root/'admission/continuation_fingerprints.json';_frozen_json(fp_path,dict(schema=SCHEMA,paths=paths))
    obj=dict(schema=SCHEMA,target_run_id=partial['run_id'],target_run_root=partial['run_root'],
        new_source_commit=partial['source_commit'],new_source_digest=partial['source_digest'],
        donor_root=str(DONOR),donor_config=identity(DONOR/'configs/run.json'),donor_source_manifest=identity(DONOR/'source_manifest.json'),
        donor_admission=identity(DONOR/'admission/final.json'),donor_validation=identity(donor_validation),split_validation=identity(split_validation),
        repair_plan=identity(REPAIR_PLAN),repair_plan_audit=identity(REPAIR_AUDIT),source_compatibility=identity(compat_path),fingerprints=identity(fp_path),
        domain_manifest=identity(donor['domain_manifest']),input_manifest=identity(donor['input_manifest']),
        stages=stages,initial_root=str(DONOR/'analysis/initial'),partitions=partitions,
        partial_replay=dict(tile_receipts=proof['tile_receipts'],tile_base=proof['tile_base'],old_split=proof['old_split'],
            old_split_expected_dependencies=proof['old_split_expected_dependencies'],split_id_count=proof['split_id_count'],
            split_id_digest=dict(encoding='canonical-json-sha256',sha256=proof['split_id_hash'])),
        validated_external_reuse_bytes=credits,reuse_credit_files=credit_files,
        resource_credit_semantics='validated logical remaining work only; observed free bytes are never increased',
        recomputed=False,stage_order_unchanged=True)
    p=root/'configs/continuation.json';_frozen_json(p,obj)
    return identity(p)
