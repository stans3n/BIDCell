"""Independent source copies, environment identities and fail-closed admission."""
from __future__ import annotations
import importlib.machinery
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import sysconfig
from .common import atomic_json, check_identity, digest, identity, read, sha256, within
from .contracts import STAGE_ORDER, PLAN_MANIFEST_SHA256, PLAN_AUDIT_SHA256

WORK=Path('/home/zwan0726/work')
REPO=WORK/'projects/BIDCell_boundary_three_20261015'
OUTPUT=WORK/'outputs/bidcell_rep1_triple_simple_boundary_20261017'
BRANCH='feature/rep1-triple-simple-boundary-20261017'
PYTHON='/home/zwan0726/miniforge3/envs/bidcell_local/bin/python'
RSCRIPT='/home/zwan0726/work/envs/bidcell_egr_r/bin/Rscript'
R_LIBRARY=WORK/'outputs/bidcell_historical_dual_20261013/sources/R_library'
COMPUTE=WORK/'outputs/cos_gpu.lock'
PLAN=WORK/'docs/BIDCell_Rep1_Triple_Simple_PlanManifest_20261017.json'
INSPECTED=WORK/'docs/BIDCell_Rep1_Triple_Simple_Inputs_20261017.json'
AUDIT=WORK/'docs/BIDCell_Rep1_Triple_Simple_ChangePlan_Audit_20261017.json'
SCIENCE_AUDIT=WORK/'docs/BIDCell_Rep1_Triple_Simple_ScienceCodeAudit_20261017.json'
CONTROLLER_AUDIT=WORK/'docs/BIDCell_Rep1_Triple_Simple_ControllerCodeAudit_20261017.json'
PROMPT=WORK/'docs/BIDCell_Rep1_Triple_Simple_Boundary_LaunchPrompt_20261017.txt'
PROMPT_SHA='07052b65e43dfcbe23ee7416db2e1ccc47dce149ef2c7fd1d3c505e81bf5e4b8'
AUDITOR='/root/independent_code_auditor'
RECOVERY_AUDITOR='/root/independent_code_auditor'
RECOVERY_AUDIT=WORK/'docs/BIDCell_Rep1_Triple_Simple_Recovery01_CodeAudit_20261017.json'
RECOVERY_RESOURCE=WORK/'docs/BIDCell_Rep1_Triple_Simple_Recovery01_ResourceTimeRevision_20261017.json'
RECOVERY2_AUDIT=WORK/'docs/BIDCell_Rep1_Triple_Simple_Recovery02_CodeAudit_20261018.json'
RECOVERY2_RESOURCE=WORK/'docs/BIDCell_Rep1_Triple_Simple_Recovery02_ResourceTimeRevision_20261018.json'
RECOVERY3_AUDIT=WORK/'docs/BIDCell_Rep1_Triple_Simple_Recovery03_CodeAudit_20261018.json'
RECOVERY3_RESOURCE=WORK/'docs/BIDCell_Rep1_Triple_Simple_Recovery03_ResourceTimeRevision_20261018.json'
RECOVERY3_AUDITOR='/root/recovery03_auditor'
RECOVERY4_AUDIT=WORK/'docs/BIDCell_Rep1_Triple_Simple_Recovery04_CodeAudit_20261018.json'
RECOVERY4_RESOURCE=WORK/'docs/BIDCell_Rep1_Triple_Simple_Recovery04_ResourceTimeRevision_20261018.json'
RESOURCE_AUTH=WORK/'docs/BIDCell_Rep1_Triple_Simple_Recovery04_UserResourceAuthorization_20261018.json'
RESOURCE_PLAN=WORK/'docs/BIDCell_Rep1_Triple_Simple_Recovery04_ChangePlan_20261018.md'
RESOURCE_PLAN_AUDIT=WORK/'docs/BIDCell_Rep1_Triple_Simple_Recovery04_ChangePlan_Audit_20261018.json'
RESOURCE_BASE_FINAL=WORK/'docs/BIDCell_Rep1_Triple_Simple_Recovery03_FinalCodeAudit_20261018.json'
RESOURCE_START_BYTES=29527900160
RESOURCE_BASE_COMMIT='532ff03d620b2584615d70cec5e93c163f1c0b05'
RESOURCE_FIXED={
    'user_authorization':(RESOURCE_AUTH,'0224e2900af4b5ec168989cf295fa1f5ccf9f2c7f156c6e2d4fc2fa4e5592672'),
    'plan':(RESOURCE_PLAN,'0cb5afca6ed61196c236c70f2861fa5534d7ddcac3c0a20892946a6842a34225'),
    'plan_audit':(RESOURCE_PLAN_AUDIT,'077d8b14275db42cbec9813759989d51f0f409474bebe8027991c0b9770a3a9c'),
    'base_code_audit':(RECOVERY3_AUDIT,'6b6fe6c1060887946e8bfa7a967b1a197eb750150e568110acc79431eb4c0d4d'),
    'base_final_audit':(RESOURCE_BASE_FINAL,'e86d4f0c37acda50ec283b248ae2c368e40963607187043a8af0c086fec9ed8a'),
    'resource_proposal':(RECOVERY4_RESOURCE,'508dedf21a0e2fef9f46d01b794be19b63698b08d7a3e446ef9739cf5b200701')}

def resource_amended(cfg):return 'ram_start_required_bytes' in cfg.get('resources',{})

def resource_amendment_evidence():
    evidence={}
    for key,(path,sha) in RESOURCE_FIXED.items():
        check_identity(dict(path=str(path),sha256=sha));evidence[key]=identity(path)
    auth=read(RESOURCE_AUTH);plan_audit=read(RESOURCE_PLAN_AUDIT);proposal=read(RECOVERY4_RESOURCE)
    if auth.get('ram_start_required_bytes')!=RESOURCE_START_BYTES or auth.get('independent_base_commit')!=RESOURCE_BASE_COMMIT or auth.get('latest_instruction_overrides_new_independent_review_requirement') is not True:
        raise PermissionError('startup RAM human authorization differs')
    if plan_audit.get('decision')!='PASS_FOR_RESOURCE_AMENDMENT_IMPLEMENTATION' or plan_audit.get('open_blockers')!=[] or plan_audit.get('reviewer')!='/root' or plan_audit.get('plan')!=evidence['plan'] or plan_audit.get('user_authorization')!=evidence['user_authorization']:
        raise PermissionError('startup RAM amendment plan approval differs')
    for key,auth_key,decision in (('base_code_audit','independent_base_CodeAudit','PASS_FOR_PREPARATION'),('base_final_audit','independent_base_FinalCodeAudit','PASS_FOR_LAUNCH')):
        base=read(evidence[key]['path'])
        if auth.get(auth_key)!=evidence[key] or base.get('decision')!=decision or base.get('source_commit')!=RESOURCE_BASE_COMMIT or base.get('open_blockers')!=[] or base.get('reviewer')!=reviewer({'continuation_request':{'p2_validation':True}}):
            raise PermissionError('startup RAM independent base approval differs')
    if proposal.get('plan')!=evidence['plan'] or proposal.get('plan_audit')!=evidence['plan_audit'] or proposal.get('user_resource_authorization')!=evidence['user_authorization'] or proposal.get('RAM_start_required_bytes')!=RESOURCE_START_BYTES:
        raise PermissionError('startup RAM resource proposal differs')
    check_identity(auth['base_actual_validation_failure'])
    return evidence

def verify_resource_amendment(cfg):
    if not resource_amended(cfg):
        if 'resource_amendment' in cfg:raise PermissionError('resource authorization without amended threshold')
        return None
    r=cfg['resources']
    if type(r['ram_start_required_bytes']) is not int or r['ram_start_required_bytes']!=RESOURCE_START_BYTES or r.get('ram_allowance_bytes')!=22*1024**3 or r.get('ram_reserve_bytes')!=6*1024**3 or r.get('reserve_bytes')!=14*1024**3:
        raise ValueError('unauthorized startup RAM threshold or runtime reserve')
    evidence=resource_amendment_evidence()
    if recovery_generation(cfg)!=3 or cfg.get('resource_amendment')!=evidence or cfg.get('resource_proposal')!=evidence['resource_proposal']:
        raise PermissionError('startup RAM amendment evidence/owner scope differs')
    return evidence

def recovery_generation(cfg):
    if 'p2_validation' in cfg.get('continuation_request',{}):return 3
    if 'replay_validation' in cfg.get('continuation_request',{}):return 2
    if cfg.get('continuation'):
        from .continuation import _bound_json
        schema=_bound_json(cfg['continuation']).get('schema')
        if schema=='rep1-triple-simple-continuation-v3':return 3
        if schema=='rep1-triple-simple-continuation-v2':return 2
    return 1 if cfg.get('continuation') or cfg.get('continuation_request') else 0

def audit_path(generation,amended=False):
    if amended:
        if generation!=3:raise PermissionError('startup RAM amendment requires generation3')
        return RECOVERY4_AUDIT
    return RECOVERY3_AUDIT if generation==3 else RECOVERY2_AUDIT if generation==2 else RECOVERY_AUDIT if generation else CONTROLLER_AUDIT

def reviewer(cfg=None):
    if cfg and resource_amended(cfg):
        return dict(agent_id='/root',role='user_authorized_resource_amendment_reviewer',implementation_author=False,independent_review=False)
    return dict(agent_id=RECOVERY3_AUDITOR if cfg and recovery_generation(cfg)==3 else RECOVERY_AUDITOR if cfg and (cfg.get('continuation') or cfg.get('continuation_request')) else AUDITOR,
                model='gpt-6-astra',reasoning_effort='xhigh',role='independent_code_auditor',implementation_author=False)

def git(*args):
    return subprocess.check_output(['git','-c','core.fsmonitor=false',*args],cwd=REPO,text=True).strip()

def source_files():
    from .inputs import source_dependencies
    rows={r['snapshot_relative']:r for r in source_dependencies()}
    for name in git('ls-files').splitlines():
        p=REPO/name
        if name.startswith('data/') or p.suffix.lower() in ('.png','.jpg','.jpeg','.tif','.tiff','.pdf'):continue
        rows[name]=dict(**identity(p),snapshot_relative=name)
    return [rows[k] for k in sorted(rows)]

def copy_snapshot(root,commit):
    root=Path(root);target=root/'source_snapshot';stage=root/'source_snapshot.staging'
    if target.exists() or stage.exists():raise FileExistsError('preserve existing source copy/staging; use a new reviewed run')
    stage.mkdir();rows=[]
    for r in source_files():
        rel=Path(r['snapshot_relative']);origin=Path(r['path'])
        if rel.is_absolute() or '..' in rel.parts or origin.is_symlink():raise ValueError('unsafe source path')
        check_identity(r);p=stage/rel;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(origin,p)
        if p.stat().st_nlink!=1 or (p.stat().st_dev,p.stat().st_ino)==(origin.stat().st_dev,origin.stat().st_ino) or sha256(p)!=r['sha256']:raise ValueError('source is not an independent identical copy')
        with p.open('rb') as f:os.fsync(f.fileno())
        p.chmod(0o555 if p.suffix=='.sh' else 0o444)
        rows.append(dict(relative=str(rel),original=str(origin),snapshot_path=str(target/rel),bytes=r['bytes'],sha256=r['sha256']))
    for p in sorted(stage.rglob('*'),key=lambda p:len(p.parts),reverse=True):
        if p.is_dir():p.chmod(0o555)
    stage.chmod(0o555);os.replace(stage,target)
    obj=dict(schema='rep1-triple-simple-source-copy-v1',source_commit=commit,snapshot_root=str(target),files=rows,source_digest=digest(rows))
    atomic_json(root/'source_manifest.json',obj);return obj

def module_origins(root):
    root=Path(root).resolve();result={}
    for name,mod in tuple(sys.modules.items()):
        if name not in ('scripts','bidcell') and not name.startswith(('scripts.','bidcell.')):continue
        file=getattr(mod,'__file__',None)
        if file:
            p=Path(file).resolve()
            if root not in p.parents:raise ImportError('project module escaped snapshot: '+name)
            result[name]=str(p)
        for loc in getattr(mod,'__path__',[]):
            if root not in Path(loc).resolve().parents:raise ImportError('namespace escaped snapshot: '+name)
    return result

def verify_snapshot(cfg,location=True):
    m=read(Path(cfg['run_root'])/'source_manifest.json');root=Path(cfg['source_snapshot']).resolve(strict=True)
    if str(root)!=m['snapshot_root'] or m['source_commit']!=cfg['source_commit'] or digest(m['files'])!=cfg['source_digest'] or m['source_digest']!=cfg['source_digest']:raise ValueError('source manifest binding drift')
    expected={r['relative'] for r in m['files']};actual={str(p.relative_to(root)) for p in root.rglob('*') if p.is_file() or p.is_symlink()}
    if actual!=expected:raise ValueError('source file inventory drift')
    for p in (root,*root.rglob('*')):
        if p.is_symlink() or p.stat().st_mode&0o222:raise ValueError('source snapshot linked/writable: '+str(p))
    for r in m['files']:
        p=root/r['relative']
        if r['snapshot_path']!=str(p) or p.stat().st_nlink!=1:raise ValueError('source copy location/link drift')
        check_identity(dict(path=str(p),bytes=r['bytes'],sha256=r['sha256']))
    if location and Path.cwd()!=root:raise ValueError('execution cwd must be frozen snapshot')
    return module_origins(root) if location else {}

class SnapshotFinder:
    def __init__(self,root,rows):self.root=Path(root);self.allowed={r['relative']:r for r in rows}
    def find_spec(self,fullname,path=None,target=None):
        if fullname not in ('scripts','bidcell') and not fullname.startswith(('scripts.','bidcell.')):return None
        spec=importlib.machinery.PathFinder.find_spec(fullname,path)
        if spec is None:raise ImportError('missing frozen dependency: '+fullname)
        if spec.origin:
            p=Path(spec.origin).resolve()
            if self.root not in p.parents:raise ImportError('live fallback forbidden: '+fullname)
            r=self.allowed.get(str(p.relative_to(self.root)))
            if r is None or p.is_symlink() or sha256(p)!=r['sha256']:raise ImportError('source content drift: '+fullname)
        for loc in spec.submodule_search_locations or []:
            if self.root not in Path(loc).resolve().parents:raise ImportError('live namespace forbidden')
        return spec

def install_guard(root):
    root=Path(root).resolve();m=read(root.parent/'source_manifest.json')
    if root!=Path(m['snapshot_root']):raise ValueError('bootstrap source location drift')
    if not any(isinstance(f,SnapshotFinder) for f in sys.meta_path):sys.meta_path.insert(0,SnapshotFinder(root,m['files']))
    module_origins(root)

def controlled_environment(cfg):
    root=Path(cfg['run_root'])
    return dict(PATH=str(Path(PYTHON).parent)+':'+str(Path(RSCRIPT).parent)+':/usr/bin:/bin',
        LD_LIBRARY_PATH='',LD_PRELOAD='',PYTHONPATH='',PYTHONHOME='',PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',PYTHONHASHSEED='0',
        LC_ALL='C.UTF-8',TZ='UTC',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',MPLBACKEND='Agg',
        MPLCONFIGDIR=str(root/'cache/matplotlib'),XDG_CACHE_HOME=str(root/'cache'),NUMBA_CACHE_DIR=str(root/'cache/numba'),TMPDIR=str(root/'tmp'),
        R_HOME=str(Path(RSCRIPT).parent.parent/'lib/R'),R_LIBS='',R_LIBS_USER=str(R_LIBRARY),R_LIBS_SITE='',
        R_PROFILE='/dev/null',R_ENVIRON='/dev/null',R_PROFILE_USER='/dev/null',R_ENVIRON_USER='/dev/null')

def child_environment(cfg):return dict(os.environ,**controlled_environment(cfg))

def preparation_audit(commit,records=None,recovery=False,amended=False):
    path=audit_path(recovery,amended);a=read(path)
    expected_reviewer=reviewer({'resources':{'ram_start_required_bytes':RESOURCE_START_BYTES}} if amended else {'continuation_request':{'p2_validation':True}} if recovery==3 else {'continuation_request':{}} if recovery else None)
    if a.get('decision')!='PASS_FOR_PREPARATION' or a.get('open_blockers')!=[] or a.get('reviewer')!=expected_reviewer or a.get('source_commit')!=commit:raise PermissionError('required preparation audit missing/mismatched')
    if amended and a.get('resource_amendment')!=resource_amendment_evidence():raise PermissionError('preparation audit lacks startup RAM authorization')
    if a.get('source_files_digest')!=digest(source_files() if records is None else records):raise PermissionError('preparation source inventory differs from code audit')
    for item in a.get('evidence',[]):check_identity(item)
    return identity(path)

def verify_preparation(partial,params):
    root=Path(partial['run_root']);params=Path(params)
    amended=resource_amended(partial);verify_resource_amendment(partial)
    if partial.get('schema')!='rep1-triple-simple-preparation-v1' or partial.get('resources')!=__import__(__package__+'.runtime',fromlist=['resource_contract']).resource_contract(amended):raise ValueError('preparation contract drift')
    if params!=root/'control/preparation.json' or Path(partial['source_snapshot'])!=root/'source_snapshot':raise ValueError('preparation path mismatch')
    binding=read(root/'control/preparation_binding.json');check_identity(binding['params']);check_identity(binding['source_manifest']);check_identity(partial['preparation_audit']);check_identity(partial['resource_proposal'])
    recovery=recovery_generation(partial)
    if binding['params']['path']!=str(params) or binding['source_manifest']['path']!=str(root/'source_manifest.json') or partial['preparation_audit']['path']!=str(audit_path(recovery,amended)):raise ValueError('preparation identity location drift')
    m=read(root/'source_manifest.json');rows=[dict(path=r['original'],bytes=r['bytes'],sha256=r['sha256'],snapshot_relative=r['relative']) for r in m['files']]
    if preparation_audit(partial['source_commit'],rows,recovery,amended)!=partial['preparation_audit']:raise ValueError('preparation audit changed')
    req=partial.get('continuation_request',{})
    if req and set(req)!=({'p2_validation','p2_baseline'} if recovery==3 else {'replay_validation','replay_baseline'} if recovery==2 else {'donor_validation','split_validation','baseline'}):raise ValueError('unknown continuation preparation request')
    for item in req.values():check_identity(item)
    verify_snapshot(partial)

def environment(root,lock_fds=(),cfg=None):
    import cv2,numpy,scipy,pandas,tifffile,skimage,h5py,matplotlib
    modules=(cv2,numpy,scipy,pandas,tifffile,skimage,h5py,matplotlib)
    packages=[];files=set()
    for d in importlib.metadata.distributions():
        packages.append(dict(name=d.metadata['Name'],version=d.version,path=str(d.locate_file(''))))
        for name in ('METADATA','RECORD','PKG-INFO'):
            p=Path(d._path)/name
            if p.is_file():files.add(p.resolve())
    for module in (cv2,numpy,scipy):
        folder=Path(module.__file__).parent
        patterns=('*.so',) if module is cv2 else ('**/*multiarray*.so',) if module is numpy else ('**/*ckdtree*.so','**/*_nd_image*.so')
        for pattern in patterns:files.update(p.resolve() for p in folder.glob(pattern) if p.is_file())
    prefixes={Path(PYTHON).parent.parent,Path(RSCRIPT).parent.parent}
    files.update({Path(PYTHON).resolve(),Path(RSCRIPT).resolve(),Path(RSCRIPT).parent.parent/'lib/R/bin/exec/R',Path(RSCRIPT).parent.parent/'lib/R/lib/libR.so'})
    for prefix in prefixes:
        files.update((prefix/'conda-meta').glob('*.json'))
        for pattern in ('libopenblas*.so*','libblas*.so*','liblapack*.so*','libstdc++*.so*','libgcc_s*.so*'):
            files.update(p.resolve() for p in (prefix/'lib').glob(pattern) if p.is_file())
    cfg=cfg or dict(run_root=str(Path(root).parent));script=Path(root)/'scripts/rep1_triple_simple_boundary/environment_probe.R'
    result=subprocess.run([RSCRIPT,'--vanilla',str(script)],cwd=root,env=child_environment(cfg),stdin=subprocess.DEVNULL,capture_output=True,text=True,check=True,pass_fds=tuple(lock_fds),timeout=45)
    R=json.loads(result.stdout)
    libraries=R['libraries'] if isinstance(R['libraries'],list) else [R['libraries']]
    directories={Path(p).resolve() for p in libraries}|{Path(sysconfig.get_paths()['purelib']).resolve()}
    for r in R['packages']:
        package=Path(r['library'])/r['package']
        directories.add(package.resolve())
        for p in package.rglob('*'):
            relative=p.relative_to(package)
            if not any(x in ('doc','help','html','tests','demo') for x in relative.parts):
                if p.is_file():files.add(p.resolve())
                elif p.is_dir():directories.add(p.resolve())
    return dict(schema='rep1-triple-simple-environment-v1',python=PYTHON,python_version=sys.version,rscript=RSCRIPT,
        packages=sorted(packages,key=lambda x:(x['name'].lower(),x['path'])),R=R,
        numerical_modules={m.__name__:dict(version=m.__version__,path=m.__file__) for m in modules},
        files=[identity(p) for p in sorted(files) if p.is_file()],directories=sorted(map(str,directories)),probe_sha256=sha256(script),controlled=controlled_environment(cfg))

def fingerprints(paths):
    out={}
    for path in sorted(set(map(str,paths))):
        p=Path(path);s=p.stat()
        if p.is_symlink() or not (p.is_file() or p.is_dir()):raise ValueError('fingerprint must be a regular path: '+path)
        out[path]=[s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns,s.st_mode,s.st_nlink]
    return out

def verify_contracts(cfg):
    verify_resource_amendment(cfg)
    check_identity(dict(path=str(PLAN),sha256=PLAN_MANIFEST_SHA256));check_identity(dict(path=str(AUDIT),sha256=PLAN_AUDIT_SHA256));check_identity(dict(path=str(PROMPT),sha256=PROMPT_SHA))
    if cfg['plan_manifest']!=str(PLAN) or cfg['plan_digest']!=PLAN_MANIFEST_SHA256:raise ValueError('plan binding mismatch')
    for item in read(PLAN)['files']:check_identity(item)
    s=read(check_identity(cfg['science_audit']))
    if s.get('decision')!='PASS_FOR_CONTROLLER_IMPLEMENTATION' or s.get('open_scientific_blockers')!=[]:raise ValueError('science audit not passed')
    if cfg.get('continuation'):
        from .continuation import approved_repair,verify
        approved_repair(cfg);verify(cfg,False)
        if cfg['preparation_audit']['path']!=str(audit_path(recovery_generation(cfg),resource_amended(cfg))):raise ValueError('recovery requires separately bound code audit')

def immutable_paths(cfg):
    root=Path(cfg['run_root']);m=read(root/'source_manifest.json');inputs=read(cfg['input_manifest']);env=read(root/'environment.json');domain=read(cfg['domain_manifest'])
    paths=[root/'configs/run.json',root/'source_manifest.json',root/'environment.json',cfg['input_manifest'],cfg['domain_manifest'],root/'configs/resource_budget.json',root/'configs/comparisons.json',root/'admission/domain_cardinalities.json',root/'control/preparation.json',root/'control/preparation_binding.json',cfg['preparation_audit']['path'],cfg['resource_proposal']['path'],PLAN,AUDIT,PROMPT,cfg['science_audit']['path']]
    paths += [r['snapshot_path'] for r in m['files']]+[r['path'] for r in inputs['files'].values()]+[r['path'] for r in env['files']]+[r['path'] for r in domain['outputs'].values()]
    paths += [Path(cfg['source_snapshot'])]+[p for p in Path(cfg['source_snapshot']).rglob('*') if p.is_dir()]+env['directories']
    if resource_amended(cfg):
        paths += [r['path'] for r in verify_resource_amendment(cfg).values()]
        paths += [read(RESOURCE_AUTH)['base_actual_validation_failure']['path']]
    if cfg.get('continuation'):
        from .continuation import manifest,fingerprint_paths,evidence_items
        m=manifest(cfg);paths += [cfg['continuation']['path']]+list(fingerprint_paths(m))
        paths += [item['path'] for item in evidence_items(m)]
    return paths+[r['path'] for r in read(PLAN)['files']]

def make_preflight(cfg,fds):
    from .inputs import load_inputs
    from .frozen_domains import validate_frozen
    verify_snapshot(cfg);verify_contracts(cfg);load_inputs(cfg);validate_frozen(cfg)
    actual=environment(cfg['source_snapshot'],fds,cfg)
    if digest(actual)!=cfg['environment_digest'] or actual!=read(Path(cfg['run_root'])/'environment.json'):raise ValueError('environment changed during preparation')
    obj=dict(schema='rep1-triple-simple-preflight-v1',config_digest=digest(cfg),source_digest=cfg['source_digest'],input_digest=cfg['input_digest'],domain_digest=cfg['domain_digest'],environment_digest=cfg['environment_digest'],fingerprints=fingerprints(immutable_paths(cfg)),full_content_checks=True)
    target=Path(cfg['run_root'])/'admission/preflight.json'
    if target.exists():
        if read(target)!=obj:raise ValueError('committed preparation preflight drift; preserve it')
    else:atomic_json(target,obj)
    return obj

def verify_admission(cfg):
    root=Path(cfg['run_root']);p=root/'admission/final.json';a=read(p)
    if a.get('status')!='PASS_FOR_LAUNCH' or a.get('approved_for_formal_launch') is not True or a.get('open_blockers')!=[]:raise PermissionError('final launch admission missing')
    if a.get('reviewer')!=reviewer(cfg):raise PermissionError('wrong final reviewer/auditor')
    amendment=verify_resource_amendment(cfg)
    if amendment is not None and a.get('resource_amendment')!=amendment:raise PermissionError('final admission lacks startup RAM authorization')
    for k,v in dict(source_commit=cfg['source_commit'],source_digest=cfg['source_digest'],config_digest=digest(cfg),input_digest=cfg['input_digest'],domain_digest=cfg['domain_digest'],plan_digest=cfg['plan_digest'],environment_digest=cfg['environment_digest'],source_snapshot=cfg['source_snapshot'],stage_order=list(STAGE_ORDER)).items():
        if a.get(k)!=v:raise PermissionError('final admission binding differs: '+k)
    for key,path in dict(source_manifest=root/'source_manifest.json',environment=root/'environment.json',preflight=root/'admission/preflight.json',config=root/'configs/run.json',input_manifest=Path(cfg['input_manifest']),domain_manifest=Path(cfg['domain_manifest']),resource_budget=root/'configs/resource_budget.json',domain_cardinalities=root/'admission/domain_cardinalities.json').items():
        check_identity(a[key])
        if a[key]['path']!=str(path):raise PermissionError('admission evidence location mismatch: '+key)
    if not a.get('evidence'):raise PermissionError('admission lacks fixture evidence')
    for item in a['evidence']:check_identity(item)
    if cfg.get('continuation'):
        from .continuation import verify
        m=verify(cfg)
        if a.get('continuation')!=cfg['continuation'] or a.get('continuation_source_compatibility')!=m['source_compatibility'] or a.get('continuation_fingerprints')!=m['fingerprints']:
            raise PermissionError('final admission lacks exact continuation bindings')
        if m.get('generation2'):
            expected={k:m['generation2'][k] for k in ('base_continuation','donor2_config','donor2_source_manifest','donor2_admission','donor2_validation','donor2_baseline','base_baseline')}
            if a.get('continuation_generation2')!=expected:raise PermissionError('final admission lacks exact two-donor lineage')
        if m.get('generation3'):
            from .continuation3 import ADMISSION_KEYS
            expected={k:m['generation3'][k] for k in ADMISSION_KEYS}
            if a.get('continuation_generation3')!=expected:raise PermissionError('final admission lacks exact third-producer lineage')
    return identity(p)

class ExecutionValidator:
    """An independently bound full-content preflight plus mutation fingerprints."""
    def __init__(self,cfg,fds):
        self.cfg=cfg;self.fds=fds;self.admission=verify_admission(cfg);self.proof=read(Path(cfg['run_root'])/'admission/preflight.json')
        if self.proof.get('config_digest')!=digest(cfg) or self.proof.get('full_content_checks') is not True:raise ValueError('preflight binding absent')
        verify_snapshot(cfg);verify_contracts(cfg)
        self.expected=self.proof['fingerprints'];self.extra=fingerprints([self.admission['path'],Path(cfg['run_root'])/'admission/preflight.json'])
        self()
    def __call__(self):
        from .runtime import check_fds
        check_fds(self.cfg,self.fds)
        if fingerprints(self.expected)!=self.expected or fingerprints(self.extra)!=self.extra:raise ValueError('frozen source/input/domain/config/environment fingerprint drift')
        module_origins(self.cfg['source_snapshot'])
        return self.admission
