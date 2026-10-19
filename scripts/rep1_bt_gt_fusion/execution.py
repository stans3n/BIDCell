"""Committed source copies, package identities and fail-closed admission checks."""
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
from . import NODE_ORDER
from .common import atomic_json, check_identity, digest, identity, read, sha256, within

WORK = Path('/home/zwan0726/work')
REPO = WORK/'projects/BIDCell_boundary_three_20261015'
OUTPUT = WORK/'outputs/bidcell_rep1_bt_gt_fusion_20261016'
RUN_ID = 'rep1-bt-gt-fusion-20261016-r02'
OLD_ROOT = OUTPUT/'rep1-bt-gt-fusion-20261016-r01'
IMPORT_SEED_SHA = '6f65f614e2dd5e780226e0a2c311aca7030748f4dc9a998c75c43b9f94174f7b'
AMENDMENT_SHA = '49d11a09cb26a1791c3de58b2bea1636a96f153b28c3147f6dc8132fcf11b859'
PYTHON = '/home/zwan0726/miniforge3/envs/bidcell_local/bin/python'
RSCRIPT = '/home/zwan0726/work/envs/bidcell_egr_r/bin/Rscript'
R_LIBRARY = WORK/'outputs/bidcell_historical_dual_20261013/sources/R_library'
COMPUTE = WORK/'outputs/cos_gpu.lock'
PLAN = WORK/'docs/BIDCell_Rep1_BT_GT_Fusion_ChangePlan_20261016.md'
PROMPT = PLAN.with_name('BIDCell_Rep1_BT_GT_Fusion_LaunchPrompt_20261016.txt')
PLAN_SHA = 'f24c5150a06ced0c4c1ff312a8aeaa5a2b9ed507b160d713030fcba361e671cf'
PROMPT_SHA = '252c517a0f06b7d6b210ee104a9d557e4ffddf4ba25d3399878dfe2d7aec87c8'


def git(*args):
    return subprocess.check_output(['git','-c','core.fsmonitor=false',*args],cwd=REPO,text=True).strip()


def source_files():
    from .inputs import source_dependencies
    rows = {r['snapshot_relative']:r for r in source_dependencies()}
    extra = [REPO/'scripts/rep1_bt_gt_fusion/control.sh', REPO/'scripts/rep1_bt_gt_fusion/README.md',
             REPO/'tests/__init__.py']
    extra += list((REPO/'tests').glob('*rep1_bt_gt*.*'))
    for p in extra:
        if p.is_file():
            relative = str(p.relative_to(REPO))
            rows[relative] = dict(**identity(p),snapshot_relative=relative)
    return [rows[k] for k in sorted(rows)]


def copy_snapshot(run_root, commit):
    root = Path(run_root); target = root/'source_snapshot'
    if target.exists():raise FileExistsError('source snapshot is immutable; use a new reviewed attempt/run')
    stage = root/'source_snapshot.staging'
    if stage.exists():raise FileExistsError('uncommitted snapshot staging exists; preserve and investigate')
    stage.mkdir()
    rows = []
    for source in source_files():
        relative = Path(source['snapshot_relative'])
        if relative.is_absolute() or '..' in relative.parts:raise ValueError('unsafe snapshot relative path')
        origin = Path(source['path'])
        if origin.is_symlink():raise ValueError('source symlink is not accepted')
        check_identity(source)
        dest = stage/relative; dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(origin,dest)
        if dest.stat().st_nlink != 1 or sha256(dest)!=source['sha256']:raise ValueError('snapshot copy differs')
        with dest.open('rb') as stream:os.fsync(stream.fileno())
        dest.chmod(0o555 if dest.suffix=='.sh' else 0o444)
        rows.append(dict(original=source['path'],relative=str(relative),bytes=source['bytes'],sha256=source['sha256'],
                         snapshot_path=str(target/relative),source_commit=commit if REPO in origin.parents else None))
    for p in sorted(stage.rglob('*'),key=lambda p:len(p.parts),reverse=True):
        if p.is_dir():p.chmod(0o555)
    stage.chmod(0o555); os.replace(stage,target)
    result=dict(schema='rep1-bt-gt-source-copy-v1',source_commit=commit,snapshot_root=str(target),
                files=rows,source_digest=digest(rows))
    atomic_json(root/'source_manifest.json',result)
    return result


def verify_snapshot(cfg, require_location=True):
    manifest = read(within(cfg['run_root'],'source_manifest.json'))
    root = Path(cfg['source_snapshot']).resolve(strict=True)
    if root != Path(manifest['snapshot_root']) or manifest['source_commit']!=cfg['source_commit']:
        raise ValueError('source snapshot root/commit mismatch')
    if digest(manifest['files'])!=cfg['source_digest'] or manifest['source_digest']!=cfg['source_digest']:
        raise ValueError('source manifest digest mismatch')
    actual={str(p.relative_to(root)) for p in root.rglob('*') if p.is_file() or p.is_symlink()}
    expected={r['relative'] for r in manifest['files']}
    if actual!=expected:raise ValueError('snapshot file inventory drift')
    if root.is_symlink() or root.stat().st_mode & 0o222:raise ValueError('snapshot directory is writable/linked')
    for p in root.rglob('*'):
        if p.is_symlink() or p.stat().st_mode & 0o222:raise ValueError('snapshot object is writable/linked: '+str(p))
    for row in manifest['files']:
        p = root/row['relative']
        if row['snapshot_path']!=str(p) or p.stat().st_nlink!=1:raise ValueError('snapshot is not an independent copy')
        check_identity(dict(path=str(p),bytes=row['bytes'],sha256=row['sha256']))
    if require_location:
        if Path.cwd()!=root:raise ValueError('execution cwd must equal snapshot')
        origins = module_origins(root)
    else:origins={}
    return dict(snapshot_root=str(root),source_digest=cfg['source_digest'],cwd=str(Path.cwd()),modules=origins)


def module_origins(root):
    origins={}
    for name,mod in tuple(sys.modules.items()):
        if name=='scripts' or name.startswith(('scripts.','bidcell.')) or name=='bidcell':
            file=getattr(mod,'__file__',None)
            if file:
                path=Path(file).resolve()
                if root not in path.parents:raise ValueError('live module origin: '+name+': '+str(path))
                origins[name]=str(path)
            for location in getattr(mod,'__path__',[]):
                if root not in Path(location).resolve().parents:raise ValueError('live namespace path: '+name)
    return origins


class SnapshotFinder:
    """Check delayed project imports before executing them; no live fallback."""
    def __init__(self,cfg):
        self.root=Path(cfg['source_snapshot']);m=read(Path(cfg['run_root'])/'source_manifest.json')
        self.allowed={r['relative']:r for r in m['files']}

    def find_spec(self,fullname,path=None,target=None):
        if not (fullname in ('scripts','bidcell') or fullname.startswith(('scripts.','bidcell.'))):return None
        spec=importlib.machinery.PathFinder.find_spec(fullname,path)
        if spec is None:raise ImportError('missing frozen project dependency: '+fullname)
        if spec.origin:
            p=Path(spec.origin).resolve()
            if self.root not in p.parents:raise ImportError('project import escaped snapshot: '+str(p))
            key=str(p.relative_to(self.root));row=self.allowed.get(key)
            if row is None or p.is_symlink() or sha256(p)!=row['sha256']:raise ImportError('unreviewed project import: '+str(p))
        for location in spec.submodule_search_locations or []:
            if self.root not in Path(location).resolve().parents:raise ImportError('namespace escaped snapshot')
        return spec


def install_import_guard(cfg):
    root=Path(cfg['source_snapshot'])
    allowed=[str(root),sysconfig.get_paths()['purelib']]
    allowed += [p for p in sys.path if p and 'site-packages' not in p and p.startswith(sys.base_prefix+'/lib/')]
    sys.path[:]=list(dict.fromkeys(allowed))
    module_origins(root)
    if not any(isinstance(f,SnapshotFinder) for f in sys.meta_path):sys.meta_path.insert(0,SnapshotFinder(cfg))


def environment(source_root,lock_fds=()):
    """Content identity without timestamps/cwd, stable across live preparation/copy."""
    import cv2,numpy,scipy,pandas,tifffile,skimage,h5py,matplotlib
    modules=[cv2,numpy,scipy,pandas,tifffile,skimage,h5py,matplotlib]
    packages=sorted([dict(name=d.metadata['Name'],version=d.version,path=str(d.locate_file('')))
                     for d in importlib.metadata.distributions()],key=lambda x:(x['name'].lower(),x['path']))
    binary_paths={Path(PYTHON).resolve(),Path(RSCRIPT).resolve(),Path(RSCRIPT).parent.parent/'lib/R/bin/exec/R',
                  Path(RSCRIPT).parent.parent/'lib/R/lib/libR.so'}
    # Identify the numerical extension binaries actually used by this workflow.
    for module in (cv2,numpy,scipy):
        folder=Path(module.__file__).parent
        if module is cv2:binary_paths.update(folder.glob('*.so'))
        elif module is numpy:binary_paths.update((folder/'core').glob('*multiarray*.so'))
        else:binary_paths.update((folder/'spatial').glob('*ckdtree*.so'))
    prefixes={Path(PYTHON).parent.parent,Path(RSCRIPT).parent.parent}
    for prefix in prefixes:
        for pattern in ('libopenblas*.so*','libblas*.so*','liblapack*.so*','libstdc++*.so*','libgcc_s*.so*'):
            binary_paths.update((prefix/'lib').glob(pattern))
    binary_paths={p.resolve() for p in binary_paths if p.is_file()}
    env=dict(os.environ,R_LIBS_USER=str(R_LIBRARY),R_LIBS_SITE='',R_PROFILE_USER='/dev/null',R_ENVIRON_USER='/dev/null',LC_ALL='C.UTF-8')
    script=Path(source_root)/'scripts/rep1_bt_gt_fusion/environment_probe.R'
    result=subprocess.run([RSCRIPT,'--vanilla',str(script)],cwd=source_root,env=env,stdin=subprocess.DEVNULL,
                          capture_output=True,text=True,check=True,pass_fds=tuple(lock_fds),timeout=45)
    return dict(schema='rep1-bt-gt-environment-v1',python=PYTHON,python_version=sys.version,
                python_packages=packages,numerical_modules={m.__name__:dict(version=m.__version__,path=m.__file__) for m in modules},
                binaries=[identity(p) for p in sorted(binary_paths) if p.is_file()],R=json.loads(result.stdout),
                conda_metadata=[identity(p) for prefix in sorted(prefixes) for p in sorted((prefix/'conda-meta').glob('*.json'))],
                entropy_files=[identity(p) for p in sorted(R_LIBRARY.rglob('*')) if p.is_file()],
                probe_sha256=sha256(script))


def verify_environment(cfg,lock_fds=()):
    actual=environment(cfg['source_snapshot'],lock_fds)
    saved=read(within(cfg['run_root'],'environment.json'))
    if digest(actual)!=cfg['environment_digest'] or actual!=saved:raise ValueError('frozen environment drift')
    return cfg['environment_digest']


def verify_contracts(cfg):
    check_identity(dict(path=str(PLAN),sha256=PLAN_SHA));check_identity(dict(path=str(PROMPT),sha256=PROMPT_SHA))
    if cfg['plan_sha256']!=PLAN_SHA or cfg['prompt_sha256']!=PROMPT_SHA:raise ValueError('contract identity drift')
    from .recovery import verify_plan
    verify_plan(cfg)


def verify_admission(cfg):
    p=within(cfg['run_root'],'admission/final.json');r=read(p)
    expected=dict(source_commit=cfg['source_commit'],source_digest=cfg['source_digest'],config_digest=digest(cfg),
                  input_digest=cfg['input_digest'],environment_digest=cfg['environment_digest'],
                  plan_sha256=cfg['plan_sha256'],prompt_sha256=cfg['prompt_sha256'],source_snapshot=cfg['source_snapshot'])
    if r.get('status')!='ADMITTED' or r.get('approved_for_formal_launch') is not True:raise ValueError('final launch admission absent')
    if r.get('reviewer')!=dict(model='gpt-6-astra',reasoning_effort='xhigh',implementation_author=False):
        raise ValueError('wrong independent final auditor')
    if any(r.get(k)!=v for k,v in expected.items()) or r.get('node_order')!=list(NODE_ORDER):raise ValueError('admission binding mismatch')
    if not r.get('evidence'):raise ValueError('admission lacks actual fixture/audit evidence')
    for item in r['evidence']:check_identity(item)
    check_identity(r['source_manifest']);check_identity(r['environment'])
    if r['source_manifest']['path']!=str(Path(cfg['run_root'])/'source_manifest.json') or r['environment']['path']!=str(Path(cfg['run_root'])/'environment.json'):
        raise ValueError('admission refers to another manifest/environment')
    for key,expected_path in (('import_plan',cfg['checkpoint_import_plan']['path']),
                              ('checkpoint_import',str(Path(cfg['run_root'])/'admission/checkpoint_import.json'))):
        check_identity(r[key])
        if r[key]['path']!=expected_path:raise ValueError('recovery admission location mismatch')
    from .recovery import verify_import
    verify_import(cfg)
    return identity(p)
