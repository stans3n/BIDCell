"""Run identities, exact states and stage-specific scientific producers."""
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import imgaug as ia
import numpy as np
import torch
from scripts.crgf.common import *
from scripts.new_methods.common import BaseSix, native_config, seed_all, json_safe

REPO = Path(__file__).resolve().parents[2]
INPUT = Path('/home/zwan0726/work/outputs/bidcell_new_methods_20261013')
HISTORICAL_SOURCE = Path('/home/zwan0726/work/projects/BIDCell_new_methods_20261013')
DEFAULT_ROOT = Path('/home/zwan0726/work/outputs/bidcell_v2_paired_20261014')
CONTRACT = Path('/home/zwan0726/work/outputs/bidcell_v2_fuse_decision_20261014/experiment_contract.json')
AUDIT = CONTRACT.with_name('historical_pairing_audit.json')
AUTHORITY = Path('/home/zwan0726/work/outputs/bidcell_historical_dual_20261013/protocol.json')
VERSION = 'V2-PAIRED-20261014-v1.1'
RUNS = {'BASE_MATCH': 'BASE', 'V2_INTERACT': 'V2_INTERACT'}
MORPHOLOGY_INITIAL_SHA = '64c4e90ce5e14c948ce7bd3f523fb3b50f9f1734c322625826a7c3f99bcf4957'


def read(path):
    return json.loads(Path(path).read_text())


def tensor_sha(state):
    h = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        t = tensor.detach().cpu().contiguous()
        h.update(json.dumps([name, str(t.dtype), list(t.shape)]).encode())
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def state_equal(a, b):
    if torch.is_tensor(a):
        return torch.is_tensor(b) and a.dtype == b.dtype and torch.equal(a.cpu(), b.cpu())
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and a.dtype == b.dtype and np.array_equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(state_equal(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return type(a) is type(b) and len(a) == len(b) and all(state_equal(x, y) for x, y in zip(a, b))
    return a == b


def rng_state():
    # imgaug lazily seeds its global generator from NumPy. Materialize it before
    # capturing NumPy so this snapshot describes one consistent instant.
    imgaug_state = copy.deepcopy(ia.random.get_global_rng().state)
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                imgaug=imgaug_state)


def restore_rng(state):
    random.setstate(state['python']); np.random.set_state(state['numpy']); torch.set_rng_state(state['torch'])
    if state['cuda']:
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA state cannot be restored without the admitted device')
        torch.cuda.set_rng_state_all(state['cuda'])
    ia.random.get_global_rng().state = copy.deepcopy(state['imgaug'])


def make_model(kind, device='cuda'):
    from scripts.new_methods.common import make_model as legacy
    from bidcell.model.model.interaction import CandidateInteraction
    if kind not in ('BASE', 'V2_MORPH', 'V2_INTERACT'):
        raise ValueError(kind)
    model = legacy('V2_MORPH' if kind == 'V2_INTERACT' else kind, INPUT/'prepared/shared_initial.pth', 'cpu')
    if kind == 'V2_INTERACT':
        before = rng_state()
        with torch.random.fork_rng():
            torch.manual_seed(20261014)
            model.interaction = CandidateInteraction()
        if not state_equal(before, rng_state()):
            raise ValueError('interaction construction advanced global RNG')
    return model.to(device)


def optimizer(model):
    return torch.optim.Adam(model.parameters(), lr=1e-5, betas=(.9, .999), eps=1e-8, weight_decay=1e-4)


def forward(model, batch, kind):
    return model(batch['input'], batch['aux'] if kind in ('V2_MORPH', 'V2_INTERACT') else None)


def producer(stage):
    # Do not include orchestration/report sources in the scientific producer.
    # The final independent review additionally binds the entire source tree.
    native = [str(p.relative_to(REPO)) for p in (REPO/'bidcell').rglob('*.py')]
    common = ['scripts/paired_v2/common.py', 'scripts/crgf/common.py', 'scripts/new_methods/common.py']
    core = common + native + ['scripts/paired_v2/data.py', 'scripts/new_methods/data.py']
    if stage in ('initial', 'train'):
        paths = core + ['scripts/paired_v2/training.py', 'scripts/paired_v2/diagnostics.py']
    elif stage == 'predict':
        paths = core + ['scripts/paired_v2/prediction.py', 'scripts/new_methods/prediction.py']
    elif stage == 'postprocess':
        paths = common + ['scripts/paired_v2/stages.py', 'scripts/crgf/postprocess.py', 'bidcell/model/postprocess_predictions.py']
    elif stage == 'matrix':
        paths = common + ['scripts/paired_v2/stages.py', 'scripts/crgf/matrix.py']
    elif stage == 'evaluate':
        paths = common + ['scripts/paired_v2/evaluation.py', 'scripts/new_methods/accepted_profiles.py',
                          'scripts/new_methods/evaluation.py', 'scripts/new_methods/author_marker.py']
        paths += [str(p.relative_to(REPO)) for p in (REPO/'scripts/crgf').glob('*') if p.suffix in ('.py', '.R')]
    else:
        raise ValueError(stage)
    return {p: sha256(REPO/p) for p in sorted(set(paths))}


def file_tree(root):
    root = Path(root)
    return {str(p.relative_to(root)): identity(p) for p in sorted(root.rglob('*'))
            if p.is_file() and p.name not in ('complete.json', 'run.lock') and not p.name.endswith('.tmp')}


def config_path(run_id, root=DEFAULT_ROOT):
    if run_id not in RUNS:
        raise ValueError(run_id)
    return Path(root)/run_id/'configs/run.json'


def load_config(config):
    cfg = read(config) if isinstance(config, (str, Path)) else dict(config)
    if cfg['run_id'] not in RUNS or cfg['model_kind'] != RUNS[cfg['run_id']]:
        raise ValueError('run_id/model_kind mismatch')
    if Path(cfg['input_root']).resolve() != INPUT or INPUT in Path(cfg['output_root']).resolve().parents:
        raise ValueError('immutable input/output root separation violated')
    if cfg['updates'] != 3625 or cfg['visits'] != 3998 or cfg['empty_skips'] != 373:
        raise ValueError('paired training budget drift')
    for key, filename in (('contract', 'experiment_contract.json'), ('authoritative_protocol', 'authoritative_protocol.json')):
        frozen = Path(cfg['runroot'])/'prepared'/filename
        if cfg[key] != identity(frozen):
            raise ValueError('frozen experiment/evaluation contract drift')
    return cfg


def check_environment(require_cuda=True):
    import platform
    import scipy
    import cv2
    if require_cuda and (not torch.cuda.is_available() or platform.node() != 'MCV-CGB4'):
        raise RuntimeError('admission/formal compute requires same-host MCV-CGB4 CUDA')
    seed_all(1234)
    return dict(host=platform.node(), python=platform.python_version(), torch=torch.__version__,
                numpy=np.__version__, scipy=scipy.__version__, opencv=cv2.__version__, imgaug=ia.__version__,
                cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                total_vram=torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else None,
                threads=torch.get_num_threads(), interop_threads=torch.get_num_interop_threads(),
                tf32=False, cudnn_benchmark=False, cudnn_deterministic=True,
                environment={k: os.environ.get(k) for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')})
