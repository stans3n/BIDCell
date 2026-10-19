"""Reuse passes only when the actual producer functions and full header match."""
import hashlib
import inspect
from .common import *


def development_producers():
    from .admission import developer_check,equal_state,frozen_capture,compare_capture
    return dict(functions={f.__name__:hashlib.sha256(inspect.getsource(f).encode()).hexdigest()
                          for f in (developer_check,equal_state,frozen_capture,compare_capture)},
                contracts=sha256(REPO/'tests/new_methods/contracts.py'))


def bind_development(path):
    path=Path(path);rec=read(path)
    if rec['source']!=source_digest():raise ValueError('cannot retroactively bind a pass to changed source')
    binding=dict(status='bound_to_exact_pass_source',receipt=identity(path),
                 full_source_at_pass=dict(digest=source_digest(),files=source_manifest()),**development_producers())
    atomic_json(path.with_name('development_binding.json'),binding)


def reusable_development(root,arm,cfg):
    from .training import checkpoint_header
    path=Path(root)/'admission'/arm/'development.json';binding=path.with_name('development_binding.json')
    if not path.exists() or not binding.exists():return False
    rec=read(path);bound=read(binding);producer=development_producers()
    if bound['receipt']!=identity(path):raise ValueError('development evidence changed after binding')
    if any(bound[k]!=producer[k] for k in producer):return False
    if rec['status']!='succeeded' or rec['arm']!=arm or rec['updates']!=(100 if arm=='V2_MORPH' else 64):return False
    if rec['resume']['full_header']!=checkpoint_header(cfg):return False
    if identity(rec['pilot']['path'])!=rec['pilot']:raise ValueError('pilot endpoint evidence changed')
    return True
