"""Unattended serial execution. Scientific decisions remain in the admitted nodes.

The supervisor holds the queue lock and, once per node, the workspace compute
lock. Its subprocess inherits both locks, so a surviving orphan cannot overlap a
replacement supervisor. Execution leases never change scientific run.json.
"""
from __future__ import annotations
import contextlib
import csv
import errno
import fcntl
import importlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import time
import traceback
import uuid
import threading
from datetime import datetime, timezone, timedelta

from .common import (atomic_json, check_identity, digest, identity, load_run,
                     output_path, read, source_manifest, utc)
from .contracts import GPU_LOCK, NODES, MINIMUM_NODES, SOURCE_ROOT, PYTHON, RSCRIPT

SAFETY = 1.25
PACKAGE_SECONDS = 1800
FINAL = {'COMPLETED', 'SKIPPED_SCIENTIFIC_GATE', 'DEFERRED_NOT_ADMITTED', 'DEFERRED_TIME'}
PAUSED = {'PAUSED_USER', 'PAUSED_BUDGET', 'PAUSED_RESOURCE', 'PAUSED_INTERRUPTED'}
MODULES = {n: ('a_core' if n in ('A1_CORE', 'A2_CORE') else
              'p2_transfer' if n.startswith('P2_') else 'morph') for n in NODES}
TRANSIENT_IO = {errno.EIO, errno.ESTALE, errno.ETIMEDOUT}
LOCK_ENV = ('BOUNDARY_THREE_LOCK_FDS','BOUNDARY_THREE_RUN_ROOT','BOUNDARY_THREE_LEASE')


def process_identity(pid=None):
    pid = os.getpid() if pid is None else int(pid)
    text = Path(f'/proc/{pid}/stat').read_text()
    fields = text[text.rfind(')') + 2:].split()
    return dict(host=socket.gethostname(), boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                pid=pid, start_ticks=int(fields[19]), process_state=fields[0])


def process_alive(saved):
    if not saved or saved.get('host') != socket.gethostname():
        return False
    try:
        actual = process_identity(saved['pid'])
        return (actual['process_state'] != 'Z' and
                all(actual[k] == saved[k] for k in ('host', 'boot_id', 'pid', 'start_ticks')))
    except (OSError, KeyError, ValueError):
        return False


@contextlib.contextmanager
def lock_file(path, blocking=False):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        except BlockingIOError as exc:
            raise RuntimeError('lock held: ' + str(path)) from exc
        try:
            yield stream
        finally:
            # Close, rather than LOCK_UN: inherited children retain the flock
            # until their final descriptor closes if the supervisor dies.
            pass


def event(cfg, kind, **values):
    path = output_path(cfg, 'events.jsonl'); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        stream.write(json.dumps(dict(time=utc(), event=kind, **values), allow_nan=False) + '\n')
        stream.flush(); os.fsync(stream.fileno())


def save_state(cfg, state):
    state['updated_utc'] = utc()
    atomic_json(output_path(cfg, 'queue_state.json'), state)


@contextlib.contextmanager
def semantic_lock(cfg):
    """Full mask/matrix validation is workspace compute work, even without GPU."""
    with GPU_LOCK.open('a+') as stream:
        until=time.monotonic()+cfg['resources']['resource_wait_seconds']
        while True:
            try:fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB);break
            except BlockingIOError:
                if time.monotonic()>=until:raise RuntimeError('semantic validation compute-lock wait exhausted')
                time.sleep(.5)
        yield stream


def new_state(cfg):
    return dict(schema='boundary-three-queue-state-v1', run_id=cfg['run_id'],
                config_digest=digest(cfg), source_digest=cfg['source_digest'], input_digest=cfg['input_digest'],
                host=socket.gethostname(), execution_status='PREPARED', current_node=None,
                stage=None, heartbeat_utc=None, last_progress_utc=None, progress=None,
                supervisor=None, worker=None, tmux_session=None, leases=[], active_lease=None,
                nodes={n: dict(execution_status='PREPARED', science_status=None, retries=0) for n in NODES},
                failure=None, log_path=str(output_path(cfg, 'logs', 'queue.log')))


def bound_state(cfg):
    state = read(output_path(cfg, 'queue_state.json'))
    if state['run_id'] != cfg['run_id'] or state['config_digest'] != digest(cfg):
        raise ValueError('queue state does not belong to frozen run')
    return state


def estimates(cfg):
    result = cfg['node_estimates']
    for n in NODES:
        if cfg['enabled_nodes'][n]:
            row = result[n]
            for key in ('seconds', 'retained_bytes', 'atomic_extra_bytes', 'peak_ram_bytes', 'peak_gpu_bytes'):
                if not isinstance(row[key], (int, float)) or not math.isfinite(row[key]) or row[key] < 0:
                    raise ValueError('invalid node resource estimate ' + n + ':' + key)
            if row['seconds'] <= 0 or not row.get('evidence'):
                raise ValueError('complete-chain estimate lacks time/evidence ' + n)
            for obj in row['evidence']:
                check_identity(obj)
    return result


def budget_forecast(cfg, seconds=None):
    rows = estimates(cfg)
    required = sum(rows[n]['seconds'] for n in MINIMUM_NODES) * SAFETY + PACKAGE_SECONDS
    offered = cfg['budget_seconds'] if seconds is None else seconds
    return dict(status='ADMITTED' if offered >= required else 'NOT_ADMITTED_BUDGET',
                budget_seconds=offered, minimum_required_seconds=required,
                deficit_seconds=max(0, required - offered), safety_factor=SAFETY,
                packaging_reserve_seconds=PACKAGE_SECONDS,
                complete_node_seconds={n: rows[n]['seconds'] for n in NODES if cfg['enabled_nodes'][n]})


def disk_used(root):
    # Logical retained bytes are deliberately conservative for sparse files.
    return sum(p.stat().st_size for p in Path(root).rglob('*') if p.is_file() and not p.is_symlink())


def disk_forecast(cfg, state=None, lease_seconds=None):
    rows = cfg['node_estimates']; remaining = []
    window = cfg['budget_seconds'] if lease_seconds is None else lease_seconds
    excluded = {}
    for n in NODES:
        status = state['nodes'][n]['execution_status'] if state else 'PREPARED'
        if cfg['enabled_nodes'][n] and status not in ('COMPLETED', 'SKIPPED_SCIENTIFIC_GATE'):
            required = rows[n]['seconds'] * SAFETY + PACKAGE_SECONDS
            # Do not lower a scientific estimate. A node unable to fit even the
            # entire frozen execution window cannot create artifacts this lease.
            if n not in MINIMUM_NODES and required > window:
                excluded[n] = dict(reason='provably unable to pass complete-node time gate in this lease',
                                   full_required_seconds=required, lease_budget_seconds=window)
            else:
                remaining.append(n)
    retained = {n: max(0, rows[n]['retained_bytes'] - disk_used(output_path(cfg, n))) for n in remaining}
    shared = max(0, cfg['resources'].get('shared_retained_bytes', 0) - disk_used(output_path(cfg, 'coordinate_cache')))
    atomic = max((rows[n]['atomic_extra_bytes'] for n in remaining), default=0)
    required = sum(retained.values()) + shared + atomic + cfg['resources']['disk_reserve_bytes']
    free = shutil.disk_usage(cfg['output_root']).free
    return dict(status='ADMITTED' if free >= required else 'NOT_ADMITTED_DISK', free_bytes=free,
                required_bytes=required, remaining_retained_bytes=retained, shared_remaining_bytes=shared,
                atomic_extra_bytes=atomic, reserve_bytes=cfg['resources']['disk_reserve_bytes'],
                lease_budget_seconds=window, time_ineligible_exclusions=excluded)


def resources_now(cfg):
    mem = {}
    for row in Path('/proc/meminfo').read_text().splitlines():
        key, value = row.split(':', 1); mem[key] = int(value.strip().split()[0]) * 1024
    gpu = subprocess.run(['nvidia-smi', '--query-gpu=memory.free,memory.total', '--format=csv,noheader,nounits'],
                         check=True, text=True, capture_output=True, timeout=15)
    rows = [r.strip().split(',') for r in gpu.stdout.strip().splitlines()]
    if len(rows) != 1:
        raise ValueError('admission requires exactly one physical GPU')
    return dict(time=utc(), ram_available_bytes=mem['MemAvailable'], ram_total_bytes=mem['MemTotal'],
                gpu_free_bytes=int(rows[0][0]) * 1024**2, gpu_total_bytes=int(rows[0][1]) * 1024**2,
                disk_free_bytes=shutil.disk_usage(cfg['output_root']).free)


def resource_reasons(cfg, measured, node=None, state=None, lease_seconds=None):
    r = cfg['resources']; row = cfg['node_estimates'][node] if node else {}
    ram = r['min_ram_available_bytes'] + row.get('peak_ram_bytes', 0)
    gpu = r['min_gpu_free_bytes'] + row.get('peak_gpu_bytes', 0)
    reasons = []
    if measured['ram_available_bytes'] < ram: reasons.append('available RAM below requirement')
    if measured['gpu_free_bytes'] < gpu: reasons.append('available GPU below requirement')
    if node:
        forecast = disk_forecast(cfg, state, lease_seconds)
        if forecast['status'] != 'ADMITTED': reasons.append('remaining whole-queue disk forecast insufficient')
    elif measured['disk_free_bytes'] < r['disk_reserve_bytes']:
        reasons.append('disk reserve exhausted')
    return reasons


def verify_source(cfg):
    if str(Path(cfg['source_root']).resolve()) != str(SOURCE_ROOT):
        raise ValueError('formal source root must be the integrated checkout')
    saved = read(output_path(cfg, 'source_manifest.json'))
    actual = source_manifest(cfg['source_root'])
    if digest(actual) != cfg['source_digest'] or saved['files'] != actual:
        raise ValueError('frozen source drift')
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=cfg['source_root'], text=True).strip()
    if commit != saved['source_commit']:
        raise ValueError('formal source commit drift')
    return saved


def check_run_config(cfg):
    if any(k in cfg for k in ('engineering_fixture','fixture_backend','fake_backend','a_core_mini_inputs')):
        raise ValueError('engineering fixture/mini overrides are forbidden in formal run')
    if set(cfg['enabled_nodes']) != set(NODES) or not all(cfg['enabled_nodes'][n] for n in MINIMUM_NODES):
        raise ValueError('minimum three scientific nodes must be enabled')
    if any(type(v) is not bool for v in cfg['enabled_nodes'].values()):
        raise ValueError('enabled nodes require frozen booleans')
    if cfg.get('no_ai_dependency') is not True or cfg.get('auto_ai_wakeup') is not False or cfg.get('nightly_formal_sync') is not False:
        raise ValueError('unattended execution must have no AI or official-table update dependency')
    if not cfg.get('source_digest') or not cfg.get('input_digest'):
        raise ValueError('formal run requires frozen source/input digests')
    if cfg['python']!=PYTHON or cfg['rscript']!=RSCRIPT:
        raise ValueError('formal interpreter paths must equal admitted contract paths')
    r = cfg['resources']
    if r['gpu_lock'] != str(GPU_LOCK) or not 1 <= r['cpu_workers'] <= 4 or r['worker_threads'] != 1:
        raise ValueError('workspace lock or CPU worker/thread contract drift')
    if not 0 < r['resource_wait_seconds'] <= 600:
        raise ValueError('resource waits require a bounded positive limit <=600s')
    if r['disk_reserve_bytes']<10*1024**3 or r['min_ram_available_bytes']<6*1024**3 or r['min_gpu_free_bytes']<1024**3:
        raise ValueError('formal resource reserves cannot be reduced below frozen plan minima')
    if (r.get('paper_R_threads'),r.get('paper_R_processes'),r.get('spatial_R_threads'))!=(4,1,1):
        raise ValueError('accepted-R threading policy must be frozen: paper1 process/4threads, spatial1thread')
    if not math.isfinite(cfg['budget_seconds']) or cfg['budget_seconds'] <= 0:
        raise ValueError('invalid initial execution budget')
    estimates(cfg)


def prepare(template, output_root=None, run_id=None):
    """Freeze only after scripts are committed; admission is a separate reviewer action."""
    cfg = load_run(template)
    if output_root is not None: cfg['output_root'] = str(Path(output_root).resolve())
    if run_id is not None: cfg['run_id'] = run_id
    cfg = load_run(cfg)
    root = Path(cfg['output_root']); runpath = root/'configs/run.json'
    if root.exists() and any(root.iterdir()):
        raise ValueError('prepare requires a new empty run directory; preserve existing runs')
    source = Path(cfg['source_root'])
    if source.resolve() != SOURCE_ROOT: raise ValueError('prepare must bind integrated production source root')
    dirty = subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=all'], cwd=source, text=True)
    if dirty.strip(): raise ValueError('commit final source before preparation')
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=source, text=True).strip()
    root.mkdir(parents=True, exist_ok=True)
    with lock_file(root/'control/prepare.lock'):
        try:
            cfg['source_digest'] = digest(source_manifest(source)); cfg['input_digest'] = None
            estimates(cfg)
            forecast = budget_forecast(cfg)
            atomic_json(root/'admission/budget_forecast.json', forecast)
            disk = disk_forecast(cfg); atomic_json(root/'admission/disk_forecast.json', disk)
            if forecast['status'] != 'ADMITTED': raise ValueError('NOT_ADMITTED_BUDGET: ' + json.dumps(forecast))
            if disk['status'] != 'ADMITTED': raise ValueError('NOT_ADMITTED_DISK: ' + json.dumps(disk))
            from .common import compute_lock
            with compute_lock(timeout=cfg['resources']['resource_wait_seconds']):
                from .morph import prepare_morph
                prepare_morph(cfg)
            from .provenance import capture_inputs, environment_check
            inputs = capture_inputs(cfg, root/'input_manifest.json'); cfg['input_digest'] = inputs['input_digest']
            check_run_config(cfg)
            atomic_json(runpath, cfg)
            atomic_json(root/'source_manifest.json', dict(schema='boundary-three-source-manifest-v1',
                source_commit=commit, source_digest=cfg['source_digest'], files=source_manifest(source), created_utc=utc()))
            verify_source(cfg)
            from .freeze import freeze_inputs
            freeze_inputs(cfg)
            atomic_json(root/'admission/environment.json', environment_check(cfg,require_cuda=True))
            save_state(cfg, new_state(cfg))
            event(cfg, 'PREPARED', source_commit=commit, config_digest=digest(cfg))
            return dict(status='PREPARED', config=str(runpath), source_commit=commit, budget=forecast, disk=disk)
        except Exception as exc:
            atomic_json(root/'prepare_failure.json', dict(status='FAILED_PREPARATION', reason=str(exc), time=utc()))
            raise


def verify_admission(cfg, admission):
    signed = read(admission); saved = read(output_path(cfg, 'source_manifest.json'))
    reviewer = signed.get('reviewer', {})
    if (signed.get('schema') != 'boundary-three-launch-admission-v1' or signed.get('status') != 'ADMITTED' or
        signed.get('approved_for_formal_launch') is not True or reviewer.get('implementation_author') is not False or
        reviewer.get('model') != 'gpt-6.1-sol' or reviewer.get('reasoning_effort') != 'xhigh'):
        raise ValueError('independent final reviewer admission is required')
    expected = dict(source_commit=saved['source_commit'], source_digest=cfg['source_digest'],
                    config_digest=digest(cfg), input_digest=cfg['input_digest'])
    if any(signed.get(k) != v for k, v in expected.items()): raise ValueError('final admission identity mismatch')
    enabled = [n for n in NODES if cfg['enabled_nodes'][n]]
    if signed.get('admitted_nodes') != enabled: raise ValueError('exact frozen enabled-node admission missing')
    if not signed.get('evidence') or not signed.get('environment'):
        raise ValueError('final admission lacks evidence/environment bindings')
    for obj in signed['evidence'] + [signed['environment']]: check_identity(obj)
    for n in enabled:
        row = signed.get('node_admissions', {}).get(n, {})
        if row.get('status') != 'ADMITTED' or not row.get('evidence'):
            raise ValueError('actual mini/resource admission missing ' + n)
        for obj in row['evidence']: check_identity(obj)
    return signed


def validate(config, admission):
    cfg = load_run(config); check_run_config(cfg)
    with lock_file(output_path(cfg, 'control/launch.lock')), lock_file(output_path(cfg,'control/queue.lock')), semantic_lock(cfg):
        state = bound_state(cfg)
        if process_alive(state.get('supervisor')) or process_alive(state.get('worker')):
            raise RuntimeError('cannot replace validation while scientific queue is active')
        source = verify_source(cfg); signed = verify_admission(cfg, admission)
        from .provenance import verify_inputs, environment_check
        from .freeze import validate_inputs
        inputs = verify_inputs(cfg); environment = environment_check(cfg, require_cuda=True)
        if signed['environment'] != identity(output_path(cfg, 'admission', 'environment.json')):
            raise ValueError('reviewer environment receipt must bind this run admission/environment.json')
        previous = read(signed['environment']['path'])
        if previous != environment: raise ValueError('admitted environment changed')
        domains = validate_inputs(cfg)
        # Re-run preparation's exact step-zero and initialization checks without
        # changing scientific configuration; all inputs remain their frozen files.
        from .morph import prepare_morph
        prepare_morph(cfg)
        budget = budget_forecast(cfg); disk = disk_forecast(cfg, state); measured = resources_now(cfg)
        reasons = resource_reasons(cfg, measured, MINIMUM_NODES[0], state)
        if budget['status'] != 'ADMITTED' or disk['status'] != 'ADMITTED' or reasons:
            raise ValueError('NOT_ADMITTED_RESOURCES: ' + json.dumps(dict(budget=budget, disk=disk, reasons=reasons)))
        result = dict(schema='boundary-three-validated-launch-v1', status='ADMITTED', time=utc(),
            run_id=cfg['run_id'], config=identity(config), config_digest=digest(cfg),
            source_digest=cfg['source_digest'], input_digest=cfg['input_digest'], source_commit=source['source_commit'],
            independent_admission=identity(admission), environment=environment, inputs=inputs,
            frozen_domains=domains, budget=budget, disk=disk, resources=measured)
        atomic_json(output_path(cfg, 'admission', 'validated.json'), result)
        state['execution_status'] = 'ADMITTED'; save_state(cfg, state); event(cfg, 'ADMITTED', admission=identity(admission))
        return result


def validated_launch(cfg, config):
    check_run_config(cfg); verify_source(cfg)
    rec = read(output_path(cfg, 'admission', 'validated.json'))
    if rec['status'] != 'ADMITTED' or rec['config'] != identity(config) or rec['config_digest'] != digest(cfg):
        raise ValueError('frozen config differs from validated launch')
    check_identity(rec['independent_admission']); verify_admission(cfg, rec['independent_admission']['path'])
    return rec


def thread_environment(cfg):
    env = dict(os.environ)
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        env[key] = '1'
    env['PYTHONUNBUFFERED'] = '1'
    env['PYTHONPATH'] = cfg['source_root']
    return env


def launch(config, resume=False, budget_seconds=None):
    cfg = load_run(config)
    with lock_file(output_path(cfg, 'control', 'launch.lock')):
        validated_launch(cfg, config); state = bound_state(cfg)
        if process_alive(state.get('supervisor')) or process_alive(state.get('worker')):
            raise RuntimeError('duplicate launch refused: bound supervisor or worker is alive')
        # The worker inherits this flock; a dead supervisor with live descendants
        # still prevents replacement, even if a stale process.json was written.
        with lock_file(output_path(cfg, 'control', 'queue.lock')): pass
        if resume:
            if any(row['execution_status'].startswith('FAILED') for row in state['nodes'].values()):
                raise ValueError('failed scientific nodes require a new reviewed run; no automatic OOM/integrity retry')
            if not state['leases']: raise ValueError('resume requires a previously launched execution lease')
        elif state['leases']:
            raise ValueError('use resume for an already launched run')
        budget = cfg['budget_seconds'] if budget_seconds is None else budget_seconds
        if not math.isfinite(budget) or budget <= PACKAGE_SECONDS:
            raise ValueError('execution lease must exceed packaging reserve')
        if not resume and budget != cfg['budget_seconds']:
            raise ValueError('initial launch budget must equal frozen config; use a new prepared run to change it')
        from .provenance import verify_inputs, environment_check
        verify_inputs(cfg); environment_check(cfg,require_cuda=True)
        disk = disk_forecast(cfg,state,budget)
        atomic_json(output_path(cfg,'admission','resume_disk_'+str(time.time_ns())+'.json'),disk)
        if disk['status']!='ADMITTED': raise ValueError('NOT_ADMITTED_DISK for proposed execution lease: '+json.dumps(disk))
        lease_id = uuid.uuid4().hex
        lease = dict(lease_id=lease_id, created_utc=utc(), budget_seconds=budget, resume=resume,
                     config_digest=digest(cfg), independent_admission=read(output_path(cfg,'admission/validated.json'))['independent_admission'])
        atomic_json(output_path(cfg, 'control', 'leases', lease_id + '.json'), lease)
        stop = output_path(cfg, 'control', 'stop.json')
        if stop.exists():
            archive = output_path(cfg, 'control', 'stops', str(time.time_ns()) + '.json')
            archive.parent.mkdir(parents=True, exist_ok=True); os.replace(stop, archive)
        state['leases'].append(lease); state.update(active_lease=lease_id, execution_status='STARTING',
            supervisor=None, worker=None, failure=None)
        session = 'boundary-three-' + lease_id[:12]; state['tmux_session'] = session
        save_state(cfg, state); event(cfg, 'LEASE_CREATED', **lease)
        import shlex
        command = 'exec ' + shlex.join([cfg['python'], '-m', 'scripts.boundary_three.cli', '_run',
                                      '--config', str(Path(config).resolve()), '--lease', lease_id])
        logfile = output_path(cfg, 'logs', 'queue.log'); logfile.parent.mkdir(parents=True, exist_ok=True)
        command += ' >> ' + shlex.quote(str(logfile)) + ' 2>&1'
        try:
            subprocess.run(['tmux', 'new-session', '-d', '-s', session, '-c', cfg['source_root'], command],
                           check=True, env=thread_environment(cfg), timeout=20)
            until = time.monotonic() + 30
            while time.monotonic() < until:
                current = bound_state(cfg)
                if current.get('active_lease') == lease_id and process_alive(current.get('supervisor')):
                    return dict(status='STARTED', run_id=cfg['run_id'], session=session,
                                supervisor=current['supervisor'], lease=lease,
                                log_path=current['log_path'], scientific_progress_confirmed=bool(
                                    current.get('last_progress_utc') and current['last_progress_utc']>=lease['created_utc']))
                if current['execution_status'].startswith('FAILED'): raise RuntimeError(current.get('failure'))
                time.sleep(.25)
            raise RuntimeError('tmux supervisor did not acknowledge live identity within 30 seconds')
        except Exception as exc:
            current = bound_state(cfg)
            if not process_alive(current.get('supervisor')):
                current.update(execution_status='FAILED_LAUNCH', failure=str(exc)); save_state(cfg, current)
            event(cfg, 'FAILED_LAUNCH', reason=str(exc)); raise


def request_stop(cfg, reason='USER', detail='explicit safe-stop request'):
    path = output_path(cfg, 'control', 'stop.json')
    # Preserve the first cause (including the user's reason) across monitor ticks.
    with lock_file(output_path(cfg, 'control', 'stop.lock'), blocking=True):
        if not path.exists():
            atomic_json(path, dict(reason=reason, detail=detail, requested_utc=utc(), requester=process_identity()))
            event(cfg, 'STOP_REQUESTED', reason=reason, detail=detail)
        return read(path)


def stop_status(cfg):
    p = output_path(cfg, 'control', 'stop.json')
    reason = read(p)['reason'] if p.exists() else 'INTERRUPTED'
    return {'USER':'PAUSED_USER', 'BUDGET':'PAUSED_BUDGET', 'RESOURCE':'PAUSED_RESOURCE'}.get(reason, 'PAUSED_INTERRUPTED')


def node_module(node):
    return importlib.import_module('scripts.boundary_three.' + MODULES[node])


def verify_worker_authorization(cfg, config, node, lease_id, queue_fd, compute_fd):
    validated_launch(cfg,config)
    for fd,path in ((queue_fd,output_path(cfg,'control/queue.lock')),(compute_fd,GPU_LOCK)):
        if fd is None: raise ValueError('internal node requires inherited supervisor lock descriptors')
        actual=os.fstat(fd); expected=path.stat()
        if (actual.st_dev,actual.st_ino)!=(expected.st_dev,expected.st_ino):
            raise ValueError('inherited scientific lock descriptor identity mismatch')
        # This succeeds on the inherited open-file description, and cannot
        # acquire another owner's held lock on a separately opened descriptor.
        fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state=bound_state(cfg)
    if state['active_lease']!=lease_id or state['current_node']!=node or state['execution_status']!='RUNNING':
        raise ValueError('internal node does not match active scientific lease/node')
    if not process_alive(state['supervisor']) or state['supervisor']['pid']!=os.getppid():
        raise ValueError('internal node requires its bound live supervisor parent')
    if state['nodes'][node]['execution_status']!='RUNNING':
        raise ValueError('internal node state is not RUNNING')


def inherited_lock_fds():
    """Preserve verified worker queue/compute locks across exact R subprocesses.

    Empty outside the boundary worker. Partial, stale, or mismatched markers
    fail before spawning. A protected orphan is allowed to reach a safe boundary;
    it must not require its already exited supervisor to remain alive.
    """
    values=[os.environ.get(k) for k in LOCK_ENV]
    if not any(values):return ()
    if not all(values):raise ValueError('partial boundary inherited-lock environment')
    fds=json.loads(values[0])
    if set(fds)!={'queue','compute'} or any(type(v) is not int or v<3 for v in fds.values()) or fds['queue']==fds['compute']:
        raise ValueError('invalid inherited queue/compute descriptor map')
    root=Path(values[1]).resolve();cfg=load_run(root/'configs/run.json')
    if root!=Path(cfg['output_root']).resolve() or not cfg.get('source_digest') or not cfg.get('input_digest') or cfg.get('engineering_fixture'):
        raise ValueError('inherited-lock markers do not bind a formal frozen run')
    state=bound_state(cfg);lease=read(output_path(cfg,'control/leases',values[2]+'.json'))
    if state['active_lease']!=values[2] or lease['config_digest']!=digest(cfg):
        raise ValueError('inherited R locks belong to a stale execution lease')
    if state['execution_status']!='RUNNING' or state.get('current_node') not in NODES:
        raise ValueError('inherited R locks lack an active scientific node')
    for key,path in (('queue',root/'control/queue.lock'),('compute',GPU_LOCK)):
        actual=os.fstat(fds[key]);expected=path.stat()
        if (actual.st_dev,actual.st_ino)!=(expected.st_dev,expected.st_ino):
            raise ValueError('inherited R '+key+' lock inode drift')
        fcntl.flock(fds[key],fcntl.LOCK_EX|fcntl.LOCK_NB)
    return (fds['queue'],fds['compute'])


def node_worker(config, node, result_path, lease_id=None, queue_fd=None, compute_fd=None):
    cfg = load_run(config)
    verify_worker_authorization(cfg,config,node,lease_id,queue_fd,compute_fd)
    previous_environment={k:os.environ.get(k) for k in LOCK_ENV}
    if lease_id is not None:
        os.environ.update(BOUNDARY_THREE_LOCK_FDS=json.dumps(dict(queue=queue_fd,compute=compute_fd)),
                          BOUNDARY_THREE_RUN_ROOT=cfg['output_root'],BOUNDARY_THREE_LEASE=lease_id)
    atomic_json(output_path(cfg, 'control', 'worker.json'), process_identity())
    started = time.monotonic()
    watchdog_done=threading.Event(); watchdog=None
    if lease_id is not None:
        lease=read(output_path(cfg,'control/leases',lease_id+'.json'))
        supervisor=dict(bound_state(cfg)['supervisor'])
        end=datetime.fromisoformat(lease['created_utc'])+timedelta(seconds=lease['budget_seconds'])
        deadline=time.monotonic()+max(0,(end-datetime.now(timezone.utc)).total_seconds())
        def watch_supervisor():
            while not watchdog_done.is_set():
                if not process_alive(supervisor):
                    request_stop(cfg,'INTERRUPTED','bound supervisor exited; worker preserves safe checkpoint and inherited locks')
                if time.monotonic()>=deadline-PACKAGE_SECONDS:
                    request_stop(cfg,'BUDGET','worker execution-lease watchdog reached packaging reserve')
                watchdog_done.wait(5)
        watchdog=threading.Thread(target=watch_supervisor,name='node-safe-boundary-watchdog',daemon=True)
        watchdog.start()
    try:
        module = node_module(node)
        module.run_node(cfg, node)
        rec = module.validate_node(cfg, node)
        result = dict(status='COMPLETED', science_status=rec['science_status'],
                      receipt=identity(output_path(cfg,node,'complete.json')))
    except BaseException as exc:
        # A_CORE's SafeStop inherits RuntimeError, so blanket RuntimeError
        # classification would turn a requested pause into scientific failure.
        controlled = (isinstance(exc, InterruptedError) or
                      (type(exc).__name__ == 'SafeStop' and type(exc).__module__ == 'scripts.boundary_three.a_core'))
        result = dict(status='CONTROLLED_PAUSE' if controlled else
                      'FAILED_INTEGRITY' if isinstance(exc, ValueError) else 'FAILED_RUNTIME',
                      reason=str(exc), exception_type=type(exc).__name__,
                      errno=exc.errno if isinstance(exc, OSError) else None,
                      traceback=traceback.format_exc())
        traceback.print_exc()
    result['elapsed_seconds'] = time.monotonic() - started
    watchdog_done.set()
    if watchdog: watchdog.join(timeout=6)
    for key,value in previous_environment.items():
        if value is None:os.environ.pop(key,None)
        else:os.environ[key]=value
    atomic_json(result_path, result)
    if lease_id is not None and not process_alive(supervisor):
        orphan=bound_state(cfg)
        if orphan['active_lease']==lease_id and orphan['current_node']==node:
            status=result['status']
            orphan['nodes'][node].update(execution_status=stop_status(cfg) if status=='CONTROLLED_PAUSE' else status,
                                        science_status=result.get('science_status'),reason=result.get('reason'))
            if status=='COMPLETED':orphan['nodes'][node]['receipt']=result['receipt']
            orphan.update(execution_status='PAUSED_INTERRUPTED' if status in ('COMPLETED','CONTROLLED_PAUSE') else status,
                          failure='bound supervisor exited; inspect atomic worker result before resuming')
            save_state(cfg,orphan);event(cfg,'ORPHAN_WORKER_SAFE_EXIT',node=node,result=identity(result_path))
    return 0 if result['status'] == 'COMPLETED' else 75 if result['status'] == 'CONTROLLED_PAUSE' else 1


def checkpoint_evidence(cfg, node):
    """Require existing content-verified atomic recovery units for the sole IO retry.

    Scientific run_node still performs full cursor/RNG/source validation before
    consuming these units. Bare checkpoint files and arbitrary JSON are refused.
    """
    root = output_path(cfg, node); records = []
    if node == 'A1_CORE':
        records = sorted((root/'count_chunks').glob('*.json'))
        for p in records:
            rec = read(p); check_identity(rec['output'])
            if rec['dependencies']['source'] != identity(Path(cfg['source_root'])/'scripts/boundary_three/a_core.py')['sha256']:
                raise ValueError('A1 checkpoint source drift')
    elif node.startswith('P2_'):
        from .p2_native import verify_record
        records = sorted(root.rglob('tile_*.json'))
        for p in records: verify_record(p)
    elif node == 'A2_CORE':
        records = sorted(root.rglob('state.json'))
        for p in records:
            rec = read(p); check_identity(rec['mask']); check_identity(rec['decisions'])
            if rec['cursor'] < 0: raise ValueError('negative A2 checkpoint cursor')
    else:
        latest = root/'checkpoints/latest.pth'
        if latest.exists():
            import torch
            from .morph import training_header, validate_cursor
            saved = torch.load(latest, map_location='cpu')
            if saved['header'] != training_header(cfg,node) or saved['pilot']:
                raise ValueError('Morph checkpoint frozen header drift')
            validate_cursor(cfg, saved); records = [latest]
    if not records: return None
    return dict(status='CONTENT_VERIFIED_RECOVERY_UNITS', node=node,
                scientific_revalidation_required=True, units=[identity(p) for p in records])


def _heartbeat(cfg, state, stage=None):
    state['heartbeat_utc'] = utc()
    if stage: state['stage'] = stage
    n = state.get('current_node')
    path = output_path(cfg,n,'progress.json') if n else None
    if path and path.exists():
        value = read(path)
        if value != state.get('progress'):
            state['progress'] = value; state['last_progress_utc'] = value['updated_utc']
            state['stage'] = value['stage']
    save_state(cfg,state)


def _wait_compute(cfg, state):
    stream = GPU_LOCK.open('a+'); until = time.monotonic() + cfg['resources']['resource_wait_seconds']
    while True:
        try: fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB); return stream
        except BlockingIOError:
            _heartbeat(cfg,state,'WAITING_COMPUTE_LOCK')
            if output_path(cfg,'control/stop.json').exists() or time.monotonic() >= until:
                stream.close()
                if not output_path(cfg,'control/stop.json').exists(): request_stop(cfg,'RESOURCE','compute lock wait exhausted')
                return None
            time.sleep(1)


def _wait_resources(cfg, state, node):
    until = time.monotonic() + cfg['resources']['resource_wait_seconds']
    while True:
        measured = resources_now(cfg)
        lease=next(r for r in state['leases'] if r['lease_id']==state['active_lease'])
        reasons = resource_reasons(cfg, measured, node, state, lease['budget_seconds'])
        state['resources'] = measured; state['resource_wait_reasons'] = reasons
        if not reasons: return True
        _heartbeat(cfg,state,'WAITING_RESOURCES')
        if output_path(cfg,'control/stop.json').exists() or time.monotonic() >= until:
            if not output_path(cfg,'control/stop.json').exists(): request_stop(cfg,'RESOURCE','; '.join(reasons))
            return False
        time.sleep(2)


def _run_worker(cfg, config, state, node, queue_stream, compute_stream, deadline):
    attempt = state['nodes'][node].get('attempts',0) + 1; state['nodes'][node]['attempts'] = attempt
    result_path = output_path(cfg,'control','results',f'{node}_{state["active_lease"]}_{attempt}.json')
    log = output_path(cfg,'logs',f'{node}_{attempt:03d}.log'); log.parent.mkdir(parents=True,exist_ok=True)
    state['node_log_path'] = str(log)
    args = [cfg['python'],'-m','scripts.boundary_three.cli','_node','--config',str(config),
            '--node',node,'--result',str(result_path),'--lease',state['active_lease'],
            '--queue-fd',str(queue_stream.fileno()),'--compute-fd',str(compute_stream.fileno())]
    worker_id_path = output_path(cfg,'control/worker.json')
    if worker_id_path.exists(): worker_id_path.unlink()
    with log.open('a') as stream:
        proc = subprocess.Popen(args,cwd=cfg['source_root'],env=thread_environment(cfg),stdout=stream,stderr=subprocess.STDOUT,
                                pass_fds=(queue_stream.fileno(),compute_stream.fileno()))
        state['worker'] = process_identity(proc.pid); save_state(cfg,state)
        event(cfg,'NODE_PROCESS_STARTED',node=node,worker=state['worker'],attempt=attempt,log_path=str(log))
        last_resource_check = 0
        while proc.poll() is None:
            _heartbeat(cfg,state)
            if time.monotonic() >= deadline - PACKAGE_SECONDS:
                request_stop(cfg,'BUDGET','execution lease reached packaging reserve; stop at scientific safe boundary')
            if time.monotonic() - last_resource_check > 30:
                try:
                    measured = resources_now(cfg); state['resources'] = measured
                except (OSError,subprocess.SubprocessError,ValueError) as exc:
                    request_stop(cfg,'RESOURCE','resource monitor unavailable: '+str(exc)); measured=None
                last_resource_check = time.monotonic()
                # In-progress work uses reserve thresholds, not peak+reserve:
                # its already allocated working set must not be counted twice.
                reasons = resource_reasons(cfg,measured) if measured else []
                if reasons: request_stop(cfg,'RESOURCE','; '.join(reasons))
            time.sleep(2)
        state['worker'] = None; state['worker_exit_code'] = proc.returncode
    if not result_path.exists():
        return dict(status='PAUSED_INTERRUPTED',reason='worker exited without atomic result receipt',exit_code=proc.returncode)
    result = read(result_path); state['nodes'][node]['last_worker_result'] = identity(result_path)
    if result['status'] == 'COMPLETED' and proc.returncode != 0:
        raise ValueError('completed worker result has nonzero exit code')
    if result['status'] == 'CONTROLLED_PAUSE': result['status'] = stop_status(cfg)
    return result


def run_queue(config, lease_id):
    cfg = load_run(config)
    with lock_file(output_path(cfg,'control/queue.lock')) as queue_stream:
        validated_launch(cfg, config); state = bound_state(cfg)
        if state['active_lease'] != lease_id: raise ValueError('stale execution lease cannot launch')
        lease = read(output_path(cfg,'control/leases',lease_id+'.json'))
        if lease['config_digest'] != digest(cfg): raise ValueError('execution lease config drift')
        if state['execution_status']!='STARTING': raise ValueError('lease must be newly dispatched through launch/resume')
        claim=output_path(cfg,'control/lease_claims',lease_id+'.json'); claim.parent.mkdir(parents=True,exist_ok=True)
        fd=os.open(claim,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
        with os.fdopen(fd,'w') as stream:
            stream.write(json.dumps(dict(claimed_utc=utc(),supervisor=process_identity()))+'\n');stream.flush();os.fsync(stream.fileno())
        lease_start = time.monotonic()
        created=datetime.fromisoformat(lease['created_utc'])
        deadline_time=created+timedelta(seconds=lease['budget_seconds'])
        deadline = lease_start + max(0,(deadline_time-datetime.now(timezone.utc)).total_seconds())
        state.update(execution_status='RUNNING',supervisor=process_identity(),worker=None,
                     stage='VERIFYING_FROZEN_INPUTS',heartbeat_utc=utc(),
                     lease_started_utc=utc(),deadline_utc=deadline_time.isoformat())
        save_state(cfg,state); event(cfg,'QUEUE_STARTED',lease_id=lease_id,supervisor=state['supervisor'])
        heartbeat_done=threading.Event()
        supervisor_identity=dict(state['supervisor'])
        def heartbeat_loop():
            while not heartbeat_done.is_set():
                atomic_json(output_path(cfg,'control/heartbeat.json'),dict(supervisor=supervisor_identity,heartbeat_utc=utc()))
                heartbeat_done.wait(5)
        heartbeat_thread=threading.Thread(target=heartbeat_loop,name='queue-heartbeat',daemon=True)
        heartbeat_thread.start()
        def stop_signal(signum, frame): request_stop(cfg,'USER','supervisor received signal '+str(signum))
        signal.signal(signal.SIGTERM,stop_signal); signal.signal(signal.SIGINT,stop_signal)
        try:
            from .provenance import verify_inputs, environment_check
            verify_inputs(cfg); environment_check(cfg,require_cuda=True)
            for index,node in enumerate(NODES):
                row = state['nodes'][node]; old = row['execution_status']
                if output_path(cfg,node,'complete.json').exists():
                    rec = node_module(node).validate_node(cfg,node)
                    row.update(execution_status='COMPLETED',science_status=rec['science_status'],
                               receipt=identity(output_path(cfg,node,'complete.json')))
                    save_state(cfg,state); continue
                if old == 'COMPLETED': raise ValueError('completed node lost its receipt: '+node)
                if old == 'SKIPPED_SCIENTIFIC_GATE': continue
                if old == 'DEFERRED_NOT_ADMITTED': continue
                if old == 'DEFERRED_TIME' and not lease['resume']: continue
                if not cfg['enabled_nodes'][node]:
                    row.update(execution_status='DEFERRED_NOT_ADMITTED',reason='disabled before scientific results'); save_state(cfg,state); continue
                if node == 'A2_CORE':
                    a1 = node_module('A1_CORE').validate_node(cfg,'A1_CORE')
                    if a1['science_status'] in ('NO_USEFUL_INCREMENT','INCONCLUSIVE'):
                        row.update(execution_status='SKIPPED_SCIENTIFIC_GATE',science_status='NOT_APPLICABLE',
                                   reason='complete A1 decision '+a1['science_status'],gate_receipt=identity(output_path(cfg,'A1_CORE/complete.json')))
                        event(cfg,'SCIENTIFIC_SKIP',node=node,reason=row['reason']); save_state(cfg,state); continue
                    if a1['science_status'] != 'GO_A2': raise ValueError('A1 invalid/missing decision cannot become scientific skip')
                # Every earlier node must have a validated receipt or a frozen,
                # permitted skip. A failure/pause cannot be stepped over.
                if any(state['nodes'][p]['execution_status'] not in FINAL for p in NODES[:index]):
                    raise ValueError('unfinished predecessor before '+node)
                remaining = deadline - time.monotonic()
                needed = cfg['node_estimates'][node]['seconds'] * SAFETY + PACKAGE_SECONDS
                if remaining < needed:
                    if node in MINIMUM_NODES:
                        request_stop(cfg,'BUDGET','remaining execution lease cannot fit complete minimum node '+node)
                        row.update(execution_status='PAUSED_BUDGET',reason='complete minimum node requires '+str(needed)+' seconds')
                        state['execution_status']='PAUSED_BUDGET'; save_state(cfg,state); break
                    row.update(execution_status='DEFERRED_TIME',reason='remaining complete-node budget insufficient',
                               budget_decision=dict(lease_id=lease_id,remaining_seconds=remaining,required_seconds=needed,
                                                   safety_factor=SAFETY,packaging_reserve_seconds=PACKAGE_SECONDS))
                    event(cfg,'TIME_DEFERRED',node=node,**row['budget_decision']); save_state(cfg,state); continue
                state.update(current_node=node,stage='NODE_ADMISSION',progress=None,last_progress_utc=None)
                row.update(execution_status='RUNNING',science_status=None,started_utc=utc())
                save_state(cfg,state); verify_source(cfg)
                if output_path(cfg,'control/stop.json').exists():
                    row['execution_status']=stop_status(cfg); state['execution_status']=stop_status(cfg); save_state(cfg,state); break
                compute = _wait_compute(cfg,state)
                if compute is None:
                    row['execution_status']=stop_status(cfg); state['execution_status']=stop_status(cfg); save_state(cfg,state); break
                try:
                    if not _wait_resources(cfg,state,node):
                        row['execution_status']=stop_status(cfg); state['execution_status']=stop_status(cfg); save_state(cfg,state); break
                    result = _run_worker(cfg,config,state,node,queue_stream,compute,deadline)
                    if result['status'].startswith('FAILED') and result.get('errno') in TRANSIENT_IO and row['retries']==0:
                        evidence=checkpoint_evidence(cfg,node)
                        if evidence:
                            row['retries']=1; atomic_json(output_path(cfg,'control/retries',node+'.json'),evidence)
                            event(cfg,'BOUNDED_IO_RETRY',node=node,errno=result['errno'],checkpoint=evidence)
                            result=_run_worker(cfg,config,state,node,queue_stream,compute,deadline)
                    row.update(execution_status=result['status'],science_status=result.get('science_status'),
                               reason=result.get('reason'),elapsed_seconds_this_attempt=result['elapsed_seconds'] if 'elapsed_seconds' in result else None)
                    if result['status']=='COMPLETED':
                        check_identity(result['receipt']); row['receipt']=result['receipt']; row['completed_utc']=utc()
                        event(cfg,'NODE_COMPLETED',node=node,science_status=row['science_status'],receipt=row['receipt'])
                    else:
                        state.update(execution_status=result['status'],failure=result.get('reason'))
                        event(cfg,'QUEUE_HALTED',node=node,status=result['status'],reason=result.get('reason')); save_state(cfg,state); break
                    save_state(cfg,state)
                finally:
                    compute.close()
            else:
                state['execution_status']='FINISHED_WITH_DEFERRALS' if any(r['execution_status'].startswith('DEFERRED') for r in state['nodes'].values()) else 'FINISHED'
                state['current_node']=None; state['stage']='PACKAGE'; save_state(cfg,state)
            state['worker']=None
            state['lease_elapsed_seconds']=time.monotonic()-lease_start; state['lease_finished_utc']=utc()
            save_state(cfg,state)
            check=complete_check(config,internal=True)
            invalid=[n for n in NODES if check['nodes'][n]['status'].startswith('INVALID')]
            if invalid: raise ValueError('semantic completion check failed: '+','.join(invalid))
            package(cfg,state)
            save_state(cfg,state); event(cfg,'QUEUE_LEASE_FINISHED',status=state['execution_status'],lease_id=lease_id)
            return 0 if state['execution_status'].startswith('FINISHED') else 1 if state['execution_status'].startswith('FAILED') else 75
        except BaseException as exc:
            state.update(execution_status='FAILED_INTEGRITY' if isinstance(exc,ValueError) else 'FAILED_RUNTIME',failure=str(exc))
            if state.get('current_node'):
                state['nodes'][state['current_node']].update(execution_status=state['execution_status'],reason=str(exc))
            save_state(cfg,state); event(cfg,'SUPERVISOR_FAILURE',reason=str(exc),traceback=traceback.format_exc())
            traceback.print_exc(); package(cfg,state); return 1
        finally:
            heartbeat_done.set(); heartbeat_thread.join(timeout=6)


def receipt_status(cfg,node):
    path=output_path(cfg,node,'complete.json')
    if not path.exists(): return dict(status='ABSENT')
    try:
        rec=read(path)
        from .common import receipt_dependencies
        if rec.get('execution_status')!='COMPLETED' or rec.get('dependencies')!=receipt_dependencies(cfg,node):
            return dict(status='INCOMPLETE_OR_WRONG_IDENTITY')
        return dict(status='PRESENT_UNVERIFIED_OUTPUTS',science_status=rec.get('science_status'),receipt=identity(path))
    except (ValueError,OSError,KeyError) as exc: return dict(status='INVALID_RECEIPT',reason=str(exc))


def status(config):
    cfg=load_run(config); state=bound_state(cfg)
    live=process_alive(state.get('supervisor')); worker_live=process_alive(state.get('worker'))
    state['process_alive']=live; state['worker_alive']=worker_live
    def held(path):
        if not path.exists():return False
        with path.open('a+') as stream:
            try:fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB);return False
            except BlockingIOError:return True
    state['queue_lock_held']=held(output_path(cfg,'control/queue.lock'))
    state['compute_lock_held']=held(GPU_LOCK)
    heartbeat=output_path(cfg,'control/heartbeat.json')
    if heartbeat.exists():
        observed=read(heartbeat)
        if observed['supervisor']==state.get('supervisor'):state['heartbeat_utc']=observed['heartbeat_utc']
    for n in NODES: state['nodes'][n]['receipt_status']=receipt_status(cfg,n)
    progress_time=state.get('last_progress_utc')
    if progress_time:
        age=(datetime.now(timezone.utc)-datetime.fromisoformat(progress_time)).total_seconds()
    else: age=None
    state['seconds_since_progress']=age
    if live:
        state['liveness']='LIVE_NO_PROGRESS' if age is None or age>cfg['resources'].get('no_progress_seconds',900) else 'LIVE_WITH_PROGRESS'
    elif worker_live: state['liveness']='SUPERVISOR_EXITED_WORKER_ALIVE'
    elif state['queue_lock_held']:state['liveness']='PROCESS_EXITED_DESCENDANT_LOCK_HELD'
    else: state['liveness']='PROCESS_EXITED' if state['leases'] else 'NOT_LAUNCHED'
    state['untracked_descendant_lock_owner']=bool(not live and not worker_live and state['queue_lock_held'])
    state['completed_nodes']=[n for n in NODES if state['nodes'][n]['execution_status']=='COMPLETED']
    state['deferred_nodes']=[n for n in NODES if state['nodes'][n]['execution_status'].startswith('DEFERRED')]
    state['skipped_nodes']=[n for n in NODES if state['nodes'][n]['execution_status']=='SKIPPED_SCIENTIFIC_GATE']
    return state


def complete_check(config, write=True, internal=False, compute_owned=False):
    cfg=load_run(config)
    if not internal:
        if bound_state(cfg)['execution_status']=='STARTING':
            raise RuntimeError('complete-check deferred while launch is dispatching; use fast status')
        # Full semantic validation can load masks/matrices and is substantial
        # work; it must not overlap a scientific worker or surviving orphan.
        with lock_file(output_path(cfg,'control/queue.lock')):
            return complete_check(config,write,internal=True)
    if not compute_owned:
        with semantic_lock(cfg):
            return complete_check(config,write,internal=True,compute_owned=True)
    state=bound_state(cfg); checked={}
    for n in NODES:
        if output_path(cfg,n,'complete.json').exists():
            try:
                rec=node_module(n).validate_node(cfg,n)
                checked[n]=dict(status='VALIDATED_COMPLETE',science_status=rec['science_status'],receipt=identity(output_path(cfg,n,'complete.json')))
            except Exception as exc: checked[n]=dict(status='INVALID_OR_INCOMPLETE',reason=str(exc))
        else: checked[n]=dict(status=state['nodes'][n]['execution_status'],reason=state['nodes'][n].get('reason'))
    if checked['A2_CORE']['status']=='SKIPPED_SCIENTIFIC_GATE':
        gate=checked['A1_CORE']
        if gate['status']!='VALIDATED_COMPLETE' or gate['science_status'] not in ('NO_USEFUL_INCREMENT','INCONCLUSIVE'):
            checked['A2_CORE']=dict(status='INVALID_SCIENTIFIC_SKIP',reason='A1 gate receipt is not valid negative')
    result=dict(schema='boundary-three-complete-check-v1',time=utc(),run_id=cfg['run_id'],config_digest=digest(cfg),
        minimum_three_complete=all(checked[n]['status']=='VALIDATED_COMPLETE' for n in MINIMUM_NODES),
        all_enabled_applicable_complete=all(checked[n]['status'] in ('VALIDATED_COMPLETE','SKIPPED_SCIENTIFIC_GATE') for n in NODES if cfg['enabled_nodes'][n]),
        nodes=checked,deferred=[n for n in NODES if checked[n]['status'].startswith('DEFERRED')],
        failures=[n for n in NODES if checked[n]['status'].startswith(('FAILED','INVALID'))],
        not_enabled=[n for n in NODES if not cfg['enabled_nodes'][n]])
    if write: atomic_json(output_path(cfg,'complete-check.json'),result)
    return result


def package(cfg,state):
    """Package actual node reports, including missingness; never rerun evaluation.

    Independent scientific interpretation and official table/context updates are
    deliberately performed only after the user resumes the Codex goal.
    """
    root=output_path(cfg,'reports'); root.mkdir(parents=True,exist_ok=True)
    rows=[]; sources={}; comparisons={}; transfers=[]; costs={}; tables=[]; endpoint_rows=[]
    for n in NODES:
        row=state['nodes'][n]
        rows.append(dict(node=n,execution_status=row['execution_status'],science_status=row.get('science_status'),reason=row.get('reason')))
        if row['execution_status']!='COMPLETED': continue
        recpath=output_path(cfg,n,'complete.json')
        try:
            if row.get('receipt') and identity(recpath)!=row['receipt']: raise ValueError('packaging completion receipt drift '+n)
            rec=read(recpath)
            for obj in rec['outputs'].values(): check_identity(obj)
        except (ValueError,OSError,KeyError) as exc:
            row.update(execution_status='FAILED_INTEGRITY',science_status=None,reason=str(exc))
            rows[-1]=dict(node=n,execution_status='FAILED_INTEGRITY',science_status=None,reason=str(exc))
            state.update(execution_status='FAILED_INTEGRITY',failure=str(exc));continue
        sources[n+'/complete.json']=identity(recpath)
        for name in ('comparison.json','runtime_cost.json','decision.json','transfer_diagnostics.json','proposal_and_rejection_counts.csv'):
            p=output_path(cfg,n,name)
            if p.exists():
                sources[n+'/'+name]=identity(p)
                if name=='runtime_cost.json': costs[n]=read(p)
                if name=='comparison.json': comparisons[n]=read(p)
        for p in sorted(output_path(cfg,n).rglob('endpoint_table.csv')):
            relative=str(p.relative_to(output_path(cfg,n)))
            # Every scientific table is already inside its completed node's
            # validated output inventory. Preserve each Purity comparison set.
            if relative not in rec['outputs']: raise ValueError('unreceipted scientific endpoint table '+str(p))
            scaling=p.parent/'purity_scaling.json'
            if not scaling.exists(): raise ValueError('endpoint table missing its comparison scaling identity')
            scale=read(scaling); comparison=read(p.parent/'complete.json')
            sources[n+'/'+relative]=identity(p); sources[str(scaling)]=identity(scaling)
            with p.open() as stream:
                for record in csv.DictReader(stream):
                    endpoint_rows.append(dict(record,node=n,execution_status='COMPLETED',
                        source_table=str(p),source_table_sha256=sources[n+'/'+relative]['sha256'],
                        comparison_set=json.dumps(scale.get('comparison_set',comparison.get('endpoint_ids'))),
                        purity_scaling_version=scale.get('version') or cfg['version']+'/purity/'+identity(scaling)['sha256'],
                        purity_scaling_sha256=identity(scaling)['sha256'],
                        comparison_receipt_sha256=identity(p.parent/'complete.json')['sha256']))
            tables.append(dict(node=n,table=identity(p),comparison_receipt=identity(p.parent/'complete.json'),purity_scaling=identity(scaling)))
        for pattern in ('postprocess_transfer.csv','stage_transfer_strata.csv','proposal_and_rejection_counts.csv','*_owner_categories.csv','domain_capture.csv'):
            for p in sorted(output_path(cfg,n).rglob(pattern)):
                relative=str(p.relative_to(output_path(cfg,n)))
                if relative not in rec['outputs']: raise ValueError('unreceipted boundary transfer table '+str(p))
                transfers.append(dict(node=n,table=identity(p)));sources[n+'/'+relative]=identity(p)
    with (root/'node_status.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=['node','execution_status','science_status','reason']); writer.writeheader(); writer.writerows(rows)
    present={r['method'] for r in endpoint_rows}
    for n in NODES:
        if n=='A1_CORE':continue  # Information test never becomes a mask/F1 row.
        methods=('A2_NATIVE','A2_GT','A2_GTC','A2_SHUFFLED') if n=='A2_CORE' else (n,)
        for name in methods:
            if name not in present:
                endpoint_rows.append(dict(method=name,node=n,execution_status=state['nodes'][n]['execution_status'],
                    science_status=state['nodes'][n].get('science_status'),reason=state['nodes'][n].get('reason'),
                    availability='NO_VALIDATED_METRIC_TABLE'))
    fields=list(dict.fromkeys(key for row in endpoint_rows for key in row))
    with (root/'endpoint_table.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for row in endpoint_rows: writer.writerow({k:row.get(k,'NA') if row.get(k) is not None else 'NA' for k in fields})
    atomic_json(root/'endpoint_table_provenance.json',dict(status='ACTUAL_EXISTING_COMPARISON_TABLES',tables=tables,
        purity_scaling_recomputed=False,comparison_sets_kept_separate=True,A1_mask_endpoint=False))
    atomic_json(root/'comparisons.json',dict(status='PARTIAL' if any(r['execution_status'] not in ('COMPLETED','SKIPPED_SCIENTIFIC_GATE') for r in rows) else 'COMPLETE_APPLICABLE',nodes=comparisons))
    with (root/'boundary_transfers.csv').open('w') as stream:
        fields=['node','source_table','source_table_sha256']
        for item in transfers:
            with Path(item['table']['path']).open() as source:
                fields.extend(k for k in next(csv.reader(source)) if k not in fields)
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for item in transfers:
            with Path(item['table']['path']).open() as source:
                for record in csv.DictReader(source):
                    writer.writerow(dict(record,node=item['node'],source_table=item['table']['path'],source_table_sha256=item['table']['sha256']))
    remaining=[dict(node=n,**state['nodes'][n]) for n in NODES if state['nodes'][n]['execution_status'] not in ('COMPLETED','SKIPPED_SCIENTIFIC_GATE')]
    atomic_json(root/'remaining_queue.json',dict(run_id=cfg['run_id'],remaining=remaining,scientific_config_changed=False))
    atomic_json(root/'runtime_cost.json',dict(nodes=costs,leases=state['leases'],current_lease_elapsed_seconds=state.get('lease_elapsed_seconds')))
    for label,names in [('a_core_report',['A1_CORE','A2_CORE']),('morph_ablation_report',[n for n in NODES if n.startswith('MORPH_')]),('p2_factorial_report',['P2_BASE','P2_MORPH'])]:
        lines=[label.replace('_',' ')+'\n']
        for n in names:
            row=state['nodes'][n]; lines.append(f"{n}: {row['execution_status']}; science={row.get('science_status')}; reason={row.get('reason')}\n")
            for p in sorted(output_path(cfg,n).rglob('report.md')) if row['execution_status']=='COMPLETED' else []:
                lines.append('Actual scientific report: '+str(p)+'\n\n'+p.read_text()+'\n'); sources[str(p)]=identity(p)
            if row['execution_status']=='COMPLETED':
                decision=output_path(cfg,n,'decision.json')
                if decision.exists(): lines.append('Frozen scientific decision\n\n```json\n'+json.dumps(read(decision),indent=2)+'\n```\n')
                for p in sorted(output_path(cfg,n).rglob('decision.json')):
                    if p!=decision:lines.append('Actual comparison decision '+str(p)+'\n\n```json\n'+json.dumps(read(p),indent=2)+'\n```\n')
                values=[r for r in endpoint_rows if r.get('node')==n and 'scclassify_positive_F1' in r]
                if values:
                    lines.append('|Endpoint|scClassify P|scClassify N|P finite n|N finite n|Comparison set|\n|---|---|---|---|---|---|')
                    for value in values:
                        lines.append('|'+ '|'.join(str(value.get(k,'NA')) for k in ('method','scclassify_positive_F1','scclassify_negative_F1','scclassify_positive_F1_n','scclassify_negative_F1_n','comparison_set'))+'|')
        if label=='p2_factorial_report':
            p=output_path(cfg,'P2_MORPH/factorial/p2_factorial_report.md')
            if all(state['nodes'][n]['execution_status']=='COMPLETED' for n in names) and p.exists():
                lines.append('Validated actual four-grid report: '+str(p)+'\n'); sources['p2_factorial']=identity(p)
            else: lines.append('Four-grid comparison incomplete; unavailable endpoints have no fabricated scores.\n')
        (root/(label+'.md')).write_text('\n'.join(lines))
    minimum=all(state['nodes'][n]['execution_status']=='COMPLETED' for n in MINIMUM_NODES)
    (root/'report.md').write_text('Boundary-three unattended execution package\n\n'+
        'Minimum three complete: '+str(minimum)+'\n\n'+
        '\n'.join(f"{r['node']}: {r['execution_status']}; {r.get('science_status')}" for r in rows)+
        '\n\nThis package indexes actual receipts and scientific reports. Time deferrals and engineering pauses remain incomplete. '+
        'Independent result analysis and official table/context synchronization await user goal resume.\n')
    frozen={}
    for key,path in [('config',output_path(cfg,'configs/run.json')),('source_manifest',output_path(cfg,'source_manifest.json')),('input_manifest',output_path(cfg,'input_manifest.json'))]:
        if path.exists(): frozen[key]=identity(path)
    atomic_json(root/'provenance.json',dict(config_digest=digest(cfg),frozen_identities=frozen,sources=sources,generated_utc=utc()))
    # A packaging receipt is not a scientific node completion claim.
    atomic_json(root/'package.json',dict(schema='boundary-three-package-v1',status='PACKAGED',time=utc(),
        minimum_three_complete=minimum,remaining_nodes=[r['node'] for r in remaining],
        config_digest=digest(cfg),sources=sources,outputs={p.name:identity(p) for p in sorted(root.iterdir()) if p.is_file() and p.name!='package.json'}))
    return dict(status='PACKAGED',minimum_three_complete=minimum,remaining_nodes=[r['node'] for r in remaining])
