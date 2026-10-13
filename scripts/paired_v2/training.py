"""Complete candidate FP32 Adam with frozen step-zero and exact state resume."""
import copy
import gc
import time
import imgaug as ia
from .common import *
from .data import PairedViews


def initial_header(cfg):
    return dict(version=VERSION, run_id=cfg['run_id'], model_kind=cfg['model_kind'],
                source=producer('initial'), shared_initial=identity(INPUT/'prepared/shared_initial.pth'),
                prepared=identity(INPUT/'prepared/manifest.json'), views=identity(INPUT/'configs/views.json'),
                contract=cfg['contract'], seed=1234, morphology_seed=20261013, interaction_seed=20261014,
                optimizer=dict(name='Adam', lr=1e-5, betas=[.9, .999], eps=1e-8, weight_decay=1e-4),
                precision='FP32; no AMP; no TF32; full K; original BN')


def setup(cfg, device='cuda'):
    # Lazy library initialization precedes the frozen seed/model constructor.
    ia.random.get_global_rng()
    model = make_model(cfg['model_kind'], device); opt = optimizer(model)
    ia.seed(1234)
    data = PairedViews(cfg['model_kind'])
    if cfg['model_kind'] == 'BASE':
        # A distinct read-only aux source; it never supplies the train forward.
        saved = rng_state()
        try:
            diagnostic_data = PairedViews('BASE', diagnostic=True)
        finally:
            restore_rng(saved)
    else:
        diagnostic_data = data
    return model, opt, data, diagnostic_data


def checkpoint_header(cfg):
    return dict(**initial_header(cfg), initial_snapshot=identity(cfg['initial_snapshot']),
                config=identity(cfg['config_path']), input_audit=identity(Path(cfg['runroot'])/'prepared/input_audit.json'))


def snapshot(model, opt, header, updates, cursor, skips, sequence, pilot=False):
    return dict(header=header, model=model.state_dict(), optimizer=opt.state_dict(), rng=rng_state(),
                actual_updates=updates, view_cursor=cursor, empty_skips=skips, effective_sequence=sequence,
                sequence_digest=digest(sequence), bn_in_model_state=True, pilot=pilot)


def restore(path, model, opt, header):
    state = torch.load(path, map_location='cpu')
    if state['header'] != header:
        raise ValueError('checkpoint source/config/input contract differs; no resume')
    model.load_state_dict(state['model'], strict=True); opt.load_state_dict(state['optimizer'])
    restore_rng(state['rng'])
    if state['sequence_digest'] != digest(state['effective_sequence']):
        raise ValueError('checkpoint effective sequence changed')
    validate_cursor(state)
    return state


def validate_cursor(state):
    views = read(INPUT/'configs/views.json')['views']; cursor = state['view_cursor']
    if not 0 <= cursor <= len(views):
        raise ValueError('checkpoint cursor outside frozen sequence')
    used = views[:cursor]
    if state['actual_updates'] != sum(v['K'] > 0 for v in used) or state['empty_skips'] != sum(v['K'] == 0 for v in used):
        raise ValueError('cursor/update/empty count mismatch')
    rows = state['effective_sequence']
    if len(rows) != state['actual_updates']:
        raise ValueError('effective sequence length mismatch')
    hashes = {v['path']: v['identity']['sha256'] for v in read(INPUT/'configs/predict_shift0.json')}
    for number, (v, row) in enumerate(zip((v for v in used if v['K']), rows), 1):
        expected = sequence_identity(v, number, hashes[v['path']])
        if any(row.get(k) != val for k, val in expected.items()):
            raise ValueError('effective sequence does not match frozen visit '+str(number))


def sequence_identity(v, update, patch_sha):
    return dict(update=update, cursor=v['cursor']+1, index=v['index'], path=v['path'], row=v['row'], col=v['col'],
                candidate_ids=v['candidate_ids'], K=v['K'], flip=v['flip'], rot90=v['rot90'], patch_sha256=patch_sha)


def rewrite_sequence(path, rows):
    path = Path(path); tmp = path.with_suffix('.tmp')
    with tmp.open('w') as f:
        for row in rows:
            f.write(json.dumps(row, allow_nan=False)+'\n')
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def train(config, pilot_updates=None, pilot_root=None):
    cfg = load_config(config); check_environment(); pilot = pilot_updates is not None
    from .prepare import verify_inputs
    verify_inputs(Path(cfg['runroot']), full=False)
    if pilot:
        if pilot_root is None or Path(cfg['runroot'])/'tests' not in Path(pilot_root).resolve().parents:
            raise ValueError('pilot output must be under this runroot/tests')
        target = int(pilot_updates)
        if not 1 <= target <= 100:
            raise ValueError('admission pilot may run at most 100 updates')
        output = Path(pilot_root)
    else:
        gate = read(Path(cfg['runroot'])/'admission'/f"{cfg['run_id']}.json")
        if gate.get('status') != 'admitted' or gate.get('train_producer') != producer('train'):
            raise ValueError('this arm has no current independently admitted train producer')
        target = cfg['updates']; output = Path(cfg['output_root'])
    output.mkdir(parents=True, exist_ok=True)
    out = output/'checkpoints'; out.mkdir(exist_ok=True)
    header = checkpoint_header(cfg)
    model, opt, data, diagnostic_data = setup(cfg)
    initial = torch.load(cfg['initial_snapshot'], map_location='cpu')
    if initial['header'] != initial_header(cfg) or initial['actual_updates'] != 0 or initial['optimizer']['state']:
        raise ValueError('formal step-zero contract mismatch')
    model.load_state_dict(initial['model'], strict=True); opt.load_state_dict(initial['optimizer']); restore_rng(initial['rng'])
    del initial
    updates = cursor = skips = 0; sequence = []
    if (out/'latest.pth').exists():
        state = restore(out/'latest.pth', model, opt, header)
        if bool(state['pilot']) != pilot:
            raise ValueError('pilot/formal checkpoint mixing prohibited')
        updates, cursor, skips = state['actual_updates'], state['view_cursor'], state['empty_skips']
        sequence = state['effective_sequence']; del state
    if updates > target:
        raise ValueError('cannot rewind checkpoint')
    log = output/'effective_sequence.jsonl'
    # Remove attempts after the last durable checkpoint; all committed rows are
    # recovered from that checkpoint, so a resumed file has no duplicate updates.
    if log.exists():
        previous = [json.loads(line) for line in log.read_text().splitlines()]
        if previous[:len(sequence)] != sequence:
            raise ValueError('durable sequence prefix changed')
    rewrite_sequence(log, sequence)
    views = read(INPUT/'configs/views.json')['views']
    hashes = {v['path']: v['identity']['sha256'] for v in read(INPUT/'configs/predict_shift0.json')}
    base = BaseSix('cuda'); model.train(); started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    attempt = output/f'training_attempt_{time.time_ns()}.jsonl'
    with log.open('a', buffering=1) as stream, attempt.open('w', buffering=1) as attempts:
        while updates < target:
            if cursor >= len(views):
                raise ValueError('frozen visits exhausted')
            view = views[cursor]; begin = time.monotonic(); batch = data.get_view(view, 'cuda'); cursor += 1
            if not view['K']:
                skips += 1; continue
            opt.zero_grad(set_to_none=True)
            z = forward(model, batch, cfg['model_kind']); loss, parts = base(z, batch)
            if not torch.isfinite(z).all() or not torch.isfinite(loss):
                raise FloatingPointError('nonfinite logits/loss')
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise FloatingPointError('nonfinite gradient')
            if updates == 0 and hasattr(model, 'interaction'):
                first_gradient = {n: float(p.grad.double().norm()) if p.grad is not None else 0.
                                  for n, p in model.interaction.named_parameters()}
                atomic_json(output/'first_interaction_gradients.json', first_gradient)
                if first_gradient['output.weight'] <= 0:
                    raise ValueError('first interaction output projection has no gradient')
            opt.step(); updates += 1; torch.cuda.synchronize()
            row = sequence_identity(view, updates, hashes[view['path']])
            row.update(base6=float(loss.detach()), base_terms={k:float(v.detach()) for k,v in parts.items()},
                       seconds=time.monotonic()-begin, empty_skips=skips,
                       morphology=copy.deepcopy(getattr(getattr(model,'morphology',None),'diagnostics',{})),
                       interaction=copy.deepcopy(getattr(getattr(model,'interaction',None),'diagnostics',{})))
            sequence.append(row); stream.write(json.dumps(row)+'\n'); attempts.write(json.dumps(row)+'\n')
            del z, loss, parts, batch
            boundaries = (2,32,64,100) if pilot and cfg['model_kind']=='V2_INTERACT' else (100,1000,2000,3625)
            if updates in boundaries:
                from .diagnostics import probes
                probes(model, opt, diagnostic_data, cfg, updates, output/'diagnostics')
            if updates % 25 == 0 or updates == target:
                stream.flush(); os.fsync(stream.fileno())
                atomic_torch(out/'latest.pth', snapshot(model,opt,header,updates,cursor,skips,sequence,pilot))
            atomic_json(out/'progress.json', dict(status='running',run_id=cfg['run_id'],pilot=pilot,
                actual_updates=updates,view_cursor=cursor,empty_skips=skips,total=target,
                elapsed_seconds=time.monotonic()-started,peak_vram_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(),process=process_identity()))
            if updates % 10 == 0:
                print(cfg['run_id'],updates,'/',target,'cursor',cursor,'loss',row['base6'],flush=True)
    final = snapshot(model,opt,header,updates,cursor,skips,sequence,pilot); validate_cursor(final)
    if not pilot and (updates,cursor,skips)!=(3625,3998,373):
        raise ValueError('formal endpoint budget mismatch')
    atomic_torch(out/'final.pth',final)
    rec = dict(status='succeeded',pilot=pilot,run_id=cfg['run_id'],model_kind=cfg['model_kind'],dependencies=header,
        actual_updates=updates,view_cursor=cursor,empty_skips=skips,wall_seconds=time.monotonic()-started,
        peak_vram_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        parameter_count=sum(p.numel() for p in model.parameters()),
        outputs={'checkpoint':identity(out/'final.pth'),'sequence':identity(log)})
    atomic_json(out/'complete.json',rec)
    return rec
