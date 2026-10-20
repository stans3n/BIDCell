"""Read-only fixed probes, including a separately supplied BASE count sidecar."""
import copy
import torch
from scripts.new_methods.diagnostics import soft_hard_diagnostics, structure_diagnostics
from scripts.new_methods.training import capture
from bidcell.model.model.region_losses import native_owner
from .common import *


def optimizer_sha(opt):
    state = opt.state_dict()
    tensors = {f'{i}/{k}': v for i, s in state['state'].items() for k, v in s.items() if torch.is_tensor(v)}
    return digest(dict(tensors=tensor_sha(tensors), groups=state['param_groups']))


def probes(model, opt, data, cfg, updates, out):
    saved_rng = rng_state(); mode = model.training
    data_state = (data.forced_transform, data.recorded)
    before = dict(model=tensor_sha(model.state_dict()), optimizer=optimizer_sha(opt),
                  grads=tensor_sha({n: p.grad for n, p in model.named_parameters() if p.grad is not None}))
    rows = []; model.eval()
    base = BaseSix(next(model.parameters()).device)
    initial = torch.load(cfg['initial_snapshot'], map_location='cpu')['model']
    try:
        for v in read(INPUT/'configs/probes.json')['morphology']:
            b = data.get_view(v, next(model.parameters()).device)
            z = forward(model, b, cfg['model_kind']); loss, parts = base(z, b)
            names, params = zip(*model.named_parameters())
            gs = torch.autograd.grad(loss, params, allow_unused=True)
            groups = {}
            for n, p, g in zip(names, params, gs):
                group = n.split('.')[0] if n.startswith(('morphology.', 'interaction.')) else 'backbone'
                rec = groups.setdefault(group, dict(gradient_squared=0., parameter_delta_squared=0., tensors=[]))
                if g is not None and not torch.isfinite(g).all():
                    raise FloatingPointError('nonfinite probe gradient: '+n)
                norm = float(g.double().square().sum()) if g is not None else 0.
                rec['gradient_squared'] += norm
                rec['parameter_delta_squared'] += float((p.detach().cpu()-initial[n]).double().square().sum())
                if group == 'interaction':
                    rec['tensors'].append(dict(name=n, gradient_norm=norm**.5,
                        delta_norm=float((p.detach().cpu()-initial[n]).double().norm())))
            row = dict(view=v, base6=float(loss.detach()), gradient_groups=groups,
                       capture=capture(z, b), soft_hard=soft_hard_diagnostics(z, b),
                       structure=structure_diagnostics(z, b),
                       morphology=copy.deepcopy(getattr(getattr(model, 'morphology', None), 'diagnostics', {})),
                       interaction=copy.deepcopy(getattr(getattr(model, 'interaction', None), 'diagnostics', {})))
            if hasattr(model, 'interaction'):
                module = model._modules.pop('interaction')
                try:
                    with torch.no_grad():
                        disabled = forward(model, b, cfg['model_kind'])
                finally:
                    model.add_module('interaction', module)
                changed = native_owner(z) != native_owner(disabled)
                coarse = torch.nn.functional.max_pool2d(b['aux']['native_support'][:, None].float(), 4, 4)[:, 0] > 0
                neighbor = (coarse.sum(0) >= 2)[None, None].float()
                supported = torch.nn.functional.interpolate(neighbor, size=(48, 48), mode='nearest')[0, 0].bool()
                row['interaction_disabled'] = dict(logits_l2=float((z-disabled).detach().norm()),
                    owner_changes=int(changed.sum()), owner_changes_outside_coarse_overlap=int((changed & ~supported).sum()),
                    logits_changed_outside_coarse_overlap=int(((z-disabled).detach()[:, :, ~supported] != 0).sum()))
            rows.append(row)
            del b, z, loss, gs
    finally:
        restore_rng(saved_rng); model.train(mode)
        data.forced_transform, data.recorded = data_state
    after = dict(model=tensor_sha(model.state_dict()), optimizer=optimizer_sha(opt),
                 grads=tensor_sha({n: p.grad for n, p in model.named_parameters() if p.grad is not None}))
    if before != after or not state_equal(saved_rng, rng_state()):
        raise ValueError('diagnostics mutated model/BN/Adam/gradients/RNG')
    rec = dict(status='passed', updates=updates, purity_before=before, purity_after=after,
               exact_rng_preserved=True, next_view_is_external_frozen_cursor=True, rows=rows,
               producer=producer('train'))
    atomic_json(Path(out)/f'probe_{updates:04d}.json', json_safe(rec))
    return rec
