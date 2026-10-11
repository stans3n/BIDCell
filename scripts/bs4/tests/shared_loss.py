"""Single-call historical provenance, not a larger CUDA scalar envelope."""
import argparse,json
from pathlib import Path
from unittest.mock import patch
import torch
from scripts.bs4 import training
from scripts.bs4.config import read
from scripts.bs4.loss_diagnostics import get
from scripts.bs4.method import teacher,replace_pos,frozen_teacher_hash
from scripts.cos.tests.real_model import tensor_hashes
from scripts.crgf.common import atomic_json,source_identity,identity

def run(root):
    root=Path(root);sources=training.TRAIN_SOURCES+['scripts/bs4/tests/shared_loss.py'];execution=source_identity(sources)
    torch.set_num_threads(2);cfg=read(root/'T1/configs/run.json');ds=training.dataset(cfg)
    model=training.make_model(cfg,'T1',len(ds.gene_names));model.eval()
    views=json.loads((root/'configs/views.json').read_text())['views']
    v=next(v for v in views if v['k']>=2 and any(ds.type_by_id.get(c,-1)<0 for c in v['cell_ids']))
    sample=get(ds,v);z=training.forward(model,sample);raw={k:(val.detach().cpu() if torch.is_tensor(val) else val) for k,val in sample.items() if k!='_teacher'}
    original=training.original_cos_training.losses;rows=[]
    for name,enabled,alpha,beta,empty in [('disabled',False,0.,1.,False),('zero_coefficients',True,0.,0.,False),('empty_H',True,0.,1.,True),('T1',True,0.,1.,False),('T2',True,.25,1.,False)]:
        cpu=dict(raw,alpha=alpha,beta=beta)
        if empty:cpu['exclusive_positive']=torch.zeros_like(cpu['exclusive_positive'])
        s=training.to_device(cpu);captured=[]
        def capture(*args,**kwargs):
            result=original(*args,**kwargs);captured.append(result);return result
        with patch.object(training.original_cos_training,'losses',side_effect=capture):total,parts=training.losses(z,s,True,cfg.anchor_weight,s_enabled=enabled)
        assert len(captured)==1
        old_total,old_parts=captured[0]
        common={k:parts[k] is old_parts[k] for k in old_parts if k!='pos'}
        assert all(common.values())
        target=teacher(s);changed=enabled and bool(target['replace'].any())
        if not changed:
            assert total is old_total and parts is old_parts
            expected=old_total
        else:
            assert parts['pos'] is not old_parts['pos']
            expected_parts=dict(old_parts,pos=replace_pos(old_parts['pos'],z,s,target))
            expected=sum(val for key,val in expected_parts.items() if key!='anchor')+cfg.anchor_weight*expected_parts['anchor']
            assert torch.equal(total,expected)
        actual_gradient=torch.autograd.grad(total,z,retain_graph=True)[0]
        expected_gradient=torch.autograd.grad(expected,z,retain_graph=True)[0]
        assert torch.equal(actual_gradient,expected_gradient)
        rows.append(dict(case=name,original_calls=1,non_POS_same_tensor_objects=common,identity_returns_original_total_and_dictionary=not changed,replace_pixels=int(target['replace'].sum()),same_call_original_scalars={k:float(val) for k,val in old_parts.items()},actual_scalars={k:float(val) for k,val in parts.items()},gradient=tensor_hashes(actual_gradient),teacher=frozen_teacher_hash(target)))
    assert all(next(r for r in rows if r['case']==arm)['replace_pixels']>0 for arm in ('T1','T2'))
    failures=[identity(root/f'tests/mixed_cuda_reduction_v{n}.json') for n in (2,3)]
    if source_identity(sources)!=execution:raise ValueError('shared-loss execution source drift')
    atomic_json(root/'tests/shared_loss_v1.json',dict(status='passed',execution_source=execution,patch=v,logits=tensor_hashes(z.detach()),model=tensor_hashes(model.state_dict()),cases=rows,prior_failed_implementation_evidence=failures,prior_failures_not_reclassified=True,contract='one untouched historical loss invocation; same tensor objects for all common contributions; unchanged identity total; original addition order with only POS replaced',uses_scalar_tolerance=False,expands_scalar_envelope=False))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);run(p.parse_args().root)
