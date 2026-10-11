"""Exact same real logits CPU scalar/logit-gradient math comparison, unchanged functions."""
import argparse,subprocess,sys,json
from pathlib import Path
import torch
from scripts.bs4.training import losses
from scripts.cos.tests.real_model import tensor_hashes
from scripts.crgf.common import atomic_json,identity

def main(root):
    root=Path(root);torch.set_num_threads(2)
    # GPU equivalence exports real forward logits from both actual candidate/type cases.
    path=root/'tests/cpu_loss_fixtures.pth'
    subprocess.run([sys.executable,str(Path(__file__).with_name('cpu_reference.py')),str(path),str(root/'tests/historical_cpu_loss.json')],check=True)
    reference=json.loads((root/'tests/historical_cpu_loss.json').read_text());actual={}
    for name,(z,sample) in torch.load(path,map_location='cpu').items():
        z=z.detach().requires_grad_();loss,parts=losses(z,sample,True,34.61094491628047,s_enabled=False);g=torch.autograd.grad(loss,z)[0]
        actual[name]=dict(loss=float(loss),parts={k:float(v) for k,v in parts.items()},gradient=tensor_hashes(g))
    assert actual==reference['fixtures'],'CPU real-logits exact legacy math mismatch'
    atomic_json(root/'tests/cpu_equivalence.json',dict(status='passed',all_components_total_and_output_gradients_bitwise=True,cases=list(actual),historical_import=reference['source'],fixtures=identity(path),old_reference=identity(root/'tests/historical_cpu_loss.json'),source=identity(__file__)))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();main(a.root)
