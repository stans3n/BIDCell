"""Explicit variant stage entry; invoked in its frozen worktree."""
import argparse,json,subprocess
from pathlib import Path
from .config import read
from .common import atomic_json,identity

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--stage',choices=['train','predict','postprocess','matrix'],required=True);a=p.parse_args()
    cfg=read(a.config);method=cfg.variant;root=Path(cfg.output_root)
    if method not in ('E','G','R'):raise ValueError('explicit variant required')
    contract=json.loads((Path(__file__).parent/'configs/variant_contract.json').read_text())
    if contract['variant']!=method:raise ValueError('worktree variant contract mismatch')
    committed=json.loads((Path(__file__).parent/f'configs/egr_{method}.json').read_text())
    if cfg.model_dump(mode='json')!=committed:raise ValueError('runtime config differs from committed variant config')
    checkout=Path(__file__).resolve().parents[2]
    branch=subprocess.check_output(['git','branch','--show-current'],cwd=checkout,text=True).strip()
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=checkout,text=True).strip()
    binding=json.loads((root/'source/binding.json').read_text())
    if binding!={'variant':method,'worktree':str(checkout),'branch':branch,'commit':commit}:raise ValueError('frozen branch binding mismatch')
    atomic_json(root/f'source/imports_{a.stage}.json',dict(binding=binding,entry=__file__,model=__import__('bidcell.model.model.model',fromlist=['x']).__file__))
    if a.stage=='train':
        from .training import train
        train(cfg,method)
    elif a.stage=='predict':
        from .prediction import predict
        predict(cfg,method)
    elif a.stage=='postprocess':
        from .postprocess import postprocess
        postprocess(cfg,method)
    elif a.stage=='matrix':
        from .matrix import build
        machine=json.loads((root/f'configs/{method}_machine.json').read_text())
        build(machine,root/'matrix'/method,cache=root.parent/'shared/coordinates')
if __name__=='__main__':main()
