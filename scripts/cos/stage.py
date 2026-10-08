"""Frozen arm entrypoint; strict binding before every production stage."""
import argparse,json,subprocess
from pathlib import Path
from .config import read
from scripts.crgf.common import atomic_json,sha256,identity

def verify_frozen(cfg,stage):
    root=Path(cfg.output_root)
    frozen=json.loads((root/'configs/frozen.json').read_text())
    for obj in frozen['identities'].values():
        if sha256(obj['path'])!=obj['sha256']: raise ValueError('frozen dependency changed '+obj['path'])
    if stage in ('train','predict'):
        for shift in cfg.shifts:
            inventory=json.loads((root/f'configs/patches_shift{shift}.json').read_text())
            for obj in inventory:
                if sha256(obj['path'])!=obj['sha256']:raise ValueError('patch content hash changed '+obj['path'])

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--stage',required=True,choices=['train','predict','postprocess','matrix','evaluate']);args=parser.parse_args()
    cfg=read(args.config);root=Path(cfg.output_root);arm=cfg.arm_id
    checkout=Path(__file__).resolve().parents[2]
    if not cfg.mini_test:
        binding=json.loads((root/'source/binding.json').read_text())
        actual=dict(arm_id=arm,worktree=str(checkout),branch=subprocess.check_output(['git','branch','--show-current'],cwd=checkout,text=True).strip(),commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=checkout,text=True).strip())
        if binding!=actual: raise ValueError('frozen checkout binding changed')
        committed=json.loads((checkout/'scripts/cos/configs/arm.json').read_text())
        if committed!=cfg.model_dump(mode='json'): raise ValueError('committed config changed')
        dirty=subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=checkout,text=True)
        if dirty.strip(): raise ValueError('frozen source dirty')
    verify_frozen(cfg,args.stage)
    atomic_json(root/f'source/imports_{args.stage}.json',dict(entry=__file__,model=__import__('bidcell.model.model.model',fromlist=['x']).__file__))
    if args.stage=='train':
        from .training import train
        train(cfg,arm)
    elif args.stage=='predict':
        from .prediction import predict
        predict(cfg,arm)
    elif args.stage=='postprocess':
        from .postprocess import postprocess
        postprocess(cfg,arm)
    elif args.stage=='matrix':
        from scripts.crgf.matrix import build
        build(json.loads((root/f'configs/{arm}_machine.json').read_text()),root/'matrix'/arm,cache=root.parent/'shared/coordinates')
    else:
        from .evaluation import evaluate
        evaluate(cfg)
if __name__=='__main__':main()
