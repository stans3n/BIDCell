"""Strict source and input binding for every SSP stage."""
import argparse,json,os
from pathlib import Path
from scripts.crgf.common import atomic_json,sha256
from .resources import require_space

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--stage',required=True,choices=['train','predict','postprocess','matrix','evaluate']);a=p.parse_args()
    threads=1 if a.stage=='postprocess' else 2 if a.stage in ('train','predict') else 4
    for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]=str(threads)
    from .config import read
    cfg=read(a.config);root=Path(cfg.output_root);arm=cfg.arm_id
    from scripts.cos.stage import verify_frozen
    verify_frozen(cfg,a.stage)
    if not cfg.mini_test:
        from .runner import preflight
        queue=json.loads(Path(cfg.queue_config).read_text());preflight(cfg.queue_config,queue)
    require_space(root,cfg.mini_test)
    atomic_json(root/f'source/imports_{a.stage}.json',dict(entry=__file__,model=__import__('bidcell.model.model.model',fromlist=['x']).__file__))
    if arm in ('P1','P2') and a.stage in ('train','predict'):raise ValueError('P train/predict forbidden')
    if arm in ('P1','P2') and a.stage=='postprocess':
        from .reuse import prediction
        prediction(cfg)
    if a.stage=='train':
        from .training import train
        train(cfg,arm)
    elif a.stage=='predict':
        from .prediction import predict
        predict(cfg,arm)
    elif a.stage=='postprocess':
        from .postprocess import postprocess
        postprocess(cfg,arm)
    elif a.stage=='matrix':
        from scripts.crgf.matrix import build
        build(json.loads((root/f'configs/{arm}_machine.json').read_text()),root/'matrix'/arm,cache=root.parent/'shared/coordinates',space_guard=lambda:require_space(root,cfg.mini_test))
    else:
        from .evaluation import evaluate
        evaluate(cfg)
if __name__=='__main__':main()
