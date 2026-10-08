"""Import measured mini stage internals without inventing missing CPU samples."""
import argparse,json
from pathlib import Path
from scripts.crgf.common import atomic_json,identity,process_identity
from scripts.cos.runner import stage_specs

def create(queuepath):
    q=json.loads(Path(queuepath).read_text());root=Path(q['output_root']);state=dict(run_id=q['run_id'],status='mini_evaluation_pending',runner=process_identity(),stages={},resource_scope='Training/prediction/postprocess are actual internal stage timers; matrix CPU is measured. Front subprocess startup/CPU peaks were not sampled. Evaluation runner adds its measured process-group peak. This receipt is mini function-test accounting, not full-field cost.')
    for spec in stage_specs(q):
        p=Path(spec['receipt']);rec=dict(**spec,status='pending',attempts=[])
        if spec['stage'] in ['train','predict','postprocess','matrix']:
            d=json.loads(p.read_text())
            if spec['stage']=='matrix':
                resource=json.loads((p.parent/'resources.json').read_text());wall=resource['wall_seconds']
                timing=d['timings'];attempt=dict(wall_seconds=wall,sampled_peak_rss_bytes_sum=max(x['rss'] for x in resource['samples']),sampled_peak_pss_bytes_sum=max(x['pss'] for x in resource['samples']),memory_sampling_scope=resource['scope'],resource_receipt=identity(p.parent/'resources.json'),wall_scope='matrix aggregation Monitor scope',exit_code=0)
            else:
                attempt=dict(wall_seconds=d['wall_seconds'],wall_scope='stage complete internal timer, excludes Python import and data/model setup',memory_sampling_scope='CPU not sampled for this mini stage; real model CUDA allocator peak in stage receipt',peak_vram_bytes=d.get('peak_vram_bytes'),exit_code=0)
            rec.update(status='succeeded',attempts=[attempt],receipt_sha256=identity(p)['sha256'])
        state['stages'][spec['name']]=rec
    atomic_json(root/'mini_front_costs.json',state)
    atomic_json(root/'mini_cost_scope.json',dict(scope=state['resource_scope'],sources=[identity(v['receipt']) for v in state['stages'].values() if v['status']=='succeeded']))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--queue',required=True);a=p.parse_args();create(a.queue)
