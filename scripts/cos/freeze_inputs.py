"""Before admission only: bind every actually-read arm config and shared protocol."""
import argparse,json
from pathlib import Path
from scripts.crgf.common import atomic_json,identity
from .config import ARMS

def freeze(root):
    root=Path(root);queue=json.loads((root/'configs/queue.json').read_text())
    if (root/'runner_identity.json').exists():raise ValueError('cannot refreeze launched run')
    protocol=json.loads(Path(queue['paper_protocol']).read_text())
    protocol.update(version='COS-four-arm-paper-aligned-v1',comparison_set=list(ARMS),comparisons=[['B1_COS','B1_CONT'],['G_COS','G_CONT']])
    atomic_json(root/'configs/evaluation_protocol.json',protocol);queue['paper_protocol']=str(root/'configs/evaluation_protocol.json')
    atomic_json(root/'configs/queue.json',queue)
    for arm,item in queue['methods'].items():
        ar=Path(item['cost_root']);f=ar/'configs/frozen.json';frozen=json.loads(f.read_text())
        frozen['identities']['protocol']=identity(queue['paper_protocol'])
        for p in (ar/'configs').glob('*.json'):
            if p.name not in ('frozen.json','queue.json'):frozen['identities']['arm_'+p.name]=identity(p)
        inputs=json.loads((ar/'configs/inputs.json').read_text())
        for key,obj in inputs.items():
            if isinstance(obj,dict) and 'path' in obj and 'sha256' in obj:frozen['identities']['data_'+key]=obj
        machine=json.loads(Path(item['machine']).read_text())
        for key in ('baseline_final_mask','nuclei_mask','preannotation','genes','transcripts','reference','positive_markers','negative_markers'):
            frozen['identities']['machine_input_'+key]=identity(machine[key])
        atomic_json(f,frozen)
    print(root/'configs/queue.json')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();freeze(a.root)
