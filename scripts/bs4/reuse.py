"""Read-only historical B endpoints; never manufacture P training receipts."""
import json
from pathlib import Path
import tifffile
from scripts.crgf.common import identity,sha256,atomic_json,verify_outputs

def prediction(cfg):
    source=Path(cfg.reused_prediction_root);out=Path(cfg.output_root)/'predictions'/cfg.arm_id;out.mkdir(parents=True,exist_ok=True)
    historical=json.loads((source/'endpoints.json').read_text());verify_outputs(json.loads((source/'postprocess/complete.json').read_text()))
    records={}
    for name in ('raw_shift0','shift0','raw_shift24','shift24','pre'):
        obj=historical['endpoints'][name];src=Path(obj['path'])
        if sha256(src)!=obj['sha256']:raise ValueError('historical B endpoint drift')
        dst=out/(name+'.tif')
        if dst.is_symlink():
            if dst.resolve()!=src.resolve():raise ValueError('reuse symlink drift')
        elif dst.exists():raise ValueError('refuse replace existing reused endpoint')
        else:dst.symlink_to(src)
        records[name]=dict(status='reused',historical=obj,output=identity(dst))
    # Frozen B prediction probes are provenance, not new model evaluation.
    probes=out/'probes'
    if not probes.exists():probes.symlink_to(source/'probes',target_is_directory=True)
    value=dict(version='BS4-reused-B-v2',arm=cfg.arm_id,training='not_required',prediction='reused',historical_endpoints=identity(source/'endpoints.json'),historical_postprocess=identity(source/'postprocess/complete.json'),records=records)
    path=Path(cfg.output_root)/'reused_input_manifest.json'
    if path.exists():
        if json.loads(path.read_text())!=value:raise ValueError('reused provenance changed')
    else:atomic_json(path,value)
    return value
