"""Actual frozen-input adapter and first semantic-chunk timing, isolated only."""
import argparse,fcntl,time
from pathlib import Path
from scripts.rep1_triple_simple_boundary import inputs
from scripts.rep1_triple_simple_boundary.common import *

def run(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    inspected='/home/zwan0726/work/docs/BIDCell_Rep1_Triple_Simple_Inputs_20261017.json';plan='/home/zwan0726/work/docs/BIDCell_Rep1_Triple_Simple_PlanManifest_20261017.json'
    with Path('/home/zwan0726/work/outputs/cos_gpu.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);t=time.monotonic();manifest=inputs.freeze_inputs(out,inspected,plan);prepare_seconds=time.monotonic()-t
        cfg=dict(run_id='isolated_input_adapter_fixture',run_root=str(out),source_digest='fixture_not_formal_snapshot',input_digest=digest(manifest),domain_digest='no_formal_domain_prepared',plan_digest=sha256(plan))
        t=time.monotonic();loaded=inputs.load_manifest(out/'configs/inputs_manifest.json');hash_seconds=time.monotonic()-t
        if loaded!=manifest:raise AssertionError('frozen inputs round trip mismatch')
        events=[]
        def first(event):
            events.append(event);raise InterruptedError('intentional stop after first actual endpoint semantic receipt')
        t=time.monotonic()
        try:inputs.validate_semantics(cfg,manifest,first)
        except InterruptedError:pass
        else:raise AssertionError('first semantic-chunk fixture did not stop')
        semantic_seconds=time.monotonic()-t
        if len(events)!=1 or not Path(events[0]['detail']['receipt']).is_file():raise AssertionError('no semantic witness')
    result=dict(status='passed',scope='input freeze plus first OLD BASE full-ID/nuclear/matrix semantic receipt only; no fixed domain or new endpoint prepared',
        files=len(manifest['files']),input_bytes=sum(x['bytes'] for x in manifest['files'].values()),freeze_seconds=prepare_seconds,
        full_manifest_rehash_seconds=hash_seconds,first_semantic_receipt_seconds=semantic_seconds,total_stage01_before_first_semantic_seconds=hash_seconds+semantic_seconds,
        coordinate_cache_full_producer_identity=manifest['coordinate_cache_validation'],legacy_BT_profiles=[k for k,v in manifest['endpoints'].items() if v.get('official_bt_root')],
        old_spatial_trace=manifest['old_spatial_trace'],old_BT_trace=manifest['old_official_BT_trace'],semantic_witness=identity(events[0]['detail']['receipt']),outputs=inventory(out))
    atomic_json(out/'fixture_receipt.json',result);print({k:result[k] for k in ('status','files','input_bytes','freeze_seconds','full_manifest_rehash_seconds','first_semantic_receipt_seconds','total_stage01_before_first_semantic_seconds')},flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();run(a.output)
