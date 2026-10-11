"""Real regional historical baselines plus all four real arms exercise the full report."""
import argparse,copy,json,shutil,subprocess
from pathlib import Path
import numpy as np,pandas as pd,tifffile
from scripts.bs4.comparison import aggregate
from scripts.bs4.config import read as config_read
from scripts.bs4.evaluation import run_profiles,publish
from scripts.bs4.diagnostics import stage_diagnostics
from scripts.bs4.postprocess import postprocess
from scripts.bs4.reuse import prediction
from scripts.crgf.matrix import build
from scripts.crgf.common import atomic_json,identity,source_identity,sha256,verify_outputs
SOURCES=['scripts/bs4/evaluation.py','scripts/crgf/matrix.py','scripts/bs4/comparison.py','scripts/bs4/stage_population.py','scripts/bs4/post_effects.py','scripts/bs4/tradeoffs.py','scripts/bs4/diagnostics.py','scripts/bs4/tables.py','scripts/bs4/tests/deployment_baselines.py']

def run(root,mini_path=None,receipt_path=None,prepare_queue_only=False):
    root=Path(root);execution=source_identity(SOURCES);formal=json.loads((root/'configs/queue.json').read_text());mini=Path(mini_path) if mini_path else root/'tests/mini_v5';q=json.loads((mini/'configs/queue.json').read_text());fixture=json.loads((mini/'fixture.json').read_text());y,x=fixture['origin'];size=fixture['size']
    historical={};baseline_receipts={};template=q['methods']['P1'];cfg0=config_read(template['config']);original_protocol=json.loads((Path(q['prepared'])/'complete.json').read_text())['dependencies']['protocol']['path']
    for arm in ('B','S'):
        ar=mini/'diagnostics'/('B_native_replay' if arm=='B' else 'S_historical_crop');out=ar/'predictions'/arm;out.mkdir(parents=True,exist_ok=True);(ar/'configs').mkdir(exist_ok=True)
        cfg=cfg0.model_copy(update=dict(arm_id='B',output_root=str(ar),post_rule='B'))
        old=Path(formal['historical_methods'][arm]['cost_root'])/'predictions'/arm/'endpoints.json';entries=json.loads(old.read_text())['endpoints'];cropped={}
        for name,obj in entries.items():
            if sha256(obj['path'])!=obj['sha256']:raise ValueError('historical native endpoint drift')
            dest=out/(name+'.tif');value=tifffile.memmap(obj['path'])[y:y+size,x:x+size]
            if not dest.exists():tifffile.imwrite(dest,value)
            elif not np.array_equal(value,tifffile.memmap(dest)):raise ValueError('regional native endpoint drift')
            cropped[name]=identity(dest)
        if arm=='B':
            from scripts.bs4.postprocess import validate_tile
            full=root/'diagnostics/B_native_replay/predictions/B/postprocess'
            height,width=tifffile.memmap(entries['pre']['path']).shape
            coords=[(yy,xx) for yy in list(range(0,height-1024,1024))+[height-1024] for xx in list(range(0,width-1024,1024))+[width-1024]]
            overlapping=[(yy,xx) for yy,xx in coords if yy<y+size and yy+1024>y and xx<x+size and xx+1024>x]
            if len(overlapping)!=1:raise ValueError('mini canonical crop crosses/overlaps native tiles')
            yy,xx=overlapping[0];tile=full/f'tile_{yy}_{xx}.json';receipt=json.loads(tile.read_text());validate_tile(tile,receipt['dependencies'])
            if receipt['arm']!='B' or receipt['dependencies']['base']['rule']!='B' or receipt['dependencies']['base']['pre']['sha256']!=entries['pre']['sha256']:raise ValueError('B canonical tile provenance')
            dest=out/'postconnect_canonical.tif';value=tifffile.memmap(receipt['outputs']['postconnect']['path'])[y-yy:y-yy+size,x-xx:x-xx+size]
            if not dest.exists():tifffile.imwrite(dest,value)
            elif not np.array_equal(value,tifffile.memmap(dest)):raise ValueError('canonical crop drift')
            cropped['postconnect_canonical']=identity(dest)
            atomic_json(ar/'validation.json',dict(status='passed',scope='historical full-native B endpoint crop plus strict production native postconnect tile; regional report mechanics',canonical_tile=identity(tile),input=identity(old)))
        atomic_json(out/'endpoints.json',dict(endpoints=cropped,historical_source=identity(old),origin=[y,x]))
        machine=json.loads(Path(template['machine']).read_text());machine.update(final_mask=str(out/'final.tif'),pre_mask=str(out/'pre.tif'),final_matrix=str(ar/'matrix'/arm/'expr_mat.csv'))
        machinepath=ar/'configs/machine.json';atomic_json(machinepath,machine)
        matrix=ar/'matrix'/arm;build(machine,matrix,cache=mini/'shared/coordinates')
        ev=ar/'evaluation';ev.mkdir(exist_ok=True);spec=dict(cost_root=str(ar),machine=str(machinepath),matrix=str(matrix),evaluation=str(ev),endpoints=str(out/'endpoints.json'),base_variant='B1')
        if not (ev/'profiles_receipt.json').exists():
            summary=run_profiles(spec,q,original_protocol,ev,arm);atomic_json(ev/'summary.json',summary)
            atomic_json(ev/'profiles_receipt.json',dict(status='succeeded',outputs={str(p.relative_to(ev)):identity(p) for d in ('engineering','paper_aligned_v1') for p in (ev/d).rglob('*') if p.is_file()}))
        else:verify_outputs(json.loads((ev/'profiles_receipt.json').read_text()))
        stage_diagnostics(dict(spec,historical_native_nine=arm=='S'),q['anchors'],q['prepared'],ev/'diagnostics/stages')
        publish(ev,ready=True,paper_protocol_complete=True,scope='real regional report admission baseline',historical_source=formal['historical_methods'][arm])
        historical[arm]=spec;baseline_receipts[arm]=identity(ev/'complete.json')
    methods={**historical,**q['methods']}
    ids=(sorted(set(pd.read_csv(historical['B']['matrix']+'/cell_mapping.csv').canonical_cell_id)&set(pd.read_csv(formal['population']['path']).cell_id)) if prepare_queue_only else sorted(set.intersection(*[set(pd.read_csv(Path(v['matrix'])/'cell_mapping.csv').canonical_cell_id) for v in methods.values()])))
    if not ids:raise ValueError('regional fixed population empty')
    population=mini/'configs/report_population.csv';pd.DataFrame({'cell_id':ids}).to_csv(population,index=False)
    aq=dict(q,historical_methods=historical,historical_validation_receipts=baseline_receipts,population=identity(population),comparison_domain=ids)
    config=mini/'configs/aggregate_queue.json'
    if prepare_queue_only:
        if any((Path(v['evaluation'])/'complete.json').exists() for v in q['methods'].values()):raise ValueError('deployment queue must precede biological execution')
        aq['review_receipt']=str(mini/'source/review_receipt.json');aq['admission']=str(mini/'admission.json')
        for spec in aq['methods'].values():spec['commit']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=spec['worktree'],text=True,timeout=30).strip()
        config=mini/'configs/queue.json';atomic_json(config,aq)
        atomic_json(mini/'source/baseline_preparation.json',dict(status='passed',execution_source=execution,scope='real regional historical baselines and fixed historical B1QC domain, final queue before any four-arm stage',outputs={'queue':identity(config),'population':identity(population),**{a:obj for a,obj in baseline_receipts.items()}}));return
    atomic_json(config,aq)
    aggregate(config)
    if source_identity(SOURCES)!=execution:raise ValueError('full aggregate mini source changed')
    atomic_json(Path(receipt_path) if receipt_path else root/'tests/aggregate_mini_shared_v1.json',dict(status='passed',execution_source=execution,scope='full six-method report mechanics on real regional native B replay/historical S crop and four trained/postprocessed mini arms; not the scientific 103000-ID result',population_count=len(ids),outputs={'comparison':identity(mini/'comparison/complete.json'),'queue':identity(config),'baselines':identity(population)}))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--mini');p.add_argument('--receipt');p.add_argument('--prepare-queue-only',action='store_true');a=p.parse_args();run(a.root,a.mini,a.receipt,a.prepare_queue_only)
