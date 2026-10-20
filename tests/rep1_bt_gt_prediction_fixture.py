"""Real frozen-model recovery on 12 synthetic cells; never a real endpoint.

Run with: python -m tests.rep1_bt_gt_prediction_fixture --out NEW_DIR
          --input-manifest FROZEN_INPUT_MANIFEST
"""
import argparse
import fcntl
from pathlib import Path
import shutil
import time
import numpy as np
import pandas as pd
from scripts.rep1_bt_gt_fusion.common import atomic_json,check_identity,digest,identity,read
from scripts.rep1_bt_gt_fusion.evaluation import run_r
from scripts.rep1_bt_gt_fusion.inputs import load_manifest,source_dependencies


def run_fixture(root,manifest_path):
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=False)
    manifest=load_manifest(manifest_path,verify=False)
    evaluation=manifest['evaluation'];prepared=Path(evaluation['prepared'])
    for path,entry in manifest['files'].items():
        if prepared in Path(path).parents:check_identity(entry)
    machine_path=manifest['endpoints']['BASE_MATCH']['machine'];check_identity(manifest['files'][machine_path])
    machine=read(machine_path);check_identity(manifest['files'][machine['genes']])
    check_identity(manifest['files'][evaluation['protocol']])
    protocol=read(evaluation['protocol']);protocol['prediction_batch_size']=4
    atomic_json(root/'protocol_fixture.json',protocol)
    genes=Path(machine['genes']).read_text().splitlines();ids=np.arange(1,13)
    counts=np.random.default_rng(7362).poisson(.5,size=(12,len(genes)))
    assert len(genes)==313
    frame=pd.DataFrame(counts,index=ids,columns=genes);frame.insert(0,'cell_id',ids)
    frame.to_csv(root/'synthetic_matrix.csv')
    snapshot=root/'snapshot';snapshot.mkdir();source_map={Path(x['path']).name:x for x in source_dependencies()}
    frozen=[]
    for name in ('paper_evaluation.R','publish_expression_metrics.R'):
        target=snapshot/name;source=source_map[name];check_identity(source)
        shutil.copyfile(source['path'],target);frozen.append(identity(target))
    bind=digest(dict(fixture='synthetic_prediction_recovery',n_cells=12,batch=4,seed=7362,
                    model=manifest['files'][evaluation['reference_model']],sources=frozen))
    runtime=dict(source_snapshot=str(snapshot),dependencies_digest=bind,
                 official_expression=str(snapshot/'publish_expression_metrics.R'),
                 allowed_sources=[str(snapshot/'publish_expression_metrics.R')])
    atomic_json(root/'runtime.json',runtime)
    driver=snapshot/'compare_predictions.R'
    driver.write_text('a<-commandArgs(trailingOnly=TRUE)\n'
        'stopifnot(identical(readRDS(file.path(a[1],"scClassify_raw_predictions.rds")),readRDS(file.path(a[2],"scClassify_raw_predictions.rds"))))\n'
        'for(i in 1:3){p<-sprintf("prediction_chunks/batch_%04d.rds",i);x<-readRDS(file.path(a[1],p));y<-readRDS(file.path(a[2],p));stopifnot(identical(x,y),!is.null(x$rng_state),identical(x$rng_state,y$rng_state))}\n'
        'writeLines("PASS: all 3 prediction batch objects, IDs, predictions and post-batch RNG states are R identical",file.path(a[3],"semantic_comparison.txt"))\n'
        'writeLines("succeeded",file.path(a[3],"execution_status.txt"))\n')
    frozen.append(identity(driver))
    cfg=dict(source_snapshot=str(snapshot),rscript='/home/zwan0726/work/envs/bidcell_egr_r/bin/Rscript',
        r_libs_user='/home/zwan0726/work/outputs/bidcell_historical_dual_20261013/sources/R_library',r_libs_site='')
    checks=[];events=[];started=time.monotonic()
    def validate():
        for item in frozen:check_identity(item)
        checks.append('snapshot_bytes_verified')
    full=root/'full';resumed=root/'resumed'
    with (root/'run.lock').open('a') as a,Path('/home/zwan0726/work/outputs/cos_gpu.lock').open('a') as b:
        fcntl.flock(a,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(b,fcntl.LOCK_EX|fcntl.LOCK_NB)
        fds=(a.fileno(),b.fileno())
        args=[root/'protocol_fixture.json',prepared,root/'synthetic_matrix.csv']
        run_r(cfg,snapshot/'paper_evaluation.R',args+[full,root/'runtime.json'],full,fds,validate,events.append,bind)
        target=resumed/'prediction_chunks/batch_0001.rds';target.parent.mkdir(parents=True)
        shutil.copyfile(full/'prediction_chunks/batch_0001.rds',target)
        proof=read(full/'prediction_chunks/batch_0001.rds.committed.json');proof['path']=str(target)
        atomic_json(target.with_name(target.name+'.committed.json'),proof)
        before=identity(target);event_boundary=len(events)
        run_r(cfg,snapshot/'paper_evaluation.R',args+[resumed,root/'runtime.json'],resumed,fds,validate,events.append,bind)
        assert before==identity(target)
        assert [e['cursor'] for e in events[event_boundary:]]==[8,12]
        compared=[]
        for name in ('annotation.csv','markers_per_cell.csv','marker_summary.csv','qc_all_cells.csv',
                     'type_correlations.csv','type_expression_vectors.csv'):
            assert identity(full/name)['sha256']==identity(resumed/name)['sha256'],name
            compared.append(name)
        run_r(cfg,driver,[full,resumed,root/'comparison'],root/'comparison',fds,validate)
    receipt=dict(status='passed',synthetic=True,n_cells=12,n_genes=313,prediction_batches=3,
        first_committed_batch_reused=True,resumed_newly_committed_cursors=[8,12],
        raw_prediction_objects_and_post_batch_RNG_states_R_identical=True,byte_identical_CSV=compared,
        source_files=frozen,model=manifest['files'][evaluation['reference_model']],
        semantic_comparison=identity(root/'comparison/semantic_comparison.txt'),
        elapsed_seconds=time.monotonic()-started,execution_validations=checks,progress_events=events,
        real_endpoint_evaluation_run=False)
    atomic_json(root/'fixture_receipt.json',receipt)
    return receipt


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',required=True);parser.add_argument('--input-manifest',required=True)
    args=parser.parse_args();result=run_fixture(args.out,args.input_manifest)
    print('PASS:',Path(args.out)/'fixture_receipt.json')
