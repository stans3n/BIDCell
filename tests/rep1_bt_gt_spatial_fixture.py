"""Small real-R producer fixtures; never evaluates a production endpoint."""
import fcntl
import os
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
from scripts.rep1_bt_gt_fusion.common import atomic_json,check_identity,digest,identity
from scripts.rep1_bt_gt_fusion.evaluation import run_r
from scripts.rep1_bt_gt_fusion.inputs import source_dependencies


def fixture_destination(temporary_root,name):
    external=os.environ.get('REP1_BT_GT_R_FIXTURE_ROOT')
    root=(Path(external) if external else Path(temporary_root)) / name
    if root.exists():raise ValueError('Refusing to overwrite a real-R fixture: '+str(root))
    return root


def spatial_fixture(root,case,compare_author=False):
    """Generate fixed tiny populations and run the actual new R entry point."""
    root=Path(root);snapshot=root/'snapshot';snapshot.mkdir(parents=True)
    source_map={Path(item['path']).name:item for item in source_dependencies()}
    frozen=[];runtime=dict(source_snapshot=str(snapshot),allowed_sources=[])
    for key,name in [('official_utils','publish_utils.R'),('official_spatial','publish_spatialDiversity_metrics.R')]:
        item=source_map[name];target=snapshot/'external_sources'/name;target.parent.mkdir(exist_ok=True)
        check_identity(item);shutil.copyfile(item['path'],target);frozen.append(identity(target))
        runtime[key]=str(target);runtime['allowed_sources'].append(str(target))
    wrapper=snapshot/'spatial_diversity.R'
    shutil.copyfile(source_map[wrapper.name]['path'],wrapper);frozen.append(identity(wrapper))
    runtime_path=root/'runtime.json';atomic_json(runtime_path,runtime)
    out=root/'output';paper=out/'paper_aligned_v1';paper.mkdir(parents=True);(out/'structure').mkdir()
    # Six occupied bins with variable compositions and different within-bin
    # dispersions. Empty bins remain absent, exactly as in the author table().
    sizes=[6,6,6,6,6,6]
    labels=[['B Cells']*6,['B Cells']*5+['CD4 T'],['B Cells']*4+['CD4 T']*2,
            ['B Cells']*3+['CD4 T']*3,['B Cells']*2+['CD4 T']*2+['CD8 T']*2,
            ['B Cells']*4+['CD4 T','CD8 T']]
    if case in ('single_known','all_unassigned'):
        labels=[[('B Cells' if case=='single_known' else 'unassigned')]*n for n in sizes]
    elif case=='constant_entropy':labels=[['B Cells','CD4 T','CD8 T']*2 for n in sizes]
    elif case=='one_cell_bins':
        sizes=[1]*6;labels=[[['B Cells','CD4 T','CD8 T'][i%3]] for i in range(6)]
    elif case=='one_bin':sizes=[6];labels=[['B Cells','CD4 T','CD8 T']*2]
    elif case not in ('normal','mixed_nonfinite','invalid_input'):
        raise ValueError('unknown synthetic spatial case: '+case)
    records=[]
    for b,(n,types) in enumerate(zip(sizes,labels)):
        for j,label in enumerate(types):
            records.append(dict(cell_id=len(records)+1,evaluation_label=label,
                x_um=(b%3)*1100+(j%3)*7,y_um=(b//3)*1100+(j//3)*11,
                total_transcripts=60+17*b+(j+1)**2*(b+1),total_genes=21+b+j*(b%3+1),
                area_um2=40+5*b+(j+1)**2*(b+2)))
    data=pd.DataFrame(records)
    if case=='one_bin':data[['x_um','y_um']]=0
    if case=='mixed_nonfinite':data.loc[:5,'area_um2']=1  # CV(log10) undefined; raw SD=0 defined.
    if case=='invalid_input':data.loc[0,'total_transcripts']=-1
    data[['cell_id','evaluation_label']].to_csv(paper/'annotation.csv',index=False)
    data[['cell_id','total_transcripts','total_genes']].assign(qc_pass=True).to_csv(paper/'qc_all_cells.csv',index=False)
    data[['cell_id','x_um','y_um']].to_csv(paper/'own_cell_centroids.csv',index=False)
    data[['cell_id','area_um2']].to_csv(out/'structure/cells_distribution.csv',index=False)
    data.to_csv(root/'synthetic_input.csv',index=False)
    script=wrapper;status_file='spatial_execution_status.txt';args=[runtime_path,out]
    if compare_author:
        # This fixture-only driver runs the untouched author implementation on
        # exactly the same input object after the production wrapper returns.
        script=snapshot/'compare_author.R'
        script.write_text('a<-commandArgs(trailingOnly=TRUE)\n'
            'source(a[3])\n'
            'original<-calSpatialMetricsDiversity(spe,celltype="cell_type",bin_width=500)\n'
            'original_result<-original@metadata$CellSPA$spatialMetricsDiversity\n'
            'write.csv(original_result$results,file.path(out,"original_spatial_bins.csv"),row.names=FALSE,na="NA")\n'
            'write.csv(data.frame(metric=names(original_result$statistics),pearson=unname(original_result$statistics)),file.path(out,"original_spatial_correlations.csv"),row.names=FALSE,na="NA")\n'
            'writeLines("succeeded",file.path(out,"comparison_execution_status.txt"))\n')
        frozen.append(identity(script));args.append(wrapper);status_file='comparison_execution_status.txt'
    cfg=dict(source_snapshot=str(snapshot),rscript='/home/zwan0726/work/envs/bidcell_egr_r/bin/Rscript',
             r_libs_user='/home/zwan0726/work/outputs/bidcell_historical_dual_20261013/sources/R_library',r_libs_site='')
    checks=[]
    def validate():
        for item in frozen:check_identity(item)
        checks.append('snapshot_bytes_verified')
    with (root/'run.lock').open('a') as a,Path('/home/zwan0726/work/outputs/cos_gpu.lock').open('a') as b:
        fcntl.flock(a,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(b,fcntl.LOCK_EX|fcntl.LOCK_NB)
        run_r(cfg,script,args,out,(a.fileno(),b.fileno()),validate,status_file=status_file)
    bins=pd.read_csv(paper/'spatial_diversity_bins.csv')
    correlations=pd.read_csv(paper/'spatial_diversity_correlations.csv',keep_default_na=False)
    correlations['pearson']=pd.to_numeric(correlations.pearson,errors='coerce')
    if compare_author:
        old_bins=pd.read_csv(out/'original_spatial_bins.csv')
        pd.testing.assert_frame_equal(bins[old_bins.columns],old_bins,check_dtype=False,check_exact=False,rtol=0,atol=1e-12)
        old=pd.read_csv(out/'original_spatial_correlations.csv').set_index('metric')
        actual=correlations.set_index('metric').loc[old.index]
        np.testing.assert_allclose(actual.pearson,old.pearson,rtol=0,atol=1e-12,equal_nan=True)
    proof=dict(case=case,status='passed',synthetic_cells=len(data),compare_unmodified_author=compare_author,
        author_bins_and_statistics_equal=compare_author,execution_validations=checks,sources=frozen,
        input=identity(root/'synthetic_input.csv'),outputs={p.name:identity(p) for p in paper.glob('spatial_*.csv')},
        definition='author cut(range, max(ceil(range/500), 2)); entropy; CV(log10), SD(raw); common complete-finite rows per 3-metric family',
        no_production_endpoint_evaluation=True)
    atomic_json(root/'fixture_receipt.json',proof)
    return out,bins,correlations
