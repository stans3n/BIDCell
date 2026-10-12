"""Fixed-population final scores preserve absent IDs, zero counts and empty pairs."""
import argparse,json,tempfile
from pathlib import Path
import numpy as np,pandas as pd
from scripts.bs4.comparison import population_scores
from scripts.bs4.evaluation import PAIRS
from scripts.bs4.tables import read_table
from scripts.cos.diagnostics import METRICS
from scripts.crgf.common import atomic_json,identity,source_identity

SOURCES=['scripts/bs4/author_marker.py','scripts/bs4/comparison.py','scripts/bs4/diagnostics.py','scripts/bs4/stage_population.py','scripts/bs4/tables.py','scripts/cos/diagnostics.py','scripts/bs4/tests/population_missing.py']

def verify(methods,population,out):
    frames={};missing={}
    summary=pd.read_csv(out/'metric_summary.csv')
    for name,spec in methods.items():
        ids=pd.read_csv(Path(spec['matrix'])/'cell_mapping.csv').canonical_cell_id.to_numpy(int)
        present=np.isin(population,ids);f=read_table(out/f'{name}.csv');frames[name]=f
        assert f.cell_id.tolist()==list(population) and not f.cell_id.duplicated().any()
        assert np.array_equal(f.stage_row_present.to_numpy(bool),present)
        for metric in METRICS:
            assert f.loc[~present,metric].isna().all()
            assert f.loc[~present,metric+'_reason'].eq('missing_stage_id').all()
            assert np.array_equal(np.isfinite(f[metric]),f[metric+'_reason'].eq('available'))
            row=summary[(summary.method==name)&(summary.type=='__all__')&(summary.metric==metric)].iloc[0]
            assert row.n_eligible==len(population) and row.n_evaluable==np.isfinite(f[metric]).sum()
            assert row.n_missing_stage_id==int((~present).sum())
        missing[name]=f.loc[~present,'cell_id'].astype(int).tolist()
    if len(methods)>1:
        paired=pd.read_csv(out/'paired_metrics.csv')
        for a,b in PAIRS:
            if a not in frames or b not in frames:continue
            for metric in METRICS:
                fa,fb=frames[a][metric],frames[b][metric];ok=np.isfinite(fa)&np.isfinite(fb)
                row=paired[(paired.comparison==a+'-'+b)&(paired.type=='__all__')&(paired.metric==metric)].iloc[0]
                assert row.n_eligible==len(population) and row.n_evaluable==int(ok.sum())
                if ok.any():np.testing.assert_allclose(row.mean_delta,(fa[ok]-fb[ok]).mean(),rtol=1e-12,atol=1e-14)
                else:assert pd.isna(row.mean_delta)
    return missing

def synthetic():
    with tempfile.TemporaryDirectory() as td:
        r=Path(td);prep=r/'prepared';prep.mkdir();anchor=r/'labels.csv'
        pd.DataFrame({'cell_id':[1,2,3,4],'cell_type':['T','T','T','__unavailable__:classifier_rejected']}).to_csv(anchor,index=False)
        atomic_json(prep/'markers.json',{'positive':{'T':['a','b']},'negative':{'T':['c','d']}});(prep/'reference_genes.txt').write_text('a\nb\nc\nd\n')
        methods={}
        for a,ids in [('B',[1,2,3,4]),('P1',[1,2,4])]:
            matrix=r/a/'matrix';matrix.mkdir(parents=True);ev=r/a/'evaluation/paper_aligned_v1';ev.mkdir(parents=True)
            pd.DataFrame({'canonical_cell_id':ids}).to_csv(matrix/'cell_mapping.csv',index=False);pd.DataFrame({'gene':list('abcd')}).to_csv(matrix/'gene_mapping.csv',index=False)
            counts=np.array([[10,3,0,1] if i!=2 else [0,0,0,0] for i in ids]);np.save(matrix/'counts.npy',counts)
            pd.DataFrame({'cell_id':ids}).to_csv(ev/'markers_per_cell.csv',index=False);methods[a]=dict(matrix=str(matrix),evaluation=str(ev.parent))
        q=dict(anchors={'B1':identity(anchor)},prepared=str(prep));population=[3,1,2,4,9]
        population_scores(methods,q,r/'scores',population,'synthetic');verify(methods,population,r/'scores')
        f=read_table(r/'scores/B.csv').set_index('cell_id')
        assert f.loc[2,'positive_F1']==0 and f.loc[2,'positive_Recall']==0 and pd.isna(f.loc[2,'positive_Precision'])
        assert f.loc[2,'positive_F1_reason']=='available' and f.loc[2,'positive_Recall_reason']=='available' and f.loc[2,'positive_Precision_reason']=='undefined_original_metric'
        assert f.loc[4,'positive_F1_reason']=='label_ineligible'
        from scripts.bs4.author_marker import fixed_scores
        from scripts.cos.diagnostics import load_anchor
        cached={}
        for name,spec in methods.items():
            matrix=Path(spec['matrix']);ids=pd.read_csv(matrix/'cell_mapping.csv').canonical_cell_id
            cached[name]=fixed_scores(np.load(matrix/'counts.npy'),ids,list('abcd'),load_anchor(q['anchors']['B1']),json.loads((prep/'markers.json').read_text()),list('abcd'))
        population_scores(methods,q,r/'cached',population,'synthetic',score_tables=cached)
        for name in methods:pd.testing.assert_frame_equal(read_table(r/'scores'/f'{name}.csv'),read_table(r/'cached'/f'{name}.csv'),check_exact=True)
        population_scores(methods,q,r/'empty',[9],'empty_pair');verify(methods,[9],r/'empty')
        # An entirely absent method cannot invent a zero row or drop the population.
        pd.DataFrame({'canonical_cell_id':[]}).to_csv(Path(methods['P1']['matrix'])/'cell_mapping.csv',index=False)
        np.save(Path(methods['P1']['matrix'])/'counts.npy',np.empty((0,4),dtype=int))
        population_scores(methods,q,r/'absent',population,'absent_method');verify(methods,population,r/'absent')
    return dict(missing_rows=True,zero_counts=True,label_ineligible=True,empty_pair=True,entire_method_absent=True)

def run(root,mini):
    root=Path(root);execution=source_identity(SOURCES);small=synthetic()
    mini=Path(mini);q=json.loads((mini/'configs/queue.json').read_text())
    methods={**q['historical_methods'],**q['methods']};population=pd.read_csv(q['population']['path']).cell_id.tolist();assert len(population)==130
    out=root/'tests/population_missing_v3';population_scores(methods,q,out,population,'historical_B1QC');missing=verify(methods,population,out)
    assert missing['S']==[13970,15908] and len(missing['T1'])==len(missing['T2'])==8
    if execution!=source_identity(SOURCES):raise ValueError('population missing execution source drift')
    failure=root/'tests/deployment/context_b64.log'
    atomic_json(root/'tests/population_missing_v3.json',dict(status='passed',execution_source=execution,synthetic=small,population_count=len(population),missing_ids=missing,fixture_queue=identity(mini/'configs/queue.json'),failed_original_attempt=identity(failure) if failure.exists() else None,outputs={p.name:identity(p) for p in out.iterdir() if p.is_file()}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--mini',required=True);a=p.parse_args();run(a.root,a.mini)
