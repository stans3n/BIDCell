"""Exercise all 17 population report assemblies, empty-pair output and JSON publication."""
from pathlib import Path
import argparse,json
import numpy as np,pandas as pd
from scripts.bs4.recover_author_markers import recompute_population_reports,artifact_index,output_manifest,PAIRS,SOURCES
from scripts.bs4.comparison import population_scores
from scripts.crgf.common import atomic_json,identity,source_identity

def run(out):
 out=Path(out);out.mkdir(parents=True,exist_ok=True);sources=[*SOURCES,'scripts/bs4/tests/author_driver_assembly.py'];before=source_identity(sources);prep=out/'prepared';prep.mkdir();genes=['g'+str(i) for i in range(307)];markers={'positive':{'T':['g0','g1']},'negative':{'T':['g5','g6']}};atomic_json(prep/'markers.json',markers);(prep/'reference_genes.txt').write_text('\n'.join(genes)+'\n');anchor=out/'anchor.csv';pd.DataFrame({'cell_id':[1,2,3,4],'cell_type':['T','T','T','__unavailable__:classifier_rejected']}).to_csv(anchor,index=False);q=dict(prepared=str(prep),anchors={'B1':identity(anchor)});methods={}
 for arm in ['B','S','P1','P2','T1','T2']:
  matrix=out/arm/'matrix';matrix.mkdir(parents=True);ev=out/arm/'evaluation/paper_aligned_v1';ev.mkdir(parents=True);ids=np.array([1,2,4] if arm=='S' else [1,2,3,4]);counts=np.zeros((len(ids),307),np.int64);counts[ids!=1,0]=10;counts[ids!=1,5]=5;np.save(matrix/'counts.npy',counts);pd.DataFrame({'canonical_cell_id':ids}).to_csv(matrix/'cell_mapping.csv',index=False);pd.DataFrame({'gene':genes}).to_csv(matrix/'gene_mapping.csv',index=False);pd.DataFrame({'cell_id':ids}).to_csv(ev/'markers_per_cell.csv',index=False);methods[arm]=dict(matrix=str(matrix),evaluation=str(ev.parent))
 original=out/'original';population=[1,2,3,4,9];population_scores(methods,q,original/'fixed_B1QC',population,'historical_B1QC')
 names=['historical_ABC_commonQC','six_methods_commonQC','four_new_arms_commonQC',*[a+'_'+b+'_commonQC' for a,b in PAIRS],*['ownQC_'+a for a in methods]]
 for name in names:
  selected={name[6:]:methods[name[6:]]} if name.startswith('ownQC_') else methods
  population_scores(selected,q,original/'sensitivity'/name,population,name)
 comparison=out/'recomputed';rows=recompute_population_reports(methods,q,original,comparison)
 assert len({row['scope'] for row in rows})==17 and all(row['exact_values_unchanged'] for row in rows)
 assert len(list(comparison.glob('sensitivity/ownQC_*/paired_metrics.csv')))==6
 inputs={str(p.relative_to(original)):identity(p) for p in original.rglob('*') if p.is_file()};index=artifact_index(comparison,inputs);outputs=output_manifest(out)
 payload=dict(status='passed',scope='17 synthetic population report assembly through production function and complete JSON serialization; actual full driver remains required',execution_source=before,population_reports=rows,index=index,outputs=outputs)
 json.dumps(payload,allow_nan=False);assert source_identity(sources)==before;atomic_json(out/'complete.json',payload);print('assembly passed',len(rows),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();run(a.output)
