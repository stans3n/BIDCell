"""Regression of paper definitions plus actual R QC/scClassify small-chain execution."""
import argparse
import json
from pathlib import Path
import posixpath
import subprocess
import shutil
import tifffile
import time
import zipfile
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scripts.crgf.evaluate_egr import purity, neighbour_curve, checked_r, read, paper_spatial, receipt, aggregate
from scripts.crgf.common import atomic_json, identity


def source_rows(path):
    ns={'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
    with zipfile.ZipFile(path) as z:
        strings=[]
        if 'xl/sharedStrings.xml' in z.namelist():
            for _,el in ET.iterparse(z.open('xl/sharedStrings.xml'),events=('end',)):
                if el.tag.endswith('}si'):strings.append(''.join(el.itertext()));el.clear()
        rel={x.attrib['Id']:x.attrib['Target'] for x in ET.fromstring(z.read('xl/_rels/workbook.xml.rels'))}
        sheet=next(s for s in ET.fromstring(z.read('xl/workbook.xml')).find('s:sheets',ns) if s.attrib['name']=='SuppFig4')
        target=rel[sheet.attrib['{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id']]
        path=target.lstrip('/') if target.startswith('/') else posixpath.normpath('xl/'+target)
        rows=[]
        for _,el in ET.iterparse(z.open(path),events=('end',)):
            if el.tag.endswith('}row'):
                vals={}
                for c in el:
                    v=c.find('s:v',ns);v=v.text if v is not None else ''.join(c.find('s:is',ns).itertext()) if c.find('s:is',ns) is not None else ''
                    if c.attrib.get('t')=='s':v=strings[int(v)]
                    vals[''.join(ch for ch in c.attrib['r'] if ch.isalpha())]=v
                rows.append(vals);el.clear()
        return [r for r in rows[2:] if 'F' in r]


def run(protocol,prepared,out,mini_root):
    p=read(protocol);out=Path(out);out.mkdir(parents=True,exist_ok=True);start=time.monotonic();tests={}
    source=Path(p['reference']).with_name('source_data.xlsx');rows=source_rows(source)
    raw={r['E']:{'positive_F1':float(r['F']),'negative_F1':float(r['G'])} for r in rows}
    result=purity(raw,list(raw));err=max(abs(result['methods'][r['E']]['purity']-float(r['J'])) for r in rows)
    assert len(rows)==23 and err<2e-14
    tests['source_data_23_rows']={'max_abs_error':err,'source':identity(source)}
    assert purity({})['status']=='incomplete_comparison_set'
    equal={m:{'positive_F1':1.,'negative_F1':0.} for m in ['B1','E','G','R','GR']}
    assert purity(equal)['status']=='not_identifiable_zero_range'
    tests['purity_incomplete_zero_range']='passed'
    coords=np.array([[0,0],[0,10],[0,20],[0,100],[0,101],[0,0],[0,200]],float)
    labels=np.array(['B']*5+['T']*2);settings={'source_type':'B','target_types':['T'],'markers':['g'],'breaks_um':[0,10,20,30,40,50,100],'threshold':0}
    curve,d=neighbour_curve(coords,labels,{'g':np.array([1,0,1,0,1,0,0])},settings)
    assert np.array_equal(d,cdist(coords[:5],coords[5:]).min(1))
    assert curve.n_eligible.sum()==5 and curve.iloc[0].n_eligible==2 and curve.iloc[1].n_eligible==1
    far=np.array([[0,101],[0,0]],float)
    c2,d2=neighbour_curve(far,np.array(['B','T']),{'g':np.array([1,0])},settings)
    assert c2.iloc[-1].n_eligible==1 and c2.iloc[-1].upper_um==101
    for labels2 in [np.array(['T','T']),np.array(['B','B'])]:
        empty,_=neighbour_curve(far,labels2,{'g':np.array([1,0])},settings)
        assert empty.iloc[0].status in ['empty_source_type','empty_target_type']
    tests['kdtree_right_closed_bins']='passed'
    # Official marker functions, including nonfinite P/R vs zero F1, strict R
    # quantile ties, positive .25 vs negative1 and explicit leaf mapping.
    rcode=r'''
args<-commandArgs(TRUE);p<-jsonlite::fromJSON(args[1],simplifyVector=FALSE);prepared<-args[2]
library(SingleCellExperiment);source(p$official_processing);source(p$official_expression)
x<-matrix(c(0,0,0,0, 0,0,0,10, 2,2,2,2),nrow=4,dimnames=list(c('a','b','c','d'),c('zero','known','unknown')))
v<-.calculate_marker_purity(x,c('T','T','unassigned'),list(T='d'),BiocParallel::SerialParam())
stopifnot(v['zero','F1']==0,!is.finite(v['zero','Precision']),v['zero','Recall']==0,all(v['unknown',]==0),all(v['known',]==1))
w<-matrix(seq_len(80),20,4,dimnames=list(paste0('g',1:20),LETTERS[1:4]));w[,2]<-w[20:1,2];w[,3]<-w[,1]
sce<-SingleCellExperiment(assays=list(mean=w))
for(kind in c('positive','negative')) {
 t<-if(kind=='positive').25 else 1
 actual<-generateMarkerList(sce,q=.9,t=t,type=kind,verbose=FALSE)
 expected<-lapply(seq_len(ncol(w)),function(i){a<-w[,i]-rowMeans(w[,-i,drop=FALSE]);names(which(if(kind=='positive')a>quantile(a,.9,type=7)else a<quantile(a,.1,type=7)))})
 names(expected)<-colnames(w);remove<-names(which(table(unlist(expected))>=t*ncol(w)));expected<-lapply(expected,setdiff,y=remove)
 stopifnot(identical(actual,expected))
}
meta<-readRDS(file.path(prepared,'reference_metadata.rds'));stopifnot(length(meta$genes)==307,meta$map['Firoblast']=='Fibroblast')
ref<-readRDS(p$reference);stopifnot(identical(meta$genes,rownames(ref)))
writeLines('passed',args[3])
'''
    script=out/'official_regression.R';script.write_text(rcode)
    with (out/'official_regression.log').open('w') as log:
        subprocess.run([p['rscript'],str(script),str(protocol),str(prepared),str(out/'official_regression_passed.txt')],stdout=log,stderr=subprocess.STDOUT,check=True)
    tests['official_R_marker_quantile_unknown_nonfinite_mapping']='passed'
    # Real spatial sample, with inclusive QC endpoint fixtures. Production never slices.
    old=Path('/home/zwan0726/work/outputs/crgf_1008_20261008T031528+1100')
    native=pd.read_csv(old/'matrix/B1/expr_mat.csv',index_col=0,nrows=512)
    genes=[k for k in native if k!='cell_id'];extra=[]
    for i,(total,n_genes) in enumerate([(19,19),(20,20),(2000,20),(2001,20),(100,19)]):
        counts=np.zeros(313,int);counts[:n_genes]=1;counts[0]+=total-n_genes
        extra.append(dict(cell_id=900000+i,**dict(zip(genes,counts))))
    fixture=pd.concat([native,pd.DataFrame(extra)],ignore_index=True);sourcecsv=out/'real_B1_plus_qc_endpoints.csv';fixture.to_csv(sourcecsv)
    checked_r(p,'paper_evaluation.R',[protocol,prepared,sourcecsv,out/'real_spatial'],out/'real_spatial')
    qc=pd.read_csv(out/'real_spatial/qc_all_cells.csv').set_index('cell_id')
    assert list(qc.loc[range(900000,900005),'qc_pass'])==[False,True,True,False,False]
    sf=pd.read_csv(out/'real_spatial/size_factors.csv').set_index('cell_id').size_factor
    passed=fixture[fixture.cell_id.isin(sf.index)].set_index('cell_id')[genes].sum(axis=1)
    np.testing.assert_allclose(sf.loc[passed.index],passed/passed.mean(),rtol=1e-12)
    tests['QC_inclusive_313_gene_size_factors_real_prediction']='passed'
    for method in ['E','G','R']:
        native=Path(mini_root)/method/'matrix/expr_mat.csv'
        checked_r(p,'paper_evaluation.R',[protocol,prepared,native,out/f'mini_{method}'],out/f'mini_{method}')
        markers=pd.read_csv(out/f'mini_{method}/markers_per_cell.csv')
        assert len(markers)>0 and np.isfinite(markers[['positive_F1','negative_F1']]).all().all()
        tests[f'actual_mini_{method}_QC_normalize_scClassify_markers']='passed'
    integration_fixture(protocol,prepared,out,mini_root)
    tests['python_spatial_paired_full_aggregate_and_missing_gate']='passed'
    atomic_json(out/'complete.json',dict(status='succeeded',tests=tests,wall_seconds=time.monotonic()-start,scope='small fixtures and first512 real B1 cells, not production evaluation',prepared=identity(Path(prepared)/'complete.json'),protocol=identity(protocol),implementations={name:identity(Path(__file__).resolve().parents[1]/'scripts/crgf'/name) for name in ['evaluate_egr.py','paper_evaluation.R','prepare_paper.R']}))



def actual_method_fixture(protocol,prepared,out):
    from scripts.crgf.matrix import build
    from scripts.crgf.evaluate_egr import method
    root=Path(out)/'actual_method';root.mkdir(parents=True,exist_ok=True)
    start=time.monotonic()
    repo=Path(__file__).resolve().parents[1];old=Path('/home/zwan0726/work/outputs/crgf_1008_20261008T031528+1100')
    nuclear=pd.read_csv(old/'evaluation/B1/structure/annotation_nuclear.csv')
    pearson=pd.read_csv(old/'evaluation/B1/cellspa/windows_style_maxpearson_raw/official_markers_per_cell.csv',usecols=['cell_id','cell_type'])
    ids=sorted(set(nuclear.groupby('cell_type').head(3).cell_id)|set(pearson.groupby('cell_type').head(3).cell_id))
    mapping=pd.read_csv(old/'matrix/B1/cell_mapping.csv').mask_label.to_numpy()
    rows=np.searchsorted(mapping,ids);assert np.array_equal(mapping[rows],ids)
    original=np.load(old/'matrix/B1/counts.npy',mmap_mode='r')[rows]
    c=read(repo/'scripts/crgf/configs/machine.json');genes=Path(c['genes']).read_text().splitlines()
    # Wide spacing ensures official legacy curve's appended max distance>100.
    # Synthetic raster geometry; actual historical expression and fixed ID labels.
    ncols=10;spacing=150;shape=((len(ids)//ncols+1)*spacing,ncols*spacing)
    mask=np.zeros(shape,np.uint32);nuc=mask.copy();trans=[]
    for i,cell_id in enumerate(ids):
        y=(i//ncols)*spacing+10;x=(i%ncols)*spacing+10
        mask[y:y+8,x:x+8]=cell_id;nuc[y+2:y+5,x+2:x+5]=cell_id
        for g,n in zip(genes,original[i]):
            trans.extend([(x+3.1,y+3.1,g)]*int(n))
    tifffile.imwrite(root/'final.tif',mask);tifffile.imwrite(root/'nuclei.tif',nuc)
    tx=pd.DataFrame(trans,columns=['x_location','y_location','feature_name']);tx['transcript_id']=np.arange(len(tx));tx['qv']=30;tx.to_csv(root/'transcripts.csv',index=False)
    c.update(final_mask=str(root/'final.tif'),pre_mask=str(root/'final.tif'),baseline_final_mask=str(root/'final.tif'),nuclei_mask=str(root/'nuclei.tif'),transcripts=str(root/'transcripts.csv'),final_matrix=str(root/'matrix/expr_mat.csv'))
    build(c,root/'matrix',cache=root/'cache')
    np.testing.assert_array_equal(np.load(root/'matrix/counts.npy'),original)
    atomic_json(root/'machine.json',c)
    # Actual CLI, including all strict own-matrix checks, old R profiles,
    # structural evaluator, paper R and own-centroid Python integration.
    command=[__import__('sys').executable,'-m','scripts.crgf.evaluate_egr','method','--machine',str(root/'machine.json'),'--reference',str(repo/'scripts/crgf/configs/reference.json'),'--engineering-protocol',str(repo/'scripts/crgf/configs/protocol.json'),'--protocol',str(protocol),'--prepared',str(prepared),'--matrix',str(root/'matrix'),'--out',str(root/'evaluation'),'--method','E']
    with (root/'method_cli.log').open('w') as log:subprocess.run(command,cwd=repo,stdout=log,stderr=subprocess.STDOUT,check=True)
    result=read(root/'evaluation/complete.json');assert result['paper_method_complete'] and result['evaluation_executed']
    atomic_json(root/'fixture_receipt.json',dict(status='succeeded',scope='synthetic spatial geometry/transcript locations with real selected historical expression and nuclear IDs; genuine full method CLI, not scientific output',n_cells=len(ids),command=command,wall_seconds=time.monotonic()-start,method_receipt=identity(root/'evaluation/complete.json')))

def integration_fixture(protocol,prepared,out,mini_root):
    p=read(protocol);dest=Path(out)/'synthetic_integration';dest.mkdir(exist_ok=True)
    methods={};stages={}
    ids=pd.read_csv(Path(out)/'mini_E/annotation.csv').cell_id.to_numpy()
    mask=tifffile.imread(Path(mini_root)/'E/final.tif')
    for j,m in enumerate(['B1','E','G','R','GR']):
        root=dest/m;paper=root/'paper_aligned_v1';shutil.copytree(Path(out)/'mini_E',paper,dirs_exist_ok=True)
        structure=root/'engineering/structure';structure.mkdir(parents=True,exist_ok=True)
        nuclear=pd.DataFrame(dict(cell_id=ids,cell_type=['Unknown']*len(ids),x_um=np.arange(len(ids))*20.,y_um=np.arange(len(ids))*10.))
        nuclear.to_csv(structure/'annotation_nuclear.csv',index=False)
        pd.DataFrame(dict(cell_id=ids,area_um2=np.ones(len(ids)),nucleus_area_px=np.ones(len(ids)),nucleus_major=np.ones(len(ids)),nucleus_minor=np.ones(len(ids)),nucleus_eccentricity=np.zeros(len(ids)))).to_csv(structure/'cells_distribution.csv',index=False)
        pd.DataFrame(dict(cell_id=ids)).to_csv(structure/'shape_raster_distribution.csv',index=False)
        rows=[dict(metric_id=i,metric_name='fixture_'+str(i),value=1,status='computed',definition='synthetic fixture') for i in list(range(1,6))+list(range(12,19))+[20,21]]
        pd.DataFrame(rows).to_csv(root/'engineering/metrics.csv',index=False)
        for old in ['remote_nuclear_annotation_raw','windows_style_maxpearson_raw']:
            target=root/'engineering/cellspa'/old;target.mkdir(parents=True,exist_ok=True)
            scores=pd.read_csv(paper/'markers_per_cell.csv');scores.cell_type='Unknown';scores.to_csv(target/'official_markers_per_cell.csv',index=False)
            summ=pd.read_csv(paper/'marker_summary.csv');summ=summ[summ.cell_type=='__all__'].rename(columns={'all_cell_mean':'all_cell_mean'})
            summ.to_csv(target/'official_marker_summary.csv',index=False)
        machine=root/'machine.json';atomic_json(machine,dict(final_mask=str(Path(mini_root)/'E/final.tif'),pixel_um=1.))
        paper_spatial(read(machine),paper,p)
        model=read(Path(prepared)/'complete.json')['model']['sha256']
        atomic_json(root/'summary.json',dict(method=m,n_after_qc=len(ids),n_before_qc=len(ids),qc_retention=1.,unassigned_fraction=1.,positive_F1=0.,negative_F1=0.,model_sha256=model))
        train=root/'checkpoints'/m;train.mkdir(parents=True,exist_ok=True)
        atomic_json(train/'complete.json',dict(parameters=123,updates=2,wall_seconds=1.,peak_vram_bytes=100))
        for stage in ['train','predict','postprocess']:
            stages[m+'_'+stage]=dict(method=m,stage=stage,attempts=[dict(wall_seconds=1.,sampled_peak_rss_bytes_sum=100,exit_code=0)])
        methods[m]=dict(evaluation=str(root),machine=str(machine),cost_root=str(root),run_status=str(dest/'run_status.json'))
        receipt(root,paper_method_complete=True,protocol_version=p['version'])
    atomic_json(dest/'run_status.json',dict(stages=stages))
    atomic_json(dest/'methods.json',dict(methods=methods))
    assert aggregate(dest/'methods.json',protocol,dest/'aggregate')
    cov=pd.read_csv(dest/'aggregate/metric_coverage.csv');assert set(map(str,range(1,44)))<=set(cov.metric_id)
    assert read(dest/'aggregate/purity.json')['status']=='not_identifiable_zero_range'
    assert (dest/'aggregate/final_comparison.csv').exists()
    del methods['R'];atomic_json(dest/'partial_methods.json',dict(methods=methods))
    assert not aggregate(dest/'partial_methods.json',protocol,dest/'aggregate_partial')
    assert not read(dest/'aggregate_partial/complete.json')['paper_protocol_complete']


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--protocol',required=True);a.add_argument('--prepared',required=True);a.add_argument('--out',required=True);a.add_argument('--mini-root',required=True);run(**vars(a.parse_args()))
