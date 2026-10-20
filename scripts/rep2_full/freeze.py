"""Input-derived Rep2 label views, BASE ROI, and frozen core/type eligibility."""
from pathlib import Path
import numpy as np
import pandas as pd
import tifffile
import yaml
from .common import *

def labels():
    out=ROOT/'frozen_labels';out.mkdir(exist_ok=True)
    annotation=ROOT/'endpoints/B1/evaluation/paper_aligned_v1/annotation.csv'
    ann=pd.read_csv(annotation)
    if ann.cell_id.duplicated().any():raise ValueError('duplicate B1 anchor IDs')
    # Literal ssp.bootstrap historical population projection: all own-QC rows.
    fields=['cell_id']+[k for k in ('evaluation_label','cell_type','unmatched_reason','rejected') if k in ann]
    historical=ann[fields];atomic_csv(out/'historical_fixed_population.csv',historical)
    # A separately identified view for local B/T, with the original exact mapping.
    local=historical.copy();labels=local.evaluation_label.where(local.unmatched_reason.fillna('').eq(''))
    local['lineage_code']=labels.map({'B Cells':1,'CD4 T':2,'CD8 T':2}).fillna(0).astype(np.int8)
    atomic_csv(out/'local_B1_scClassify_labels.csv',local)
    for name in ('historical_fixed_population.csv','local_B1_scClassify_labels.csv'):
        atomic_json(out/(name+'.producer.json'),dict(producer_annotation=identity(annotation),producer=identity(ROOT/'receipts/B1_annotate.json'),
            reference_model=config()['frozen_inputs']['evaluation_scClassify_model.rds'],contract=config()['frozen_inputs']['label_contract_audit'],
            rows=len(local),membership='all B1 own-matrix QC-pass; unmatched/rejected retained',view=identity(out/name)))
    return stage_outputs(out),dict(population=len(ann),local_known_BT=int((local.lineage_code>0).sum()))

def roi():
    from scripts.rep1_triple_simple_boundary.frozen_domains import build_interfaces,bootstrap_weights,block_indices,in_region
    from scripts.rep1_triple_simple_boundary.contracts import REGIONS
    from scripts.boundary_three.core_refine import interface_band
    out=ROOT/'frozen_roi';out.mkdir(exist_ok=True);s=spec(BASE)
    base=tifffile.imread(s['final']);nuc=tifffile.imread(s['nuclei']);meta=np.load(TRAINING/'shared/prepared/nucleus_metadata.npz')
    ids=meta['ids'];types=meta['type_index'];lut=np.full(int(ids.max())+1,-2,np.int16);lut[ids]=types
    d=dict(base_deps(),BASE=identity(s['final']),nuclei=identity(s['nuclei']),labels=identity(ROOT/'frozen_labels/historical_fixed_population.csv'))
    arrays,pairs=build_interfaces(base,nuc,lut,out/'interfaces',d,cfg_progress)
    selected=arrays['assignment']>=0
    if not np.array_equal(selected,interface_band(base)&(nuc==0)):raise ValueError('BASE fixed interface geometry differs')
    bits=np.full(base.shape,1<<REGIONS.index('all'),np.uint16)
    bits[selected]|=1<<REGIONS.index('interface');bits[nuc>0]|=1<<REGIONS.index('nuclear')
    bits[selected&(base>0)]|=1<<REGIONS.index('interface_positive')
    atomic_npy(out/'region_bits.npy',bits)
    blocks,index,W=bootstrap_weights(meta['centroid_xy'],1000,replicates=2000,seed=20261019)
    atomic_npz(out/'bootstrap_weights_1000.npz',blocks=blocks,W=W)
    cells=pd.DataFrame(dict(cell_id=ids,x_um=meta['centroid_xy'][:,0],y_um=meta['centroid_xy'][:,1],block_1000=index,original_type_index=types))
    atomic_csv(out/'cells.csv',cells)
    distance=np.full(base.size,255,np.uint8)
    for path in sorted((out/'interfaces').glob('exposure_*.npz')):
        with np.load(path) as z:
            p=z['pixel'];keep=arrays['assignment'].ravel()[p]==z['interface'];distance[p[keep]]=z['distance_squared'][keep]
    distance=distance.reshape(base.shape)
    memberships=[]
    for region,use in [('GLOBAL',np.ones(base.shape,bool)),('INTERFACE_0_2',selected),('INTERFACE_0_1',selected&(distance<=1)),('INTERFACE_1_2',selected&(distance>1))]:
        chosen=np.unique(base[use]);chosen=chosen[chosen>0]
        memberships.extend((region,int(cid)) for cid in chosen)
    atomic_csv(out/'cell_memberships.csv',pd.DataFrame(memberships,columns=['region','cell_id']))
    genes=(TRAINING/'shared/data/all_gene_names.txt').read_text().splitlines()
    marker=np.isin(genes,['CD4','CD8A','CD8B','CD79A','CD79B','MS4A1'])
    if int(marker.sum())!=6:raise ValueError('six-marker panel incomplete')
    den=dict(global_all_gene=0,interface_all_gene=0,interface_six_marker=0)
    for i,rec in enumerate(coordinates()):
        a=np.load(rec['path'],mmap_mode='r');good=a['x']>=0;y,x=a['y'][good],a['x'][good]
        mbits=np.full(len(a),(1<<REGIONS.index('all'))|(1<<REGIONS.index('oob')),np.uint16);mbits[good]=bits[y,x]
        interface=np.full(len(a),-1,np.int32);interface[good]=arrays['assignment'][y,x]
        bi=block_indices(np.c_[a['x'],a['y']],blocks,1000,True)
        if np.any(bi<0):raise ValueError('raw RNA outside frozen nuclear block universe; no silent block imputation')
        path=out/'molecule_strata'/f'chunk_{i:04d}.npz';rp=path.with_suffix('.json');cd=dict(d,coordinate=rec)
        if rp.exists():check_receipt(rp,cd)
        else:
            atomic_npz(path,region_bits=mbits,interface_id=interface,block_1000=bi)
            write_receipt(rp,cd,{'strata':identity(path)})
        use=in_region(mbits,'interface');den['global_all_gene']+=len(a);den['interface_all_gene']+=int(use.sum());den['interface_six_marker']+=int((use&marker[a['gene']]).sum())
        progress('freeze_roi',i+1,len(coordinates()),**den)
    if den['global_all_gene']!=config()['eligible']:raise ValueError('ROI raw universe changed')
    atomic_json(out/'manifest.json',dict(**den,blocks=len(blocks),interfaces=len(pairs),seed=20261019,replicates=2000,side_um=1000,
        geometry='BASE positive-positive 8-neighbour interfaces; Euclidean<=2; nonnuclear; nearby background retained; unique distance_squared,i,j',
        local_F1_membership='cells with any BASE-owned pixel in the fixed region; complete own cell expression spectrum; overlaps retained',
        source=identity(SOURCE/'scripts/rep1_triple_simple_boundary/frozen_domains.py'),dependencies=d))
    return stage_outputs(out),dict(**den,blocks=len(blocks),interfaces=len(pairs))

def support_data():
    from scripts.boundary_three.core_refine import NuclearSupport
    from scripts.boundary_three.core_scores import reference_arrays
    meta=np.load(TRAINING/'shared/prepared/nucleus_metadata.npz');ids=meta['ids'];types=meta['type_index']
    ref=read(TRAINING/'shared/configs/reference_region.json');genes=(TRAINING/'shared/data/all_gene_names.txt').read_text().splitlines()
    indices,theta,valid=reference_arrays(ref,genes);nuc=tifffile.imread(TRAINING/'shared/data/nuclei.tif')
    settings=yaml.safe_load((TRAINING/'configs/native.yaml').read_text())
    support=NuclearSupport(nuc,ids,types,ref['type_names'],settings['model_params']['elongated'])
    if not np.array_equal(support.area,meta['area']) or not np.allclose(support.centroids,meta['centroid_xy'],rtol=0,atol=1e-8):raise ValueError('full nuclear support metadata differs')
    return dict(ids=ids,types=types,reference=ref,genes=genes,gene_indices=indices,theta=theta,valid=valid,nuclei=nuc,support=support,
        coordinates=coordinates(),expected_molecules=config()['eligible'])

def core_chunk(data,core,a):
    """Literal frozen _count_core chunk arithmetic, exposed for differential tests."""
    from scripts.boundary_three.core_scores import molecule_fold
    n=len(data['ids']);ng=len(data['gene_indices']);lookup=np.full(len(data['genes']),-1,int);lookup[data['gene_indices']]=np.arange(ng)
    good=a['x']>=0;owner=np.zeros(len(a),np.int64);owner[good]=core[a['y'][good],a['x'][good]]
    use=(owner>0)&(lookup[a['gene']]>=0);rr=np.searchsorted(data['ids'],owner[use])
    if not np.array_equal(data['ids'][rr],owner[use]):raise ValueError('core ID mapping')
    fold=molecule_fold(a['transcript_id'][use]);g=lookup[a['gene'][use]]
    keys,values=np.unique((fold.astype(np.int64)*n+rr)*ng+g,return_counts=True)
    return dict(keys=keys,counts=values,transcript_id=a['transcript_id'][use],fold=fold,row=rr.astype(np.int32),gene=g.astype(np.int16),pixel=a['y'][use].astype(np.int64)*core.shape[1]+a['x'][use])

def reference():
    from scripts.boundary_three.core_scores import molecule_fold,cross_scores,paired_bootstrap,decide,PARAMETERS
    data=support_data();out=ROOT/'method_reference';out.mkdir(exist_ok=True);d=base_deps()
    n=len(data['ids']);ng=len(data['gene_indices']);core,area=data['support'].core_image();lookup=np.full(313,-1,int);lookup[data['gene_indices']]=np.arange(ng)
    counts=np.zeros((2,n,ng),np.int64);rows=0
    for i,rec in enumerate(data['coordinates']):
        p=out/'count_chunks'/f'{i:04d}.npz';rp=p.with_suffix('.json');cd=dict(d,coordinate=rec,parameters=PARAMETERS)
        a=np.load(rec['path'],mmap_mode='r')
        if rp.exists():check_receipt(rp,cd)
        else:
            part=core_chunk(data,core,a);atomic_npz(p,**part)
            write_receipt(rp,cd,{'counts':identity(p)},core_G=int(part['counts'].sum()))
        with np.load(p) as z:counts.ravel()[z['keys']]+=z['counts']
        rows+=len(a);progress('core_count',i+1,len(data['coordinates']),molecules=rows)
    if rows!=config()['eligible']:raise ValueError('core raw denominator')
    atomic_npz(out/'core_counts.npz',counts=counts,ids=data['ids'],gene_indices=data['gene_indices'],core_area=area)
    stats=cross_scores(counts,data['types'],data['theta'],data['valid']);paired=stats['paired'];types=data['types']
    shuffle=np.full(n,np.nan);shuffle[paired]=stats['shuffle'][:,paired].mean(0)
    atomic_npz(out/'donor_tables.npz',donors=stats['donors'],fold_bins=stats['bins'],ids=data['ids'],eligible=stats['eligible'],paired=paired)
    atomic_csv(out/'cell_coverage.csv',pd.DataFrame(dict(cell_id=data['ids'],type_index=types,evaluable=stats['eligible'],paired=paired,core_area=area,F0=stats['fold_n'][0],F1=stats['fold_n'][1])))
    atomic_csv(out/'per_cell_scores.csv',pd.DataFrame(dict(cell_id=data['ids'],core=stats['core'],type=stats['type'],shuffle_mean=shuffle,evaluable=stats['eligible'],paired=paired)))
    def uncertainty(use,side,tag):
        v=np.c_[stats['core'][use],stats['type'][use],shuffle[use]]
        result,samples=paired_bootstrap(v,data['support'].centroids[use],side,lambda i,total:progress('reference_'+tag,i,total))
        atomic_npy(out/('bootstrap_'+tag+'.npy'),samples);return result
    uncertainties={str(side):uncertainty(paired,side,str(side)) for side in (500,1000,2000)}
    decision=decide(uncertainties['1000']);scope={};applicable=[]
    for typ in np.unique(types[paired]):
        result=decide(uncertainty(paired&(types==typ),1000,'type_'+str(typ)))
        scope[data['reference']['type_names'][int(typ)]]=dict(type_index=int(typ),**result)
        if result['science_status']=='GO_A2':applicable.append(int(typ))
    decision.update(A2_applicable_type_indices=applicable,parameters=PARAMETERS,total_original_cells=n,evaluable_cells=int(stats['eligible'].sum()),paired_cells=int(paired.sum()),
        A2_scope_rule='unchanged global/per-type A1 gate; unestablished types abstain',source=identity(SOURCE/'scripts/boundary_three/a_core.py'))
    atomic_json(out/'spatial_uncertainty.json',uncertainties);atomic_json(out/'type_scope.json',scope);atomic_json(out/'decision.json',decision)
    if decision['science_status']!='GO_A2':raise RuntimeError('Rep2 frozen method-reference global A1 gate '+decision['science_status']+'; no candidate refinement or compute completion')
    return stage_outputs(out),dict(global_gate=decision['science_status'],applicable_type_indices=applicable)

def domain_data():
    from scripts.boundary_three.core_scores import legal_types,matched_donors,SHUFFLE_SEEDS,MIN_FOLD,PARAMETERS
    d=support_data();decision=read(ROOT/'method_reference/decision.json')
    if decision['science_status']!='GO_A2' or decision['parameters']!=PARAMETERS:raise ValueError('method-reference gate/parameters')
    saved=np.load(ROOT/'method_reference/core_counts.npz');counts=saved['counts']
    if not np.array_equal(saved['ids'],d['ids']) or not np.array_equal(saved['gene_indices'],d['gene_indices']):raise ValueError('core IDs/genes differ')
    eligible=legal_types(d['types'],d['valid'])&np.all(counts.sum(2)>=MIN_FOLD,axis=0)&np.isin(d['types'],decision['A2_applicable_type_indices'])
    donor,bins=matched_donors(counts.sum(0),d['types'],eligible,SHUFFLE_SEEDS[0])
    d.update(baseline=tifffile.imread(spec(P2)['final']),parent=P2,spec=spec(P2),machine=machine(P2),eligible=eligible,donor=donor,bins=bins,
        prior=d['theta'][np.where(legal_types(d['types'],d['valid']),d['types'],0)])
    if not np.array_equal(d['baseline'][d['nuclei']>0],d['nuclei'][d['nuclei']>0]):raise ValueError('parent nuclear owner')
    return d

def domain():
    from scripts.rep1_triple_simple_boundary import gt
    d=domain_data();gt.APPLICABLE_TYPES=read(ROOT/'method_reference/decision.json')['A2_applicable_type_indices']
    out=ROOT/'shared_method/domain'
    _,_,info=gt.build_domain(d,out,dict(base_deps(),parent=identity(spec(P2)['final']),reference=identity(ROOT/'method_reference/decision.json')),cfg_progress)
    return stage_outputs(out),info

def refine(beta):
    from scipy import sparse
    from scripts.rep1_triple_simple_boundary import gt
    d=domain_data();name=B0 if beta==0 else BETA1;out=ROOT/'endpoints'/name/'gt';out.mkdir(parents=True,exist_ok=True)
    domainroot=ROOT/'shared_method/domain';link=out/'domain'
    if not link.exists():link.symlink_to(domainroot,target_is_directory=True)
    elif link.resolve()!=domainroot:raise ValueError('sibling shared domain differs')
    z=dict(np.load(domainroot/'domain.npz'));expr=sparse.load_npz(domainroot/'expression.npz')
    dep=dict(base_deps(),parent=identity(spec(P2)['final']),domain=identity(domainroot/'complete.json'),beta=beta)
    proposals,scoring=gt.score_domain(d,z,expr,out/'scores',dep,cfg_progress,beta=beta)
    mask,application=gt.apply_gt(d['baseline'],d['nuclei'],proposals,out/'recovery',dict(dep,proposals=identity(out/'scores/proposals.npy')),cfg_progress)
    final=Path(spec(name)['final']);atomic_tiff(final,mask.astype(np.uint32));materialize_machine(name)
    invariants=gt.validate_topology(d['baseline'],mask,d['nuclei'],d['ids'],proposals,out/'validation',dict(dep,final=identity(final)),cfg_progress)
    decisions=np.load(read(out/'recovery/state.json')['decisions']['path']);table=pd.DataFrame.from_records(proposals)
    table['decision']=[gt.REASONS[int(x)] for x in decisions];table['accepted']=decisions==1
    atomic_csv(out/'transfers.csv',table);atomic_json(out/'summary.json',dict(scoring=scoring,application=application,invariants=invariants,dependencies=dep))
    outputs=stage_outputs(out);outputs['final']=identity(final);outputs['machine']=identity(spec(name)['machine'])
    return outputs,dict(scoring=scoring,application=application,invariants=invariants)
