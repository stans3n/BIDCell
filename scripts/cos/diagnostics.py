"""COS-ownership-diagnostic-v1: explicit ID, frozen-label and stage diagnostics.

Rank marker scoring follows the accepted historical COS diagnostic (diagnose.py).
Nuclear density uses all fixed nuclei, independent of a comparison's QC population.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from scripts.crgf.common import atomic_json, identity, sha256, verify_outputs

VERSION = 'COS-ownership-diagnostic-v1'
CATS = ['background_to_background', 'background_to_cell', 'cell_to_background', 'same_cell', 'cell_to_cell']
BANDS = ['<=1', '(1,2]', '(2,3]', '(3,5]', '>5']
METRICS = [s+'_'+m for s in ('positive','negative') for m in ('F1','Precision','Recall')]
STATES = ['neutral','positive','negative','both','classifier_rejected','hierarchical_nonleaf_or_unmatched','unavailable_other','missing_historical_label','background','outside_reference_gene']


def read(path): return json.loads(Path(path).read_text())


def classify(new, old):
    new, old = np.asarray(new), np.asarray(old)
    if new.shape != old.shape or np.any(new < 0) or np.any(old < 0):
        raise ValueError('owner shape/domain mismatch')
    c = np.full(new.shape, 4, np.uint8)
    c[(old == 0) & (new == 0)] = 0
    c[(old == 0) & (new > 0)] = 1
    c[(old > 0) & (new == 0)] = 2
    c[(old == new) & (old > 0)] = 3
    return c


def load_anchor(spec):
    if sha256(spec['path']) != spec['sha256']: raise ValueError('frozen historical anchor hash changed')
    frame = pd.read_csv(spec['path'])
    if frame.cell_id.duplicated().any(): raise ValueError('duplicate historical anchor cell ID')
    column = 'evaluation_label' if 'evaluation_label' in frame else 'cell_type'
    labels=frame.set_index('cell_id')[column].fillna('__unavailable__:unavailable_other')
    if 'unmatched_reason' in frame:
        reasons=frame.set_index('cell_id').unmatched_reason.fillna('')
        for reason in ['classifier_rejected','hierarchical_nonleaf_or_unmatched']:
            labels.loc[reasons==reason]='__unavailable__:'+reason
    return labels


def fixed_scores(counts, ids, genes, labels, markers, reference_genes):
    """Same known-label score and NA rules as historical diagnostic, no new QC."""
    counts = np.asarray(counts)
    if counts.shape != (len(ids),len(genes)) or len(set(ids)) != len(ids): raise ValueError('counts/ID mapping mismatch')
    if not set(reference_genes) <= set(genes): raise ValueError('reference gene missing')
    x = counts[:,pd.Index(genes).get_indexer(reference_genes)]
    frame = pd.DataFrame(index=pd.Index(ids,name='cell_id'))
    frame['anchor_label'] = labels.reindex(ids).fillna('missing_historical_label')
    frame['label_availability'] = np.where(frame.anchor_label.eq('missing_historical_label'),'missing_historical_label',
        np.where(frame.anchor_label.isin(set(markers['positive']) & set(markers['negative'])),'known',frame.anchor_label.str.replace('__unavailable__:','',regex=False)))
    for sign in ('positive','negative'):
        values = np.full((len(ids),3),np.nan)
        for label in sorted(set(markers['positive']) & set(markers['negative'])):
            rows = np.flatnonzero(frame.anchor_label.to_numpy() == label)
            if not len(rows): continue
            marker = np.isin(reference_genes,markers[sign][label])
            xx=x[rows]
            predicted = xx > np.quantile(xx,1-marker.mean(),axis=1,method='linear')[:,None]
            tp = predicted[:,marker].sum(1)
            with np.errstate(divide='ignore',invalid='ignore'):
                precision=tp/predicted.sum(1); recall=tp/marker.sum(); f1=2*precision*recall/(precision+recall)
            f1[~np.isfinite(f1)]=0
            values[rows]=np.column_stack((f1,precision,recall))
        for col,name in enumerate(('F1','Precision','Recall')): frame[sign+'_'+name]=values[:,col]
    return frame


def fixed_comparison(methods, anchors, prepared, out, pairs, primary, population=None):
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    markers=read(Path(prepared)/'markers.json'); refgenes=(Path(prepared)/'reference_genes.txt').read_text().splitlines()
    tables={m:pd.read_csv(Path(s['evaluation'])/'paper_aligned_v1/markers_per_cell.csv').set_index('cell_id') for m,s in methods.items()}
    for t in tables.values():
        if t.index.has_duplicates: raise ValueError('duplicate QC cell ID')
    common=sorted(set.intersection(*(set(t.index) for t in tables.values()))) if population is None else sorted(population)
    pd.DataFrame({'cell_id':common}).to_csv(out/'common_qc_cell_ids.csv',index=False)
    values={}; summary=[]; availability=[]
    for anchor_name, anchor_spec in anchors.items():
        labels=load_anchor(anchor_spec); values[anchor_name]={}
        for m,spec in methods.items():
            matrix=Path(spec['matrix']); ids=pd.read_csv(matrix/'cell_mapping.csv').canonical_cell_id.to_numpy()
            genes=pd.read_csv(matrix/'gene_mapping.csv').gene.tolist(); positions=pd.Index(ids).get_indexer(common)
            if np.any(positions<0): raise ValueError('QC ID absent from own matrix')
            frame=fixed_scores(np.load(matrix/'counts.npy',mmap_mode='r')[positions],common,genes,labels,markers,refgenes)
            values[anchor_name][m]=frame
            frame.to_csv(out/f'fixed_{anchor_name}_{m}.csv',na_rep='NA')
            for state,n in frame.label_availability.value_counts().items(): availability.append(dict(anchor=anchor_name,method=m,status=state,n_cells=int(n),n_common_qc=len(common)))
        for a,b in pairs:
            delta=values[anchor_name][a][METRICS]-values[anchor_name][b][METRICS]
            delta.insert(0,'anchor_label',values[anchor_name][a].anchor_label)
            delta.to_csv(out/f'delta_{anchor_name}_{a}_minus_{b}.csv',na_rep='NA')
            for label,part in [('__all__',delta),*list(delta.groupby('anchor_label',dropna=False))]:
                for metric in METRICS:
                    finite=np.isfinite(part[metric]); n=int(finite.sum())
                    summary.append(dict(comparison=a+'-'+b,anchor=anchor_name,role='primary' if primary.get((a,b))==anchor_name else 'exploratory_all_anchor',population='common_qc_of_'+','.join(methods),type=label,metric=metric,n_eligible=len(part),n_evaluable=n,n_unavailable=len(part)-n,mean_delta=float(part.loc[finite,metric].mean()) if n else None,status='computed' if n else 'empty_evaluable_domain'))
    pd.DataFrame(summary).to_csv(out/'fixed_anchor_summary.csv',index=False,na_rep='NA')
    pd.DataFrame(availability).to_csv(out/'label_availability.csv',index=False)
    return dict(common_qc_n=len(common),anchors={k:identity(v['path']) for k,v in anchors.items()},summary=str(out/'fixed_anchor_summary.csv'))


def boundary_bins(mask, pixel_um):
    boundary=np.zeros(mask.shape,bool)
    v=mask[1:]!=mask[:-1]; boundary[1:]|=v; boundary[:-1]|=v
    v=mask[:,1:]!=mask[:,:-1]; boundary[:,1:]|=v; boundary[:,:-1]|=v
    if not boundary.any(): return np.full(mask.shape,4,np.uint8)
    return np.searchsorted([1,2,3,5],ndi.distance_transform_edt(~boundary,sampling=pixel_um),side='left').astype(np.uint8)


def components(mask, ids):
    result=np.zeros(len(ids),np.int32); boxes=ndi.find_objects(mask,max_label=int(max(ids,default=0)))
    for j,cid in enumerate(ids):
        box=boxes[int(cid)-1] if int(cid)<=len(boxes) else None
        if box is not None: result[j]=ndi.label(mask[box]==cid,structure=np.ones((3,3),bool))[1]
    return result


def marker_lookup(labels, markers, genes, refgenes, maximum):
    typenames=sorted(set(markers['positive'])&set(markers['negative']))
    unavailable=['classifier_rejected','hierarchical_nonleaf_or_unmatched','unavailable_other','missing_historical_label','background']
    names=typenames+unavailable
    code=np.full(maximum+1,len(typenames)+3,np.int16);code[0]=len(typenames)+4
    lookup=np.zeros((len(names),len(genes)),np.uint8)
    for j,state in enumerate(unavailable):lookup[len(typenames)+j,:]=STATES.index(state)
    for i,t in enumerate(typenames):
        valid=labels.index[(labels==t)&(labels.index<=maximum)&(labels.index>0)].to_numpy(int);code[valid]=i
        lookup[i]=[int(g in markers['positive'][t])+2*int(g in markers['negative'][t]) if g in refgenes else STATES.index('outside_reference_gene') for g in genes]
    for cid,label in labels.items():
        if 0<int(cid)<=maximum and label not in typenames:
            reason=str(label).replace('__unavailable__:','')
            if reason not in unavailable[:3]:reason='unavailable_other'
            code[int(cid)]=len(typenames)+unavailable.index(reason)
    return lookup,code,names


def ownership_pair(new_spec, old_spec, out, anchor_spec, prepared, comparison):
    """Full pixel/transcript accounting with explicit uint64 row identity and counts."""
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    specs=[old_spec,new_spec];cfg=[read(s['machine']) for s in specs];roots=[Path(s['matrix']) for s in specs]
    manifests=[read(r/'complete.json') for r in roots]
    for m in manifests:
        if m.get('status')!='succeeded' or m.get('validation_kind')!='independent_full_native_resize_lookup_and_conservation': raise ValueError('unaccepted own matrix')
        verify_outputs(m)
    if manifests[0]['dependencies']['transcripts']['sha256']!=manifests[1]['dependencies']['transcripts']['sha256']:raise ValueError('different transcript source')
    if manifests[0]['contract']!=manifests[1]['contract']:raise ValueError('coordinate contract mismatch')
    maps=[pd.read_csv(r/'cell_mapping.csv').canonical_cell_id.to_numpy(int) for r in roots]
    genes=[pd.read_csv(r/'gene_mapping.csv').gene.tolist() for r in roots]
    if not np.array_equal(*maps) or genes[0]!=genes[1]:raise ValueError('row/gene identity mismatch')
    ids=maps[0];genes=genes[0];ng=len(genes);maximum=int(ids.max(initial=0));n=maximum+1
    masks=[tifffile.imread(c['final_mask']) for c in cfg]
    if masks[0].shape!=masks[1].shape or cfg[0]['pixel_um']!=cfg[1]['pixel_um']:raise ValueError('mask geometry mismatch')
    for i in range(2):
        if sha256(cfg[i]['final_mask'])!=manifests[i]['dependencies']['mask']['sha256']:raise ValueError('own mask hash changed')
    labels=load_anchor(anchor_spec);mk=read(Path(prepared)/'markers.json');refgenes=(Path(prepared)/'reference_genes.txt').read_text().splitlines()
    lookup,typecode,types=marker_lookup(labels,mk,genes,refgenes,maximum)
    coordinates=pd.read_csv(Path(old_spec['evaluation'])/'engineering/structure/annotation_nuclear.csv').set_index('cell_id')
    if coordinates.index.has_duplicates or not set(ids)<=set(coordinates.index):raise ValueError('fixed nuclear centroids missing/duplicate')
    radius=np.full(n,np.nan);density=np.full(n,4,np.uint8)
    if len(coordinates)>=6:
        r=cKDTree(coordinates[['x_um','y_um']]).query(coordinates[['x_um','y_um']],k=6,workers=1)[0][:,-1]
        quartiles=pd.qcut(pd.Series(r),4,labels=False,duplicates='drop').fillna(0).to_numpy(np.uint8)
        radius[coordinates.index]=r;density[coordinates.index]=quartiles
    pd.DataFrame(dict(cell_id=ids,fifth_neighbour_radius_um=radius[ids],density_quartile=density[ids])).to_csv(out/'fixed_nuclear_density.csv',index=False)
    frozen_boundary=tifffile.imread(cfg[0]['baseline_final_mask'])
    if frozen_boundary.shape!=masks[0].shape or sha256(cfg[0]['baseline_final_mask'])!=sha256(cfg[1]['baseline_final_mask']):raise ValueError('fixed historical boundary reference mismatch')
    bands=boundary_bins(frozen_boundary,cfg[0]['pixel_um']);pixel=np.zeros(5,np.int64);pixel_band=np.zeros((5,5),np.int64);pgain=np.zeros(n,np.int64);ploss=pgain.copy()
    for y in range(0,masks[0].shape[0],256):
        old,new=[m[y:y+256].ravel() for m in masks];cat=classify(new,old);changed=new!=old
        pixel+=np.bincount(cat,minlength=5);pixel_band+=np.bincount(cat*5+bands[y:y+256].ravel(),minlength=25).reshape(5,5)
        pgain+=np.bincount(new[changed],minlength=n);ploss+=np.bincount(old[changed],minlength=n)
    counts=[np.load(r/'counts.npy',mmap_mode='r') for r in roots]
    rebuilt=[np.zeros((n,ng),np.int64) for _ in range(2)]
    trans=np.zeros(5,np.int64);gene=np.zeros((5,ng),np.int64);band=np.zeros((5,5),np.int64);mark=np.zeros((5,len(STATES),len(STATES)),np.int64)
    dens=np.zeros((5,5,5),np.int64);typed=np.zeros((5,len(types),len(types)),np.int64);rows=0;gain=np.zeros(n,np.int64);loss=gain.copy()
    if len(manifests[0]['receipts'])!=len(manifests[1]['receipts']):raise ValueError('chunk coverage differs')
    for rec0,rec1 in zip(manifests[0]['receipts'],manifests[1]['receipts']):
        for k in ('row_start','row_stop','rows','coordinate_sha256'):
            if rec0[k]!=rec1[k]:raise ValueError('transcript row identity differs: '+k)
        if rec0['row_start']!=rows:raise ValueError('transcript rows not contiguous')
        co=np.load(rec0['coordinate_file'],mmap_mode='r')
        if co.dtype['transcript_id']!=np.dtype('uint64') or sha256(rec0['coordinate_file'])!=rec0['coordinate_sha256']:raise ValueError('explicit uint64 transcript identity missing/corrupt')
        valid=(co['x']>=0)&(co['y']>=0)&(co['x']<masks[0].shape[1])&(co['y']<masks[0].shape[0]);owners=[]
        for i,rec in enumerate((rec0,rec1)):
            if sha256(rec['coordinate_file'])!=rec['coordinate_sha256']:raise ValueError('coordinate input changed')
            path=roots[i]/rec['owner_file'];owner=np.load(path);expected=np.zeros(len(co),np.uint32);expected[valid]=masks[i][co['y'][valid],co['x'][valid]]
            if sha256(path)!=rec['sha256'] or not np.array_equal(owner,expected):raise ValueError('own owner mismatch')
            np.add.at(rebuilt[i],(owner,co['gene']),1);owners.append(owner)
        old,new=owners;cat=classify(new,old);g=co['gene'];rows+=len(co)
        trans+=np.bincount(cat,minlength=5);gene+=np.bincount(cat.astype(np.int64)*ng+g,minlength=5*ng).reshape(5,ng)
        b=np.full(len(co),4,np.uint8);b[valid]=bands[co['y'][valid],co['x'][valid]];band+=np.bincount(cat*5+b,minlength=25).reshape(5,5)
        source=lookup[typecode[old],g];dest=lookup[typecode[new],g]
        ns=len(STATES);mark+=np.bincount(cat.astype(np.int64)*ns*ns+source*ns+dest,minlength=5*ns*ns).reshape(5,ns,ns)
        dens+=np.bincount(cat.astype(np.int64)*25+density[old]*5+density[new],minlength=125).reshape(5,5,5)
        nt=len(types);typed+=np.bincount(cat.astype(np.int64)*nt*nt+typecode[old]*nt+typecode[new],minlength=5*nt*nt).reshape(5,nt,nt)
        changed=old!=new;gain+=np.bincount(new[changed],minlength=n);loss+=np.bincount(old[changed],minlength=n)
    for i in range(2):
        if rows!=manifests[i]['totals']['eligible'] or not np.array_equal(rebuilt[i][ids],counts[i]):raise ValueError('cell-gene conservation failed')
    areas=[np.bincount(m.ravel(),minlength=n) for m in masks]
    if not np.array_equal((pgain-ploss)[ids],(areas[1]-areas[0])[ids]) or not np.array_equal((gain-loss)[ids],counts[1].sum(1)-counts[0].sum(1)):raise ValueError('gain-loss conservation failed')
    if int(trans[1]-trans[2])!=int(counts[1].sum()-counts[0].sum()):raise ValueError('capture conservation failed')
    comps=[components(m,ids) for m in masks]
    ledger=pd.DataFrame(dict(cell_id=ids,old_area_pixels=areas[0][ids],new_area_pixels=areas[1][ids],pixel_gain=pgain[ids],pixel_loss=ploss[ids],transcript_gain=gain[ids],transcript_loss=loss[ids],old_components_8=comps[0],new_components_8=comps[1],new_disconnected=(comps[0]<=1)&(comps[1]>1),restored_connected=(comps[0]>1)&(comps[1]==1),persistently_disconnected=(comps[0]>1)&(comps[1]>1)))
    ledger.to_csv(out/'cell_ledger.csv',index=False)
    transitions=[];boundaries=[]
    for unit,ct,bt,den in [('pixel',pixel,pixel_band,masks[0].size),('transcript',trans,band,rows)]:
        if ct.sum()!=den:raise ValueError('transition partition failed')
        for i,name in enumerate(CATS):
            transitions.append(dict(comparison=comparison,unit=unit,transition=name,n=int(ct[i]),denominator=int(den),fraction=float(ct[i]/den) if den else None))
            for j,label in enumerate(BANDS):boundaries.append(dict(unit=unit,transition=name,distance_bin_um=label,n=int(bt[i,j]),denominator_bin=int(bt[:,j].sum()),denominator_transition=int(ct[i])))
    pd.DataFrame(transitions).to_csv(out/'ownership_transitions.csv',index=False);pd.DataFrame(boundaries).to_csv(out/'boundary_transitions.csv',index=False)
    pd.DataFrame([dict(transition=CATS[c],gene=genes[g],n=int(gene[c,g]),denominator=int(trans[c])) for c in range(5) for g in range(ng)]).to_csv(out/'gene_composition.csv',index=False)
    for name,array,names in [('marker_consistency',mark,STATES),('density_transitions',dens,list(range(5))),('type_ownership_transitions',typed,types)]:
        pd.DataFrame([dict(transition=CATS[c],source=names[s],destination=names[d],n=int(array[c,s,d]),denominator=int(trans[c])) for c in range(5) for s in range(len(names)) for d in range(len(names))]).to_csv(out/(name+'.csv'),index=False)
    result=dict(status='succeeded',version=VERSION,comparison=comparison,transcript_denominator=rows,pixel_denominator=int(masks[0].size),integer_conservation=True,boundary_reference=identity(cfg[0]['baseline_final_mask']),marker_interpretation='fixed-label marker consistency; not true-owner accuracy',density_domain='all fixed nuclear centroids; fifth-neighbour distance; quartiles; 4 means unavailable',inputs={'anchor':identity(anchor_spec['path']),**{str(i):identity(r/'complete.json') for i,r in enumerate(roots)}})
    atomic_json(out/'validation.json',result);return result


def stage_diagnostics(spec, anchors, prepared, out):
    """Saved whole-slide ID endpoints; intermediate counts use frozen labels only."""
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    machine=read(spec['machine']); matrix=Path(spec['matrix']);manifest=read(matrix/'complete.json')
    endpoint_path=Path(spec.get('endpoints',Path(machine['final_mask']).parent/'endpoints.json'))
    endpoints=read(endpoint_path); entries=endpoints['endpoints']
    required={'raw_shift0','raw_shift24','shift0','shift24','pre','tile_merge','split_cleanup','before_global_nuclear_restore','final'}
    if not required<=set(entries):raise ValueError('missing native ownership endpoints: '+str(sorted(required-set(entries))))
    ids=pd.read_csv(matrix/'cell_mapping.csv').canonical_cell_id.to_numpy(int);maximum=int(ids.max(initial=0));n=maximum+1
    genes=pd.read_csv(matrix/'gene_mapping.csv').gene.tolist();ng=len(genes)
    mk=read(Path(prepared)/'markers.json');refgenes=(Path(prepared)/'reference_genes.txt').read_text().splitlines()
    anchor_labels={name:load_anchor(spec) for name,spec in anchors.items()}
    masks={}
    for name,obj in entries.items():
        if sha256(obj['path'])!=obj['sha256']:raise ValueError('stage endpoint changed: '+name)
        masks[name]=tifffile.imread(obj['path'])
        if np.any(masks[name]>maximum):raise ValueError('unknown cell ID in endpoint')
    shape=masks['final'].shape
    if any(x.shape!=shape for x in masks.values()):raise ValueError('endpoint shapes differ')
    nuclei=tifffile.imread(machine['nuclei_mask'])
    if not np.array_equal(masks['final'],np.where(nuclei>0,nuclei,masks['before_global_nuclear_restore'])):raise ValueError('global nuclear restore endpoint mismatch')
    if not np.array_equal(masks['split_cleanup'],masks['before_global_nuclear_restore']):raise ValueError('undocumented step before global nuclear restore')
    links=[('raw_shift0','shift0','patch_nuclear_restore_shift0'),('raw_shift24','shift24','patch_nuclear_restore_shift24'),('shift0','pre','dual_shift_stitch'),('pre','tile_merge','ordered_tile_merge'),('tile_merge','split_cleanup','split_cleanup'),('before_global_nuclear_restore','final','global_nuclear_restore'),('pre','final','aggregate_postprocessing')]
    transitions={op:{unit:np.zeros(5,np.int64) for unit in ('pixel','transcript')} for _,_,op in links}
    snapshots={};scores=[];cells={};count_files={}
    common=pd.read_csv(Path(spec['evaluation'])/'paper_aligned_v1/markers_per_cell.csv').cell_id.to_numpy(int)
    positions=pd.Index(ids).get_indexer(common)
    if np.any(positions<0):raise ValueError('final QC outside cell-ID domain')
    # Each endpoint independently accounts every transcript; no annotation reruns.
    for name,mask in masks.items():
        counts=np.zeros((n,ng),np.int64);den=0
        for rec in manifest['receipts']:
            if sha256(rec['coordinate_file'])!=rec['coordinate_sha256']:raise ValueError('coordinate identity changed')
            co=np.load(rec['coordinate_file'],mmap_mode='r');valid=(co['x']>=0)&(co['y']>=0)&(co['x']<shape[1])&(co['y']<shape[0]);owner=np.zeros(len(co),np.uint32);owner[valid]=mask[co['y'][valid],co['x'][valid]]
            np.add.at(counts,(owner,co['gene']),1);den+=len(co)
        if den!=manifest['totals']['eligible'] or int(counts.sum())!=den:raise ValueError('endpoint transcript conservation')
        if name=='final' and not np.array_equal(counts[ids],np.load(matrix/'counts.npy',mmap_mode='r')):raise ValueError('final endpoint own-count mismatch')
        area=np.bincount(mask.ravel(),minlength=n);comp=components(mask,ids)
        table=pd.DataFrame(dict(cell_id=ids,area_pixels=area[ids],components_8=comp,assigned_transcripts=counts[ids].sum(1)))
        table.to_csv(out/f'cells_{name}.csv',index=False);cells[name]=table
        count_file=out/f'counts_{name}.npz';np.savez_compressed(count_file,cell_id=ids,counts=counts[ids]);count_files[name]=count_file
        snapshots[name]=dict(stage=name,n_domain_cells=len(ids),n_present=int((area[ids]>0).sum()),n_disconnected=int((comp>1).sum()),pixel_denominator=int(mask.size),foreground_pixels=int(area[ids].sum()),transcript_denominator=den,assigned_transcripts=int(counts[ids].sum()))
        for anchor,anchor_spec in anchors.items():
            frame=fixed_scores(counts[ids][positions],common,genes,anchor_labels[anchor],mk,refgenes)
            frame.to_csv(out/f'fixed_{anchor}_{name}.csv',na_rep='NA')
            for metric in METRICS:
                finite=np.isfinite(frame[metric]);count=int(finite.sum());scores.append(dict(anchor=anchor,stage=name,metric=metric,n_eligible=len(frame),n_evaluable=count,n_unavailable=len(frame)-count,mean=float(frame.loc[finite,metric].mean()) if count else None,population='own_final_qc_fixed_across_stages'))
        del counts
    # Stream links so denominators and per-cell gain/loss can be checked separately.
    rows=[];connect=[];composition=[];gene_rows=[];boundary_rows=[]
    centroids=pd.read_csv(Path(spec['evaluation'])/'engineering/structure/annotation_nuclear.csv').set_index('cell_id')
    density=np.full(n,4,np.uint8)
    if len(centroids)>=6:
        radius=cKDTree(centroids[['x_um','y_um']]).query(centroids[['x_um','y_um']],k=6,workers=1)[0][:,-1]
        density[centroids.index]=pd.qcut(pd.Series(radius),4,labels=False,duplicates='drop').fillna(0).to_numpy(np.uint8)
    primary_name=spec.get('base_variant',Path(machine['final_mask']).parent.name.split('_')[0])
    primary_names=[primary_name] if primary_name in anchors else ['B1','G']
    lookups={a:marker_lookup(anchor_labels[a],mk,genes,refgenes,maximum) for a in primary_names}
    fixed_boundary=tifffile.imread(machine['baseline_final_mask'])
    if fixed_boundary.shape!=shape:raise ValueError('fixed historical boundary reference shape mismatch')
    bounds=boundary_bins(fixed_boundary,machine['pixel_um'])
    for old,new,op in links:
        bt=np.zeros((5,5),np.int64)
        gene_transition=np.zeros((5,ng),np.int64);gene_delta=np.zeros((n,ng),np.int64)
        dimensions={a:{'marker_consistency':(np.zeros((5,len(STATES),len(STATES)),np.int64),STATES),'type_ownership':(np.zeros((5,len(lookup[2]),len(lookup[2])),np.int64),lookup[2]),'density':(np.zeros((5,5,5),np.int64),list(range(5)))} for a,lookup in lookups.items()}
        for y in range(0,shape[0],256):transitions[op]['pixel']+=np.bincount(classify(masks[new][y:y+256],masks[old][y:y+256]).ravel(),minlength=5)
        gain=np.zeros(n,np.int64);loss=gain.copy()
        for rec in manifest['receipts']:
            co=np.load(rec['coordinate_file'],mmap_mode='r');v=(co['x']>=0)&(co['y']>=0)&(co['x']<shape[1])&(co['y']<shape[0]);a=np.zeros(len(co),np.uint32);b=a.copy();a[v]=masks[new][co['y'][v],co['x'][v]];b[v]=masks[old][co['y'][v],co['x'][v]]
            category=classify(a,b);transitions[op]['transcript']+=np.bincount(category,minlength=5);changed=a!=b;gain+=np.bincount(a[changed],minlength=n);loss+=np.bincount(b[changed],minlength=n)
            g=co['gene'];np.add.at(gene_delta,(a[changed],g[changed]),1);np.add.at(gene_delta,(b[changed],g[changed]),-1)
            gene_transition+=np.bincount(category.astype(np.int64)*ng+g,minlength=5*ng).reshape(5,ng)
            band=np.full(len(co),4,np.uint8);band[v]=bounds[co['y'][v],co['x'][v]];bt+=np.bincount(category*5+band,minlength=25).reshape(5,5)
            for anchor,(lookup,code,types) in lookups.items():
                pairs={'marker_consistency':(lookup[code[b],g],lookup[code[a],g]),'type_ownership':(code[b],code[a]),'density':(density[b],density[a])}
                for measure,(source,dest) in pairs.items():
                    arr,names=dimensions[anchor][measure];size=len(names);arr+=np.bincount(category.astype(np.int64)*size*size+source*size+dest,minlength=5*size*size).reshape(5,size,size)
        if not np.array_equal((gain-loss)[ids],cells[new].assigned_transcripts-cells[old].assigned_transcripts):raise ValueError('stage per-cell count conservation')
        with np.load(count_files[new]) as cc1,np.load(count_files[old]) as cc0:
            if not np.array_equal(gene_delta[ids],cc1['counts']-cc0['counts']):raise ValueError('stage per-cell/gene gain-loss conservation')
        for category in range(5):
            for j in range(5):boundary_rows.append(dict(operation=op,transition=CATS[category],distance_bin_um=BANDS[j],n=int(bt[category,j]),denominator_bin=int(bt[:,j].sum()),denominator_transition=int(bt[category].sum())))
            for j,gene in enumerate(genes):gene_rows.append(dict(operation=op,transition=CATS[category],gene=gene,n=int(gene_transition[category,j]),denominator=int(gene_transition[category].sum())))
        for anchor,arrays in dimensions.items():
            for measure,(arr,names) in arrays.items():
                for category in range(5):
                    for source in range(len(names)):
                        for dest in range(len(names)):composition.append(dict(operation=op,anchor=anchor,measure=measure,transition=CATS[category],source=names[source],destination=names[dest],n=int(arr[category,source,dest]),denominator=int(arr[category].sum())))
        for unit,ct in transitions[op].items():
            den=shape[0]*shape[1] if unit=='pixel' else manifest['totals']['eligible'];key='foreground_pixels' if unit=='pixel' else 'assigned_transcripts'
            if ct.sum()!=den or int(ct[1]-ct[2])!=snapshots[new][key]-snapshots[old][key]:raise ValueError('stage transition conservation')
            for category,value in zip(CATS,ct):rows.append(dict(from_stage=old,to_stage=new,operation=op,unit=unit,transition=category,n=int(value),denominator=int(den)))
        c0=cells[old].components_8;c1=cells[new].components_8
        connect.append(dict(operation=op,n_domain_cells=len(ids),new_disconnected=int(((c0<=1)&(c1>1)).sum()),restored_connected=int(((c0>1)&(c1==1)).sum()),persistently_disconnected=int(((c0>1)&(c1>1)).sum())))
    pd.DataFrame(rows).to_csv(out/'stage_effects.csv',index=False);pd.DataFrame(list(snapshots.values())).to_csv(out/'stage_snapshots.csv',index=False)
    pd.DataFrame(scores).to_csv(out/'stage_fixed_anchor_summary.csv',index=False,na_rep='NA');pd.DataFrame(connect).to_csv(out/'stage_connectivity.csv',index=False)
    pd.DataFrame(composition).to_csv(out/'stage_marker_type_density_transitions.csv',index=False)
    pd.DataFrame(gene_rows).to_csv(out/'stage_gene_composition.csv',index=False);pd.DataFrame(boundary_rows).to_csv(out/'stage_boundary_transitions.csv',index=False)
    result=dict(status='succeeded',version=VERSION,endpoints=identity(endpoint_path),boundary_reference=identity(machine['baseline_final_mask']),integer_conservation=True,cell_gene_gain_loss_verified=True,all_required_endpoints=True,annotation_reruns=0,count_artifacts={k:identity(p) for k,p in count_files.items()})
    atomic_json(out/'validation.json',result);return result


def probe_diagnostics(probes,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    manifest=probes['manifest']
    if sha256(manifest['path'])!=manifest['sha256']:raise ValueError('frozen probe manifest changed')
    frozen=read(manifest['path']);expected={(int(x['shift']),int(x['index'])) for x in frozen['probes']};seen=set();rows=[]
    import re
    required={'binary_logits','corrected_logits','base_binary_probabilities','corrected_binary_probabilities','residual','candidate_ids','candidate_valid','decoder_support','background_score','q','margin','cell_only_margin','entropy','nuclei','anchor_mask','nuclear_owner_coverage','raw_owner','restored_owner','k','search_overlap','top1_id','top2_id','top1_score','top2_score','cell_top1_id','cell_top2_id','anchor_ce'}
    for obj in probes['files']:
        if sha256(obj['path'])!=obj['sha256']:raise ValueError('probe file hash changed')
        match=re.fullmatch(r'shift(\d+)_(\d+)\.npz',Path(obj['path']).name)
        if match is None:raise ValueError('unrecognised probe identity')
        key=tuple(map(int,match.groups()))
        if key not in expected or key in seen:raise ValueError('unexpected/duplicate probe')
        seen.add(key)
        with np.load(obj['path']) as z:
            if not required<=set(z.files):raise ValueError('required probe fields missing '+str(required-set(z.files)))
            k=int(z['k']);ids=z['candidate_ids']
            if len(ids)!=k or np.any(ids<=0) or not np.array_equal(ids,np.sort(np.unique(ids))):raise ValueError('probe candidate contract')
            for name in ['binary_logits','corrected_logits','base_binary_probabilities','corrected_binary_probabilities','residual','q','background_score','margin','entropy']:
                if not np.isfinite(z[name]).all():raise ValueError('nonfinite probe '+name)
            if not np.allclose(z['q'].sum(0),1,atol=1e-6):raise ValueError('probe ownership normalization')
            if not np.array_equal(z['raw_owner'],z['top1_id']):raise ValueError('probe decode/ranking mismatch')
            anchor=z['anchor_mask'].astype(bool)
            if not z['nuclear_owner_coverage'][anchor].all():raise ValueError('nuclear true owner outside candidates')
            ce=np.full(anchor.shape,np.nan);ce[anchor]=z['anchor_ce'];overlap=np.asarray(z['search_overlap'])
            if overlap.size!=anchor.size:raise ValueError('probe SearchArea overlap shape mismatch')
            support=overlap.reshape(anchor.shape)>0
            for region,sel in [('all',np.ones(anchor.shape,bool)),('nuclear',anchor),('extranuclear',~anchor)]:
                for search,domain in [('all',sel),('searcharea_inside',sel&support),('searcharea_outside',sel&~support)]:
                    n=int(domain.sum());nuclear=domain&anchor;count=int(nuclear.sum())
                    rows.append(dict(shift=key[0],patch_index=key[1],k=k,k_stratum='0' if k==0 else '1' if k==1 else '2-4' if k<=4 else '5+',region=region,searcharea=search,n_pixels=n,n_anchor_pixels=count,raw_correct_nuclear=int(((z['raw_owner']==z['nuclei'])&nuclear).sum()),nuclear_top1=float((z['raw_owner'][nuclear]==z['nuclei'][nuclear]).mean()) if count else None,anchor_ce=float(ce[nuclear].mean()) if count else None,mean_margin=float(z['margin'][domain].mean()) if n else None,mean_cell_only_margin=float(z['cell_only_margin'][domain].mean()) if n and k>1 else None,mean_entropy=float(z['entropy'][domain].mean()) if n else None,raw_background_pixels=int(((z['raw_owner']==0)&domain).sum()),restoration_changed_pixels=int(((z['raw_owner']!=z['restored_owner'])&domain).sum()),nuclear_owner_coverage_n=int(z['nuclear_owner_coverage'][nuclear].sum()),nuclear_owner_coverage_den=count,outside_nuclear_true_owner_coverage='unavailable_no_whole_cell_truth',log_probability_owner_mismatch=int(z.get('log_probability_owner_mismatch',0))))
    if seen!=expected:raise ValueError('missing frozen probes '+str(expected-seen))
    frame=pd.DataFrame(rows);frame.to_csv(out/'probe_strata.csv',index=False,na_rep='NA')
    atomic_json(out/'validation.json',dict(status='succeeded',version=VERSION,n_expected=len(expected),n_observed=len(seen),manifest=manifest,selection=frozen.get('selection'),selection_denominators=frozen.get('denominator'),uncalibrated_q=True,ce_lower_bound='log(2 - 1/K) for K>0; density CE is not directly comparable',whole_cell_true_owner='unavailable'))
    return dict(n_expected=len(expected),n_observed=len(seen))


def postprocess_increment(methods,anchors,pairs,common_ids,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);rows=[]
    for a,b in pairs:
        for anchor in anchors:
            values={}
            for arm in [a,b]:
                for stage in ['pre','final']:
                    table=pd.read_csv(Path(methods[arm]['evaluation'])/'diagnostics/stages'/f'fixed_{anchor}_{stage}.csv').set_index('cell_id')
                    values[(arm,stage)]=table.loc[common_ids]
            for metric in METRICS:
                pre=values[(a,'pre')][metric]-values[(b,'pre')][metric];final=values[(a,'final')][metric]-values[(b,'final')][metric]
                finite=np.isfinite(pre)&np.isfinite(final);n=int(finite.sum())
                rows.append(dict(comparison=a+'-'+b,anchor=anchor,metric=metric,n_common_qc=len(common_ids),n_evaluable=n,n_unavailable=len(common_ids)-n,pre_cos_increment=float(pre[finite].mean()) if n else None,final_cos_increment=float(final[finite].mean()) if n else None,postprocess_rewrite=float((final-pre)[finite].mean()) if n else None,status='computed' if n else 'empty_evaluable_domain'))
    pd.DataFrame(rows).to_csv(out/'postprocess_cos_increment.csv',index=False,na_rep='NA')
