"""Read-only adapters from the audited 30-method registry and explicit parents."""
from pathlib import Path
import numpy as np
import pandas as pd
import tifffile
from .contracts import (INPUT_SCHEMA,INSPECTED_SHA256,PLAN_MANIFEST_SHA256,DAG_SHA256,
                        EXISTING,NEW,PARENTS,GT_ENDPOINTS,N_CELLS,N_GENES,ELIGIBLE)
from .common import (read,identity,check_identity,sha256,digest,atomic_json,within,
                     check_receipt,chunk_receipt,progress,write_receipt)

OLD_SPATIAL_TRACE='/home/zwan0726/work/docs/BIDCell_Rep1_Triple_Simple_RootOld6SpatialTrace_20261017.json'
OLD_SPATIAL_TRACE_SHA256='63656c51af6860498c4a19421500f598cb7846ca3d0642f9eefc4ad154c15040'
OLD_BT_TRACE='/home/zwan0726/work/docs/BIDCell_Rep1_Triple_Simple_RootOld7BTTrace_20261017.json'
OLD_BT_TRACE_SHA256='6799216927b886d722cd6812e33451391342ecdcf2e0370963647ee2fb4123bc'

def validate_coordinate_cache(cache_path, coordinates, machine, matrix_receipt,
                              expected_rows=ELIGIBLE, expected_chunks=69):
    """Admit only the complete historical coordinate producer identity.

    The independent CSV oracle remains mandatory downstream. This check is a
    separate preflight guard against accepting a cache built with other rounding,
    resize, sources, library versions, shape or row coverage.
    """
    import hashlib,json,cv2
    from scripts.crgf import matrix as builder,common as builder_common
    cache_path=Path(cache_path);cache=read(cache_path)
    with tifffile.TiffFile(machine['final_mask']) as tif:shape=list(tif.series[0].shape)
    expected=dict(transcripts=sha256(machine['transcripts']),genes=sha256(machine['genes']),
                  shape=shape,scale_x=machine['scale_x'],scale_y=machine['scale_y'],
                  builder=sha256(builder.__file__),common=sha256(builder_common.__file__),
                  numpy=np.__version__,opencv=cv2.__version__,pandas=pd.__version__)
    if cache.get('dependencies')!=expected or cache.get('duplicate_ids')!=0:
        raise ValueError('coordinate cache complete producer identity differs')
    key=hashlib.sha256(json.dumps(expected,sort_keys=True).encode()).hexdigest()
    if cache_path.parent.name!=key:raise ValueError('coordinate cache key/directory differs')
    contract=matrix_receipt['contract']
    if any(contract.get(k)!=expected[k] for k in ('shape','scale_x','scale_y')) or contract.get('source_sha256')!=expected['builder']:
        raise ValueError('coordinate cache and original matrix producer contracts differ')
    if len(cache['chunks'])!=expected_chunks or len(coordinates)!=expected_chunks:
        raise ValueError('coordinate cache chunk count differs')
    dtype=np.dtype([('transcript_id','<u8'),('x','<i4'),('y','<i4'),('gene','<i4')])
    cursor=0
    for i,(old,rec) in enumerate(zip(cache['chunks'],coordinates)):
        path=cache_path.parent/old['file']
        if old['file']!=f'chunk_{i:04d}.npy' or Path(rec['path']).resolve()!=path.resolve() or old['sha256']!=rec['sha256'] or old['rows']!=rec['rows'] or rec['row_start']!=cursor or rec['row_stop']!=cursor+rec['rows']:
            raise ValueError('coordinate cache chunk/hash/row span mismatch')
        check_identity(rec);a=np.load(path,mmap_mode='r')
        if a.dtype!=dtype or a.shape!=(rec['rows'],):raise ValueError('coordinate cache dtype/row count differs')
        good=(a['x']>=0)&(a['y']>=0)
        if np.any((a['x']<0)!=(a['y']<0)) or np.any(a['x'][good]>=shape[1]) or np.any(a['y'][good]>=shape[0]) or np.any(a['gene']<0) or np.any(a['gene']>=N_GENES):
            raise ValueError('coordinate cache native bounds/gene index differ')
        cursor+=len(a)
    if cursor!=expected_rows:raise ValueError('coordinate cache total eligible coverage differs')
    return dict(producer_dependencies=expected,key=key,chunks=expected_chunks,rows=cursor,all_row_spans_verified=True)

def matrix_arrays(spec):
    root=Path(spec['matrix']);cm=pd.read_csv(root/'cell_mapping.csv');gm=pd.read_csv(root/'gene_mapping.csv')
    ids=cm.canonical_cell_id.to_numpy(np.int64)
    if len(np.unique(ids))!=len(ids) or np.any(ids<=0) or np.any(np.diff(ids)<=0):
        raise ValueError('matrix IDs must be sorted unique positive integers')
    if not np.array_equal(cm.canonical_cell_id,ids) or not np.array_equal(cm.mask_label,ids) or not np.array_equal(cm.matrix_row,np.arange(len(ids))):
        raise ValueError('matrix canonical/mask/row association differs')
    if gm.gene.duplicated().any() or gm.gene.isna().any() or not np.array_equal(gm.gene_column,np.arange(len(gm))):
        raise ValueError('matrix gene mapping differs')
    counts=np.load(root/'counts.npy',mmap_mode='r')
    if counts.shape!=(len(ids),len(gm)) or counts.dtype!=np.dtype('int64') or np.any(counts<0):
        raise ValueError('full matrix shape/int64/nonnegative contract violation')
    return ids,gm.gene.tolist(),counts

def matrix_evidence(spec, expected_cells=N_CELLS, expected_genes=N_GENES, expected_molecules=ELIGIBLE, verify_owners=True):
    root=Path(spec['matrix']);rec=read(root/'complete.json')
    if rec.get('status')!='succeeded' or rec.get('validation_kind')!='independent_full_native_resize_lookup_and_conservation':
        raise ValueError('parent lacks independent complete matrix oracle')
    check_identity(rec['validation']);v=read(rec['validation']['path'])
    if v.get('status')!='succeeded' or v.get('rows')!=expected_molecules or v.get('owner_differences')!=0 or v.get('integer_count_differences')!=0:
        raise ValueError('parent oracle semantic failure')
    for item in rec['outputs'].values():
        if verify_owners or not Path(item['path']).name.startswith(('owners_','aggregate_')):check_identity(item)
    dep=rec.get('matrix_dependencies',rec.get('dependencies',{}))
    check_identity(dep['mask'])
    if Path(dep['mask']['path']).resolve()!=Path(spec['final']).resolve():
        raise ValueError('parent endpoint identity differs from matrix mask producer')
    ids,genes,counts=matrix_arrays(spec)
    assigned=int(counts.sum(dtype=np.int64))
    if (len(ids),len(genes))!=(expected_cells,expected_genes) or rec['totals']['eligible']!=expected_molecules or assigned!=rec['totals']['assigned']:
        raise ValueError('parent full matrix population or assigned mismatch')
    if rec['totals']['unassigned']!=expected_molecules-assigned:
        raise ValueError('parent eligible partition mismatch')
    return dict(endpoint=spec['name'],final=identity(spec['final']),matrix_receipt=identity(root/'complete.json'),
                counts=identity(root/'counts.npy'),assigned=assigned,gene_totals=counts.sum(0,dtype=np.int64).tolist(),
                eligible=expected_molecules,shape=list(counts.shape),dtype=str(counts.dtype))

def new_spec(cfg,name):
    if name not in NEW:raise ValueError('new endpoint outside authorized four')
    root=within(cfg['run_root'],'endpoints',name)
    return dict(name=name,root=str(root),machine=str(root/'machine.json'),matrix=str(root/'matrix'),
                evaluation=str(root/'evaluation'),final=str(root/'final.tif'),
                sc_scores=str(root/'evaluation/paper_aligned_v1/markers_per_cell.csv'),
                fixed_scores=str(root/'evaluation/fixed/fixed_population.csv'))

def resolve_parent(cfg,manifest,endpoint):
    if endpoint not in NEW:raise ValueError('unrecognized new endpoint')
    name=PARENTS[endpoint]
    if name in EXISTING:
        spec=dict(manifest['endpoints'][name])
    else:
        if endpoint!='P2_SIMPLE_GT' or name!='P2_SIMPLE':raise ValueError('unexpected dynamic parent edge')
        spec=new_spec(cfg,name)
        prod=check_receipt(Path(spec['root'])/'production_complete.json',output_root=cfg['run_root'])
        if prod.get('endpoint')!=name or prod.get('parent')!='MORPH_SIMPLE_FUSION' or prod.get('stage')!='06':
            raise ValueError('dynamic P2_SIMPLE parent lacks admitted stage 06 production')
        if prod['dependencies'].get('config_digest')!=digest(cfg):raise ValueError('dynamic parent belongs to another run')
        spec['nuclei']=read(spec['machine'])['nuclei_mask']
    if spec['name']!=name:raise ValueError('parent name masquerade')
    evidence=matrix_evidence(spec)
    spec['parent_evidence']=evidence
    return spec

def _walk_identities(value):
    if isinstance(value,dict):
        if isinstance(value.get('path'),str) and 'sha256' in value:
            yield value
        else:
            for child in value.values():yield from _walk_identities(child)
    elif isinstance(value,list):
        for child in value:yield from _walk_identities(child)

def source_dependencies():
    """Whole tracked Python/R/shell packages eliminate delayed-import omissions."""
    import subprocess
    repo=Path(__file__).resolve().parents[2]
    names=subprocess.check_output(['git','ls-files','scripts','bidcell'],cwd=repo,text=True).splitlines()
    paths=[repo/p for p in names if Path(p).suffix in ('.py','.R','.sh','.json','.yaml','.yml')]
    # During source assembly the committed new package is already tracked.
    external=('/home/zwan0726/work/outputs/bidcell_metric_protocol_review_20261008/publish_expression_metrics.R',
              '/home/zwan0726/work/outputs/bidcell_metric_protocol_review_20261008/publish_processing.R',
              '/home/zwan0726/work/outputs/bidcell_metric_protocol_review_20261008/publish_utils.R',
              '/home/zwan0726/work/outputs/bidcell_historical_dual_20261013/sources/publish_spatialDiversity_metrics.R')
    return sorted([dict(**identity(p),snapshot_relative=str(p.relative_to(repo))) for p in paths]+
                  [dict(**identity(p),snapshot_relative='external_sources/'+Path(p).name) for p in external],key=lambda x:x['snapshot_relative'])

def freeze_inputs(run_root,inspected_inputs,plan_manifest,progress=None):
    """Revalidate planned identities, extend to all owner/evaluation leaves, freeze."""
    if sha256(inspected_inputs)!=INSPECTED_SHA256 or sha256(plan_manifest)!=PLAN_MANIFEST_SHA256:
        raise ValueError('plan/input approval binding mismatch')
    for item in read(plan_manifest)['files']:check_identity(item)
    inspected=read(inspected_inputs);out=within(run_root,'configs/inputs_manifest.json')
    if out.exists():return load_manifest(out)
    files={}
    def add(item):
        path=str(Path(item['path']).resolve(strict=True))
        if path not in files:
            check_identity(item);files[path]=identity(path)
        elif files[path]['sha256']!=item['sha256']:raise ValueError('conflicting input identity')
        return path
    for item in inspected['files'].values():add(item)
    regpath=next(p for p in files if p.endswith('bidcell_historical_dual_bt_gt_fusion_20261017/artifact_registry.json'))
    registry=read(regpath)
    if len(registry['methods'])!=30:raise ValueError('latest approved registry is not the 30-method baseline')
    endpoints={}
    for name in EXISTING:
        source=inspected['endpoints'][name];parts=source['registry_pointer'].split('/')
        entry=registry
        for part in parts:entry=entry[part]
        f=source['files']
        for key in ('matrix_receipt','evaluation_receipt','sc_scores','fixed_scores'):
            if entry[key]['sha256']!=f[key]['sha256']:raise ValueError('registry endpoint identity differs')
        spec=dict(name=name,machine=add(f['machine']),matrix=source['matrix_root'],evaluation=source['evaluation_root'],
                  final=add(f['final_mask']),pre=add(f['pre_mask']),nuclei=add(f['nuclei_mask']),
                  checkpoint=add(f['checkpoint']),sc_scores=add(f['sc_scores']),fixed_scores=add(f['fixed_scores']))
        er=read(f['evaluation_receipt']['path']);mr=read(f['matrix_receipt']['path'])
        if not er.get('paper_method_complete'):raise ValueError('old endpoint incomplete paper evaluation')
        for item in _walk_identities(er.get('outputs',{})):add(item)
        for item in _walk_identities(mr.get('outputs',{})):add(item)
        add(mr['validation'])
        for chunk in mr['receipts']:
            add(dict(path=str(Path(spec['matrix'])/chunk['owner_file']),sha256=chunk['sha256']))
        endpoints[name]=spec
        # Fresh dynamic count check; no endpoint-to-assigned lookup table.
        spec['validated_parent']=matrix_evidence(spec,verify_owners=False)
    shared=inspected['shared']
    if sha256(OLD_SPATIAL_TRACE)!=OLD_SPATIAL_TRACE_SHA256:raise ValueError('root historical spatial trace drift')
    trace=read(OLD_SPATIAL_TRACE);add(identity(OLD_SPATIAL_TRACE))
    if trace['problems'] or trace['status']!='PASS_READ_ONLY_REUSE_TRACE':raise ValueError('old spatial trace has unresolved problems')
    for item in _walk_identities(trace):add(item)
    for name,spec in endpoints.items():
        if name in trace['endpoints']:
            entry=trace['endpoints'][name]
            for filename,obj in entry['original_ownQC_inputs'].items():
                if not obj['same_bytes'] or sha256(Path(spec['evaluation'])/'paper_aligned_v1'/filename)!=obj['source']['sha256']:raise ValueError('historical spatial original QC/annotation changed')
            spatial=entry['artifacts'][f'author_spatial/methods/{name}/paper/spatial_diversity_correlations.csv']
            spec['spatial_root']=str(Path(spatial['path']).parent)
            spec['spatial_receipt']=entry['comparison_receipt']
        else:spec['spatial_root']=str(Path(spec['evaluation'])/'paper_aligned_v1')
    # Only original own-QC/own-centroid curves are population-compatible here.
    # Historical commonID6/commonQC6/commonID8 populations are never imported.
    if sha256(OLD_BT_TRACE)!=OLD_BT_TRACE_SHA256:raise ValueError('root historical B/T trace drift')
    bt_trace=read(OLD_BT_TRACE);add(identity(OLD_BT_TRACE))
    for item in _walk_identities(bt_trace):add(item)
    for key,names in [('evaluation8',('BASE_MATCH','V2_MORPH','P2_BASE','A2_GT','P2_MORPH','V2_MORPH_GT','P2_BASE_GT'))]:
        outer_path=add(registry[key]);outer=read(outer_path)
        if outer.get('status')!='succeeded':raise ValueError('historical B/T producer incomplete')
        for name in names:
            profile='methods/'+name+'/official_sc_ownQC_owncentroid/'
            producer=outer['outputs'][profile+'complete.json'];add(producer)
            if producer!=bt_trace['endpoints'][name]['profile_receipt']:raise ValueError('historical B/T trace producer differs')
            saved=read(producer['path'])
            table=pd.read_csv(Path(producer['path']).parent/'full_curves.csv')
            from .bt import PANELS
            expected={(p['direction'],g,b) for p in PANELS for g in p['markers'] for b in range(7)}
            actual=list(table[['direction','marker','bin_index']].itertuples(index=False,name=None))
            if saved.get('status')!='succeeded' or saved.get('profile')!='official_sc_ownQC_owncentroid' or len(actual)!=42 or set(actual)!=expected:raise ValueError('historical official six-marker profile incomplete')
            for filename,obj in saved['outputs'].items():
                if outer['outputs'][profile+filename]['sha256']!=obj['sha256']:raise ValueError('historical B/T leaf producer chain differs')
                add(obj)
            endpoints[name]['official_bt_root']=str(Path(producer['path']).parent)
            endpoints[name]['official_bt_producer']=producer
            endpoints[name]['official_bt_registry_chain']=registry[key]
    # The registry supplies frozen reference/model and B0 background ancestry.
    add(registry['frozen_input_manifest'])
    legacy=read(registry['frozen_input_manifest']['path'])
    b0=dict(legacy['endpoints']['B0'])
    for key in ('machine','sc_scores','fixed_scores'):add(identity(b0[key]))
    for item in _walk_identities(read(Path(b0['evaluation'])/'complete.json').get('outputs',{})):add(item)
    historical_strata=next(str(Path(p).parent) for p in files if p.endswith('frozen_strata/support_counts.npy'))
    historical_counts=next(str(Path(p).parent) for p in files if p.endswith('frozen_counts/fixed_count_groups.csv'))
    dapi=next(p for p in files if p.endswith('/dapi_resized.tif'))
    for item in _walk_identities(read(Path(shared['evaluation']['prepared'])/'complete.json')):add(item)
    base=read(endpoints['BASE_MATCH']['machine'])
    cache_validation=validate_coordinate_cache(shared['coordinate_cache'],shared['coordinates'],base,read(Path(endpoints['BASE_MATCH']['matrix'])/'complete.json'))
    obj=dict(schema=INPUT_SCHEMA,**shared,endpoints=endpoints,background={'B0':b0},files=files,
             expected_cells=N_CELLS,expected_genes=N_GENES,expected_molecules=ELIGIBLE,
             historical_strata=historical_strata,historical_counts=historical_counts,dapi=dapi,
             registry=files[regpath],plan_manifest=identity(plan_manifest),inspected_inputs=identity(inspected_inputs),
             existing_order=list(EXISTING),new_order=list(NEW),coordinate_cache_validation=cache_validation,old_spatial_trace=identity(OLD_SPATIAL_TRACE),old_official_BT_trace=identity(OLD_BT_TRACE))
    atomic_json(out,obj)
    if progress is not None:progress(dict(stage='input_freeze',cursor=len(files),total=len(files),detail={'manifest':str(out)}))
    return obj

def load_manifest(path,verify=True):
    obj=read(path)
    if obj.get('schema')!=INPUT_SCHEMA or obj.get('existing_order')!=list(EXISTING) or obj.get('new_order')!=list(NEW):
        raise ValueError('input schema/method set drift')
    if set(obj['endpoints'])!=set(EXISTING) or (obj['expected_cells'],obj['expected_genes'],obj['expected_molecules'])!=(N_CELLS,N_GENES,ELIGIBLE):
        raise ValueError('input universe drift')
    if verify:
        for item in obj['files'].values():check_identity(item)
    return obj

def load_inputs(cfg,verify=True):
    obj=load_manifest(cfg['input_manifest'],verify)
    if digest(obj)!=cfg['input_digest']:raise ValueError('config/input digest mismatch')
    return obj

def validate_semantics(cfg,manifest,callback=None):
    """Committed endpoint/chunk semantic evidence, suitable for launch observation."""
    from .common import dependencies
    out=within(cfg['run_root'],'validation/inputs');out.mkdir(parents=True,exist_ok=True)
    meta=np.load(manifest['metadata']);ids=meta['ids'];nuc=tifffile.imread(manifest['endpoints']['BASE_MATCH']['nuclei'])
    for index,name in enumerate(EXISTING):
        path=out/(name+'.json');dep=dict(dependencies(cfg,'01'),endpoint=name)
        if path.exists():check_receipt(path,dep);continue
        spec=manifest['endpoints'][name];mi,genes,counts=matrix_arrays(spec)
        mask=tifffile.imread(spec['final'])
        if not np.array_equal(mi,ids) or not np.array_equal(mask[nuc>0],nuc[nuc>0]) or not np.array_equal(np.unique(mask)[1:],ids):
            raise ValueError('input mask/matrix nuclear ID semantics differ')
        evidence=matrix_evidence(spec,verify_owners=False)
        summary=out/(name+'_semantics.json');atomic_json(summary,evidence)
        chunk_receipt(path,dep,[summary],endpoint=name,n_cells=len(mi),n_genes=len(genes),assigned=int(counts.sum()))
        progress(callback,'input_endpoint_semantics',index+1,len(EXISTING),receipt=str(path))
    return out
