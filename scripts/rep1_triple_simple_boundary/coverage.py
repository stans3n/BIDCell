"""Semantic completeness, never a file-existence or favorable-result gate."""
from .continuation import frozen_root,read_data
from pathlib import Path
from itertools import product
import numpy as np
import pandas as pd
from .contracts import DAG,NEW,EXISTING,METHODS,METRICS,SCALES,REGIONS,comparisons,comparison_id
from .common import (identity,check_identity,check_receipt,dependencies,read,atomic_json,id_hash,
                     inventory,write_receipt)
from .bt import MARKERS,PANELS,EDGES

def exact_keys(frame,columns,expected,description):
    """Validate the actual unique table keys, independently of receipt claims."""
    if not set(columns).issubset(frame):raise ValueError(description+' required key columns absent')
    actual=list(frame[list(columns)].itertuples(index=False,name=None));wanted=set(expected)
    if len(actual)!=len(wanted) or len(set(actual))!=len(actual) or set(actual)!=wanted:
        raise ValueError(description+' actual unique key coverage differs')

def interval_rows(frame,description):
    required={'point','ci_low','ci_high','n_units','NA_reason','replicates','valid_replicates','contributing_blocks','universe_blocks'}
    if not required.issubset(frame):raise ValueError(description+' interval fields absent')
    if not frame.replicates.eq(2000).all() or (frame.n_units<0).any() or not frame.valid_replicates.between(0,2000).all():raise ValueError(description+' invalid denominator/replicate count')
    missing=frame[['point','ci_low','ci_high']].isna().any(axis=1)
    if (missing&frame.NA_reason.fillna('').eq('')).any():raise ValueError(description+' NA interval lacks reason')
    if ((frame.n_units>0)&~np.isfinite(frame.point)).any() or ((frame.n_units==0)&frame.point.notna()).any():raise ValueError(description+' point denominator semantics differ')

def bt_curve_rows(frame,description):
    expected=[(p['direction'],m,b) for p in PANELS for m in p['markers'] for b in range(7)]
    exact_keys(frame,('direction','marker','bin_index'),expected,description)
    if not {'n_eligible','n_positive','fraction','status','lower_um','upper_um','recipient_ids_sha256','target_ids_sha256','population_ids_sha256'}.issubset(frame):raise ValueError(description+' curve fields absent')
    if (frame.n_eligible<0).any() or (frame.n_positive<0).any() or (frame.n_positive>frame.n_eligible).any():raise ValueError(description+' impossible marker numerator')
    expected_fraction=frame.n_positive.div(frame.n_eligible.replace(0,np.nan))
    if not np.allclose(frame.fraction,expected_fraction,rtol=0,atol=1e-14,equal_nan=True):raise ValueError(description+' marker fraction mismatch')
    if not np.array_equal(frame.lower_um,EDGES[frame.bin_index.to_numpy(int)]):raise ValueError(description+' fixed bin boundary mismatch')
    part=frame.bin_index<6
    if not np.array_equal(frame.loc[part,'upper_um'],EDGES[frame.loc[part,'bin_index'].to_numpy(int)+1]):raise ValueError(description+' fixed bin boundary mismatch')
    if ((frame.n_eligible==0)&frame.status.eq('computed')).any():raise ValueError(description+' empty bin status absent')

def comparison_rows(frame,groups,coefficient,population,pop_hashes):
    import json
    exact_keys(frame,('metric','profile','scale_um','stratum'),product(METRICS,('sc','fixed'),SCALES,groups),'score comparison')
    interval_rows(frame,'score comparison')
    if not frame.population.eq(population).all() or any(json.loads(x)!=coefficient for x in frame.coefficient.unique()):raise ValueError('comparison actual coefficient/population differs')
    for (metric,profile),part in frame.groupby(['metric','profile']):
        if not part.population_ids_sha256.eq(pop_hashes[(metric,profile)]).all():raise ValueError('comparison actual population ID hash differs')

def paired_bt_coverage(folder,key,methods,pair,frozen,expected_ids):
    rec=check_receipt(read_data(folder/'complete.json'));q=pd.read_csv(read_data(folder/'population.csv')).cell_id.to_numpy(np.int64)
    if rec.get('methods')!=list(methods) or not np.array_equal(q,expected_ids) or id_hash(q)!=rec['dependencies']['population']:raise ValueError('paired B/T population/method binding differs')
    for name in methods:
        frame=pd.read_csv(read_data(folder/name/'full_curves.csv'));bt_curve_rows(frame,'paired B/T '+key+'/'+name)
        if not frame.population_ids_sha256.eq(id_hash(q)).all():raise ValueError('paired B/T table uses another population')
    if pair:
        f=pd.read_csv(read_data(folder/'paired_0_40um.csv'));exact_keys(f,('direction','marker','scale_um'),[(p['direction'],m,s) for p in PANELS for m in p['markers'] for s in SCALES],'paired B/T fixed-window CI');interval_rows(f,'paired B/T fixed-window CI')
        if not f.candidate.eq(methods[0]).all() or not f.control.eq(methods[1]).all() or not f.comparison.eq(key).all():raise ValueError('paired B/T candidate minus control direction differs')
        if not f.window_um.eq('0-40 inclusive').all():raise ValueError('paired B/T reporting window changed')
        for panel in PANELS:
            d=pd.read_csv(read_data(frozen/'bt_targets'/(panel['direction']+'_frozen_distances.csv')));ids=np.intersect1d(q,d.loc[d.distance_um.le(40),'cell_id'])
            if not f.loc[f.direction==panel['direction'],'ids_sha256'].eq(id_hash(ids)).all():raise ValueError('paired B/T fixed-window recipient population differs')

def boundary_tables(cfg,manifest,a,b,folder,frozen,cells,frozen_groups,final):
    from .boundary_analysis import TRANSFER_COLUMNS
    from .local_statistics import spatial_masks
    from .inputs import matrix_arrays
    ids,genes,pc=matrix_arrays(b);ai,ag,cc=matrix_arrays(a)
    if not np.array_equal(ai,ids) or ag!=genes:raise ValueError('boundary matrix axes differ')
    groups=dict(frozen_groups)
    for dimension in ('block_500','block_1000','block_2000','RNA_density_quartile','nuclear_crowding_quartile'):
        for k,part in cells.groupby(['original_type',dimension],dropna=False):groups['type__'+dimension+'__'+str(k)]=part.index.to_numpy(np.int64)
    cost=pd.read_csv(read_data(folder/'local_costs.csv.gz'));exact_keys(cost,('stratum',),[(g,) for g in groups],'local cell costs')
    cell=pd.read_csv(read_data(folder/'cells.csv.gz'));exact_keys(cell,('cell_id',),[(int(i),) for i in ids],'full cell transfer distribution')
    if not np.array_equal(cell.incoming_RNA-cell.outgoing_RNA,cell.net_RNA) or (cell[['incoming_RNA','outgoing_RNA']]<0).any().any():raise ValueError('gross/net cell transfer inconsistent')
    delta=np.zeros(pc.shape,np.int64)
    for i,coordinate in enumerate(manifest['coordinates']):
        receipt=check_receipt(read_data(folder/'chunks'/f'{i:04d}.json'));z=np.load(read_data(folder/'chunks'/f'{i:04d}.npz'));keys,n=z['keys'],z['molecules']
        if keys.ndim!=2 or keys.shape[1]!=len(TRANSFER_COLUMNS) or n.shape!=(len(keys),) or np.any(n<=0) or receipt['changed']!=int(n.sum()) or receipt['rows']!=coordinate['rows']:raise ValueError('complete sparse gene transfer shape/coverage differs')
        if np.any(keys[:,0]<0) or np.any(keys[:,0]>=len(genes)):raise ValueError('sparse ledger gene index differs')
        for col,sign in ((1,-1),(2,1)):
            use=keys[:,col]>0;row=np.searchsorted(ids,keys[use,col])
            if np.any(row>=len(ids)) or not np.array_equal(ids[row],keys[use,col]):raise ValueError('sparse ledger unknown owner ID')
            np.add.at(delta,(row,keys[use,0]),sign*n[use])
    if not np.array_equal(delta,cc-pc):raise ValueError('validated sparse ledger does not reconstruct full313 matrix delta')
    gene=pd.read_csv(read_data(folder/'gene_totals.csv'));exact_keys(gene,('gene',),[(g,) for g in genes],'all313 transfer totals')
    if not np.array_equal(gene.set_index('gene').loc[genes,'net'],delta.sum(0)):raise ValueError('all313 transfer totals differ')
    regions=pd.read_csv(read_data(folder/'regions.csv'));exact_keys(regions,('region',),[(r,) for r in REGIONS],'complete molecule regions')
    pixels=pd.read_csv(read_data(folder/'pixels/regions.csv'));exact_keys(pixels,('region',),[(r,) for r in REGIONS],'complete pixel regions')
    regional_ci=pd.read_csv(read_data(folder/'region_spatial_intervals.csv'));exact_keys(regional_ci,('region','metric','scale_um'),product(REGIONS,('capture_delta_pp','pooled_retention','gross_transfer_rate'),SCALES),'regional intervals');interval_rows(regional_ci,'regional intervals')
    import tifffile
    source=tifffile.imread(b['final']);target=tifffile.imread(a['final'])
    for start in range(0,len(source),128):
        p=folder/'pixels'/f'chunk_{start:05d}.json';r=check_receipt(read_data(p));z=np.load(read_data(p.with_suffix('.npz')));keys,n=z['keys'],z['pixels']
        changed=int(np.count_nonzero(source[start:start+128]!=target[start:start+128]))
        if keys.ndim!=2 or keys.shape[1]!=len(TRANSFER_COLUMNS)-1 or n.shape!=(len(keys),) or np.any(n<=0) or int(n.sum())!=changed or r['changed_pixels']!=changed:raise ValueError('complete pixel transfer coverage differs')
    del source,target
    cell_ci=pd.read_csv(read_data(folder/'cell_cost_intervals/intervals.csv.gz'));exact_keys(cell_ci,('metric','scale_um','stratum'),product(('incoming_RNA','outgoing_RNA','net_RNA','pooled_RNA_retention'),SCALES,groups),'cell cost intervals');interval_rows(cell_ci,'cell cost intervals')
    empty={k:np.empty(0,np.int64) for k in ('BASE_owner','region_bits','competition_type','interface_type','K','local_density')};empty['single_unknown']=np.empty(0,bool)
    spatial_names=[name for name,_ in spatial_masks(empty,cells)];metrics=('capture_delta_pp','owner_retention','cell_to_cell_rate','cell_to_background_rate','background_to_cell_rate','gross_transfer_rate')
    for unit in ('molecule','pixel'):
        part=folder/'spatial_strata'/unit;counts=pd.read_csv(read_data(part/'stratified_counts.csv.gz'));exact_keys(counts,('stratum',),[(g,) for g in spatial_names],unit+' fixed spatial counts')
        f=pd.read_csv(read_data(part/'stratified_intervals.csv.gz'));exact_keys(f,('metric','scale_um','stratum'),product(metrics,SCALES,spatial_names),unit+' fixed spatial intervals');interval_rows(f,unit+' fixed spatial intervals')
    pairs=pd.read_csv(read_data(frozen/'interfaces/pairs.csv'));interface_groups=['all','type_same','type_heterotypic','type_Unknown']+['pair_exposed_to_'+k for k in frozen_groups]
    assigned=pd.read_csv(read_data(folder/'interfaces/unique_assignment_molecule_transfers.csv.gz'));exact_keys(assigned,('interface_id',),[(int(i),) for i in pairs.interface_id],'interface unique assignment')
    overlap=pd.read_csv(read_data(folder/'interfaces/overlapping_exposure_transfers.csv.gz'));exact_keys(overlap,('interface_id','unit'),product(pairs.loc[pairs.exposure_pixels>0,'interface_id'],('pixel','molecule')),'complete overlapping interface exposure')
    interface=pd.read_csv(read_data(folder/'interfaces/paired_spatial_intervals.csv.gz'));exact_keys(interface,('exposure_unit','metric','scale_um','stratum'),product(('pixel','molecule'),('capture_delta_pp','owner_retention','gross_transfer_rate'),SCALES,interface_groups),'interface intervals');interval_rows(interface,'interface intervals')
    if final:
        metrics=('area_um2','components_4','components_8','transcripts_per_cell','genes_per_cell')
        costs=pd.read_csv(read_data(folder/'structure/costs.csv.gz'));exact_keys(costs,('metric','stratum'),product(metrics,groups),'full structure local costs')
        f=pd.read_csv(read_data(folder/'structure/intervals.csv.gz'));exact_keys(f,('metric','scale_um','stratum'),product(metrics,SCALES,groups),'full structure local intervals');interval_rows(f,'full structure local intervals')

def roi_coverage(folder,frozen,methods,expected_pairs):
    selected=read(read_data(frozen/'roi_selection.json'));coverage=read(read_data(folder/'coverage.json'))
    if coverage['selection']!=selected or len(selected['rows'])!=32:raise ValueError('ROI selection drift')
    for row in selected['rows']:
        if not row['selected']:
            if not row['reason']:raise ValueError('empty ROI lacks reason')
            continue
        dest=folder/row['id'];check_receipt(read_data(dest/'complete.json'));m=read(read_data(dest/'manifest.json'))
        if m['selection']!=row or m['methods']!=list(methods) or m['comparisons']!=expected_pairs:raise ValueError('actual ROI method/comparison/selection coverage differs')
        z=np.load(read_data(dest/'data.npz'));raw=np.load(read_data(frozen/'roi_transcripts'/(row['id']+'.npz')))['transcripts']
        if set(z.files)!={'nuclei','dapi','transcripts',*methods} or not np.array_equal(z['transcripts'],raw) or m['molecule_points']!=len(raw):raise ValueError('actual ROI full raw molecule/mask coverage differs')
        for name in ('panels.pdf','masks.png',*[f'changes_{i:02d}.png' for i in range(0,len(expected_pairs),8)]):identity(read_data(dest/name))

def validate_endpoint_products(spec,structure_root=None,bt_root=None,spatial_root=None,manifest=None,cfg=None):
    root=Path(spec['evaluation']);matrix=Path(spec['matrix'])
    paired=None
    if spec['name'] in NEW:
        if not (root/'paired/complete.json').exists():raise ValueError('full evaluate requires parent/BASE paired evidence')
        if cfg is None or manifest is None:raise ValueError('full paired evaluation requires frozen config and inputs')
        from .evaluation_pairs import validate as validate_pairs
        from .continuation import producer_cfg
        paired=validate_pairs(producer_cfg(cfg,spec['name']),manifest,spec['name'])
    mr=read(read_data(matrix/'complete.json'));validation=read(read_data(matrix/'validation.json'))
    if mr.get('validation_kind')!='independent_full_native_resize_lookup_and_conservation' or validation.get('integer_count_differences')!=0 or validation.get('owner_differences')!=0:
        raise ValueError('full evaluate requires complete independent matrix oracle')
    check_identity(mr['validation'])
    structure=Path(structure_root) if structure_root else root/'structure';bt=Path(bt_root) if bt_root else root/'bt'
    spatial_dir=Path(spatial_root or spec.get('spatial_root',root/'paper_aligned_v1'))
    required={'own_matrix_full_oracle':matrix/'validation.json','ownQC313_logNormCounts':root/'paper_aligned_v1/size_factors.csv',
              'frozen_scClassify_reference307':root/'paper_aligned_v1/annotation.csv','author_marker_six_metrics':Path(spec['sc_scores']),
              'fixed_full_panel_six_metrics':Path(spec['fixed_scores']),'QC_unassigned_missing_unknown':root/'paper_aligned_v1/qc_all_cells.csv',
              'capture_cell_RNA_genes_area':structure/'cells_distribution.csv','4_and_8_connectivity_nucleus_ID_integrity':structure/'structure.json',
              'reference_expression_detection_correlations':root/'paper_aligned_v1/type_correlations.csv',
              'author_spatial_diversity500um':spatial_dir/'spatial_diversity_correlations.csv',
              'six_marker_BT_official_and_fixed':bt/'complete.json','stage_dependency_and_population_receipts':matrix/'complete.json'}
    if set(required)!=set(DAG['required_full_evaluate']):raise AssertionError('full evaluation checklist drift')
    records=[dict(endpoint=spec['name'],metric=k,artifact=identity(read_data(p)),status='computed_or_verified_reuse') for k,p in required.items()]
    for profile,p in [('sc',spec['sc_scores']),('fixed',spec['fixed_scores'])]:
        frame=pd.read_csv(read_data(p))
        if frame.cell_id.duplicated().any() or not set(METRICS).issubset(frame):raise ValueError('complete scoring six metrics/unique IDs absent')
        if profile=='fixed' and len(frame)!=103000:raise ValueError('fixed full103000 panel missing')
        for m in METRICS:
            finite=np.isfinite(frame[m]);records.append(dict(endpoint=spec['name'],metric=m,profile=profile,population='raw_ownQC' if profile=='sc' else 'raw_fixed_full_panel',n_population=len(frame),n_finite=int(finite.sum()),artifact=identity(read_data(p)),status='computed',NA_reason='' if finite.any() else 'no_finite_values'))
    st=read(read_data(structure/'structure.json'))
    if st.get('nuclear_preservation_fraction')!=1 or st.get('n_cells')!=107955:raise ValueError('full structural nuclear/ID integrity failed')
    from .inputs import matrix_arrays
    ids,genes,counts=matrix_arrays(spec)
    dist=pd.read_csv(read_data(structure/'cells_distribution.csv'))
    if not {'components_4','components_8','transcripts_per_cell','genes_per_cell','area_um2'}.issubset(dist):raise ValueError('full 4/8 structural distributions absent')
    exact_keys(dist,('cell_id',),[(int(i),) for i in ids],'full structural distribution')
    ordered=dist.set_index('cell_id').loc[ids]
    if not np.array_equal(ordered.transcripts_per_cell,counts.sum(1)) or not np.array_equal(ordered.genes_per_cell,(counts>0).sum(1)) or not ordered.nuclei_overlapped.eq(1).all():raise ValueError('full structural RNA/gene/nucleus association differs')
    total=counts.sum(1);qc=(total>=20)&(total<=2000)&((counts>0).sum(1)>=20);qcids=ids[qc]
    for filename in ('annotation.csv','size_factors.csv','single_cell_max_pearson.csv','markers_per_cell.csv'):
        frame=pd.read_csv(read_data(root/'paper_aligned_v1'/filename));exact_keys(frame,('cell_id',),[(int(i),) for i in qcids],'ownQC '+filename)
    allqc=pd.read_csv(read_data(root/'paper_aligned_v1/qc_all_cells.csv'));exact_keys(allqc,('cell_id',),[(int(i),) for i in ids],'complete QC ledger')
    corr=pd.read_csv(read_data(root/'paper_aligned_v1/type_correlations.csv'))
    if not {'mean_logcounts','detection','official_source_detection_vs_mean'}.issubset(set(corr.definition)):raise ValueError('reference correlation definitions missing')
    if manifest is None:raise ValueError('full evaluation coverage requires frozen reference manifest')
    prepared=Path(manifest['evaluation']['prepared']);types=pd.read_csv(read_data(prepared/'reference_mean_logcounts.csv'),nrows=0).columns[1:]
    reference_genes=(prepared/'reference_genes.txt').read_text().splitlines()
    vectors=pd.read_csv(read_data(root/'paper_aligned_v1/type_expression_vectors.csv'))
    exact_keys(corr,('cell_type','definition'),product(types,('mean_logcounts','detection','official_source_detection_vs_mean')),'reference expression/detection definitions')
    exact_keys(vectors,('cell_type','gene'),product(types,reference_genes),'full reference gene/type vectors')
    detection=pd.read_csv(read_data(root/'paper_aligned_v1/gene_detection_313.csv'));exact_keys(detection,('gene',),[(g,) for g in genes],'complete313 detection')
    if (corr.pearson.isna()&corr.status.fillna('').isin(['','computed'])).any():raise ValueError('reference correlation NA lacks reason')
    spatial=pd.read_csv(read_data(spatial_dir/'spatial_diversity_correlations.csv'))
    if 'family' not in spatial:spatial['family']=spatial.metric.str.split('_').str[0]
    spatial_metrics=[f+'_'+m for f in ('cv','sd') for m in ('total_transciprts','total_genes','cell_area')]
    exact_keys(spatial,('metric',),[(m,) for m in spatial_metrics],'author spatial six metrics')
    bins=pd.read_csv(read_data(spatial_dir/'spatial_diversity_bins.csv'))
    if bins.bin.duplicated().any() or not {'entropy','n_cells',*spatial_metrics}.issubset(bins) or not spatial.n_bins.eq(len(bins)).all() or not spatial.n_qc_cells.eq(len(qcids)).all() or int(bins.n_cells.sum())!=len(qcids) or not spatial.bin_width.eq(500).all():raise ValueError('full author spatial bins/population differ')
    for _,r in spatial.iterrows():
        if not np.isfinite(r.pearson) and ('reason' not in spatial or pd.isna(r.reason) or not str(r.reason)):raise ValueError('spatial NA lacks reason')
    br=check_receipt(read_data(bt/'complete.json'))
    if set(br.get('markers',[]))!=set(MARKERS):raise ValueError('complete evaluation missing six-marker B/T')
    if set(br.get('profiles',[]))!={'official_sc_ownQC_owncentroid','fixed_commonID12_nuclear_ref','fixed_ownQC_nuclear_ref'} or br.get('endpoint')!=spec['name']:raise ValueError('complete evaluation missing official/fixed B/T profiles')
    for profile in br['profiles']:
        f=pd.read_csv(read_data(bt/profile/'full_curves.csv'))
        bt_curve_rows(f,'endpoint full BT curves')
        if ((f.n_eligible==0)&f.fraction.notna()).any() or ((f.n_eligible>0)&~np.isfinite(f.fraction)).any():raise ValueError('BT empty-bin semantics differ')
    if paired is not None:records.extend(paired['score_records'])
    return dict(status='passed',endpoint=spec['name'],required_products=list(required),records=records,paired_parent_BASE=paired,full_evaluate=True)

def validate_analysis(cfg,phase,methods):
    if cfg.get('continuation') and phase=='initial':
        from .continuation import validate_adopted_stage
        if list(methods)!=DAG['initial_methods9']:raise ValueError('adopted initial method scope changed')
        return validate_adopted_stage(cfg,'04')
    from .inputs import load_inputs,new_spec
    from .populations import selected,common_finite
    root=Path(cfg['run_root'])/'analysis'/phase
    rec=check_receipt(read_data(root/'complete.json'),dependencies(cfg,'04' if phase=='initial' else '13'),cfg['run_root'])
    if rec.get('methods')!=list(methods):raise ValueError('analysis method coverage mismatch')
    checks={'populations':'sets','comparisons':'jobs','boundary':'pairs','bt':'sets'}
    for part in checks:check_receipt(read_data(root/part/'complete.json'),output_root=cfg['run_root'])
    cmp=read(read_data(root/'comparisons/coverage.json'));boundary=read(read_data(root/'boundary/coverage.json'));bt=read(read_data(root/'bt/coverage.json'))
    expected=[comparison_id(a,b) for a,b in comparisons(methods)]
    if cmp['comparisons']!=expected or boundary['pairs']!=expected or not set(expected).issubset(bt['sets']):raise ValueError('prespecified pair coverage missing')
    if cmp['metrics']!=list(METRICS) or cmp['profiles']!=['sc','fixed'] or cmp['scales']!=list(SCALES):raise ValueError('comparison six metrics/two profiles/three scales missing')
    expected_jobs=set(expected)|{'raw_'+m for m in methods}|{c['id'] for c in DAG['factorial_contrasts']}
    if phase=='all12':expected_jobs|={'all12_'+m for m in methods}
    if set(cmp['jobs'])!=expected_jobs:raise ValueError('raw profiles/paired/factorial/all12 job coverage differs')
    frozen=frozen_root(cfg);manifest=load_inputs(cfg,verify=False);specs={m:manifest['endpoints'][m] if m in EXISTING else new_spec(cfg,m) for m in methods}
    cells=pd.read_csv(read_data(frozen/'cells.csv')).set_index('cell_id');gf=np.load(read_data(frozen/'cell_groups.npz'));groups={k:gf[k] for k in gf.files}
    denominators=pd.read_csv(read_data(frozen/'denominators.csv'));denominators=denominators[denominators.unit.isin(['pixel','molecule'])].set_index(['region','unit']).denominator
    for name in methods:
        capture=pd.read_csv(read_data(root/'boundary/methods'/name/'capture_foreground.csv'));exact_keys(capture,('region','unit'),product(REGIONS,('pixel','molecule')),'endpoint complete regional capture')
        observed=capture.set_index(['region','unit']).denominator.reindex(denominators.index)
        if not np.array_equal(observed,denominators) or not capture.endpoint.eq(name).all():raise ValueError('endpoint capture uses different frozen denominator/name')
    fixed_ids=pd.read_csv(read_data(manifest['fixed_population'])).cell_id.to_numpy(np.int64);tables={}
    for name,spec in specs.items():
        tables[name]={profile:pd.read_csv(read_data(spec[profile+'_scores'])).set_index('cell_id').sort_index() for profile in ('sc','fixed')}
        qc=pd.read_csv(read_data(Path(spec['evaluation'])/'paper_aligned_v1/qc_all_cells.csv'));tables[name]['qc_ids']=np.sort(qc.loc[qc.qc_pass,'cell_id'].to_numpy(np.int64))
    def bt_ids(names):return np.asarray(sorted(set(fixed_ids)&set.intersection(*(set(tables[m]['qc_ids']) for m in names))),np.int64)
    if cmp['strata']!=list(groups):raise ValueError('comparison claimed strata differ from frozen definitions')
    jobs={comparison_id(a,b):({a:1,b:-1},'pairwise') for a,b in comparisons(methods)}
    jobs.update({'raw_'+m:({m:1},'raw_full_profile') for m in methods});jobs.update({c['id']:(c['coefficient'],'factorial8') for c in DAG['factorial_contrasts']})
    if phase=='all12':jobs.update({'all12_'+m:({m:1},'all12') for m in methods})
    raw_hashes={};population_hashes={}
    for key in expected_jobs:
        job=check_receipt(read_data(root/'comparisons'/key/'complete.json'))
        if not job.get('reuse_binding') or job.get('rows',0)!=len(cmp['strata'])*len(METRICS)*2*len(SCALES):raise ValueError('comparison complete stratum/profile/scale rows absent')
        coefficient,population=jobs[key];hashes={}
        if population=='raw_full_profile':
            name=next(iter(coefficient))
            if name not in raw_hashes:
                raw_hashes[name]={}
                for profile in ('sc','fixed'):
                    frame=tables[name][profile]
                    for metric in METRICS:raw_hashes[name][(metric,profile)]=id_hash(frame.index[np.isfinite(frame[metric])])
            hashes=raw_hashes[name]
        else:
            popkey=key if population=='pairwise' else population
            if popkey not in population_hashes:
                from .contracts import FACTORIAL8
                names=tuple(coefficient) if population=='pairwise' else FACTORIAL8 if population=='factorial8' else METHODS
                population_hashes[popkey]={}
                for metric in METRICS:
                    _,correct,_,_=common_finite(tables,names,fixed_ids,metric);actual=selected(root/'populations',popkey,metric)
                    if not np.array_equal(actual,correct):raise ValueError('actual common finite population differs from frozen source scores/QC')
                    population_hashes[popkey][metric]=id_hash(actual)
            for metric,h in population_hashes[popkey].items():
                for profile in ('sc','fixed'):hashes[(metric,profile)]=h
        comparison_rows(pd.read_csv(read_data(root/'comparisons'/key/'stratified_intervals.csv.gz')),groups,coefficient,population,hashes)
    for key in expected:
        folder=root/'boundary/comparisons'/key;pair=check_receipt(read_data(folder/'complete.json'));schema=read(read_data(folder/'schema.json'))
        if not pair.get('matrix_gene_cell_delta_exact') or not pair.get('all313_gene_ledger') or pair.get('sparse_chunks')!=69 or len(schema.get('gene_names',[]))!=313:raise ValueError('complete 313 gene transfer ledger missing')
        check_receipt(read_data(folder/'spatial_strata/complete.json'))
        if phase=='all12':
            if not pair.get('structure_completed'):raise ValueError('final structural local costs pending')
            check_receipt(read_data(folder/'structure/complete.json'))
        a,b=next((a,b) for a,b in comparisons(methods) if comparison_id(a,b)==key)
        boundary_tables(cfg,manifest,specs[a],specs[b],folder,frozen,cells,groups,phase=='all12')
        paired_bt_coverage(root/'bt'/key,key,(a,b),True,frozen,bt_ids((a,b)))
    rois=read(read_data(root/'roi/coverage.json'));check_receipt(read_data(root/'roi/complete.json'))
    if rois['target']!=32 or rois['methods']!=list(methods) or rois['rendered']+len(rois['empty_reasons'])!=32:raise ValueError('32 predefined ROI coverage incomplete')
    roi_coverage(root/'roi',frozen,methods,expected)
    from .contracts import FACTORIAL8
    paired_bt_coverage(root/'bt/commonQC8','commonQC8',FACTORIAL8,False,frozen,bt_ids(FACTORIAL8))
    if phase=='all12':
        if cmp['factorial']!=19 or not set(c['id'] for c in DAG['factorial_contrasts']).issubset(cmp['jobs']):raise ValueError('12 conditional +6 second +1 third contrasts missing')
        pop=read(read_data(root/'populations/complete.json'))
        if not {'factorial8','all12'}.issubset(pop['sets']) or not {'commonQC8','commonQC12'}.issubset(bt['sets']):raise ValueError('eight/twelve common populations missing')
        paired_bt_coverage(root/'bt/commonQC12','commonQC12',METHODS,False,frozen,bt_ids(METHODS))
    return rec

def complete_check(cfg,manifest):
    from .science import validate_stage,receipt_path
    for stage in [f'{i:02d}' for i in range(1,14)]:validate_stage(cfg,stage,manifest)
    evidence={s:identity(read_data(receipt_path(cfg,s))) for s in [f'{i:02d}' for i in range(1,14)]}
    for item in manifest['files'].values():check_identity(item)
    out=Path(cfg['run_root'])/'compute_complete.json'
    return write_receipt(out,dependencies(cfg,'14'),evidence,status_phase='COMPUTE_COMPLETE_AWAITING_MANUAL_RESUME',
                         new_endpoints=list(NEW),full_evaluations=list(NEW),reused_endpoints=list(EXISTING),methods=list(METHODS),
                         comparisons=26,factorial_contrasts=19,manual_resume_for_publication_required=True,
                         interpretation_complete=False,publication_complete=False,context_sync_complete=False,
                         scientific_values_are_not_completion_gates=True)
