"""Actual old BASE/Simple max-change chunk with conservative 29-column payload.

Only the resource fixture's not-yet-frozen region/density/interface fields are
synthetic high-entropy values. They are never scientific domain assignments.
The actual source/target owners, genes, types and coordinates remain unchanged.
"""
import argparse,fcntl,time
from pathlib import Path
import numpy as np
import pandas as pd
import tifffile
from scripts.rep1_triple_simple_boundary.common import *
from scripts.rep1_triple_simple_boundary.boundary_analysis import sparse_transfer,TRANSFER_COLUMNS,_owner_chunk
from scripts.rep1_triple_simple_boundary.frozen_domains import quartiles,block_indices,bootstrap_weights,interface_edges
from scripts.rep1_triple_simple_boundary.inputs import matrix_arrays
from scripts.rep1_triple_simple_boundary.contracts import SCALES
from tests.rep1_triple_real_fixtures import MemorySampler


def run(out,manifest_path,statistics_path):
    out=Path(out);out.mkdir(parents=True,exist_ok=False);manifest=read(manifest_path)
    base=manifest['endpoints']['BASE_MATCH'];simple=manifest['endpoints']['MORPH_SIMPLE_FUSION'];start=time.monotonic();sources={}
    def bound(path):
        obj=identity(path);sources[str(Path(path).resolve())]=obj
        old=manifest['files'].get(str(Path(path).resolve()))
        if old is not None and obj['sha256']!=old['sha256']:raise ValueError('resource input identity drift')
        return obj
    with Path('/home/zwan0726/work/outputs/cos_gpu.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with MemorySampler((lock.fileno(),)) as memory:
            ar=read(Path(base['matrix'])/'complete.json');br=read(Path(simple['matrix'])/'complete.json');bound(Path(base['matrix'])/'complete.json');bound(Path(simple['matrix'])/'complete.json');catalogue=[]
            for i,(a,b,c) in enumerate(zip(ar['receipts'],br['receipts'],manifest['coordinates'])):
                x=_owner_chunk(base,a,c);y=_owner_chunk(simple,b,c);catalogue.append(dict(chunk=i,rows=len(x),changed=int(np.count_nonzero(x!=y))))
            if len(catalogue)!=69 or sum(r['rows'] for r in catalogue)!=34442716:raise ValueError('actual owner chunk coverage differs')
            chosen=max(catalogue,key=lambda r:(r['changed'],-r['chunk']));i=chosen['chunk'];coordinate=manifest['coordinates'][i];bound(coordinate['path']);co=np.load(coordinate['path'])
            x=_owner_chunk(base,ar['receipts'][i],coordinate);y=_owner_chunk(simple,br['receipts'][i],coordinate)
            for spec,r in ((base,ar['receipts'][i]),(simple,br['receipts'][i])):bound(Path(spec['matrix'])/r['owner_file'])
            bound(manifest['metadata']);meta=np.load(manifest['metadata']);ids=meta['ids'];cells=pd.DataFrame(index=pd.Index(ids,name='cell_id'));cells['original_type_index']=meta['type_index']
            bound(base['final']);mask=tifffile.imread(base['final']);area=np.bincount(mask.ravel(),minlength=int(ids.max())+1)[ids]
            _,_,counts=matrix_arrays(base);tot=counts.sum(1);del counts
            old=Path(manifest['historical_strata'])/'fixed_cell_groups.csv';bound(old);crowd=pd.read_csv(old).set_index('cell_id').loc[ids,'density_30um'].to_numpy()
            eig=np.linalg.eigvalsh(meta['covariance']);ecc=np.sqrt(np.maximum(0,1-np.divide(eig[:,0],eig[:,1],out=np.full(len(ids),np.nan),where=eig[:,1]>0)))
            for name,value in [('BASE_count_quartile',tot),('RNA_density_quartile',tot/area),('nuclear_crowding_quartile',crowd),('nuclear_area_quartile',meta['area']),('eccentricity_quartile',ecc)]:cells[name]=quartiles(value)[1]
            rng=np.random.default_rng(20261017);N=len(co)
            # Upper-cardinality resource burden; these fields are not claims
            # about scientific fixed domains, interfaces, or density groups.
            st=dict(region_bits=rng.integers(1,1024,N,dtype=np.uint16),local_density=rng.integers(0,4,N,dtype=np.int8),interface_id=rng.integers(-1,len(ids)*4,N,dtype=np.int32))
            kp=Path(manifest['historical_strata'])/'support_counts.npy';bound(kp);K=np.load(kp,mmap_mode='r');good=(co['x']>=0)&(co['y']>=0);st['K']=np.zeros(N,np.uint8);st['K'][good]=np.minimum(K[co['y'][good],co['x'][good]],4)
            for side in SCALES:
                blocks,ix,W=bootstrap_weights(meta['centroid_xy'],side);cells['block_'+str(side)]=ix;st['block_'+str(side)]=block_indices(np.c_[co['x'],co['y']],blocks,side,True)
            type_count=len(cells.original_type_index.unique())
            local_extra=sum(cells.groupby(['original_type_index',dim]).ngroups for dim in ('block_500','block_1000','block_2000','RNA_density_quartile','nuclear_crowding_quartile'))
            t=time.monotonic();keys,n=sparse_transfer(x,y,co['gene'],st,cells);aggregation_seconds=time.monotonic()-t
            if int(n.sum())!=chosen['changed'] or keys.shape[1]!=29:raise ValueError('resource full sparse ledger missing rows/columns')
            atomic_npz(out/'representative_full29_sparse.npz',keys=keys,molecules=n)
            atomic_npz(out/'representative_pixel28_sparse.npz',keys=keys[:,1:],pixels=n)
            # Real full masks only count changes and interface cardinality;
            # no formal new final, domain, comparison, or scientific ROI exists.
            bound(simple['final']);other=tifffile.imread(simple['final']);changed_pixels=int(np.count_nonzero(mask!=other));del other
            edges=interface_edges(mask);n_interfaces=len(np.unique(edges[:,:2],axis=0));del edges,mask
    stats=read(statistics_path);statroot=Path(statistics_path).parent
    files=[p for p in statroot.rglob('*') if p.is_file() and p.name!='fixture_receipt.json'];statistics_bytes=sum(p.stat().st_size for p in files)
    statistics_allocated=sum(p.stat().st_blocks*512 for p in files)
    molecule_bytes=(out/'representative_full29_sparse.npz').stat().st_size;pixel_bytes=(out/'representative_pixel28_sparse.npz').stat().st_size
    ratio=molecule_bytes/max(len(keys),1);pixel_ratio=pixel_bytes/max(len(keys),1);changed_total=sum(r['changed'] for r in catalogue)
    timings=stats['timings'];membership=next(r['seconds'] for r in timings if r['operation']=='worst_case_membership');metric=sum(r['seconds'] for r in timings if r['operation']!='worst_case_membership')
    G=stats['group_count'];cell_G=G+local_extra;spatial_G=10*(1+4*(1+6*5)+6*5+5+5+1)+type_count*11;interface_G=G+4
    groups_by_task=dict(score=69*12*G,cell_and_structure=26*9*cell_G,spatial_pixel_and_molecule=26*12*spatial_G,interface=26*6*interface_G,regional=26*3*10,paired_BT=26*6)
    stat_projections={k:dict(group_metric_units_three_scales=v,CI_rows=v*3,persistent_bytes_projection=statistics_bytes*v/G) for k,v in groups_by_task.items()}
    trace=read('/home/zwan0726/work/docs/BIDCell_Rep1_Triple_Simple_RootOld7BTTrace_20261017.json')
    bt_profile_sizes={name:sum(p.stat().st_size for p in Path(row['profile_root']).rglob('*') if p.is_file()) for name,row in trace['endpoints'].items()}
    result=dict(status='passed_resource_probe',scope='old BASE to Simple read-only max-change owner chunk, resource payload only; not a scientific transfer table',
        selected_chunk=chosen,owner_catalogue=catalogue,full_29_columns=list(TRANSFER_COLUMNS),actual_sparse_rows=len(keys),actual_molecules_changed=changed_total,actual_pixels_changed=changed_pixels,
        actual_BASE_interface_pairs=n_interfaces,aggregation_seconds=aggregation_seconds,wall_seconds=time.monotonic()-start,resource=memory.result(),
        synthetic_resource_only_fields=['region_bits high entropy 1..1023','local_density high entropy 0..3','interface_id high entropy -1..4*Ncells'],
        compressed_molecule_bytes_per_unique_row=ratio,compressed_pixel_bytes_per_unique_row=pixel_ratio,
        compressed_26pair_old_change_workload_projection_bytes=26*(changed_total*ratio+changed_pixels*pixel_ratio),
        absolute_full_molecule_29_int64_raw_bound_all26_bytes=26*34442716*30*8,
        absolute_full_pixel_28_int64_raw_bound_all26_bytes=26*5478*7526*29*8,
        scoring_projection=dict(jobs=69,profile_metrics=12,groups_bound=stats['group_count'],rows=stats['comparison_intervals_rows'],
            persisted_all_files_per_metric_three_scales_bytes=statistics_bytes,persisted_allocated_bytes=statistics_allocated,
            persisted_all69x12_bytes=statistics_bytes*69*12,persisted_all69x12_allocated_bytes=statistics_allocated*69*12,
            seconds_estimate=69*membership+69*12*metric,scope='all table gzip, block sufficient statistics, intervals JSON, final receipts and retired-batch identities'),
        full_statistics_projection=dict(frozen_cell_groups_bound=G,actual_type_property_additional_groups=local_extra,local_cell_groups_bound=cell_G,position_groups=spatial_G,interface_groups_bound=interface_G,
            by_task=stat_projections,total_bytes=sum(r['persistent_bytes_projection'] for r in stat_projections.values()),
            definition='scale every persisted file in measured dense statistic by actual/bounded complete group count; this is a resource estimate, not an absolute compression bound'),
        BT_distance_curve_source_profiles_bytes=bt_profile_sizes,BT_108_profiles_projection_bytes=max(bt_profile_sizes.values())*108,
        other_retained_allowances_bytes=dict(BT_new_plots_and_receipts=128*1024**2,ROI_32x128um_data_all_methods_and_all26_panels=512*1024**2,
            full_cell_distributions_raw_tables_populations_endpoint_regions=512*1024**2,nested_inventory_receipts_long_stratum_names_block_state_snapshots=1024**3),
        known_additional_budget_terms=dict(boundary_pairs=26,cell_cost_metrics=4,structure_metrics=5,spatial_units=2,spatial_metrics=6,interface_units=2,interface_metrics=3,
            BT_curves_per_profile=42,BT_profiles_per_endpoint=3,endpoints=12,paired_BT_sets=26,common_BT_sets=2,paired_BT_metrics=6,scales=3,roi_slots=32,roi_method_masks=12,
            temporary_bootstrap_one_job_sample_array_bytes=2000*stats['group_count']*8,temporary_one_batch_bytes=100*stats['group_count']*8,
            raw_sparse_working_selected_chunk_bytes=int(keys.nbytes+n.nbytes)),
        limitations=['old changed workloads and measured compression are estimates, not upper bounds on new endpoints',
          'absolute raw full-change bounds exceed disk capacity; launch must use measured workload estimates plus per-chunk admission and preserve failed evidence',
          'complete all69 scoring projection includes persistent duplicated intervals JSON and all receipts; boundary/BT/ROI add separately',
          'pixel compression uses the same actual ID combinations as a conservative cardinality proxy; it is not an actual pixel transfer table',
          'RAM includes actual max-change chunk aggregation only, not a full scientific stage or concurrent tasks'],
        inputs=sources,statistical_fixture=identity(statistics_path),source=identity(__file__),outputs=inventory(out))
    atomic_json(out/'fixture_receipt.json',result);print({k:result[k] for k in ('status','selected_chunk','actual_sparse_rows','actual_molecules_changed','actual_pixels_changed','resource','scoring_projection','compressed_26pair_old_change_workload_projection_bytes')},flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--manifest',required=True);p.add_argument('--statistics',required=True);a=p.parse_args();run(a.output,a.manifest,a.statistics)
