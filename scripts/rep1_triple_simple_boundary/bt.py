"""Six B/T markers, exact merged T pools, immutable fixed nuclear distances."""
from .continuation import frozen_root
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from .common import (atomic_csv,atomic_json,atomic_npz,check_receipt,chunk_receipt,
                     dependencies,digest,identity,id_hash,progress,read,resume_chunk,
                     within,write_receipt,inventory,committed,check_identity,atomic)
from .inputs import matrix_arrays

PANELS = (
    dict(direction='T_marker_in_B', recipient_types=['B Cells'], target_types=['CD4 T', 'CD8 T'], markers=['CD4', 'CD8A', 'CD8B']),
    dict(direction='B_marker_in_T', recipient_types=['CD4 T', 'CD8 T'], target_types=['B Cells'], markers=['CD79A', 'CD79B', 'MS4A1']),
)
MARKERS = [g for panel in PANELS for g in panel['markers']]
EDGES = np.array([0, 10, 20, 30, 40, 50, 100], dtype=float)
PAPER_TYPES = {'B Cells': 'B Cells', 'CD4 T': 'CD4 T', 'CD8 T': 'CD8 T'}
ENGINEERING_TYPES = {'B': 'B Cells', 'CD4Tconv/Treg': 'CD4 T', 'CD8T/CD8Tex': 'CD8 T'}


def canonical_bt(labels, profile='paper_fixed'):
    if profile not in ('paper_fixed', 'engineering'):
        raise ValueError('unknown exact label dictionary')
    mapping = PAPER_TYPES if profile == 'paper_fixed' else ENGINEERING_TYPES
    return pd.Series(labels).map(mapping)


def checked_ids(frame, name):
    if 'cell_id' not in frame or frame.cell_id.isna().any() or frame.cell_id.duplicated().any():
        raise ValueError('missing or duplicate IDs: ' + name)
    ids = frame.cell_id.to_numpy(np.int64)
    if not np.array_equal(ids, frame.cell_id.to_numpy()) or np.any(ids <= 0):
        raise ValueError('noncanonical cell IDs: ' + name)
    return frame.set_index('cell_id').sort_index()


def nearest(records, recipient_ids, target_ids):
    """Euclidean nearest targets; exact-distance ties use the smallest target ID."""
    recipient_ids = np.sort(np.asarray(recipient_ids, np.int64))
    target_ids = np.sort(np.asarray(target_ids, np.int64))
    coords = records[['x_um', 'y_um']]
    out = pd.DataFrame(index=pd.Index(recipient_ids, name='cell_id'))
    out['nearest_target_id'] = pd.Series(pd.NA, index=out.index, dtype='Int64')
    out['distance_um'] = np.nan
    out['bin_index'] = pd.Series(pd.NA, index=out.index, dtype='Int64')
    out['status'] = 'empty_target_pool' if not len(target_ids) else 'computed'
    if not len(recipient_ids) or not len(target_ids):
        return out
    rxy = coords.loc[recipient_ids].to_numpy(float)
    txy = coords.loc[target_ids].to_numpy(float)
    if not np.isfinite(rxy).all() or not np.isfinite(txy).all():
        raise ValueError('nonfinite coordinates in B/T pool')
    tree = cKDTree(txy)
    distances, chosen = tree.query(rxy, workers=1)
    # query_ball_point may contain slightly more points due to its tolerance;
    # squared distances are then compared exactly, rather than treating near ties as ties.
    for i, (xy, d) in enumerate(zip(rxy, distances)):
        candidates = np.asarray(tree.query_ball_point(xy, np.nextafter(d, np.inf)), int)
        dsq = np.sum((txy[candidates] - xy) ** 2, axis=1)
        best = dsq.min()
        ties = candidates[dsq == best]
        chosen[i] = ties[np.argmin(target_ids[ties])]
        distances[i] = np.sqrt(best)
    out['nearest_target_id'] = target_ids[chosen]
    out['distance_um'] = distances
    out['bin_index'] = np.searchsorted(EDGES[1:], distances, side='left')
    return out


def curve(records, recipient_ids, target_ids, marker_values, panel, distances=None):
    """Always return 7 bins per marker, including empty recipients and targets."""
    recipient_ids = np.sort(np.asarray(recipient_ids, np.int64))
    target_ids = np.sort(np.asarray(target_ids, np.int64))
    if len(np.unique(recipient_ids)) != len(recipient_ids) or len(np.unique(target_ids)) != len(target_ids):
        raise ValueError('duplicate B/T pool ID')
    for gene in panel['markers']:
        if gene not in marker_values:
            raise ValueError('required B/T gene missing: ' + gene)
    if not set(recipient_ids).issubset(marker_values.index) or not set(recipient_ids).issubset(records.index) or not set(target_ids).issubset(records.index):
        raise ValueError('B/T join would silently drop required IDs')
    if distances is None:
        distances = nearest(records, recipient_ids, target_ids)
    elif not set(recipient_ids).issubset(distances.index):
        raise ValueError('frozen distance table lacks recipient IDs')
    dist = distances.loc[recipient_ids]
    observed_max = float(dist.distance_um.max()) if len(dist) and dist.distance_um.notna().any() else None
    last = max(100.0, observed_max or 100.0)
    rows = []
    r_hash, t_hash = id_hash(recipient_ids), id_hash(target_ids)
    for b in range(7):
        use = dist.bin_index.eq(b).fillna(False).to_numpy(bool)
        lo, hi = EDGES[b], EDGES[b + 1] if b < 6 else last
        for marker in panel['markers']:
            vals = marker_values.loc[recipient_ids, marker].to_numpy()
            if np.any(vals < 0) or not np.isfinite(vals).all():
                raise ValueError('invalid marker counts')
            n = int(use.sum())
            hit = int(np.count_nonzero(vals[use] > 0))
            status = ('empty_recipient_pool' if not len(recipient_ids) else 'empty_target_pool' if not len(target_ids)
                      else 'empty_bin' if not n else 'computed')
            rows.append(dict(direction=panel['direction'], marker=marker, bin_index=b,
                             bin=f'{"[" if b == 0 else "("}{lo:g},{hi:g}]', lower_um=lo, upper_um=hi,
                             n_eligible=n, n_positive=hit, fraction=hit / n if n else np.nan, status=status,
                             recipient_pool_n=len(recipient_ids), target_pool_n=len(target_ids),
                             recipient_ids_sha256=r_hash, target_ids_sha256=t_hash,
                             observed_max_um=observed_max,
                             endpoint_handling='fixed_bins_last_empty_when_max_le_100'))
    return pd.DataFrame(rows), dist


def mask_moments(mask, ids, out, deps, callback=None):
    """Checkpoint full-image label area and first moments, one committed row stripe."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    n = int(ids.max()) + 1
    state = out / 'state.json'
    count = np.zeros(n, np.int64)
    sx = np.zeros(n, np.float64)
    sy = np.zeros(n, np.float64)
    cursor = 0
    if state.exists():
        rec = check_receipt(state, deps)
        a = np.load(rec['snapshot']['path'])
        count, sx, sy = a['count'], a['sx'], a['sy']
        cursor = rec['cursor']
        if not 0 <= cursor <= mask.shape[0]:
            raise ValueError('moment recovery cursor invalid')
    for start in range(cursor, mask.shape[0], 512):
        stop = min(start + 512, mask.shape[0])
        part = mask[start:stop]
        if part.min(initial=0) < 0 or part.max(initial=0) >= n:
            raise ValueError('unexpected mask ID')
        flat = part.ravel()
        count += np.bincount(flat, minlength=n)
        sx += np.bincount(flat, weights=np.tile(np.arange(mask.shape[1]), stop-start), minlength=n)
        sy += np.bincount(flat, weights=np.repeat(np.arange(start, stop), mask.shape[1]), minlength=n)
        prior=read(state) if state.exists() else None
        snap = out / f'moments_{stop:06d}.npz'
        atomic_npz(snap, count=count, sx=sx, sy=sy)
        obj = identity(snap)
        atomic_json(state, dict(status='succeeded', dependencies=deps, cursor=stop, snapshot=obj, outputs={'snapshot':obj}))
        committed(state)
        if prior:
            old=prior['snapshot'];check_identity(old)
            if Path(old['path'])!=snap:Path(old['path']).unlink()
        progress(callback, 'mask_moments', stop, mask.shape[0], mask=deps.get('mask'))
    actual = np.flatnonzero(count > 0)
    actual = actual[actual > 0]
    if not np.array_equal(actual, ids):
        raise ValueError('mask does not preserve complete nuclear ID namespace')
    return pd.DataFrame(dict(cell_id=ids, area=count[ids], x_um=sx[ids]/count[ids], y_um=sy[ids]/count[ids])).set_index('cell_id')


def freeze_targets(cells,fixed,out,deps):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').exists():return check_receipt(out/'complete.json',deps)
    available=fixed.unmatched_reason.fillna('').eq('')
    records=cells[['x_um','y_um']].copy();records['label']=fixed.evaluation_label.reindex(cells.index)
    domain=np.intersect1d(cells.index,fixed.index[available]);stats={}
    for panel in PANELS:
        label=records.loc[domain,'label'];rids=label.index[label.isin(panel['recipient_types'])].to_numpy(np.int64);tids=label.index[label.isin(panel['target_types'])].to_numpy(np.int64)
        d=nearest(records,rids,tids);direction=panel['direction'];atomic_csv(out/(direction+'_frozen_distances.csv'),d.reset_index())
        for role,pool in [('recipients',rids),('targets',tids)]:atomic_csv(out/(direction+'_'+role+'.csv'),pd.DataFrame({'cell_id':pool}))
        stats[direction]=dict(recipient_ids_sha256=id_hash(rids),target_ids_sha256=id_hash(tids),recipients=len(rids),targets=len(tids))
    return write_receipt(out/'complete.json',deps,inventory(out),target_pool='F intersect complete nuclear IDs with available frozen B1 B/T labels',pools=stats,new_QC_used=False)


def endpoint_frame(cfg,manifest,spec):
    ids,genes,counts=matrix_arrays(spec)
    if not set(MARKERS).issubset(genes):raise ValueError('six full313 markers missing (CD8B required)')
    total=counts.sum(1);detected=(counts>0).sum(1);qc=(total>=20)&(total<=2000)&(detected>=20)
    ann=checked_ids(pd.read_csv(Path(spec['evaluation'])/'paper_aligned_v1/annotation.csv'),'BT ownQC annotation')
    coords=checked_ids(pd.read_csv(Path(spec['evaluation'])/'paper_aligned_v1/own_cell_centroids.csv'),'BT own centroids')
    if not np.array_equal(ann.index,ids[qc]) or not np.array_equal(coords.index,ids):raise ValueError('BT ownQC/centroid population mismatch')
    fixed=checked_ids(pd.read_csv(manifest['fixed_population']),'BT fixed labels')
    frame=pd.DataFrame(index=pd.Index(ids,name='cell_id'));frame['own_qc']=qc;frame['sc_label']=ann.evaluation_label.reindex(ids)
    frame['fixed_label']=fixed.evaluation_label.reindex(ids);frame[['x_um','y_um']]=coords[['x_um','y_um']]
    for g in MARKERS:frame[g]=counts[:,genes.index(g)]
    return frame


def run_endpoint(cfg,manifest,spec,out,callback=None):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);deps=dependencies(cfg,'BT_'+spec['name'])
    if (out/'complete.json').exists():return check_receipt(out/'complete.json',deps)
    frozen=frozen_root(cfg);reference=checked_ids(pd.read_csv(frozen/'cells.csv'),'fixed nuclear geometry')
    target=frozen/'bt_targets';check_receipt(target/'complete.json')
    frame=endpoint_frame(cfg,manifest,spec);fixed=pd.read_csv(manifest['fixed_population']).cell_id.to_numpy(np.int64)
    atomic_csv(out/'cells.csv',frame,index=True)
    populations=[('official_sc_ownQC_owncentroid',frame.index[frame.own_qc].to_numpy(),False),
                 ('fixed_commonID12_nuclear_ref',np.intersect1d(fixed,frame.index),True),
                 ('fixed_ownQC_nuclear_ref',np.intersect1d(fixed,frame.index[frame.own_qc]),True)]
    for i,(profile,ids,fx) in enumerate(populations):
        dest=out/profile;dep=dict(deps,profile=profile,population=id_hash(ids))
        if (dest/'complete.json').exists():check_receipt(dest/'complete.json',dep)
        else:
            legacy=None
            if not fx and spec.get('official_bt_root'):
                table=reuse_official(frame,ids,spec,dest)
                legacy=dict(producer=spec['official_bt_producer'],registry_chain=spec['official_bt_registry_chain'])
            else:table=write_curves(frame,dest,profile,ids,reference if fx else frame,target if fx else None)
            if table.marker.nunique()!=6 or len(table)!=42:raise ValueError('BT incomplete six-marker seven-bin coverage')
            old=Path(spec['evaluation'])/'paper_aligned_v1/neighbour_curves.csv'
            if not fx and old.exists():
                before=pd.read_csv(old).sort_values(['marker','lower_um']);after=table[table.direction=='T_marker_in_B'].sort_values(['marker','bin_index'])
                if len(before)!=len(after) or not np.array_equal(before.n_eligible,after.n_eligible) or not np.array_equal(before.n_positive,after.n_positive) or not np.allclose(before.fraction,after.fraction,rtol=0,atol=1e-14,equal_nan=True):raise ValueError('official prior B-recipient curve compatibility failed')
            write_receipt(dest/'complete.json',dep,inventory(dest),six_markers=True,rows=len(table),historical_official_reuse=legacy)
        progress(callback,'six_marker_BT',i+1,len(populations),receipt=str(dest/'complete.json'))
    return write_receipt(out/'complete.json',deps,inventory(out),endpoint=spec['name'],markers=MARKERS,profiles=[p[0] for p in populations],fixed_target_pool_does_not_use_new_QC=True)


def paired(cfg,manifest,specs,out,callback=None,reuse_from=None,pair_list=None):
    from .common import reuse_partition,partition_source
    from .contracts import comparisons,comparison_id,FACTORIAL8,METHODS,SCALES
    from .comparisons import pooled_interval
    out=Path(out);out.mkdir(parents=True,exist_ok=True);deps=dependencies(cfg,'BT_PAIRED_'+str(len(specs)))
    if (out/'complete.json').exists():return check_receipt(out/'complete.json',deps)
    frames={m:endpoint_frame(cfg,manifest,s) for m,s in specs.items()};frozen=frozen_root(cfg)
    ref=checked_ids(pd.read_csv(frozen/'cells.csv'),'nuclear reference');target=frozen/'bt_targets'
    fixed=set(pd.read_csv(manifest['fixed_population']).cell_id);sets=[]
    pairs=comparisons(tuple(specs)) if pair_list is None else pair_list
    if any(a not in specs or b not in specs for a,b in pairs):raise ValueError('paired B/T endpoint unavailable')
    for a,b in pairs:
        q=np.asarray(sorted(fixed&set(frames[a].index[frames[a].own_qc])&set(frames[b].index[frames[b].own_qc])),np.int64)
        sets.append((comparison_id(a,b),(a,b),q))
    for name,methods in [('commonQC8',FACTORIAL8),('commonQC12',METHODS)]:
        if pair_list is None and set(methods).issubset(specs):sets.append((name,methods,np.asarray(sorted(fixed&set.intersection(*(set(frames[m].index[frames[m].own_qc]) for m in methods))),np.int64)))
    rows=[]
    for index,(key,methods,q) in enumerate(sets):
        dest=out/key;dest.mkdir(parents=True,exist_ok=True);dep=dict(deps,key=key,population=id_hash(q))
        binding=dict(methods=list(methods),population_ids_sha256=id_hash(q),target=identity(target/'complete.json'),
            coefficient={methods[0]:1,methods[1]:-1} if len(methods)==2 else None,
            weights={str(s):identity(frozen/f'bootstrap_weights_{s}.npz') for s in SCALES},
            matrix={m:identity(Path(specs[m]['matrix'])/'complete.json') for m in methods})
        if (dest/'complete.json').exists():check_receipt(dest/'complete.json',dep);continue
        source=partition_source(reuse_from,key)
        if source is not None and (source/'complete.json').exists():reuse_partition(source,dest,dep,binding,cfg=cfg);continue
        atomic_csv(dest/'population.csv',pd.DataFrame({'cell_id':q}))
        for m in methods:
            write_curves(frames[m],dest/m,'fixed_'+key+'_nuclear_ref',q,ref,target)
        summary=[]
        if len(methods)==2:
            a,b=methods
            for panel in PANELS:
                dist=checked_ids(pd.read_csv(target/(panel['direction']+'_frozen_distances.csv')),'BT frozen distance')
                ids=np.intersect1d(q,dist.index[dist.distance_um.le(40)])
                for g in panel['markers']:
                    delta=(frames[a].loc[ids,g].to_numpy()>0).astype(float)-(frames[b].loc[ids,g].to_numpy()>0).astype(float)
                    for side in SCALES:
                        W=np.load(frozen/f'bootstrap_weights_{side}.npz')['W'];bi=ref.loc[ids,'block_'+str(side)].to_numpy(int)
                        n=np.bincount(bi,minlength=W.shape[1]);s=np.bincount(bi,weights=delta,minlength=W.shape[1])
                        row=dict(comparison=key,candidate=a,control=b,marker=g,direction=panel['direction'],window_um='0-40 inclusive',scale_um=side,
                                 ids_sha256=id_hash(ids),population='fixed_pairwise_commonQC_no_F1_finite_filter',**pooled_interval(s,n,W))
                        summary.append(row)
            atomic_csv(dest/'paired_0_40um.csv',pd.DataFrame(summary))
        write_receipt(dest/'complete.json',dep,inventory(dest),methods=list(methods),six_markers=True,reuse_binding=binding)
        progress(callback,'BT_shared_populations',index+1,len(sets),receipt=str(dest/'complete.json'))
    atomic_json(out/'coverage.json',dict(sets=[s[0] for s in sets],markers=MARKERS,full_curves=True,paired_fixed_window=[0,40],scales=list(SCALES)))
    return write_receipt(out/'complete.json',deps,inventory(out),methods=list(specs),sets=[s[0] for s in sets])


def validate_official(frame,population_ids,source,recompute_nearest=False):
    """Check every original pool and seven-bin integer count against current inputs.

    The optional nearest recomputation is used by the actual legacy fixture.
    Production reuses the hash-bound historical distance producer, additionally
    checking distances against its selected target and the current centroids.
    """
    source=Path(source);saved=check_receipt(source/'complete.json');table=pd.read_csv(source/'full_curves.csv')
    if len(table)!=42 or set(table.marker)!=set(MARKERS) or not table.population_ids_sha256.eq(id_hash(population_ids)).all():
        raise ValueError('historical official B/T population/marker coverage mismatch')
    labels=frame.loc[population_ids,'sc_label']
    for panel in PANELS:
        direction=panel['direction'];rids=labels.index[labels.isin(panel['recipient_types'])].to_numpy(np.int64)
        tids=labels.index[labels.isin(panel['target_types'])].to_numpy(np.int64)
        for role,ids in [('recipients',rids),('targets',tids)]:
            if not np.array_equal(pd.read_csv(source/(direction+'_'+role+'.csv')).cell_id,ids):raise ValueError('historical official B/T pool IDs changed')
        distances=checked_ids(pd.read_csv(source/(direction+'_distances.csv')),'historical official distance')
        if not np.array_equal(distances.index,rids):raise ValueError('historical official B/T distance population mismatch')
        if len(tids) and len(rids):
            selected=distances.nearest_target_id.to_numpy(np.int64)
            if not np.isin(selected,tids).all():raise ValueError('historical nearest target outside original pool')
            xy=frame[['x_um','y_um']]
            actual=np.sqrt(np.sum((xy.loc[rids].to_numpy()-xy.loc[selected].to_numpy())**2,axis=1))
            if not np.allclose(actual,distances.distance_um,rtol=0,atol=1e-10):raise ValueError('historical B/T centroid distance differs')
            if not np.array_equal(np.searchsorted(EDGES[1:],actual,side='left'),distances.bin_index):raise ValueError('historical B/T bin boundary differs')
        if recompute_nearest:
            exact=nearest(frame,rids,tids)
            if not exact.nearest_target_id.astype('Int64').equals(distances.nearest_target_id.astype('Int64')) or not np.allclose(exact.distance_um,distances.distance_um,rtol=0,atol=1e-10,equal_nan=True):raise ValueError('historical exact nearest-target compatibility failed')
        new,_=curve(frame,rids,tids,frame,panel,distances)
        old=table[table.direction==direction].sort_values(['marker','bin_index']).reset_index(drop=True)
        new=new.sort_values(['marker','bin_index']).reset_index(drop=True)
        for field in ('marker','bin_index','n_eligible','n_positive','recipient_ids_sha256','target_ids_sha256','status'):
            if not np.array_equal(old[field],new[field]):raise ValueError('historical B/T compatibility differs: '+field)
        if not np.allclose(old.fraction,new.fraction,rtol=0,atol=1e-14,equal_nan=True):raise ValueError('historical B/T fractions differ')
    return table

def reuse_official(frame,ids,spec,out):
    import shutil
    source=Path(spec['official_bt_root']);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    check_identity(spec['official_bt_producer']);check_identity(spec['official_bt_registry_chain'])
    saved=check_receipt(source/'complete.json');table=validate_official(frame,ids,source)
    for name,obj in saved['outputs'].items():
        src=check_identity(obj);dest=out/name
        # Copy, never hard-link historical data into a writable run.
        def copy(f,src=src):
            with Path(src).open('rb') as handle:shutil.copyfileobj(handle,f)
        atomic(dest,copy)
        if identity(dest)['sha256']!=obj['sha256']:raise ValueError('historical B/T copy drift')
    plot_curves(table,out,'official_sc_ownQC_owncentroid')
    return table

def write_curves(frame, out, profile, population_ids, reference, fixed_root=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    tables = []
    for panel in PANELS:
        direction = panel['direction']
        if fixed_root is None:
            labels = frame.loc[population_ids, 'sc_label']
            rids = labels.index[labels.isin(panel['recipient_types'])].to_numpy(np.int64)
            tids = labels.index[labels.isin(panel['target_types'])].to_numpy(np.int64)
            dist = None
        else:
            fixed_root = Path(fixed_root)
            full_dist = checked_ids(pd.read_csv(fixed_root/(direction+'_frozen_distances.csv')), 'fixed distances')
            full_r = pd.read_csv(fixed_root/(direction+'_recipients.csv')).cell_id.to_numpy(np.int64)
            rids = np.intersect1d(full_r, population_ids)
            tids = pd.read_csv(fixed_root/(direction+'_targets.csv')).cell_id.to_numpy(np.int64)
            dist = full_dist
        table, distances = curve(reference, rids, tids, frame, panel, dist)
        table['profile'] = profile
        table['population_ids_sha256'] = id_hash(population_ids)
        table['population_n'] = len(population_ids)
        tables.append(table)
        atomic_csv(out/(direction+'_distances.csv'), distances.reset_index())
        for role, pool in (('recipients',rids),('targets',tids)):
            atomic_csv(out/(direction+'_'+role+'.csv'), pd.DataFrame({'cell_id':pool}))
    table = pd.concat(tables, ignore_index=True)
    atomic_csv(out/'full_curves.csv', table)
    plot_curves(table,out,profile)
    return table

def plot_curves(table,out,profile):
    out=Path(out)
    # The plotting backend writes only to the destination, never the snapshot.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2,3,figsize=(12,7),sharex=True,sharey=True)
    for ax, marker in zip(axes.ravel(), MARKERS):
        part = table[(table.marker==marker)&(table.upper_um<=40)]
        ax.plot(part.upper_um, part.fraction, marker='o')
        ax.set_title(marker)
        ax.set_xlim(0,40)
        ax.set_ylim(0,1)
        ax.set_xlabel('Nearest opposite type distance (µm)')
        ax.set_ylabel('Fraction with count > 0')
    fig.suptitle(profile)
    fig.tight_layout()
    fig.savefig(out/'curves_0_40um.png',dpi=160)
    plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(14,8),sharey=True)
    for ax,marker in zip(axes.ravel(),MARKERS):
        part=table[table.marker==marker].sort_values('bin_index')
        ax.plot(part.bin_index,part.fraction,marker='o');ax.set_xticks(part.bin_index,part.bin,rotation=35)
        ax.set_title(marker);ax.set_ylim(0,1);ax.set_ylabel('Fraction with count > 0')
    fig.suptitle(profile+' — all seven prespecified distance bins');fig.tight_layout()
    fig.savefig(out/'full_curves.png',dpi=160);fig.savefig(out/'full_curves.pdf');plt.close(fig)
