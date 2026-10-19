"""Six prespecified B/T panels with explicit, hashed recipient and target pools."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import tifffile
from scipy.spatial import cKDTree
from . import EXISTING, N_CELLS, N_GENES, comparisons
from .common import (atomic_csv, atomic_json, atomic_npz, check_receipt, chunk_receipt,
                     dependencies, digest, identity, id_hash, progress, read, resume_chunk,
                     within, write_receipt, inventory)
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
        snap = out / f'moments_{stop:06d}.npz'
        atomic_npz(snap, count=count, sx=sx, sy=sy)
        obj = identity(snap)
        atomic_json(state, dict(status='succeeded', dependencies=deps, cursor=stop, snapshot=obj, outputs={'snapshot':obj}))
        for old in out.glob('moments_*.npz'):
            if old != snap:
                old.unlink()
        progress(callback, 'mask_moments', stop, mask.shape[0], mask=deps.get('mask'))
    actual = np.flatnonzero(count > 0)
    actual = actual[actual > 0]
    if not np.array_equal(actual, ids):
        raise ValueError('mask does not preserve complete nuclear ID namespace')
    return pd.DataFrame(dict(cell_id=ids, area=count[ids], x_um=sx[ids]/count[ids], y_um=sy[ids]/count[ids])).set_index('cell_id')


def check_old_scores(spec, fixed_ids):
    sc = checked_ids(pd.read_csv(spec['sc_scores']), 'sc scores')
    fx = checked_ids(pd.read_csv(spec['fixed_scores']), 'fixed scores')
    if not np.array_equal(fx.index.to_numpy(), np.sort(fixed_ids)):
        raise ValueError('fixed score panel is not the complete historical 103000 IDs')
    result = {}
    for prefix, frame in (('scclassify', sc), ('fixed', fx)):
        for metric in ('positive_F1', 'negative_F1', 'positive_Precision', 'positive_Recall', 'negative_Precision', 'negative_Recall'):
            key = prefix + '_' + metric
            vals = frame[metric].to_numpy(float)
            finite = np.isfinite(vals)
            expected = spec['expected_metrics'][key]
            mean = float(vals[finite].mean())
            if int(finite.sum()) != expected['n_finite'] or not np.isclose(mean, expected['mean'], rtol=0, atol=1e-12):
                raise ValueError('historical F1/P/R aggregate mismatch: ' + spec['name'] + '/' + key)
            if prefix == 'fixed' and metric.endswith('_F1') and int(finite.sum()) != 85421:
                raise ValueError('historical fixed finite population differs from 85421')
            result[key] = dict(mean=mean, n_finite=int(finite.sum()), n_total=len(vals))
    return result


def prepare_population(cfg, manifest, callback=None):
    out = within(cfg['run_root'], 'bt_existing6', 'populations')
    deps = dependencies(cfg, 'BT_EXISTING6_POPULATIONS')
    if (out / 'complete.json').exists():
        check_receipt(out / 'complete.json', deps, cfg['run_root'])
        return out
    out.mkdir(parents=True, exist_ok=True)
    fixed = checked_ids(pd.read_csv(manifest['fixed_population']), 'fixed labels')
    if len(fixed) != 103000:
        raise ValueError('historical fixed population is not 103000')
    ids_fixed = fixed.index.to_numpy(np.int64)
    meta = np.load(manifest['metadata'])
    ids = meta['ids'].astype(np.int64)
    if len(ids) != N_CELLS or len(np.unique(ids)) != len(ids) or not np.array_equal(np.sort(ids), ids):
        raise ValueError('invalid 107955 nucleus metadata namespace')
    base_spec = manifest['endpoints']['BASE_MATCH']
    nuclei = tifffile.imread(base_spec['nuclei'])
    nuclear = mask_moments(nuclei, ids, out / 'nuclear_geometry', dict(deps, mask=manifest['files'][str(Path(base_spec['nuclei']).resolve())]['sha256']), callback)
    if not np.array_equal(nuclear.area, meta['area']) or not np.allclose(nuclear[['x_um','y_um']], meta['centroid_xy'], rtol=0, atol=1e-8):
        raise ValueError('complete nucleus image and metadata coordinates/area disagree')
    atomic_csv(out / 'nuclear_reference.csv', nuclear.reset_index())
    populations = {}
    for i, name in enumerate(EXISTING):
        spec = manifest['endpoints'][name]
        epout = out / name
        epdeps = dict(deps, endpoint=name)
        if (epout / 'complete.json').exists():
            rec = check_receipt(epout / 'complete.json', epdeps, cfg['run_root'])
            populations[name] = read(epout / 'population.json')
            continue
        epout.mkdir(exist_ok=True)
        mids, genes, counts = matrix_arrays(spec)
        if len(genes) != N_GENES or not np.array_equal(mids, ids):
            raise ValueError('six endpoint complete matrix ID/gene namespace mismatch: ' + name)
        if genes != Path(read(spec['machine'])['genes']).read_text().splitlines() or not set(MARKERS).issubset(genes):
            raise ValueError('full 313 gene order/required B/T markers mismatch')
        own_nuc = tifffile.imread(spec['nuclei'])
        if not np.array_equal(own_nuc, nuclei):
            raise ValueError('nuclear owner pixels differ: ' + name)
        del own_nuc
        mask = tifffile.imread(spec['final'])
        if mask.shape != nuclei.shape or not np.array_equal(mask[nuclei > 0], nuclei[nuclei > 0]):
            raise ValueError('nuclear owner is not preserved by old endpoint: ' + name)
        props = mask_moments(mask, ids, epout / 'geometry', dict(epdeps, mask=manifest['files'][str(Path(spec['final']).resolve())]['sha256']), callback)
        del mask
        historical = checked_ids(pd.read_csv(Path(spec['evaluation']) / 'paper_aligned_v1/own_cell_centroids.csv'), name + ' centroids')
        hn = checked_ids(pd.read_csv(Path(spec['evaluation']) / 'engineering/structure/annotation_nuclear.csv'), name + ' nuclear coordinates')
        if not np.array_equal(historical.index, ids) or not np.array_equal(hn.index, ids):
            raise ValueError('coordinate join would drop full matrix IDs')
        if not np.allclose(historical[['x_um','y_um']], props[['x_um','y_um']], rtol=0, atol=1e-8) or not np.allclose(hn[['x_um','y_um']], nuclear[['x_um','y_um']], rtol=0, atol=1e-8):
            raise ValueError('historical own/nuclear coordinates disagree with mask')
        total = np.asarray(counts.sum(1, dtype=np.int64))
        detected = np.asarray((counts > 0).sum(1))
        own_qc = (total >= 20) & (total <= 2000) & (detected >= 20)
        qc = checked_ids(pd.read_csv(Path(spec['evaluation']) / 'paper_aligned_v1/qc_all_cells.csv'), name + ' QC')
        if not np.array_equal(qc.index, ids) or not np.array_equal(qc.qc_pass.to_numpy(bool), own_qc) or not np.array_equal(qc.total_transcripts, total) or not np.array_equal(qc.total_genes, detected):
            raise ValueError('ownQC does not match frozen 313-gene QC definition')
        ann = checked_ids(pd.read_csv(Path(spec['evaluation']) / 'paper_aligned_v1/annotation.csv'), name + ' annotation')
        if not np.array_equal(ann.index.to_numpy(), ids[own_qc]):
            raise ValueError('annotation/QC exact population mismatch')
        sc = checked_ids(pd.read_csv(spec['sc_scores']), name + ' scores')
        if not np.array_equal(sc.index, ann.index):
            raise ValueError('sc score/annotation population mismatch')
        label = ann.evaluation_label.reindex(ids)
        frame = pd.DataFrame(index=pd.Index(ids, name='cell_id'))
        frame['own_qc'] = own_qc
        frame['sc_label'] = label
        frame['fixed_label'] = fixed.evaluation_label.reindex(ids)
        frame['fixed_unmatched_reason'] = fixed.unmatched_reason.reindex(ids)
        frame[['x_um','y_um']] = props[['x_um','y_um']]
        for gene in MARKERS:
            frame[gene] = counts[:, genes.index(gene)]
        atomic_csv(epout / 'cells.csv', frame.reset_index())
        metrics = check_old_scores(spec, ids_fixed)
        stats = dict(endpoint=name, full_ids_n=len(ids), own_qc_n=int(own_qc.sum()), full_ids_sha256=id_hash(ids),
                     own_qc_ids_sha256=id_hash(ids[own_qc]), missing_matrix_rows=0, missing_coordinates=0,
                     missing_sc_labels_in_qc=int(ann.evaluation_label.isna().sum()),
                     sc_unassigned=int(ann.evaluation_label.eq('unassigned').sum()),
                     sc_unknown=int(ann.evaluation_label.eq('Unknown').sum()),
                     fixed_missing_labels=int(fixed.evaluation_label.isna().sum()),
                     fixed_unassigned=int(fixed.evaluation_label.eq('unassigned').sum()),
                     fixed_unknown=int(fixed.evaluation_label.eq('Unknown').sum()),
                     assigned=int(total.sum()), historical_metrics_recomputed=metrics)
        if stats['assigned'] != spec['assigned']:
            raise ValueError('old matrix capture differs from registry')
        atomic_json(epout / 'population.json', stats)
        write_receipt(epout / 'complete.json', epdeps, inventory(epout), population=stats)
        populations[name] = stats
        progress(callback, 'BT_EXISTING6_identity_and_QC', i+1, len(EXISTING), endpoint=name)
    frames = {m: checked_ids(pd.read_csv(out / m / 'cells.csv'), m) for m in EXISTING}
    common_id = set.intersection(*(set(f.index) for f in frames.values()))
    common_qc = set.intersection(*(set(f.index[f.own_qc]) for f in frames.values()))
    sets = {'commonID6':sorted(common_id), 'commonQC6':sorted(common_qc),
            'fixed_commonID6':sorted(set(ids_fixed) & common_id), 'fixed_commonQC6':sorted(set(ids_fixed) & common_qc)}
    for a,b in comparisons(EXISTING):
        q = set(frames[a].index[frames[a].own_qc]) & set(frames[b].index[frames[b].own_qc])
        sets[f'pairwise_commonQC__{a}__{b}'] = sorted(q)
        sets[f'fixed_pairwise_commonQC__{a}__{b}'] = sorted(q & set(ids_fixed))
        migration = pd.crosstab(frames[b].loc[sorted(q),'sc_label'].fillna('__missing__'), frames[a].loc[sorted(q),'sc_label'].fillna('__missing__'), dropna=False)
        atomic_csv(out / f'label_transition__{a}__{b}.csv', migration, index=True)
        transition = pd.DataFrame(dict(cell_id=ids, candidate_qc=frames[a].own_qc, control_qc=frames[b].own_qc))
        atomic_csv(out / f'QC_transition__{a}__{b}.csv', transition)
    for key, val in sets.items():
        atomic_csv(out / (key + '.csv'), pd.DataFrame({'cell_id':val}))
    target_domain = np.asarray(sets['fixed_commonID6'], np.int64)
    fixed_records = nuclear.copy()
    fixed_records['label'] = fixed.evaluation_label.reindex(ids)
    for panel in PANELS:
        good_fixed = fixed.unmatched_reason.fillna('').eq('')
        valid_ids = set(fixed.index[good_fixed])
        recips = np.asarray([i for i in target_domain if i in valid_ids and fixed_records.loc[i,'label'] in panel['recipient_types']],np.int64)
        targets = np.asarray([i for i in target_domain if i in valid_ids and fixed_records.loc[i,'label'] in panel['target_types']],np.int64)
        dist = nearest(fixed_records, recips, targets)
        direction = panel['direction']
        atomic_csv(out / (direction + '_frozen_distances.csv'), dist.reset_index())
        for role, pool in (('recipients',recips),('targets',targets)):
            atomic_csv(out / (direction + '_' + role + '.csv'), pd.DataFrame({'cell_id':pool}))
    atomic_json(out / 'sets.json', {k:dict(n=len(v),ids_sha256=id_hash(v),file=identity(out/(k+'.csv'))) for k,v in sets.items()})
    write_receipt(out / 'complete.json', deps, inventory(out), methods=list(EXISTING),
                  coordinate='same complete BASE_MATCH nuclei; exact ID/pixel/metadata/centroid validation',
                  fixed_population_n=103000, fixed_target_domain='F_intersect_M6', populations=populations)
    return out


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
    return table


def run_existing6(cfg, manifest, callback=None):
    out = within(cfg['run_root'], 'bt_existing6')
    deps = dependencies(cfg, 'BT_EXISTING6')
    if (out/'complete.json').exists():
        return check_receipt(out/'complete.json', deps, cfg['run_root'])
    pop = prepare_population(cfg, manifest, callback)
    nuclear = checked_ids(pd.read_csv(pop/'nuclear_reference.csv'), 'nuclear reference')
    for i,name in enumerate(EXISTING):
        dest = out/'methods'/name
        edeps = dict(deps, endpoint=name, populations=identity(pop/'complete.json'))
        if (dest/'complete.json').exists():
            check_receipt(dest/'complete.json', edeps, cfg['run_root'])
            continue
        dest.mkdir(parents=True, exist_ok=True)
        frame = checked_ids(pd.read_csv(pop/name/'cells.csv'), name)
        profiles = [('official_sc_ownQC_owncentroid',frame.index[frame.own_qc].to_numpy(),False),
                    ('fixed_commonID6_nuclear_ref',pd.read_csv(pop/'fixed_commonID6.csv').cell_id.to_numpy(),True),
                    ('fixed_commonQC6_nuclear_ref',pd.read_csv(pop/'fixed_commonQC6.csv').cell_id.to_numpy(),True)]
        for a,b in comparisons(EXISTING):
            if name in (a,b):
                profiles.append((f'fixed_pairwise_commonQC_nuclear_ref__{a}__{b}',pd.read_csv(pop/f'fixed_pairwise_commonQC__{a}__{b}.csv').cell_id.to_numpy(),True))
        for j,(profile,ids,fixed) in enumerate(profiles):
            target = dest/profile
            pdeps = dict(edeps, profile=profile, population=id_hash(ids))
            if (target/'complete.json').exists():
                check_receipt(target/'complete.json',pdeps,cfg['run_root'])
            else:
                table = write_curves(frame,target,profile,ids,nuclear if fixed else frame,pop if fixed else None)
                if not fixed:
                    old = pd.read_csv(Path(manifest['endpoints'][name]['evaluation'])/'paper_aligned_v1/neighbour_curves.csv')
                    a = table[table.direction=='T_marker_in_B'].sort_values(['marker','bin_index'])
                    b = old.sort_values(['marker','lower_um'])
                    if len(a)!=len(b) or not np.array_equal(a.n_eligible,b.n_eligible) or not np.array_equal(a.n_positive,b.n_positive) or not np.allclose(a.fraction,b.fraction,rtol=0,atol=1e-14,equal_nan=True):
                        raise ValueError('original B-recipient three panels do not reproduce: '+name)
                write_receipt(target/'complete.json',pdeps,inventory(target),n_curves=len(table),profile=profile)
            progress(callback,'BT_EXISTING6_curves',j+1,len(profiles),endpoint=name,profile=profile)
        write_receipt(dest/'complete.json',edeps,inventory(dest),endpoint=name,profiles=[p[0] for p in profiles])
        progress(callback,'BT_EXISTING6_endpoint',i+1,len(EXISTING),endpoint=name)
    # Shared-ID dual-profile F1 diagnostics are part of the six-old population gate.
    from .reports import paired_f1
    paired_f1(cfg,manifest['endpoints'],pop,out/'paired_f1',EXISTING,callback)
    return write_receipt(out/'complete.json',deps,inventory(out),methods=list(EXISTING),
                         gate='identity_data_computation_only; unfavorable scientific values do not block production')
