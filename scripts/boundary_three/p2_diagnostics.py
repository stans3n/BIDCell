"""Canonical (non-duplicated) pixel/molecule transfer and conflict diagnostics."""
from pathlib import Path
import numpy as np
import pandas as pd
import tifffile
from .common import read, atomic_json, identity, sha256
from .p2_native import STAGES


def frozen_groups(cfg):
    cells = pd.read_csv(Path(cfg['inputs']['frozen_strata'])/'fixed_cell_groups.csv')
    counts = pd.read_csv(Path(cfg['inputs']['frozen_counts'])/'fixed_count_groups.csv')
    if cells.cell_id.duplicated().any() or counts.cell_id.duplicated().any():
        raise ValueError('duplicate frozen strata IDs')
    table = cells.merge(counts, on='cell_id', how='outer', validate='one_to_one', indicator=True)
    if (table['_merge'] != 'both').any():
        raise ValueError('frozen cell/count populations differ')
    meta = np.load(Path(cfg['inputs']['historical_input_root'])/'prepared/nucleus_metadata.npz')
    ids = meta['ids'].astype(np.int64)
    covariance = meta['covariance']
    eig = np.linalg.eigvalsh(covariance)
    ecc = np.sqrt(np.maximum(0, 1-eig[:, 0]/np.maximum(eig[:, 1], 1e-8)))
    shape = pd.DataFrame(dict(cell_id=ids, nucleus_eccentricity=ecc,
                             nucleus_shape_bin=np.searchsorted([.5, .8], ecc, side='left')))
    return table.drop(columns='_merge').merge(shape, on='cell_id', validate='one_to_one').set_index('cell_id')


def transition(a, b, weight=None, domain=None):
    use = np.ones(a.shape, bool) if domain is None else domain
    changed = (a != b) & use
    masks = dict(changed=changed, cell_to_cell=changed & (a > 0) & (b > 0),
                 cell_to_background=use & (a > 0) & (b == 0),
                 background_to_cell=use & (a == 0) & (b > 0))
    if weight is None:
        return {k: int(v.sum()) for k, v in masks.items()}
    return {k: int(weight[v].sum(dtype=np.uint64)) for k, v in masks.items()}


def transfer_diagnostics(cfg, node, pred, native_pred, machine, matrix, *, origin=(0,0)):
    """Use the exact own-matrix coordinate records and unique ordered tile view.

    Type/density/RNA/shape conflict groups refer to the native postconnect owner.
    Shape bins are descriptive fixed eccentricity intervals <=.5, (.5,.8], >.8.
    Molecules are counted once at their canonical full-field pixel, including
    multiple distinct molecules occupying the same pixel.
    """
    root = Path(pred).parents[1]
    pred, native_pred = Path(pred), Path(native_pred)
    mrec = read(Path(matrix)/'complete.json')
    nuc = tifffile.imread(machine['nuclei_mask'])
    density = np.zeros(nuc.shape, np.uint32)
    for chunk in mrec['receipts']:
        if sha256(chunk['coordinate_file']) != chunk['coordinate_sha256']:
            raise ValueError('coordinate record drift during P2 diagnostics')
        arr = np.load(chunk['coordinate_file'], mmap_mode='r')
        valid = (arr['x'] >= 0) & (arr['y'] >= 0)
        np.add.at(density, (arr['y'][valid], arr['x'][valid]), 1)
    if int(density.sum(dtype=np.uint64)) + mrec['totals']['out_of_bounds'] != mrec['totals']['eligible']:
        raise ValueError('canonical molecule conservation')
    groups = frozen_groups(cfg)
    maxid = int(groups.index.max())
    types = np.full(maxid+1, -1, np.int32)
    types[groups.index.to_numpy(int)] = groups.original_type_index.to_numpy(int)
    native_connect = tifffile.imread(native_pred/'postconnect_canonical.tif')
    p2_connect = tifffile.imread(pred/'postconnect_canonical.tif')
    intervention = native_connect != p2_connect
    manifest = read(pred/'postprocess/manifest.json')
    coords = manifest['coords']
    ys = sorted({yx[0] for yx in coords})
    xs = sorted({yx[1] for yx in coords})
    ymax = {y: (ys[i+1] if i+1 < len(ys) else nuc.shape[0]) for i,y in enumerate(ys)}
    xmax = {x: (xs[i+1] if i+1 < len(xs) else nuc.shape[1]) for i,x in enumerate(xs)}
    conflicts = np.zeros(nuc.shape, bool)
    tables = []
    total_proposals = 0
    for y, x in coords:
        with np.load(pred/'postprocess'/f'tile_{y}_{x}_diagnostics.npz') as z:
            count = z['proposal_count']
            total_proposals += int(count[:ymax[y]-y, :xmax[x]-x].sum(dtype=np.uint64))
            resolution = z['conflict_resolution']
            proposals = z['conflict_proposals']
            # Full candidate composition, including nonwinning proposals.
            first = np.full(count.shape, -2, np.int32)
            unknown = np.zeros(count.shape, bool)
            hetero = np.zeros(count.shape, bool)
            for cid in np.unique(proposals[:,2]) if len(proposals) else []:
                q = proposals[proposals[:,2] == cid]
                yy, xx = q[:,0], q[:,1]
                typ = types[int(cid)]
                unknown[yy,xx] |= typ < 0
                if typ >= 0:
                    previous = first[yy,xx]
                    hetero[yy,xx] |= (previous >= 0) & (previous != typ)
                    first[yy,xx] = np.where(previous == -2, typ, previous)
            r = resolution[(resolution[:,0] < ymax[y]-y) & (resolution[:,1] < xmax[x]-x)]
            if not len(r):
                continue
            ly,lx = r[:,0],r[:,1]
            gy,gx = ly+y,lx+x
            conflicts[gy,gx] = True
            old = native_connect[gy,gx]
            if not np.isin(old, groups.index).all():
                raise ValueError('conflict native owner outside frozen population')
            selected = groups.reindex(old)
            frame = pd.DataFrame(dict(y_global=gy+origin[0], x_global=gx+origin[1],
                proposals=count[ly,lx], native_owner=old, p2_before_nuclear_restore=r[:,2],
                squared_nearest=r[:,3], runner_up=r[:,4], squared_second=r[:,5],
                exact_tie=r[:,6].astype(bool), p2_owner=r[:,7],
                nuclear=nuc[gy,gx] > 0, molecules=density[gy,gx],
                proposal_relation=np.where(unknown[ly,lx], 'Unknown', np.where(hetero[ly,lx], 'heterotypic', 'same_type')),
                native_owner_type=selected.original_type.to_numpy(),
                density_quartile=selected.density_quartile.to_numpy(),
                baseline_RNA_quartile=selected.count_quartile.to_numpy(),
                nucleus_shape_bin=selected.nucleus_shape_bin.to_numpy(),
                nucleus_eccentricity=selected.nucleus_eccentricity.to_numpy()))
            frame['changed'] = frame.native_owner != frame.p2_owner
            tables.append(frame)
    columns = ['y_global','x_global','proposals','native_owner','p2_before_nuclear_restore','squared_nearest',
               'runner_up','squared_second','exact_tie','p2_owner','nuclear','molecules','proposal_relation',
               'native_owner_type','density_quartile','baseline_RNA_quartile','nucleus_shape_bin','nucleus_eccentricity','changed']
    conflict_table = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame(columns=columns)
    conflict_table.to_csv(root/'conflict_pixels.csv', index=False)
    if not np.array_equal(p2_connect > 0, native_connect > 0) or np.any(intervention & ~((nuc == 0) & conflicts)):
        raise ValueError('canonical postconnect P2 invariants violated')
    strata = []
    for dimension in ('proposal_relation','native_owner_type','density_quartile','baseline_RNA_quartile','nucleus_shape_bin','nuclear'):
        for value, frame in conflict_table.groupby(dimension, dropna=False):
            changed = frame.changed.to_numpy(bool)
            strata.append(dict(dimension=dimension, group=str(value), conflict_pixels=len(frame),
                conflict_molecules=int(frame.molecules.sum()), changed_pixels=int(changed.sum()),
                changed_molecules=int(frame.loc[changed,'molecules'].sum()),
                exact_tie_pixels=int(frame.exact_tie.sum()),
                squared_nearest_mean=float(frame.squared_nearest.mean()),
                squared_second_mean=float(frame.squared_second.mean())))
    pd.DataFrame(strata, columns=['dimension','group','conflict_pixels','conflict_molecules','changed_pixels',
        'changed_molecules','exact_tie_pixels','squared_nearest_mean','squared_second_mean']).to_csv(root/'conflict_strata.csv',index=False)
    rows = []
    stage_group_rows = []
    for stage in STAGES:
        a = native_connect if stage == STAGES[0] else tifffile.imread(native_pred/(stage+'.tif'))
        b = p2_connect if stage == STAGES[0] else tifffile.imread(pred/(stage+'.tif'))
        changed = a != b
        retained = intervention & changed
        cancelled = intervention & ~changed
        reverse = retained & (b == native_connect) & (a == p2_connect)
        for unit, weight in [('pixels',None),('molecules',density)]:
            count = lambda mask: int(mask.sum()) if weight is None else int(weight[mask].sum(dtype=np.uint64))
            row = dict(stage=stage, unit=unit, **transition(a,b,weight),
                intervention_retained=count(retained), intervention_cancelled=count(cancelled),
                exact_owner_reversal=count(reverse), new_difference=count(~intervention & changed),
                conflict_domain_difference=count(conflicts & changed),
                outside_conflict_domain_difference=count(~conflicts & changed),
                denominator=int(nuc.size) if weight is None else mrec['totals']['eligible'])
            rows.append(row)
        aa,bb = a[changed].astype(int),b[changed].astype(int)
        ww = density[changed]
        relation = np.where((aa==0)|(bb==0),'background',np.where((types[aa]<0)|(types[bb]<0),'Unknown',np.where(types[aa]==types[bb],'same_type','heterotypic')))
        for kind in np.unique(relation):
            use = relation == kind
            stage_group_rows.append(dict(stage=stage,dimension='owner_transfer_type',group=str(kind),
                                         pixels=int(use.sum()),molecules=int(ww[use].sum(dtype=np.uint64))))
        # Retention/loss/change strata use each stage's native owner, with a
        # distinct background group for newly gained foreground.
        for dimension in ('density_quartile','count_quartile','nucleus_shape_bin'):
            value = groups[dimension].reindex(aa).fillna(-1).to_numpy(int)
            for group in np.unique(value):
                use = value == group
                stage_group_rows.append(dict(stage=stage,dimension=dimension,group=str(group),
                                             pixels=int(use.sum()),molecules=int(ww[use].sum(dtype=np.uint64))))
    pd.DataFrame(rows).to_csv(root/'postprocess_transfer.csv',index=False)
    pd.DataFrame(stage_group_rows,columns=['stage','dimension','group','pixels','molecules']).to_csv(root/'stage_transfer_strata.csv',index=False)
    summary = dict(canonical_overlap_semantics='native ordered overwrite, each pixel/molecule counted once',
        proposal_pixels=total_proposals, conflict_pixels=int(conflicts.sum()),
        conflict_molecules=int(density[conflicts].sum(dtype=np.uint64)),
        ties=int(conflict_table.exact_tie.sum()),
        postconnect_support_equal=True, postconnect_changes_confined_to_extranuclear_conflicts=True,
        input_postconnect_intervention_pixels=int(intervention.sum()),
        eligible=mrec['totals']['eligible'], out_of_bounds=mrec['totals']['out_of_bounds'],
        shape_bin_definition='complete nuclear covariance eccentricity <=.5, (.5,.8], >.8; descriptive only',
        group_owner_definition='native postconnect owner for conflicts; same-stage native owner for transfer strata',
        final_foreground_invariant_asserted=False,
        scope='mini' if origin != (0,0) else 'full_field',
        outputs={p.name:identity(p) for p in [root/'conflict_pixels.csv',root/'conflict_strata.csv',root/'postprocess_transfer.csv',root/'stage_transfer_strata.csv']})
    atomic_json(root/'transfer_diagnostics.json',summary)
    return summary
