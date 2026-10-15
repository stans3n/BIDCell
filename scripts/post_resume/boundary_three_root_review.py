"""Post-resume independent reductions. Never imported by frozen execution.

Checks actual matrices and per-cell scores rather than trusting packaged means.
Native-mask/owner/support semantics are delegated explicitly to complete-check.
"""
from pathlib import Path
import csv
import hashlib
import json
import math
import numpy as np
import pandas as pd

BASE = Path('/home/zwan0726/work/outputs/bidcell_boundary_three_20261015')
RUN = BASE / 'boundary-three-20261015-r02'
OUT = BASE / 'analysis/root_review_20261016'
METRICS = ('positive_F1', 'negative_F1', 'positive_Precision', 'positive_Recall',
           'negative_Precision', 'negative_Recall')
NEW = ('P2_BASE', 'P2_MORPH', 'MORPH_NO_DAPI', 'MORPH_SIMPLE_FUSION',
       'MORPH_NO_GEO', 'A2_GT', 'A2_GTC', 'A2_SHUFFLED')


def read(path):
    return json.loads(Path(path).read_text())


def identity(path):
    path = Path(path)
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=h.hexdigest())


def close(a, b):
    assert math.isclose(float(a), float(b), rel_tol=1e-10, abs_tol=1e-12), (a, b)


def write(name, data):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def main():
    check = read(RUN / 'complete-check.json')
    assert check['all_enabled_applicable_complete'] and not check['failures'] and not check['deferred']
    rows = list(csv.DictReader((RUN / 'reports/endpoint_table.csv').open()))
    by = {}
    excluded = {'scclassify_Purity_F1', 'fixed_Purity_F1', 'node', 'execution_status',
                'source_table', 'source_table_sha256', 'comparison_set', 'purity_scaling_version',
                'purity_scaling_sha256', 'comparison_receipt_sha256'}
    for row in rows:
        method = row['method']
        if method in by:
            assert all(row[k] == by[method][k] for k in row if k not in excluded), method
        by.setdefault(method, row)
    assert set(by) == set(NEW) | {'BASE_MATCH', 'V2_MORPH', 'A2_NATIVE'}
    assert all(by['A2_NATIVE'][k] == by['BASE_MATCH'][k]
               for k in by['BASE_MATCH'] if k not in excluded | {'method', 'geometry_source'})
    results = {}
    populations = {}
    gene_sums = {}
    mapping_hashes = set()
    gene_hashes = set()
    for method in ('BASE_MATCH', 'V2_MORPH', *NEW):
        row = by[method]
        mp, ep = Path(row['matrix_path']), Path(row['evaluation_path'])
        counts = np.load(mp / 'counts.npy', mmap_mode='r')
        ids = pd.read_csv(mp / 'cell_mapping.csv').canonical_cell_id.to_numpy()
        assert counts.shape == (107955, 313) and np.issubdtype(counts.dtype, np.integer)
        assert len(np.unique(ids)) == 107955 and (counts >= 0).all()
        totals = counts.sum(axis=1); genes = (counts > 0).sum(axis=1)
        gene_sums[method] = counts.sum(axis=0)
        assigned = int(totals.sum())
        receipt = read(mp / 'complete.json')
        assert assigned == receipt['totals']['assigned']
        assert receipt['totals']['eligible'] == 34442716
        assert assigned + receipt['totals']['unassigned'] == 34442716
        close(100 * assigned / 34442716, row['capture_pct'])
        close(totals.mean(), row['mean_transcripts_per_cell'])
        close(genes.mean(), row['mean_genes_per_cell'])
        qc = (totals >= 20) & (totals <= 2000) & (genes >= 20)
        assert int(qc.sum()) == int(row['scclassify_qc_cells'])
        ann = pd.read_csv(ep / 'paper_aligned_v1/annotation.csv',
                          usecols=['cell_id', 'evaluation_label']).set_index('cell_id')
        assert ann.index.is_unique and set(ann.index) == set(ids[qc])
        close(100 * ann.evaluation_label.eq('unassigned').mean(), row['scclassify_unassigned_pct'])
        raw_scores = {}
        pop = pd.DataFrame({'qc': qc}, index=ids)
        for profile, file in [('scclassify', ep / 'paper_aligned_v1/markers_per_cell.csv'),
                              ('fixed', ep / 'fixed/fixed_population.csv')]:
            raw = pd.read_csv(file).set_index('cell_id')
            assert raw.index.is_unique and set(raw.index) <= set(ids)
            if profile == 'scclassify':
                assert set(raw.index) == set(ids[qc])
            else:
                assert len(raw) == 103000
            values = raw.reindex(ids)
            for metric in METRICS:
                v = values[metric].to_numpy(float)
                finite = v[np.isfinite(v)]
                close(finite.mean(), row[profile + '_' + metric])
                assert len(finite) == int(row[profile + '_' + metric + '_n'])
                raw_scores[profile + '_' + metric] = dict(mean=float(finite.mean()),
                                                        n_finite=len(finite), n_missing=len(ids)-len(finite))
                pop[profile + '_' + metric] = v
        population_path = Path(row['source_table']).parent / (method + '_all_ID_population.csv')
        packaged = pd.read_csv(population_path).set_index('cell_id').reindex(ids)
        assert len(packaged) == 107955 and packaged.index.is_unique
        assert np.array_equal(packaged.qc_pass.to_numpy(bool), qc)
        assert np.array_equal(packaged.total_transcripts.to_numpy(), totals)
        assert np.array_equal(packaged.total_genes.to_numpy(), genes)
        for profile, prefix in [('scclassify', 'sc'), ('fixed', 'fixed')]:
            for metric in METRICS:
                a = pop[profile + '_' + metric].to_numpy()
                b = packaged[prefix + '_' + metric].to_numpy()
                assert np.array_equal(np.isfinite(a), np.isfinite(b))
                assert np.allclose(a, b, rtol=1e-10, atol=1e-12, equal_nan=True)
        for column, stat in [('mean_cell_area_um2', packaged.area_um2.mean()),
                             ('median_cell_area_um2', packaged.area_um2.median()),
                             ('p95_cell_area_um2', packaged.area_um2.quantile(.95)),
                             ('disconnected_cells_8', packaged.components_8.gt(1).sum()),
                             ('disconnected_cells_4', packaged.components_4.gt(1).sum())]:
            close(row[column], stat)
        mi, gi = identity(mp / 'cell_mapping.csv'), identity(mp / 'gene_mapping.csv')
        mapping_hashes.add(mi['sha256']); gene_hashes.add(gi['sha256'])
        results[method] = dict(assigned=assigned, qc_ids=int(qc.sum()), metrics=raw_scores,
            matrix=identity(mp/'counts.npy'), matrix_receipt=identity(mp/'complete.json'),
            cell_mapping=mi, gene_mapping=gi, evaluation_receipt=identity(ep/'complete.json'),
            annotation=identity(ep/'paper_aligned_v1/annotation.csv'),
            sc_scores=identity(ep/'paper_aligned_v1/markers_per_cell.csv'),
            fixed_scores=identity(ep/'fixed/fixed_population.csv'), population=identity(population_path))
        assert results[method]['matrix']['sha256'] == row['matrix_sha256']
        assert results[method]['evaluation_receipt']['sha256'] == row['evaluation_sha256']
        populations[method] = pop
        print('validated', method, assigned, int(qc.sum()), flush=True)
        del counts, packaged
    assert len(mapping_hashes) == len(gene_hashes) == 1
    for method in ('A2_GT', 'A2_GTC', 'A2_SHUFFLED'):
        assert np.array_equal(gene_sums[method], gene_sums['BASE_MATCH'])
    comparisons = []
    pairs = [('P2_BASE','BASE_MATCH'), ('P2_MORPH','V2_MORPH'),
             ('P2_MORPH','P2_BASE'), ('V2_MORPH','BASE_MATCH'),
             ('A2_GT','BASE_MATCH'), ('A2_GTC','A2_GT'), ('A2_GTC','BASE_MATCH'),
             ('A2_GTC','A2_SHUFFLED')]
    pairs += [(m,b) for m in ('MORPH_NO_DAPI','MORPH_SIMPLE_FUSION','MORPH_NO_GEO')
              for b in ('BASE_MATCH','V2_MORPH')]
    for a,b in pairs:
        pa,pb=populations[a],populations[b]
        assert pa.index.equals(pb.index)
        common=pa.qc & pb.qc
        for metric in METRICS:
            av,bv=pa['scclassify_'+metric],pb['scclassify_'+metric]
            both=common & np.isfinite(av) & np.isfinite(bv)
            comparisons.append(dict(candidate=a,control=b,metric=metric,
                own_QC_delta=float(av.mean()-bv.mean()),common_QC_n=int(both.sum()),
                common_QC_delta=float((av[both]-bv[both]).mean()),
                QC_entry=int((pa.qc & ~pb.qc).sum()),QC_exit=int((pb.qc & ~pa.qc).sum()),
                fixed_delta=float(pa['fixed_'+metric].mean()-pb['fixed_'+metric].mean())))
    a1=pd.read_csv(RUN/'A1_CORE/per_cell_scores.csv').set_index('cell_id')
    coverage=pd.read_csv(RUN/'A1_CORE/cell_coverage.csv').set_index('cell_id')
    assert a1.index.is_unique and a1.index.equals(coverage.index) and len(a1)==107955
    paired=a1.paired
    decision=read(RUN/'A1_CORE/decision.json')
    assert int(paired.sum())==decision['paired_cells']==88102
    for col in ('core_minus_type','core_minus_shuffle'):
        close(a1.loc[paired,col].mean(),decision['primary'][col]['mean'])
    uncertainty=read(RUN/'A1_CORE/spatial_uncertainty.json'); bootstrap={}
    values=a1.loc[paired,['core','type','shuffle_mean']].to_numpy()
    xy=coverage.loc[paired,['x_um','y_um']].to_numpy()
    for side in (500,1000,2000):
        block, inverse=np.unique(np.floor(xy/side).astype(int),axis=0,return_inverse=True)
        n=np.bincount(inverse,minlength=len(block))
        sums=np.column_stack([np.bincount(inverse,weights=values[:,i],minlength=len(block)) for i in range(3)])
        rng=np.random.Generator(np.random.PCG64(20261015)); stats=[]
        for _ in range(2000):
            weights=np.bincount(rng.integers(0,len(block),size=len(block)),minlength=len(block))
            means=(weights@sums)/(weights@n)
            stats.append([means[0]-means[1],means[0]-means[2]])
        ci=np.quantile(stats,[.025,.975],axis=0)
        for i,col in enumerate(('core_minus_type','core_minus_shuffle')):
            assert np.allclose(ci[:,i],uncertainty[str(side)][col]['ci95'],rtol=1e-10,atol=1e-12)
        bootstrap[str(side)]=dict(n_blocks=len(block),core_minus_type_ci95=ci[:,0].tolist(),
                                  core_minus_shuffle_ci95=ci[:,1].tolist())
    # Latest user instruction: both fixed and scClassify F1 deserve parallel
    # scientific assessment. Frozen program decisions are retained separately.
    strata_path=Path('/home/zwan0726/work/outputs/bidcell_v2_paired_20261014/frozen_strata/fixed_cell_groups.csv')
    strata=pd.read_csv(strata_path).set_index('cell_id').loc[coverage.index]
    blocks=strata.block_1000.to_numpy(int); nb=int(blocks.max()+1)
    draws=np.random.Generator(np.random.PCG64(20261013)).integers(0,nb,size=(2000,nb))
    dual_ci=[]
    for a,b in pairs:
        for profile in ('scclassify','fixed'):
            for metric in ('positive_F1','negative_F1'):
                av=populations[a][profile+'_'+metric].to_numpy()
                bv=populations[b][profile+'_'+metric].to_numpy()
                fa,fb=np.isfinite(av),np.isfinite(bv)
                sa=np.bincount(blocks,weights=np.where(fa,av,0),minlength=nb)
                sb=np.bincount(blocks,weights=np.where(fb,bv,0),minlength=nb)
                na=np.bincount(blocks,weights=fa,minlength=nb)
                nn=np.bincount(blocks,weights=fb,minlength=nb)
                boot=sa[draws].sum(1)/na[draws].sum(1)-sb[draws].sum(1)/nn[draws].sum(1)
                assert np.isfinite(boot).all()
                ci=np.quantile(boot,[.025,.975])
                dual_ci.append(dict(candidate=a,control=b,profile=profile,metric=metric,
                    delta=float(np.nanmean(av)-np.nanmean(bv)),ci95=ci.tolist(),
                    n_candidate=int(fa.sum()),n_control=int(fb.sum()),n_blocks=nb,n_valid=2000))
    write('raw_validation.json',dict(status='passed',producer=identity(__file__),
        scope='Ten unique endpoints independently reduced from full integer matrices, raw per-cell metrics and annotations; packaged structural rows checked against full population CSVs. Mask/support semantics use official complete-check, not independently reread here.',
        complete_check=identity(RUN/'complete-check.json'),endpoints=results,
        A2_NATIVE_alias='BASE_MATCH',A2_gene_totals_exactly_conserved=True,
        A1=dict(paired_cells=int(paired.sum()),evaluable_cells=int(a1.evaluable.sum()),
                core_minus_type=float(a1.loc[paired,'core_minus_type'].mean()),
                core_minus_shuffle=float(a1.loc[paired,'core_minus_shuffle'].mean()),
                bootstrap=bootstrap,per_cell=identity(RUN/'A1_CORE/per_cell_scores.csv')),
        comparisons=comparisons,dual_profile_1000um=dual_ci,strata=identity(strata_path),
        current_interpretation='Fixed and scClassify P/N F1 assessed in parallel under user instruction 2026-10-16; old program gate remains frozen sc-primary. Spatial intervals are within-slide conditional uncertainty, not training/specimen replication.',
        historical_scope='Other historical18 endpoints not rerun; preserve prior20 table strings and its original acceptance.'))


if __name__ == '__main__':
    main()
