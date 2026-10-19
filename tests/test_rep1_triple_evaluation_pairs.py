"""SCI-01: parent/BASE evidence is mandatory at each FULL_EVALUATE node."""
import os,tempfile,unittest,importlib
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
from scripts.rep1_triple_simple_boundary import contracts,bt,populations,evaluation_pairs as ep
from scripts.rep1_triple_simple_boundary.common import *
from scripts.rep1_triple_simple_boundary.inputs import new_spec
from scripts.rep1_triple_simple_boundary.coverage import validate_endpoint_products

class EvaluationPairFixtures(unittest.TestCase):
    def setUp(self):
        self.temp=None;retained=os.environ.get('REP1_TRIPLE_SCIENCE_FIXTURE_ROOT')
        if retained:self.root=Path(retained)/self._testMethodName;self.root.mkdir(parents=True,exist_ok=False)
        else:
            folder=Path('/home/zwan0726/work/tmp/rep1_triple_science');folder.mkdir(parents=True,exist_ok=True)
            self.temp=tempfile.TemporaryDirectory(dir=folder);self.root=Path(self.temp.name)
    def tearDown(self):
        if self.temp:self.temp.cleanup()
    def fixture(self,empty=False):
        root=self.root;name='P2_SIMPLE_GT';parent='P2_SIMPLE';cfg=dict(run_id='paired_fixture',run_root=str(root),source_digest='s',input_digest='i',domain_digest='d',plan_digest='p')
        ids=np.arange(1,7);genes=bt.MARKERS+['G'+str(i) for i in range(307)];labels=['B Cells','CD4 T','CD8 T','B Cells','CD8 T','B Cells']
        fixed=pd.DataFrame(dict(cell_id=ids,evaluation_label=labels,unmatched_reason=''));fp=root/'fixed.csv';atomic_csv(fp,fixed)
        frozen=root/'frozen_domain';cells=pd.DataFrame(dict(cell_id=ids,x_um=[0,10,20,30,40,200],y_um=0,**{'block_'+str(s):np.array([0,1,2,3,4,11]) for s in contracts.SCALES})).set_index('cell_id')
        atomic_csv(frozen/'cells.csv',cells,index=True);atomic_npz(frozen/'cell_groups.npz',all=ids,small=ids[:2],empty=np.empty(0,np.int64))
        rng=np.random.default_rng(20261013)
        for side in contracts.SCALES:atomic_npz(frozen/f'bootstrap_weights_{side}.npz',blocks=np.c_[np.arange(12),np.zeros(12)],W=rng.multinomial(12,np.ones(12)/12,size=2000))
        bt.freeze_targets(cells,fixed.set_index('cell_id'),frozen/'bt_targets',dependencies(cfg,'fixture_targets'))
        specs={name:new_spec(cfg,name),parent:new_spec(cfg,parent),'BASE_MATCH':dict(name='BASE_MATCH',matrix=str(root/'old_BASE/matrix'),evaluation=str(root/'old_BASE/evaluation'),sc_scores=str(root/'old_BASE/evaluation/paper_aligned_v1/markers_per_cell.csv'),fixed_scores=str(root/'old_BASE/evaluation/fixed/fixed_population.csv'))}
        for offset,(method,spec) in enumerate(specs.items()):
            matrix=Path(spec['matrix']);paper=Path(spec['evaluation'])/'paper_aligned_v1';counts=np.zeros((len(ids),313),np.int64);counts[:,:25]=1
            if method==name:
                counts[0,genes.index('CD4')]=0;counts[3,genes.index('CD4')]=0
                if empty:counts[:]=0
            if method==parent:counts[-1]=0
            q=(counts.sum(1)>=20)&((counts>0).sum(1)>=20)
            atomic_npy(matrix/'counts.npy',counts);atomic_csv(matrix/'cell_mapping.csv',pd.DataFrame(dict(matrix_row=np.arange(len(ids)),mask_label=ids,canonical_cell_id=ids)))
            atomic_csv(matrix/'gene_mapping.csv',pd.DataFrame(dict(gene_column=np.arange(313),gene=genes)));write_receipt(matrix/'complete.json',dependencies(cfg,'fixture_matrix_'+method),inventory(matrix))
            atomic_csv(paper/'qc_all_cells.csv',pd.DataFrame(dict(cell_id=ids,qc_pass=q)));atomic_csv(paper/'annotation.csv',pd.DataFrame(dict(cell_id=ids[q],evaluation_label=np.array(labels)[q])))
            atomic_csv(paper/'own_cell_centroids.csv',cells[['x_um','y_um']].reset_index())
            scores=pd.DataFrame(dict(cell_id=ids,cell_type=labels,**{m:np.full(len(ids),.2+.1*offset) for m in contracts.METRICS}))
            scores.loc[0,'positive_F1']=0.;atomic_csv(spec['sc_scores'],scores.loc[q])
            # Use the actual frozen metric spelling, including a nonfinite
            # fixed value that must not remove this ID from B/T commonQC.
            scores.loc[2,contracts.METRICS[0]]=np.nan;atomic_csv(spec['fixed_scores'],scores)
        return cfg,dict(fixed_population=str(fp),endpoints={'BASE_MATCH':specs['BASE_MATCH']}),specs

    def test_SCI01_four_true_parent_BASE_mappings(self):
        expected={'P2_MORPH_GT':'P2_MORPH','P2_SIMPLE':'MORPH_SIMPLE_FUSION','SIMPLE_GT':'MORPH_SIMPLE_FUSION','P2_SIMPLE_GT':'P2_SIMPLE'}
        for name,parent in expected.items():
            self.assertEqual(contracts.evaluation_pairs(name),[(name,parent),(name,'BASE_MATCH')])
            self.assertTrue(set(contracts.evaluation_pairs(name)).issubset(set(contracts.comparisons())))
            cfg={'run_root':str(self.root)};base={'name':'BASE_MATCH'};p={'name':parent,'marker':'true_parent'}
            with patch.object(ep,'resolve_parent',return_value=p) as resolve:
                specs=ep.specs_for(cfg,{'endpoints':{'BASE_MATCH':base}},name)
            resolve.assert_called_once_with(cfg,{'endpoints':{'BASE_MATCH':base}},name)
            self.assertIs(specs[parent],p);self.assertIs(specs['BASE_MATCH'],base);self.assertEqual(set(specs),{name,parent,'BASE_MATCH'})

    def test_SCI01_complete_actual_pairs_required_and_reused(self):
        cfg,manifest,specs=self.fixture();name='P2_SIMPLE_GT';spec=specs[name]
        with self.assertRaisesRegex(ValueError,'parent/BASE paired evidence'):validate_endpoint_products(spec,manifest=manifest,cfg=cfg)
        with patch.object(ep,'specs_for',return_value=specs),patch('scripts.rep1_triple_simple_boundary.science.ensure_reused',side_effect=AssertionError('old8 completion called')),patch('scripts.rep1_triple_simple_boundary.evaluation.run_r',side_effect=AssertionError('reclassification called')):
            rec=ep.run(cfg,manifest,name);validated=ep.validate(cfg,manifest,name)
        self.assertEqual(rec['pairs'],[contracts.comparison_id(*x) for x in contracts.evaluation_pairs(name)])
        self.assertEqual(len(validated['score_records']),24)
        out=Path(spec['evaluation'])/'paired';key=contracts.comparison_id(name,'P2_SIMPLE')
        q=pd.read_csv(out/'bt'/key/'population.csv').cell_id.to_numpy();finite=np.load(out/'populations'/key/(contracts.METRICS[0]+'.npz'))['ids']
        self.assertIn(3,q);self.assertNotIn(3,finite);self.assertIn(1,finite)
        ci=pd.read_csv(out/'bt'/key/'paired_0_40um.csv');self.assertTrue(ci.candidate.eq(name).all());self.assertTrue(ci.control.eq('P2_SIMPLE').all());self.assertTrue((ci[ci.marker=='CD4'].point<0).all())
        # The same eight existing scientific jobs are linked into04/13.
        pair_list=contracts.evaluation_pairs(name);compare=importlib.import_module('scripts.rep1_triple_simple_boundary.comparisons')
        target=self.root/'initial_consumer';tables=populations.prepare(cfg,manifest,specs,target/'populations',reuse_from=ep.reuse_sources(cfg,specs,'populations'),pair_list=pair_list)
        compare.run(cfg,manifest,specs,tables,target/'populations',target/'comparisons',reuse_from=ep.reuse_sources(cfg,specs,'comparisons'),pair_list=pair_list)
        with patch.object(bt,'write_curves',side_effect=AssertionError('reused BT recomputed')):bt.paired(cfg,manifest,specs,target/'bt',reuse_from=ep.reuse_sources(cfg,specs,'bt'),pair_list=pair_list)
        for part,relative in [('populations',contracts.METRICS[0]+'.npz'),('comparisons','stratified_intervals.csv.gz'),('bt','paired_0_40um.csv')]:
            producer=out/part/key;consumer=target/part/key
            self.assertEqual(os.stat(producer/relative).st_ino,os.stat(consumer/relative).st_ino);self.assertEqual(read(consumer/'complete.json')['reused_from'],identity(producer/'complete.json'))
        final=self.root/'all12_consumer'
        sources={part:ep.reuse_sources(cfg,specs,part,initial=target) for part in ('populations','comparisons','bt')}
        for part in sources:self.assertEqual(sources[part][key],target/part/key)
        tables=populations.prepare(cfg,manifest,specs,final/'populations',reuse_from=sources['populations'],pair_list=pair_list)
        with patch.object(compare,'_intervals_for_groups',side_effect=AssertionError('reused interval recomputed')):
            compare.run(cfg,manifest,specs,tables,final/'populations',final/'comparisons',reuse_from=sources['comparisons'],pair_list=pair_list)
        with patch.object(bt,'write_curves',side_effect=AssertionError('reused BT recomputed')):bt.paired(cfg,manifest,specs,final/'bt',reuse_from=sources['bt'],pair_list=pair_list)
        for part,relative in [('populations',contracts.METRICS[0]+'.npz'),('comparisons','stratified_intervals.csv.gz'),('bt','paired_0_40um.csv')]:
            self.assertEqual(os.stat(out/part/key/relative).st_ino,os.stat(final/part/key/relative).st_ino)
            self.assertEqual(read(final/part/key/'complete.json')['reused_from'],identity(target/part/key/'complete.json'))
            check_receipt(final/part/'complete.json')
        # Measure both apparent and physically allocated incremental bytes before
        # the intentional corruption below. The fixture does not bound full data.
        seen=set();storage=[]
        for label,path in [('evaluation_pair_producer',out),('initial_consumer',target),('all12_consumer',final)]:
            records=[]
            for p in sorted(path.rglob('*')):
                if p.is_file():
                    st=p.stat();inode=(st.st_dev,st.st_ino);fresh=inode not in seen;seen.add(inode)
                    records.append(dict(path=str(p),bytes=st.st_size,allocated_bytes=st.st_blocks*512,new_inode=fresh))
            storage.append(dict(role=label,files=len(records),apparent_bytes=sum(r['bytes'] for r in records),incremental_unique_bytes=sum(r['bytes'] for r in records if r['new_inode']),incremental_allocated_bytes=sum(r['allocated_bytes'] for r in records if r['new_inode']),files_detail=records))
        atomic_json(self.root/'paired_reuse_storage.json',dict(status='measured_synthetic_only',groups=3,IDs=6,pairs=2,storage=storage,full_job_count=69,scientific_pair_count=26))
        # Reject a fresh hash around a truncated table, not merely stale hash.
        bad=out/'comparisons'/key/'stratified_intervals.csv.gz';frame=pd.read_csv(bad);compare.atomic_csv_gzip(bad,frame.iloc[:-1])
        from scripts.rep1_triple_simple_boundary.coverage import comparison_rows
        hashes={(r['metric'],r['profile']):r['ids_sha256'] for r in validated['score_records'] if r['control']=='P2_SIMPLE'}
        with self.assertRaisesRegex(ValueError,'actual unique key'):comparison_rows(pd.read_csv(bad),['all','small','empty'],{name:1,'P2_SIMPLE':-1},'pairwise',hashes)
        # Simulate an erroneous producer signing the truncated data afresh.
        # Full-evaluation validation must still fail on semantic coverage.
        for receipt in (bad.parent/'complete.json',out/'comparisons/complete.json',out/'complete.json'):
            original=read(receipt);atomic_json(self.root/('before_corruption_'+str(len(list(self.root.glob('before_corruption_*'))))+'.json'),original)
            original['outputs']={k:identity(v['path']) for k,v in original['outputs'].items()};atomic_json(receipt,original)
        with patch.object(ep,'specs_for',return_value=specs),self.assertRaisesRegex(ValueError,'actual unique key'):
            validate_endpoint_products(spec,manifest=manifest,cfg=cfg)
        self.assertFalse((self.root/'reused').exists());self.assertFalse((self.root/'analysis').exists())

    def test_SCI01_empty_common_population_is_complete_NA(self):
        cfg,manifest,specs=self.fixture(empty=True)
        with patch.object(ep,'specs_for',return_value=specs):
            ep.run(cfg,manifest,'P2_SIMPLE_GT');result=ep.validate(cfg,manifest,'P2_SIMPLE_GT')
        self.assertTrue(all(r['n_finite']==0 and r['NA_reason'] for r in result['score_records']))
        out=Path(specs['P2_SIMPLE_GT']['evaluation'])/'paired'
        for key in result['pairs']:
            ci=pd.read_csv(out/'bt'/key/'paired_0_40um.csv');self.assertEqual(len(ci),18);self.assertTrue(ci.point.isna().all());self.assertTrue(ci.NA_reason.eq('NA_zero_denominator').all())
            for name in read(out/'bt'/key/'complete.json')['methods']:
                curves=pd.read_csv(out/'bt'/key/name/'full_curves.csv');self.assertEqual(len(curves),42);self.assertTrue(curves.n_eligible.eq(0).all());self.assertTrue(curves.fraction.isna().all())

if __name__=='__main__':unittest.main()
