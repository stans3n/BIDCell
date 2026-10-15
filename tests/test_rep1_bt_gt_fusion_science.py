"""Small deterministic scientific fixtures. Never starts a formal endpoint."""
import json
from pathlib import Path
import shutil
import tempfile
import unittest
import numpy as np
import pandas as pd
import tifffile
from scipy import sparse
from scripts.rep1_bt_gt_fusion import EXISTING, NEW, comparisons
from scripts.rep1_bt_gt_fusion.common import (atomic_json, atomic_npy, identity, dependencies,
                                             check_receipt, digest, require_manual_resume, id_hash)
from scripts.rep1_bt_gt_fusion.bt_curves import PANELS, MARKERS, curve, nearest, canonical_bt, mask_moments
from scripts.rep1_bt_gt_fusion.gt_refine import build_domain, score_domain, apply_gt, validate_topology
from scripts.rep1_bt_gt_fusion.matrix_stage import build, validate_full_oracle
from scripts.rep1_bt_gt_fusion.publication import (purity, preserve_old_raw, mask_content, read_strings,
    new_table_row, require_publication_masks, _copy_frozen_mask, recover_historical_mask)
from scripts.rep1_bt_gt_fusion.inputs import source_dependencies, TABLE
from scripts.boundary_three.core_refine import NuclearSupport, interface_band, fixed_domain, score_proposals, topology_oracle
from scripts.crgf.matrix import grid, validate_production


class ScientificFixtures(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='rep1_bt_gt_science_fixture_')
        self.root=Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_bt_boundaries_and_empty_rows(self):
        ids=np.arange(1,9)
        records=pd.DataFrame(dict(cell_id=np.r_[ids,100],x_um=np.r_[[0,10,20,30,40,50,100,101],0],y_um=np.zeros(9))).set_index('cell_id')
        values=pd.DataFrame(1,index=records.index,columns=MARKERS)
        values.loc[1,'CD4']=0
        table,dist=curve(records,ids,[100],values,PANELS[0])
        part=table[table.marker.eq('CD4')]
        self.assertEqual(part.n_eligible.tolist(),[2,1,1,1,1,1,1])
        self.assertEqual(part.n_positive.tolist(),[1,1,1,1,1,1,1])
        self.assertEqual(dist.bin_index.tolist(),[0,0,1,2,3,4,5,6])
        short,_=curve(records,ids[:7],[100],values,PANELS[0])
        self.assertTrue(short[short.bin_index.eq(6)].fraction.isna().all())
        self.assertTrue(short[short.bin_index.eq(6)].upper_um.eq(100).all())
        no_source,_=curve(records,[],[100],values,PANELS[0]);self.assertEqual(len(no_source),21)
        self.assertTrue(no_source.status.eq('empty_recipient_pool').all())
        no_target,_=curve(records,ids,[],values,PANELS[0]);self.assertEqual(len(no_target),21)
        self.assertTrue(no_target.fraction.isna().all());self.assertTrue(no_target.status.eq('empty_target_pool').all())
        with self.assertRaises(ValueError):curve(records,ids,[100],values.drop(columns='CD8B'),PANELS[0])
        with self.assertRaises(ValueError):curve(records,[1,1],[100],values,PANELS[0])
        with self.assertRaises(ValueError):curve(records,[999],[100],values,PANELS[0])

    def test_merged_reverse_hand_calculation_and_ties(self):
        rec=pd.DataFrame(dict(cell_id=[1,2,3,10,11],x_um=[0,0,10,-10,10],y_um=[0]*5)).set_index('cell_id')
        val=pd.DataFrame(0,index=rec.index,columns=MARKERS);val.loc[2,'CD79A']=1;val.loc[3,'CD79B']=3
        labels=pd.Series(['B Cells','CD4 T','CD8 T','B','CD4Tconv/Treg'],index=rec.index)
        rids=labels.index[labels.isin(PANELS[1]['recipient_types'])]
        tab,_=curve(rec,rids,[1],val,PANELS[1])
        self.assertEqual(tab[(tab.marker=='CD79A')&(tab.bin_index==0)].n_eligible.item(),2)
        self.assertEqual(tab[(tab.marker=='CD79A')&(tab.bin_index==0)].fraction.item(),.5)
        self.assertTrue(canonical_bt(['B','CD4Tconv/Treg','CD8T/CD8Tex']).isna().all())
        self.assertEqual(canonical_bt(['B','CD4Tconv/Treg','CD8T/CD8Tex'],'engineering').tolist(),['B Cells','CD4 T','CD8 T'])
        self.assertEqual(nearest(rec,[1],[11,10]).nearest_target_id.iloc[0],10)
        # Removing a QC recipient must not move another fixed nearest target.
        frozen=nearest(rec,[2,3],[1]);copy=rec.copy();copy.loc[1,'x_um']=1000
        fixed,_=curve(copy,[3],[1],val,PANELS[1],frozen)
        self.assertEqual(fixed[fixed.marker.eq('CD79A')&fixed.bin_index.eq(0)].n_eligible.item(),1)

    def _tiny_gt(self,boundary=10):
        shape=(18,22)
        nuclei=np.zeros(shape,np.uint32);nuclei[8:10,5:7]=1;nuclei[8:10,15:17]=2
        parent=np.ones(shape,np.uint32);parent[:,boundary:]=2
        parent[nuclei>0]=nuclei[nuclei>0]
        ids=np.array([1,2]);types=np.array([0,1])
        sup=NuclearSupport(nuclei,ids,types,['A','B'])
        records=[]
        for y in (6,7,8,9,10,11):
            for x in range(6,17):
                records.extend([(len(records)+1,x,y,(x+y)%3),(len(records)+2,x,y,1)])
        arr=np.array(records,dtype=[('transcript_id','<u8'),('x','<i4'),('y','<i4'),('gene','<i4')])
        path=self.root/f'coords_{boundary}.npy';atomic_npy(path,arr)
        return dict(parent='tiny_'+str(boundary),baseline=parent,nuclei=nuclei,ids=ids,types=types,support=sup,
                    genes=['A','B','C'],gene_indices=np.array([0,1]),prior=np.array([[.99,.01],[.01,.99]]),
                    eligible=np.ones(2,bool),donor=np.array([1,0]),bins=np.zeros(2,int),expected_molecules=len(arr),
                    coordinates=[dict(**identity(path),rows=len(arr),row_start=0,row_stop=len(arr))])

    def test_gt_parent_domains_exact_algorithm_and_resume(self):
        data=self._tiny_gt(10);other=self._tiny_gt(12)
        events=[]
        domain,expr,info=build_domain(data,self.root/'domain',{'fixture':'one'},events.append)
        second,_,_=build_domain(other,self.root/'domain2',{'fixture':'two'})
        self.assertNotEqual(domain['pixels'].tolist(),second['pixels'].tolist())
        pixels=np.load(self.root/'domain/query_pixels.npy')
        # Independently form expression and invoke the unmodified old definition.
        arr=np.load(data['coordinates'][0]['path']);flat=arr['y'].astype(np.int64)*data['nuclei'].shape[1]+arr['x']
        mat=np.zeros((len(pixels),2),np.int64)
        for p,g in zip(flat,arr['gene']):
            if p in pixels and g<2:mat[np.flatnonzero(pixels==p)[0],g]+=1
        off,rows,_=data['support'].candidate_csr(pixels)
        pos,eo,er,reasons=fixed_domain(data['baseline'],data['nuclei'],pixels,off,rows,data['ids'],data['eligible'],data['donor'],sparse.csr_matrix(mat))
        np.testing.assert_array_equal(domain['pixels'],pixels[pos]);np.testing.assert_array_equal(domain['offsets'],eo)
        np.testing.assert_array_equal(domain['candidate_rows'],er);self.assertEqual(info['abstentions'],reasons)
        np.testing.assert_array_equal(expr.toarray(),mat[pos])
        p1,s1=score_domain(data,domain,expr,self.root/'scores',{'fixture':'score'})
        self.assertGreater(len(p1),1)
        old,_,_=score_proposals(data['baseline'],domain['pixels'],domain['offsets'],domain['candidate_rows'],domain['geometry'],expr,data['ids'],data['prior'])
        np.testing.assert_array_equal(p1,old)
        full,summary=apply_gt(data['baseline'],data['nuclei'],p1,self.root/'full',{'fixture':'apply'},oracle=True)
        self.assertGreater(summary['accepted'],0)
        interrupted,_=apply_gt(data['baseline'],data['nuclei'],p1,self.root/'resume',{'fixture':'apply'},stop_after=1,oracle=True)
        self.assertIsNone(interrupted)
        recovered,summary2=apply_gt(data['baseline'],data['nuclei'],p1,self.root/'resume',{'fixture':'apply'},oracle=True)
        np.testing.assert_array_equal(full,recovered);self.assertEqual(summary,summary2)
        validate_topology(data['baseline'],full,data['nuclei'],data['ids'],p1,self.root/'topology',{'fixture':'topology'})
        self.assertTrue(events);self.assertTrue(all('cursor' in e for e in events))
        # Committed domain receipt detects corruption; it does not silently regenerate.
        (self.root/'domain/domain.json').write_text('{}')
        with self.assertRaises(ValueError):build_domain(data,self.root/'domain',{'fixture':'one'})

    def test_gt_eligibility_margin_tie_and_rejections(self):
        data=self._tiny_gt();data['eligible'][1]=False
        dom,expr,info=build_domain(data,self.root/'ineligible',{'fixture':'ineligible'})
        self.assertEqual(len(dom['pixels']),0)
        self.assertGreater(info['abstentions'].get('candidate_not_A1_evaluable',0),0)
        baseline=np.array([[1,1,2],[1,1,2],[1,1,2]],np.uint32);nuc=np.zeros_like(baseline);nuc[1,0]=1;nuc[1,2]=2
        pixels=np.array([4]);off=np.array([0,2]);rows=np.array([0,1]);counts=sparse.csr_matrix([[1,0]])
        props,_,_=score_proposals(baseline,pixels,off,rows,np.zeros(2),counts,np.array([1,2]),np.array([[.5,.5],[.5,.5]]))
        self.assertEqual(len(props),0)
        props,_,_=score_proposals(baseline,pixels,off,rows,np.array([0,.1]),counts,np.array([1,2]),np.array([[.5,.5],[.5,.5]]))
        self.assertEqual(len(props),0)
        self.assertEqual(topology_oracle(baseline,nuc,1,0,1,2),(False,'nuclear_pixel'))
        # Removing the centre bridge splits the donor.
        bridge=np.array([[1,0,1],[1,1,1],[0,2,2]],np.uint32);bn=np.zeros_like(bridge);bn[0,0]=1;bn[2,2]=2
        self.assertEqual(topology_oracle(bridge,bn,1,1,1,2),(False,'donor_components_increased'))

    def test_matrix_full_oracle_and_recovery(self):
        mask=np.array([[1,1,1,2,2,2,0],[1,1,1,2,2,2,0],[1,1,1,2,2,2,0],[0,0,0,2,2,2,0],[0,0,0,0,0,0,0]],np.uint32)
        tifffile.imwrite(self.root/'final.tif',mask)
        (self.root/'genes.txt').write_text('A\nB\nC\n');(self.root/'baseline.yaml').write_text('fixture: true\n')
        transcript=pd.DataFrame(dict(transcript_id=np.arange(1,13),x_location=[0,.10625,.31875,.53125,1,2.5,3,4.1,5,6.5,7.5,-.4],
            y_location=[0,0,0,0,1,1,2,3,3,4,4,0],feature_name=['A','B','C']*4,qv=[30]*12))
        transcript.to_csv(self.root/'transcripts.csv',index=False)
        cfg=dict(final_mask=str(self.root/'final.tif'),genes=str(self.root/'genes.txt'),baseline_config=str(self.root/'baseline.yaml'),
                 transcripts=str(self.root/'transcripts.csv'),scale_x=.2125,scale_y=.2125,pixel_um=1.,chunksize=5)
        coords=[];cursor=0
        for i,frame in enumerate(pd.read_csv(cfg['transcripts'],chunksize=5)):
            x,y,_=grid(frame,mask.shape,.2125,.2125)
            arr=np.empty(len(frame),dtype=[('transcript_id','<u8'),('x','<i4'),('y','<i4'),('gene','<i4')])
            arr['transcript_id']=frame.transcript_id;arr['x']=x;arr['y']=y;arr['gene']=pd.Index(['A','B','C']).get_indexer(frame.feature_name)
            p=self.root/f'coordinate_{i}.npy';atomic_npy(p,arr)
            coords.append(dict(**identity(p),rows=len(frame),row_start=cursor,row_stop=cursor+len(frame)));cursor+=len(frame)
        out=self.root/'matrix';events=[]
        class Interrupt(Exception):pass
        def stop(event):
            events.append(event)
            if event['stage']=='matrix_own_owner_and_counts' and event['cursor']==1:raise Interrupt()
        with self.assertRaises(Interrupt):build(cfg,coords,out,{'fixture':'matrix'},stop,expected_molecules=12)
        result=build(cfg,coords,out,{'fixture':'matrix'},events.append,expected_molecules=12)
        self.assertEqual(result['totals']['eligible'],12)
        self.assertEqual(result['validation_kind'],'independent_full_native_resize_lookup_and_conservation')
        # Run the unmodified legacy full 2D oracle on the same tiny generated artifacts.
        legacy=self.root/'legacy';legacy.mkdir()
        for name in ['counts.npy','expr_mat.csv','cell_mapping.csv','gene_mapping.csv',*[r['owner_file'] for r in result['receipts']]]:shutil.copyfile(out/name,legacy/name)
        old=dict(result);old['dependencies']=result['matrix_dependencies']
        validate_production(cfg,legacy,old)
        self.assertEqual(json.loads((legacy/'validation.json').read_text())['rows'],12)
        self.assertTrue(any(e['stage']=='matrix_independent_full_oracle' for e in events))
        with (out/'owners_0000.npy').open('ab') as f:f.write(b'corruption')
        with self.assertRaises(ValueError):build(cfg,coords,out,{'fixture':'matrix'},expected_molecules=12)

    def test_gate_purity_content_and_source_inventory(self):
        cfg=dict(run_id='isolated_fixture',run_root=str(self.root),source_digest='fixture_source',input_digest='fixture_input',plan_sha256='fixture_plan')
        with self.assertRaises(PermissionError):require_manual_resume(cfg,None)
        grant=self.root/'control/manual_resume.json';atomic_json(grant,dict(authorization='explicit_user_resume',dependencies={'run_id':'old_run'}))
        with self.assertRaises(PermissionError):require_manual_resume(cfg,grant)
        old=[dict(method='A',scclassify_positive_F1='0.1000',scclassify_negative_F1='0.0400',fixed_positive_F1='0.3000',fixed_negative_F1='0.0500',raw='01.2000')]
        rows=old+[dict(method='B',scclassify_positive_F1='0.2',scclassify_negative_F1='0.03',fixed_positive_F1='0.4',fixed_negative_F1='0.04',raw='1.2')]
        self.assertEqual(purity(rows)['profiles']['scclassify']['methods']['B'],1.0)
        preserve_old_raw(old,rows,list(old[0]))
        changed=[dict(rows[0],raw='1.2'),rows[1]]
        with self.assertRaises(ValueError):preserve_old_raw(old,changed,list(old[0]))
        array=np.array([[0,1],[2,3]],np.uint32);a=self.root/'a.tif';b=self.root/'b.tif'
        tifffile.imwrite(a,array,description='first');tifffile.imwrite(b,array,description='different metadata')
        self.assertEqual(mask_content(a),mask_content(b));self.assertNotEqual(identity(a)['sha256'],identity(b)['sha256'])
        sources=source_dependencies();self.assertTrue(sources)
        self.assertTrue(all(Path(x['path']).is_file() for x in sources))
        self.assertIn('scripts/boundary_three/core_refine.py',[x['snapshot_relative'] for x in sources])
        self.assertIn('external_sources/publish_spatialDiversity_metrics.R',[x['snapshot_relative'] for x in sources])
        pairs=comparisons();self.assertEqual(len(pairs),len(set(pairs)));self.assertFalse(any(a==b for a,b in pairs))
        # Real source table carries a UTF-8 BOM; data strings remain literal.
        columns,real_rows=read_strings(TABLE)
        self.assertEqual(columns[0],'method');self.assertEqual(len(real_rows),28)
        self.assertEqual(len({r['method'] for r in real_rows}),28)

    def test_four_way_common_finite_F1(self):
        from scripts.rep1_bt_gt_fusion.reports import paired_f1, METRICS
        specs={}
        for name in ('B0','BASE_MATCH'):
            path=self.root/name;path.mkdir()
            sc=pd.DataFrame({'cell_id':[1,2,3,4]});fx=sc.copy()
            for metric in METRICS:
                sc[metric]=[.2,.3,.4,.5] if name=='B0' else [.3,.4,.5,.6]
                fx[metric]=[.4,.5,.6,.7] if name=='B0' else [.6,.7,.8,.9]
            if name=='B0':sc.loc[1,'positive_F1']=np.nan
            else:fx.loc[2,'positive_F1']=np.nan
            sc.to_csv(path/'sc.csv',index=False);fx.to_csv(path/'fixed.csv',index=False)
            specs[name]=dict(sc_scores=str(path/'sc.csv'),fixed_scores=str(path/'fixed.csv'))
        cfg=dict(run_id='fixture',run_root=str(self.root),source_digest='s',input_digest='i',plan_sha256='p')
        paired_f1(cfg,specs,self.root,self.root/'paired',('B0','BASE_MATCH'))
        ids=pd.read_csv(self.root/'paired/BASE_MATCH__B0/positive_F1_four_way_finite_ids.csv').cell_id.tolist()
        self.assertEqual(ids,[1,4])
        summary=pd.read_csv(self.root/'paired/paired_summary.csv')
        q=summary[(summary.control=='B0')&(summary.metric=='positive_F1')]
        self.assertTrue(q.n_finite.eq(2).all());self.assertEqual(q.ids_sha256.nunique(),1)
        np.testing.assert_allclose(q.sort_values('profile').mean_delta,[.2,.1])

    def test_historical_missing_mask_exact_recovery(self):
        cfg=dict(run_id='fixture',run_root=str(self.root/'run'),source_digest='s',input_digest='i',plan_sha256='p')
        original=self.root/'original/final.tif';original.parent.mkdir();original.write_bytes(b'frozen mask bytes')
        expected=identity(original);original.unlink()
        mr=self.root/'original/matrix_complete.json';atomic_json(mr,dict(dependencies={'mask':expected}))
        record=dict(mask=expected,matrix_receipt=identity(mr),available=False,recovery_relative='recovered_historical_masks/C/final.tif')
        manifest=dict(historical_masks={'C':record})
        with self.assertRaises(ValueError):require_publication_masks(cfg,manifest)
        archive=self.root/'archive.tif';archive.write_bytes(b'wrong bytes')
        with self.assertRaises(ValueError):_copy_frozen_mask(cfg,'C',record,archive)
        self.assertFalse((Path(cfg['run_root'])/record['recovery_relative']).exists())
        archive.write_bytes(b'frozen mask bytes')
        recovered=_copy_frozen_mask(cfg,'C',record,archive)
        resolved=require_publication_masks(cfg,manifest)
        self.assertEqual(resolved['C']['sha256'],expected['sha256'])
        self.assertFalse(original.exists());self.assertTrue(recovered['exact_original_artifact'])
        self.assertEqual(recovered['dependencies']['original_matrix_receipt'],identity(mr))
        with self.assertRaises(PermissionError):recover_historical_mask(cfg,None,'C',archive)
        Path(resolved['C']['path']).write_bytes(b'corrupt after recovery')
        with self.assertRaises(ValueError):require_publication_masks(cfg,manifest)

    def test_new_publication_row_matches_full_real_schema(self):
        from scripts.rep1_bt_gt_fusion.evaluation import new_spec
        from scripts.rep1_bt_gt_fusion.reports import METRICS,endpoint_comparison_tables
        from scripts.rep1_bt_gt_fusion.publication import SPATIAL_COLUMNS
        from tests.rep1_bt_gt_spatial_fixture import fixture_destination,spatial_fixture
        actual_r,bins,correlations=spatial_fixture(fixture_destination(self.root,'normal_spatial'),'normal',compare_author=True)
        self.assertTrue(correlations.status.eq('computed').all())
        self.assertTrue(correlations.pearson.notna().all())
        columns,_=read_strings(TABLE)
        cfg=dict(run_id='fixture',run_root=str(self.root),source_digest='s',input_digest='i',plan_sha256='p')
        structure={key:1 for key in ('mean_transcripts_per_cell','mean_genes_per_cell','mean_cell_area_um2',
            'median_cell_area_um2','p95_cell_area_um2','disconnected_cells_4','disconnected_cells_8','nuclear_preservation_fraction')}
        metrics={profile+'_'+metric:dict(mean=.1,n_finite=7,n_population=8) for profile in ('scclassify','fixed') for metric in METRICS}
        checkpoint=self.root/'parent_checkpoint';checkpoint.write_bytes(b'parent checkpoint')
        machine=self.root/'parent_machine.json';atomic_json(machine,dict(checkpoint=str(checkpoint)))
        manifest=dict(endpoints={p:{'machine':str(machine)} for p in ('P2_BASE','V2_MORPH')})
        actual_rows={}
        for name in NEW:
            spec=new_spec(cfg,name);matrix=Path(spec['matrix']);ev=Path(spec['evaluation'])
            matrix.mkdir(parents=True);(ev/'paper_aligned_v1').mkdir(parents=True)
            for rel in ('counts.npy','cell_mapping.csv','gene_mapping.csv'):(matrix/rel).write_bytes(b'fixture')
            atomic_json(matrix/'complete.json',dict(totals={'assigned':21654424 if name=='P2_BASE_GT' else 21449617}))
            atomic_json(ev/'complete.json',{'fixture':True});(ev/'paper_aligned_v1/annotation.csv').write_text('cell_id\n1\n')
            atomic_json(ev/'summary.json',dict(structure=structure,metrics=metrics,n_after_qc=8,qc_retention=.8,unassigned_fraction=.2,wall_seconds=.1))
            pd.DataFrame(dict(definition=['mean_logcounts','detection'],pearson=[.6,.7])).to_csv(ev/'paper_aligned_v1/type_correlations.csv',index=False)
            target=ev/'paper_aligned_v1/spatial_diversity_correlations.csv'
            shutil.copyfile(actual_r/'paper_aligned_v1/spatial_diversity_correlations.csv',target)
            row=new_table_row(cfg,manifest,name,columns)
            self.assertEqual(set(row),set(columns));self.assertEqual(row['method'],name)
            self.assertEqual(row['fixed_positive_F1_n'],'7')
            for key in ['matrix_receipt_sha256','evaluation_receipt_sha256','annotation_sha256','checkpoint_sha256']:
                self.assertEqual(len(row[key]),64)
            for record in correlations.to_dict('records'):
                self.assertAlmostEqual(float(row[SPATIAL_COLUMNS[record['metric']]]),record['pearson'])
            self.assertTrue(all(isinstance(v,str) for v in row.values()))
            actual_rows[name]=row
        endpoint_comparison_tables(cfg,manifest,self.root)
        contrasts=pd.read_csv(self.root/'structure_reference_spatial_comparisons.csv')
        direct=contrasts[(contrasts.candidate==NEW[0])&(contrasts.control==NEW[1])&contrasts.metric.isin(SPATIAL_COLUMNS.values())]
        self.assertEqual(len(direct),6);self.assertTrue(direct.candidate_minus_control.eq(0).all())
        # A typo, duplicate or omitted actual producer metric must still fail.
        bad=correlations.copy();bad.loc[0,'metric']='unregistered_metric'
        for invalid in (bad,correlations.iloc[:-1],pd.concat([correlations,correlations.iloc[:1]],ignore_index=True)):
            invalid.to_csv(target,index=False)
            with self.assertRaises(ValueError):new_table_row(cfg,manifest,NEW[-1],columns)
        shutil.copyfile(actual_r/'paper_aligned_v1/spatial_diversity_correlations.csv',target)
        shutil.copyfile(self.root/'structure_reference_spatial_comparisons.csv',actual_r.parent/'consumer_comparisons.csv')
        atomic_json(actual_r.parent/'consumer_receipt.json',dict(status='passed',synthetic=True,
            new_table_rows=actual_rows,real_table_columns=columns,explicit_spatial_mapping=SPATIAL_COLUMNS,
            original_R_spatial_output=identity(actual_r/'paper_aligned_v1/spatial_diversity_correlations.csv'),
            comparisons=identity(actual_r.parent/'consumer_comparisons.csv'),
            unknown_duplicate_missing_metric_rejected=True,no_publication_run=True))

    def test_real_R_spatial_degeneracies_and_common_family_population(self):
        from tests.rep1_bt_gt_spatial_fixture import fixture_destination,spatial_fixture
        for case in ('single_known','all_unassigned','constant_entropy','one_cell_bins','one_bin','mixed_nonfinite'):
            with self.subTest(case=case):
                out,bins,cor=spatial_fixture(fixture_destination(self.root,case),case,compare_author=(case=='mixed_nonfinite'))
                self.assertEqual(len(cor),6);self.assertTrue(cor.bin_width.eq(500).all())
                if case in ('single_known','all_unassigned','constant_entropy'):
                    self.assertTrue(cor.pearson.isna().all());self.assertTrue(cor.reason.eq('constant_entropy').all())
                    self.assertTrue(cor.n_common_finite_bins.eq(6).all())
                    if case!='constant_entropy':self.assertTrue(bins.entropy.eq(0).all())
                    self.assertTrue(np.isfinite(bins.filter(regex='^(cv|sd)_(total_transciprts|total_genes|cell_area)$')).all().all())
                elif case in ('one_cell_bins','one_bin'):
                    self.assertTrue(cor.pearson.isna().all());self.assertTrue(cor.reason.eq('fewer_than_two_common_finite_bins').all())
                    self.assertTrue(cor.n_common_finite_bins.eq(0 if case=='one_cell_bins' else 1).all())
                    if case=='one_cell_bins':self.assertTrue(bins.filter(regex='^(cv|sd)_(total_transciprts|total_genes|cell_area)$').isna().all().all())
                else:
                    cv=cor[cor.family.eq('cv')];sd=cor[cor.family.eq('sd')]
                    self.assertTrue(cv.n_common_finite_bins.eq(5).all());self.assertTrue(sd.n_common_finite_bins.eq(6).all())
                    self.assertEqual(cv.set_index('metric').loc['cv_total_transciprts','n_metric_finite_bins'],6)
                    self.assertEqual(bins.cv_cell_area_status.eq('zero_mean_log10_metric').sum(),1)
                    self.assertTrue(cor.status.eq('computed').all())
        dest=fixture_destination(self.root,'invalid_spatial_input')
        with self.assertRaises(RuntimeError):spatial_fixture(dest,'invalid_input')
        self.assertFalse((dest/'output/spatial_execution_status.txt').exists())

    def test_formal_stage_requires_execution_validator_and_locks(self):
        from scripts.rep1_bt_gt_fusion.science import run_stage
        from scripts.rep1_bt_gt_fusion import NODE_ORDER
        cfg=dict(run_id='fixture',run_root=str(self.root),source_digest='s',input_digest='i',plan_sha256='p',
                 input_manifest=str(self.root/'inputs.json'),source_snapshot=str(self.root/'snapshot'),node_order=list(NODE_ORDER))
        with self.assertRaises(PermissionError):run_stage(cfg,'BT_EXISTING6')
        with (self.root/'lock').open('w') as lock:
            with self.assertRaises(PermissionError):run_stage(cfg,'BT_EXISTING6',lock_fds=(lock.fileno(),))

    def test_paired_spatial_bootstrap_resume(self):
        from scripts.rep1_bt_gt_fusion.reports import block_interval
        delta=np.linspace(-1,1,120);xy=np.column_stack((np.repeat(np.arange(12)*1000,10),np.zeros(120)))
        full=block_interval(delta,xy,self.root/'full',{'fixture':'bootstrap'})
        class Stop(Exception):pass
        def stop(event):
            if event['cursor']==100:raise Stop()
        with self.assertRaises(Stop):block_interval(delta,xy,self.root/'resumed',{'fixture':'bootstrap'},stop)
        resumed=block_interval(delta,xy,self.root/'resumed',{'fixture':'bootstrap'})
        self.assertEqual(full,resumed);self.assertEqual(resumed['valid_replicates'],2000)
        self.assertEqual(resumed['seed'],20261013)
        missing=block_interval([],np.empty((0,2)),self.root/'empty',{'fixture':'empty'})
        self.assertEqual(missing['status'],'NA_insufficient_spatial_blocks');self.assertEqual(missing['ci95'],[None,None])

    def test_local_RNA_gene_spatial_density_ledger(self):
        from scripts.rep1_bt_gt_fusion.reports import local_transfers
        cfg=dict(run_id='fixture',run_root=str(self.root),source_digest='s',input_digest='i',plan_sha256='p')
        dummy=self.root/'production.txt';dummy.write_text('fixture production')
        atomic_json(self.root/'production_complete.json',dict(status='succeeded',dependencies=dependencies(cfg,'PRODUCTION_VALIDATE'),outputs={'fixture':identity(dummy)}))
        grant=self.root/'control/manual_resume.json'
        atomic_json(grant,dict(dependencies=dependencies(cfg,'MANUAL_RESUME'),authorization='explicit_user_resume',
             user_instruction_utc='isolated test only',production_complete=identity(self.root/'production_complete.json')))
        coord=self.root/'coords.npy';atomic_npy(coord,np.array([(0,),(1,),(0,),(1,)],dtype=[('gene','<i4')]))
        parent=self.root/'parent_matrix';child=self.root/'child_matrix'
        for p,owners,counts in [(parent,[1,1,2,2],[[1,1],[1,1]]),(child,[2,1,2,2],[[0,1],[2,1]])]:
            p.mkdir();atomic_npy(p/'owners.npy',np.asarray(owners,np.uint32));atomic_npy(p/'counts.npy',np.asarray(counts,np.int64))
            pd.DataFrame(dict(matrix_row=[0,1],canonical_cell_id=[1,2],mask_label=[1,2])).to_csv(p/'cell_mapping.csv',index=False)
            pd.DataFrame(dict(gene_column=[0,1],gene=['A','B'])).to_csv(p/'gene_mapping.csv',index=False)
            atomic_json(p/'complete.json',dict(receipts=[dict(rows=4,coordinate_sha256=identity(coord)['sha256'],
                coordinate_file=str(coord),owner_file='owners.npy',sha256=identity(p/'owners.npy')['sha256'])]))
        metadata=self.root/'metadata.npz';np.savez(metadata,ids=[1,2],type_index=[0,1],centroid_xy=[[0,0],[2000,1000]])
        ref=self.root/'reference.json';atomic_json(ref,dict(type_names=['Type A','Type B']))
        coverage=self.root/'coverage.csv';pd.DataFrame(dict(cell_id=[1,2],density_group=['low','high'])).to_csv(coverage,index=False)
        manifest=dict(endpoints={'P2_BASE':dict(matrix=str(parent))},metadata=str(metadata),reference_region=str(ref),a1={'cell_coverage.csv':str(coverage)})
        out=self.root/'local';result=local_transfers(cfg,manifest,dict(matrix=str(child)),'P2_BASE_GT',out,grant)
        cells=pd.read_csv(out/'cells.csv');self.assertEqual(cells.outgoing_RNA.tolist(),[1,0]);self.assertEqual(cells.incoming_RNA.tolist(),[0,1])
        edge=pd.read_csv(out/'type_spatial_density_gene_transfers.csv').iloc[0]
        self.assertEqual((edge.source_type,edge.target_type,edge.gene,edge.molecules),('Type A','Type B','A',1))
        self.assertEqual((edge.source_block_x,edge.target_block_x,edge.source_density_group,edge.target_density_group),(0,2,'low','high'))
        self.assertTrue(result['capture_exact']);self.assertTrue(result['guardrails_unchanged'])

    def test_R_child_explicit_FD_and_snapshot_boundary(self):
        import fcntl
        from scripts.rep1_bt_gt_fusion.evaluation import run_r
        snapshot=self.root/'snapshot';snapshot.mkdir();out=self.root/'R_out'
        r=snapshot/'fixture.R'
        r.write_text('args<-commandArgs(trailingOnly=TRUE)\n'
                     'fds<-strsplit(Sys.getenv("REP1_BT_GT_LOCK_FDS"),",")[[1]]\n'
                     'actual<-Sys.readlink(paste0("/proc/self/fd/",fds))\n'
                     'stopifnot(length(fds)==2L,all(nchar(actual)>0))\n'
                     'dir.create(args[1],recursive=TRUE,showWarnings=FALSE)\n'
                     'writeLines(actual,file.path(args[1],"inherited.txt"))\n'
                     'writeLines("succeeded",file.path(args[1],"execution_status.txt"))\n')
        checks=[]
        cfg=dict(source_snapshot=str(snapshot),rscript='/home/zwan0726/work/envs/bidcell_egr_r/bin/Rscript')
        with (self.root/'run.lock').open('a') as a,(self.root/'compute.lock').open('a') as b:
            fcntl.flock(a,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(b,fcntl.LOCK_EX|fcntl.LOCK_NB)
            run_r(cfg,r,[out],out,(a.fileno(),b.fileno()),lambda:checks.append(True))
        self.assertEqual(len(checks),2)
        self.assertEqual((out/'inherited.txt').read_text().splitlines(),[str(self.root/'run.lock'),str(self.root/'compute.lock')])
        with self.assertRaises(PermissionError):run_r(cfg,r,[out],out,(),lambda:None)


if __name__=='__main__':unittest.main()
