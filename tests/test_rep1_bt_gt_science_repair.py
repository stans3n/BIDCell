"""Tiny final-TIFF recovery and missing-history publication fixtures, never a real run."""
import copy
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import tifffile
from scripts.rep1_bt_gt_fusion import EXISTING, NEW, PARENTS, N_CELLS, N_GENES, ELIGIBLE, comparisons
from scripts.rep1_bt_gt_fusion import common, gt_refine as gt, publication as pub, context_sync as context
from scripts.rep1_bt_gt_fusion.common import (atomic_json, atomic_npy, atomic_tiff, check_receipt,
    dependencies, digest, identity, read, write_receipt)
from scripts.rep1_bt_gt_fusion.evaluation import new_spec
from scripts.boundary_three.core_refine import NuclearSupport, topology_oracle


class ScienceRepairFixtures(unittest.TestCase):
    def setUp(self):
        retained=os.environ.get('REP1_BT_GT_REPAIR_FIXTURE_ROOT')
        self.tmp=None
        if retained:
            self.root=Path(retained)/self._testMethodName
            self.root.mkdir(parents=True,exist_ok=False)
        else:
            self.tmp=tempfile.TemporaryDirectory(prefix='rep1_bt_gt_repair_fixture_')
            self.root=Path(self.tmp.name)

    def tearDown(self):
        if self.tmp:self.tmp.cleanup()

    def _config(self,name):
        root=self.root/name
        return dict(run_id='isolated_'+name,run_root=str(root),source_digest='fixture_source',
                    source_commit='fixture_commit',input_digest='fixture_input',plan_sha256='fixture_plan',
                    input_manifest=str(root/'configs/inputs_manifest.json'))

    def _refine_inputs(self,name,boundary=10):
        cfg=self._config(name);source=self.root/(name+'_input');source.mkdir()
        nuclei=np.zeros((18,22),np.uint32);nuclei[8:10,5:7]=1;nuclei[8:10,15:17]=2
        parent=np.ones_like(nuclei);parent[:,boundary:]=2;parent[nuclei>0]=nuclei[nuclei>0]
        ids=np.array([1,2]);types=np.array([0,1]);records=[]
        for y in range(6,12):
            for x in range(6,17):
                records.extend([(len(records)+1,x,y,(x+y)%3),(len(records)+2,x,y,1)])
        arr=np.array(records,dtype=[('transcript_id','<u8'),('x','<i4'),('y','<i4'),('gene','<i4')])
        coord=source/'coordinates.npy';atomic_npy(coord,arr)
        final=source/'parent.tif';tifffile.imwrite(final,parent)
        reference=source/'reference.json';atomic_json(reference,{'synthetic':'reference identity'})
        cache=source/'cache.json';atomic_json(cache,{'synthetic':'coordinate identity'})
        a1=source/'a1.json';atomic_json(a1,{'synthetic':'A1 eligibility loaded by fixture'})
        manifest=dict(a1={'fixture':str(a1)},reference_region=str(reference),coordinate_cache=str(cache),
                      files={str(p.resolve()):identity(p) for p in (final,reference,cache,a1)})
        data=dict(parent='P2_BASE' if boundary==10 else 'V2_MORPH',spec={'final':str(final)},
                  baseline=parent,nuclei=nuclei,ids=ids,types=types,
                  support=NuclearSupport(nuclei,ids,types,['A','B']),genes=['A','B','C'],
                  gene_indices=np.array([0,1]),prior=np.array([[.99,.01],[.01,.99]]),
                  eligible=np.ones(2,bool),donor=np.array([1,0]),bins=np.zeros(2,int),
                  expected_molecules=len(arr),coordinates=[dict(**identity(coord),rows=len(arr),row_start=0,row_stop=len(arr))])
        return cfg,manifest,data

    def _upstream(self,root):
        return {str(p.relative_to(root)):{**identity(p),'mtime_ns':p.stat().st_mtime_ns}
                for group in ('domain','scores','recovery') for p in sorted((root/group).rglob('*')) if p.is_file()}

    def _assert_refine(self,cfg,data,endpoint,result):
        root=Path(cfg['run_root'])/'endpoints'/endpoint
        final=tifffile.imread(root/'final.tif')
        self.assertEqual(final.dtype,np.dtype(np.uint32))
        np.testing.assert_array_equal(final>0,data['baseline']>0)
        np.testing.assert_array_equal(final[data['nuclei']>0],data['nuclei'][data['nuclei']>0])
        np.testing.assert_array_equal(np.unique(final),data['ids'])
        oracle=data['baseline'].copy();accepted=0
        proposals=np.load(root/'gt/scores/proposals.npy')
        for p in proposals:
            y,x,source,target=map(int,(p['y'],p['x'],p['source'],p['target']))
            ok,_=topology_oracle(oracle,data['nuclei'],y,x,source,target)
            if ok:oracle[y,x]=target;accepted+=int(ok)
        self.assertGreater(accepted,0)
        np.testing.assert_array_equal(final,oracle)
        self.assertEqual(result['application']['cursor'],len(proposals))
        self.assertEqual(result['application']['accepted'],accepted)
        self.assertTrue(result['invariants']['nuclear_connections_preserved'])
        check_receipt(root/'gt/complete.json',dependencies(cfg,endpoint+'_GT'),cfg['run_root'])
        return root

    def test_atomic_tiff_exact_uint32_and_interrupted_install(self):
        path=self.root/'final.tif'
        array=np.arange(260*7,dtype=np.uint32).reshape(260,7);array[-1,-1]=2**32-1
        atomic_tiff(path,array)
        np.testing.assert_array_equal(tifffile.imread(path),array)
        self.assertEqual(tifffile.imread(path).dtype,np.dtype(np.uint32))
        before=identity(path)
        with patch.object(common.os,'replace',side_effect=OSError('fixture before replace')):
            with self.assertRaises(OSError):atomic_tiff(path,array+np.uint32(1))
        self.assertEqual(identity(path),before)
        real_write=tifffile.imwrite
        def partial(path,data):
            Path(path).write_bytes(b'partial TIFF');raise OSError('fixture write interruption')
        with patch.object(tifffile,'imwrite',side_effect=partial):
            with self.assertRaises(OSError):atomic_tiff(path,array)
        self.assertEqual(identity(path),before)
        def wrong_pixels(path,data):real_write(path,np.zeros_like(data))
        with patch.object(tifffile,'imwrite',side_effect=wrong_pixels):
            with self.assertRaisesRegex(ValueError,'pixels differ'):atomic_tiff(path,array)
        self.assertEqual(identity(path),before)
        with self.assertRaises(ValueError):atomic_tiff(path,array.astype(np.uint16))
        self.assertEqual(list(self.root.glob('.final.tif.*')),[])
        atomic_json(self.root/'fixture_receipt.json',dict(status='passed',uint32_exact=True,
            before_replace_and_during_write_failures_preserve_destination=True,
            temporary_pixel_corruption_rejected=True,final=identity(path)))

    def test_refine_actual_final_tiff_and_receipt_for_both_parents(self):
        for endpoint,boundary in zip(NEW,(10,12)):
            cfg,manifest,data=self._refine_inputs(endpoint,boundary)
            with patch.object(gt,'load_data',return_value=data):result=gt.refine(cfg,manifest,endpoint)
            root=self._assert_refine(cfg,data,endpoint,result)
            before={str(p):identity(p) for p in root.rglob('*') if p.is_file()}
            with patch.object(gt,'load_data',side_effect=AssertionError('completed refine must only verify')):
                self.assertEqual(gt.refine(cfg,manifest,endpoint),result)
            self.assertEqual(before,{str(p):identity(p) for p in root.rglob('*') if p.is_file()})
        atomic_json(self.root/'fixture_receipt.json',dict(status='passed',both_parent_domains=True,
            actual_refine_final_TIFF_and_complete_receipt=True,full_image_topology_oracle_equal=True,
            completed_receipt_reuse_verified=True))

    def test_refine_resume_before_and_after_final_replace(self):
        for point in ('before_replace','after_replace'):
            cfg,manifest,data=self._refine_inputs(point)
            endpoint=NEW[0];root=Path(cfg['run_root'])/'endpoints'/endpoint;final=root/'final.tif'
            real_replace=common.os.replace;real_sync=common.fsync_dir
            def replace(source,target):
                if point=='before_replace' and Path(target)==final:raise OSError('fixture before final replace')
                return real_replace(source,target)
            def sync(path):
                if point=='after_replace' and Path(path)==root:raise OSError('fixture after final replace')
                return real_sync(path)
            with patch.object(gt,'load_data',return_value=data),patch.object(common.os,'replace',side_effect=replace),patch.object(common,'fsync_dir',side_effect=sync):
                with self.assertRaises(OSError):gt.refine(cfg,manifest,endpoint)
            self.assertEqual(final.exists(),point=='after_replace')
            self.assertFalse((root/'gt/complete.json').exists())
            state=read(root/'gt/recovery/state.json');proposals=np.load(root/'gt/scores/proposals.npy')
            self.assertEqual(state['cursor'],len(proposals))
            before=self._upstream(root/'gt')
            with patch.object(gt,'load_data',return_value=data),patch.object(gt,'score_proposals',side_effect=AssertionError('recomputed scores')),patch.object(gt.CompleteCellTopology,'check',side_effect=AssertionError('reapplied proposals')):
                if point=='after_replace':
                    with patch.object(gt,'atomic_tiff',side_effect=AssertionError('rewrote installed TIFF')):
                        result=gt.refine(cfg,manifest,endpoint)
                else:result=gt.refine(cfg,manifest,endpoint)
            self._assert_refine(cfg,data,endpoint,result)
            self.assertEqual(self._upstream(root/'gt'),before)
            atomic_json(self.root/(point+'_receipt.json'),dict(status='passed',interruption=point,
                committed_cursor=state['cursor'],upstream_hashes_and_mtimes_unchanged=True,
                no_score_or_topology_replay=True,complete=identity(root/'gt/complete.json'),upstream=before))

    def test_refine_rejects_installed_tiff_drift_after_interruption(self):
        cfg,manifest,data=self._refine_inputs('drift');root=Path(cfg['run_root'])/'endpoints'/NEW[0]
        real_sync=common.fsync_dir
        def sync(path):
            if Path(path)==root:raise OSError('fixture after final replace')
            return real_sync(path)
        with patch.object(gt,'load_data',return_value=data),patch.object(common,'fsync_dir',side_effect=sync):
            with self.assertRaises(OSError):gt.refine(cfg,manifest,NEW[0])
        final=root/'final.tif';bad=tifffile.imread(final);bad[0,0]=99;tifffile.imwrite(final,bad)
        with patch.object(gt,'load_data',return_value=data):
            with self.assertRaisesRegex(ValueError,'final mask differs'):gt.refine(cfg,manifest,NEW[0])
        self.assertFalse((root/'gt/complete.json').exists())

    def _publication_fixture(self):
        cfg=self._config('publication');root=Path(cfg['run_root']);source=self.root/'history';source.mkdir()
        columns,old=pub.read_strings(pub.TABLE)
        table=source/'main_table.csv';pub.write_strings(table,columns,old)
        registry=source/'artifact_registry.json';atomic_json(registry,{'synthetic':'preserved historical28 chain'})
        historical={};files={str(p):identity(p) for p in (table,registry)}
        base=np.array([[0,1,1],[2,3,3]],np.uint32)
        for i,row in enumerate(old):
            name=row['method'];folder=source/name;folder.mkdir();mask=folder/'final.tif'
            tifffile.imwrite(mask,base+np.uint32(i%3));expected=identity(mask)
            mr=folder/'matrix_complete.json';atomic_json(mr,{'dependencies':{'mask':expected}})
            available=name not in ('C','SP')
            if available:files[str(mask)]=expected
            else:mask.unlink()
            files[str(mr)]=identity(mr)
            historical[name]=dict(mask=expected,matrix_receipt=identity(mr),available=available,
                recovery_relative=None if available else 'recovered_historical_masks/'+name+'/final.tif',
                unavailable_reason='' if available else 'original_final_mask_missing; publication_unique_content_verification_blocked')
        manifest=dict(schema='rep1-bt-gt-inputs-v1',existing_order=list(EXISTING),new_order=list(NEW),
            comparisons=[list(x) for x in comparisons()],endpoints={n:{} for n in EXISTING},
            expected_cells=N_CELLS,expected_genes=N_GENES,expected_molecules=ELIGIBLE,
            historical_masks=historical,author_reference={},files=files)
        atomic_json(cfg['input_manifest'],manifest);cfg['input_digest']=digest(manifest)
        data=root/'fixture.data';data.write_bytes(b'isolated synthetic production and evaluation')
        for target,stage in [('production_complete.json','PRODUCTION_VALIDATE'),('evaluation8/complete.json','EVALUATION8'),('bt_existing6/complete.json','BT_EXISTING6')]:
            write_receipt(root/target,dependencies(cfg,stage),{'fixture':identity(data)})
        grant=root/'control/manual_resume.json';atomic_json(grant,dict(dependencies=dependencies(cfg,'MANUAL_RESUME'),
            authorization='explicit_user_resume',user_instruction_utc='isolated fixture only',production_complete=identity(root/'production_complete.json')))
        new_rows={}
        for i,name in enumerate(NEW):
            spec=new_spec(cfg,name);endpoint=Path(spec['root']);matrix=Path(spec['matrix']);ev=Path(spec['evaluation'])
            matrix.mkdir(parents=True);(ev/'paper_aligned_v1').mkdir(parents=True);(ev/'fixed').mkdir()
            mask=tifffile.imread(historical[PARENTS[name]]['mask']['path']) if i==0 else np.array([[0,1,2],[3,4,5]],np.uint32)
            tifffile.imwrite(spec['final'],mask);atomic_json(spec['machine'],{'synthetic':True})
            atomic_npy(matrix/'counts.npy',np.array([[1,2],[3,4],[5,6]],np.int64))
            pd.DataFrame(dict(matrix_row=[0,1,2],canonical_cell_id=[1,2,3],mask_label=[1,2,3])).to_csv(matrix/'cell_mapping.csv',index=False)
            pd.DataFrame(dict(gene_column=[0,1],gene=['A','B'])).to_csv(matrix/'gene_mapping.csv',index=False)
            for path in (matrix/'complete.json',matrix/'validation.json',ev/'complete.json',endpoint/'production_complete.json'):
                atomic_json(path,{'synthetic':True})
            (ev/'paper_aligned_v1/annotation.csv').write_text('cell_id,evaluation_label\n1,B Cells\n2,CD4 T\n')
            row={key:'' for key in columns};row['method']=name
            for profile,path in (('scclassify',spec['sc_scores']),('fixed',spec['fixed_scores'])):
                positive=.35+.1*i;negative=.04-.01*i
                pd.DataFrame(dict(cell_id=[1,2],positive_F1=[positive]*2,negative_F1=[negative]*2)).to_csv(path,index=False)
                row[profile+'_positive_F1']=str(positive);row[profile+'_negative_F1']=str(negative)
                row[profile+'_positive_F1_n']='2';row[profile+'_negative_F1_n']='2'
            row['capture_pct']='62';new_rows[name]=row
        return cfg,manifest,grant,columns,old,new_rows,table,registry

    def _stage(self,fixture):
        cfg,manifest,grant,columns,old,new_rows,table,registry=fixture
        patches=patch.multiple(pub,TABLE=table,REGISTRY=registry,TABLE_SHA=identity(table)['sha256'],REGISTRY_SHA=identity(registry)['sha256'])
        patches.start();self.addCleanup(patches.stop)
        with patch.object(pub,'new_table_row',side_effect=lambda c,m,n,cols:dict(new_rows[n])):
            stage=pub.prepare_publication(cfg,grant)
        return stage

    def _restamp(self,stage):
        receipt=read(stage/'staging_complete.json')
        receipt['outputs']={name:{k:v for k,v in identity(stage/name).items() if k!='path'} for name in receipt['outputs']}
        atomic_json(stage/'staging_complete.json',receipt)

    def test_missing_C_SP_publication_and_context_keep_30_rows_and_NA(self):
        fixture=self._publication_fixture();cfg,manifest,grant,columns,old,_,_,_=fixture
        frozen=identity(cfg['input_manifest']);stage=self._stage(fixture)
        pub.validate_bundle(stage)
        reg=read(stage/'artifact_registry.json');notes=read(stage/'publication_notes.json')
        _,rows=pub.read_strings(stage/'main_table.csv')
        self.assertEqual(len(rows),30);pub.preserve_old_raw(old,rows,columns)
        self.assertEqual(read(stage/'purity_scaling.json')['comparison_set'],[r['method'] for r in rows])
        self.assertEqual(reg['available_mask_method_entries'],28)
        self.assertEqual(reg['missing_mask_methods'],['C','SP'])
        self.assertEqual(reg['available_unique_mask_contents'],4)
        self.assertIsNone(reg['unique_mask_contents']);self.assertFalse(reg['unique_mask_contents_verified'])
        self.assertIsNone(notes['unique_mask_contents'])
        for name in ('C','SP'):
            self.assertEqual(reg['mask_content'][name]['mask'],manifest['historical_masks'][name]['mask'])
            self.assertIsNone(reg['mask_content'][name]['content_sha256'])
            self.assertIsNone(reg['mask_content'][name]['alias_of'])
        self.assertTrue(reg['new_endpoints'][NEW[0]]['no_op_with_parent'])
        self.assertFalse(reg['new_endpoints'][NEW[1]]['no_op_with_parent'])
        self.assertEqual(identity(cfg['input_manifest']),frozen)
        # Exercise the actual append/final-receipt consumer with every target
        # redirected into this synthetic fixture, never the real context tree.
        ctx=self.root/'context';ctx.mkdir();targets=[ctx/Path(p).name for p in context.TARGETS]
        for p in targets:p.write_bytes(b'original synthetic history\n')
        (ctx/'experiment_registry.csv').write_text('experiment_id,logical_method,status,scope_reason,evidence_root\n')
        analysis=dict(dependencies=dependencies(cfg,'RESULTS_ANALYSIS'),reviewer='primary',
            evaluation8=identity(Path(cfg['run_root'])/'evaluation8/complete.json'),
            dual_F1_parallel=True,negative_results_retained=True,local_RNA_guardrails_reviewed=True,no_owner_truth_claim=True,
            comparisons_reviewed=[list(x) for x in comparisons()],selection_summary='Synthetic negative results retained.',
            local_RNA_explanation='Synthetic local cost remains explicit.')
        atomic_json(Path(cfg['run_root'])/'results_analysis.json',analysis)
        proof=dict(published_root=str(stage),old28_unchanged=True,outputs={'table':identity(stage/'main_table.csv')})
        atomic_json(stage/'publication_complete.json',proof)
        with patch.multiple(context,CONTEXT=ctx,TARGETS=targets,RESULT_ROOT=stage):
            final=context.synchronize(cfg,grant,proof)
        self.assertIsNone(final['unique_mask_contents']);self.assertEqual(final['available_unique_mask_contents'],4)
        self.assertEqual(final['missing_mask_methods'],['C','SP'])
        self.assertFalse(final['unique_mask_contents_verified'])
        text=targets[0].read_text()
        self.assertIn('实际唯一 mask 内容数 NA（未完全核验',text)
        self.assertIn('可读 mask 的方法条目 28',text);self.assertIn('缺失方法 C、SP',text)
        self.assertNotIn('内容数 None',text);self.assertIn('original synthetic history',text)
        atomic_json(self.root/'fixture_receipt.json',dict(status='passed',synthetic=True,
            old28_nonpurity_strings_preserved=True,thirty_rows_and_Purity_inputs=True,
            frozen_manifest_unchanged=True,whole30_unique_is_null=True,readable_entries=28,readable_unique=4,
            context_NA_and_final_receipt_fields_verified=True,staging=identity(stage/'staging_complete.json'),
            final_registration=identity(stage/'context_sync_complete.json'),no_real_publication_or_context_write=True))

    def test_missing_mask_allowlist_and_identity_drift_fail_closed(self):
        fixture=self._publication_fixture();cfg,manifest=fixture[:2]
        self.assertIsNone(pub.require_publication_masks(cfg,manifest)['C'])
        altered=copy.deepcopy(manifest);altered['historical_masks']['C']['available']=True
        with self.assertRaises(ValueError):pub.require_publication_masks(cfg,altered)
        altered=copy.deepcopy(manifest);altered['historical_masks']['C']['mask']['sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'original matrix receipt'):pub.require_publication_masks(cfg,altered)
        altered=copy.deepcopy(manifest);record=altered['historical_masks'].pop('C');record['recovery_relative']='recovered_historical_masks/OTHER/final.tif'
        altered['historical_masks']['OTHER']=record
        with self.assertRaisesRegex(ValueError,'outside the frozen'):pub.require_publication_masks(cfg,altered)
        target=Path(cfg['run_root'])/'recovered_historical_masks/C/final.tif';target.parent.mkdir(parents=True)
        target.write_bytes(b'incomplete recovery')
        with self.assertRaisesRegex(ValueError,'incomplete'):pub.require_publication_masks(cfg,manifest)
        target.unlink()
        b0=Path(manifest['historical_masks']['B0']['mask']['path']);b0.write_bytes(b'drift')
        with self.assertRaises(ValueError):pub.require_publication_masks(cfg,manifest)
        b0.unlink()
        with self.assertRaises(ValueError):pub.require_publication_masks(cfg,manifest)

    def test_exact_recovered_history_restores_verified_whole_set_count(self):
        fixture=self._publication_fixture();cfg,manifest,grant,columns,old=fixture[:5]
        base=np.array([[0,1,1],[2,3,3]],np.uint32)
        for name in ('C','SP'):
            i=next(i for i,row in enumerate(old) if row['method']==name)
            original=self.root/(name+'_exact_archive.tif');tifffile.imwrite(original,base+np.uint32(i%3))
            pub.recover_historical_mask(cfg,grant,name,original)
        stage=self._stage(fixture);pub.validate_bundle(stage)
        reg=read(stage/'artifact_registry.json')
        self.assertEqual(reg['missing_mask_methods'],[])
        self.assertEqual(reg['available_mask_method_entries'],30)
        self.assertEqual(reg['unique_mask_contents'],4)
        self.assertTrue(reg['unique_mask_contents_verified'])
        self.assertEqual(set(reg['historical_mask_recovery_receipts']),{'C','SP'})

    def test_bundle_rejects_false_unique_claims_notes_drift_and_missing_new_mask(self):
        fixture=self._publication_fixture();stage=self._stage(fixture)
        original=read(stage/'artifact_registry.json');notes=read(stage/'publication_notes.json')
        for value in (30,28,0):
            bad=copy.deepcopy(original);bad['unique_mask_contents']=value
            atomic_json(stage/'artifact_registry.json',bad);self._restamp(stage)
            with self.assertRaisesRegex(ValueError,'unique-content counts'):pub.validate_bundle(stage)
        bad=copy.deepcopy(original);del bad['unique_mask_contents']
        atomic_json(stage/'artifact_registry.json',bad);self._restamp(stage)
        with self.assertRaisesRegex(ValueError,'unique-content counts'):pub.validate_bundle(stage)
        bad=copy.deepcopy(original);bad['mask_content']['C']['content_sha256']='0'*64
        atomic_json(stage/'artifact_registry.json',bad);self._restamp(stage)
        with self.assertRaisesRegex(ValueError,'missing mask cannot'):pub.validate_bundle(stage)
        bad=copy.deepcopy(original);bad['mask_content']['C']['mask']['sha256']='0'*64
        atomic_json(stage/'artifact_registry.json',bad);self._restamp(stage)
        with self.assertRaisesRegex(ValueError,'original identity'):pub.validate_bundle(stage)
        atomic_json(stage/'artifact_registry.json',original)
        atomic_json(stage/'publication_notes.json',dict(notes,unique_mask_contents=30));self._restamp(stage)
        with self.assertRaisesRegex(ValueError,'notes mask verification'):pub.validate_bundle(stage)
        atomic_json(stage/'publication_notes.json',notes);self._restamp(stage)
        pub.validate_bundle(stage)
        Path(original['new_endpoints'][NEW[0]]['mask']['path']).unlink()
        with self.assertRaises(ValueError):pub.validate_bundle(stage)


if __name__=='__main__':unittest.main()
