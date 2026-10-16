"""Recovery boundary tests: interrupted adoption, identities and compatibility."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from scripts.rep1_bt_gt_fusion import recovery as rec
from scripts.rep1_bt_gt_fusion.common import atomic_json, dependencies, identity, read, sha256
from scripts.rep1_bt_gt_fusion import execution as ex


class CheckpointImport(unittest.TestCase):
    def setup_tree(self,tmp):
        base=Path(tmp);old=base/'old';new=base/'new';old.mkdir();new.mkdir()
        def cfg(root,name):
            return dict(run_root=str(root),run_id=name,source_commit=name,source_digest=name,
                        input_digest='same-inputs',plan_sha256='same-plan')
        producer=cfg(old,'old');consumer=cfg(new,'new')
        config=old/'config.json';atomic_json(config,producer)
        source=old/'manifest.json';atomic_json(source,{'commit':'old'})
        plan=dict(source_config=identity(config),source_manifest=identity(source))
        planfile=new/'plan.json';atomic_json(planfile,plan);consumer['checkpoint_import_plan']=identity(planfile)
        folder=old/'bt_existing6';folder.mkdir()
        binary=folder/'science.csv';binary.write_bytes(b'cell,value\n1,2\n')
        leaf=folder/'chunk.json';atomic_json(leaf,dict(status='succeeded',dependencies=dependencies(producer,'BT_EXISTING6'),outputs={'data':identity(binary)}))
        complete=folder/'complete.json';atomic_json(complete,dict(status='succeeded',dependencies=dependencies(producer,'BT_EXISTING6'),outputs={'chunk':identity(leaf),'data':identity(binary)}))
        rows=[dict(**identity(p),relative=str(p.relative_to(old))) for p in (complete,leaf,binary)]
        return producer,consumer,plan,rows

    def test_leaf_root_rehash_provenance_and_independent_scientific_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            old,new,plan,rows=self.setup_tree(tmp);before={r['path']:Path(r['path']).read_bytes() for r in rows}
            mapping=rec.Importer(new,old,plan,rows).run();root=Path(new['run_root'])/'bt_existing6'
            receipt=read(root/'complete.json');leaf=read(root/'chunk.json')
            self.assertEqual(receipt['outputs']['chunk'],identity(root/'chunk.json'))
            self.assertEqual(leaf['outputs']['data'],identity(root/'science.csv'))
            self.assertEqual(receipt['dependencies'],dependencies(new,'BT_EXISTING6'))
            self.assertEqual(leaf['_checkpoint_import']['producer_dependencies'],dependencies(old,'BT_EXISTING6'))
            self.assertFalse(receipt['_checkpoint_import']['scientific_computation_reexecuted'])
            self.assertEqual(receipt['_checkpoint_import']['producer'],{k:rows[0][k] for k in ('path','bytes','sha256')})
            for row in mapping:
                self.assertEqual(Path(row['producer']['path']).read_bytes(),before[row['producer']['path']])
                self.assertNotEqual(Path(row['producer']['path']).stat().st_ino,Path(row['consumer']['path']).stat().st_ino)
            self.assertEqual((root/'science.csv').read_bytes(),before[str(Path(old['run_root'])/'bt_existing6/science.csv')])
            self.assertEqual(rec.Importer(new,old,plan,rows).run(),mapping)

    def test_interruption_resumes_without_replacing_committed_children(self):
        with tempfile.TemporaryDirectory() as tmp:
            old,new,plan,rows=self.setup_tree(tmp);real=rec.atomic
            def stop(path,writer,*args,**kwargs):
                if Path(path).name=='complete.json':raise OSError('simulated power loss before root commit')
                return real(path,writer,*args,**kwargs)
            with patch.object(rec,'atomic',side_effect=stop):
                with self.assertRaises(OSError):rec.Importer(new,old,plan,rows).run()
            child=Path(new['run_root'])/'bt_existing6/chunk.json';before=(child.read_bytes(),child.stat().st_ino)
            rec.Importer(new,old,plan,rows).run()
            self.assertEqual((child.read_bytes(),child.stat().st_ino),before)

    def test_source_and_target_tampering_are_preserved_and_rejected(self):
        for side in ('source','target'):
            with self.subTest(side=side),tempfile.TemporaryDirectory() as tmp:
                old,new,plan,rows=self.setup_tree(tmp)
                if side=='target':rec.Importer(new,old,plan,rows).run()
                root=Path(old['run_root'] if side=='source' else new['run_root']);p=root/'bt_existing6/science.csv'
                p.write_bytes(b'changed-science');before=p.read_bytes()
                with self.assertRaises(ValueError):rec.Importer(new,old,plan,rows).run()
                self.assertEqual(p.read_bytes(),before)

    def test_unselected_identity_bad_stage_and_path_escape_rejected(self):
        for fault in ('unselected','stage','path','symlink'):
            with self.subTest(fault=fault),tempfile.TemporaryDirectory() as tmp:
                old,new,plan,rows=self.setup_tree(tmp);leaf=Path(rows[1]['path']);obj=read(leaf)
                if fault=='unselected':obj['outputs']['external']=identity(Path(old['run_root'])/'config.json')
                elif fault=='stage':obj['dependencies']['stage']='V2_MORPH_GT_GT'
                elif fault=='path':rows[1]['relative']='../escape'
                elif fault=='symlink':
                    outside=Path(tmp)/'outside';outside.mkdir();(Path(new['run_root'])/'bt_existing6').symlink_to(outside,target_is_directory=True)
                if fault in ('unselected','stage'):
                    atomic_json(leaf,obj);rows[1]=dict(**identity(leaf),relative='bt_existing6/chunk.json')
                    complete=Path(rows[0]['path']);obj=read(complete);obj['outputs']['chunk']=identity(leaf);atomic_json(complete,obj)
                    rows[0]=dict(**identity(complete),relative='bt_existing6/complete.json')
                with self.assertRaises(ValueError):rec.Importer(new,old,plan,rows).run()

    def test_full_module_AST_rejects_global_semantics_and_dtype_changes(self):
        root=Path(__file__).resolve().parents[1]/'scripts/rep1_bt_gt_fusion'
        common=(root/'common.py').read_text();gt=(root/'gt_refine.py').read_text()
        self.assertEqual(rec.module_ast(common,'common'),'d65a14b4e931a1c7c71ba676ee9f1cbe5e8947099a068a4de94fea6a10a3b6d3')
        self.assertEqual(rec.module_ast(gt,'gt'),'abade49c08f74fdf7a3855f86f0a429c556e1196042a6e4a16e3ea4472ff2d6c')
        for changed in (common+'\nSCHEMA="changed"\n',common+'\ndependencies=lambda *a: {}\n'):
            self.assertNotEqual(rec.module_ast(changed,'common'),rec.module_ast(common,'common'))
        self.assertNotEqual(rec.module_ast(gt.replace('mask.astype(np.uint32)','mask.astype(np.uint16)'),'gt'),rec.module_ast(gt,'gt'))
        with self.assertRaises(ValueError):rec.module_ast(common+'\ndef atomic_tiff(): pass\n','common')
        with self.assertRaises(ValueError):rec.module_ast(common.replace('def atomic_tiff','@staticmethod\ndef atomic_tiff'),'common')

    def test_immutable_plan_support_scope_and_seed_binding(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);old=root/'old';old.mkdir();new=root/'new';(new/'configs').mkdir(parents=True)
            support=root/'support.json';atomic_json(support,{'audited':'fixture-only'})
            seed=dict(source_root=str(old),target_root=str(new),allowed_trees=list(rec.TREES),input_digest='fixed-input')
            for key in rec.SUPPORT+('source_config','source_manifest','source_admission','source_environment','source_input_manifest','source_failed_state'):
                seed[key]=identity(support)
            seedfile=root/'seed.json';atomic_json(seedfile,seed)
            with patch.object(ex,'OLD_ROOT',old),patch.object(ex,'IMPORT_SEED_SHA',sha256(seedfile)),patch.object(ex,'AMENDMENT_SHA',sha256(support)):
                plan=rec.prepare_plan(new,seedfile);cfg=dict(run_root=str(new),input_digest='fixed-input',checkpoint_import_plan=plan)
                value=rec.verify_plan(cfg)
                self.assertEqual(value['source_config'],seed['source_config'])
                self.assertNotEqual(value['source_inventory']['path'],seed['source_inventory']['path'])
                self.assertEqual(value['source_inventory']['sha256'],seed['source_inventory']['sha256'])
                with self.assertRaises(ValueError):rec.verify_plan(dict(cfg,input_digest='wrong-input'))
                supportcopy=Path(value['source_inventory']['path']);supportcopy.chmod(0o644);supportcopy.write_text('{}')
                with self.assertRaises(ValueError):rec.verify_plan(cfg)


if __name__=='__main__':unittest.main()
