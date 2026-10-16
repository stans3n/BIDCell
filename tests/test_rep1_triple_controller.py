"""Focused controller safety/identity contracts; no full biological tests."""
from pathlib import Path
from contextlib import ExitStack
from unittest.mock import patch
import fcntl,json,os,tempfile,unittest
from scripts.rep1_triple_simple_boundary import runtime as rt,execution as ex
from scripts.rep1_triple_simple_boundary.common import atomic_json,digest,identity,dependencies,write_receipt

class ControllerContracts(unittest.TestCase):
    def test_fingerprints_detect_same_size_timestamp_restored_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'input';p.write_bytes(b'original');old=ex.fingerprints([p]);st=p.stat();p.write_bytes(b'modified');os.utime(p,ns=(st.st_atime_ns,st.st_mtime_ns));self.assertNotEqual(old,ex.fingerprints([p]))
    def test_fingerprint_directory_detects_added_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            old=ex.fingerprints([tmp]);(Path(tmp)/'new.py').write_text('x=1');self.assertNotEqual(old,ex.fingerprints([tmp]))
    def test_lock_inherited_OFD_and_independent_fd_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg=dict(run_root=tmp);work=Path(tmp)/'workspace.lock'
            with patch.object(ex,'COMPUTE',work),ExitStack() as stack:
                fds=tuple(stack.enter_context(rt.lock(p)) for p in rt.paths(cfg));rt.check_fds(cfg,fds)
                other=os.open(rt.paths(cfg)[0],os.O_RDONLY)
                try:
                    with self.assertRaises(PermissionError):rt.check_fds(cfg,(other,*fds[1:]))
                finally:os.close(other)
    def test_allocated_hardlink_count_once_and_failed_temp_not_subtracted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);a=root/'analysis/initial/data';a.parent.mkdir(parents=True);a.write_bytes(b'x'*10000);b=root/'analysis/all12/data';b.parent.mkdir(parents=True);os.link(a,b)
            temp=root/'endpoints/P2_SIMPLE/.counts.failed';temp.parent.mkdir(parents=True);temp.write_bytes(b'x'*20000)
            sizes=rt.allocations(dict(run_root=tmp));self.assertEqual(sizes['analysis'],a.stat().st_blocks*512);self.assertEqual(sizes['p2_replay'],0);self.assertEqual(sizes['uncommitted_or_temporary'],temp.stat().st_blocks*512)
    def test_formal_loader_rejects_fixture_path_and_options(self):
        with tempfile.TemporaryDirectory(prefix='controller-fixtures-') as tmp:
            p=Path(tmp)/'run.json';atomic_json(p,dict(run_root=tmp,run_id='fake',schema='rep1-triple-engineering-fixture-v1',fixture=True))
            with self.assertRaises(ValueError):rt.load_config(p)
    def test_actual_process_identity_namespace_and_ticks(self):
        p=rt.process_identity();self.assertTrue(rt.alive(p));self.assertFalse(rt.alive(dict(p,start_ticks=p['start_ticks']+1)));self.assertIsNone(rt.alive(dict(p,pid_namespace=p['pid_namespace']+1)))
    def test_semantic_witness_rejects_hash_list_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg=dict(run_root=tmp,run_id='fixture',source_digest='s',input_digest='i',domain_digest='d',plan_digest='p');data=Path(tmp)/'data';data.write_text('manifest only');p=Path(tmp)/'chunk.json';write_receipt(p,dependencies(cfg,'01'),{'data':identity(data)})
            with self.assertRaisesRegex(ValueError,'semantic'):rt.witness(cfg,'01',p)
    def test_status_namespace_unobservable_is_not_false_process_exit(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg=dict(run_root=tmp,run_id='fixture',source_digest='s',source_commit='s',input_digest='i',domain_digest='d',plan_digest='p',environment_digest='e',source_snapshot=tmp,resources=dict(stale_seconds=60,no_progress_seconds=900));s=rt.new_state(cfg);s.update(status='RUNNING',supervisor=dict(rt.process_identity(),pid_namespace=rt.namespace()+1),last_heartbeat=rt.utc());atomic_json(Path(tmp)/'state.json',s);before=(Path(tmp)/'state.json').read_bytes()
            with patch.object(rt,'load_config',return_value=cfg):result=rt.status('unused')
            self.assertIsNone(result['process_observations']['supervisor']);self.assertEqual(result['observed_status'],'RUNNING');self.assertEqual(before,(Path(tmp)/'state.json').read_bytes())
    def test_wrong_auditor_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'admission/final.json';atomic_json(p,dict(status='PASS_FOR_LAUNCH',approved_for_formal_launch=True,open_blockers=[],reviewer=dict(implementation_author=True)))
            with self.assertRaisesRegex(PermissionError,'auditor'):ex.verify_admission(dict(run_root=tmp))
    def test_controlled_environment_overrides_loader_R_and_cache(self):
        env=ex.controlled_environment(dict(run_root='/isolated/run'));self.assertEqual(env['R_LIBS'],'');self.assertEqual(env['LD_PRELOAD'],'');self.assertEqual(env['TMPDIR'],'/isolated/run/tmp');self.assertEqual(env['R_HOME'],'/home/zwan0726/work/envs/bidcell_egr_r/lib/R')

    def test_preparation_audit_and_original_source_params_cannot_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'run';root.mkdir();(root/'control').mkdir();origin=Path(tmp)/'original.py';origin.write_text('x=1\n');budget=Path(tmp)/'budget.json';atomic_json(budget,{})
            rows=[dict(**identity(origin),snapshot_relative='scripts/payload.py')];audit=Path(tmp)/'audit.json';reviewer=dict(agent_id=ex.AUDITOR,model='gpt-6-astra',reasoning_effort='xhigh',role='independent_code_auditor',implementation_author=False)
            atomic_json(audit,dict(decision='PASS_FOR_PREPARATION',source_commit='fixture',open_blockers=[],reviewer=reviewer,source_files_digest=digest(rows),evidence=[]))
            with patch.object(ex,'source_files',return_value=rows),patch.object(ex,'CONTROLLER_AUDIT',audit):
                m=ex.copy_snapshot(root,'fixture');partial=dict(schema='rep1-triple-simple-preparation-v1',run_root=str(root),source_snapshot=m['snapshot_root'],source_commit='fixture',source_digest=m['source_digest'],resources=rt.resource_contract(),preparation_audit=identity(audit),resource_proposal=identity(budget));params=root/'control/preparation.json';atomic_json(params,partial);atomic_json(root/'control/preparation_binding.json',dict(params=identity(params),source_manifest=identity(root/'source_manifest.json')))
                with patch.object(Path,'cwd',return_value=Path(m['snapshot_root'])),patch.object(ex,'module_origins',return_value={}):
                    ex.verify_preparation(partial,params)
                    p=Path(m['snapshot_root'])/'scripts/payload.py';p.chmod(0o644);p.write_text('x=2\n');p.chmod(0o444)
                    with self.assertRaises(ValueError):ex.verify_preparation(partial,params)
                with self.assertRaises(PermissionError):ex.preparation_audit('wrong',rows)
                atomic_json(params,dict(partial,run_id='tampered'))
                with self.assertRaises(ValueError):ex.verify_preparation(partial,params)

    def test_resume_preparation_after_domain_and_after_config_before_preflight(self):
        from scripts.rep1_triple_simple_boundary import inputs,frozen_domains
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'control').mkdir();(root/'configs').mkdir();(root/'admission').mkdir();budget=root/'budget.json';atomic_json(budget,{});params=root/'control/preparation.json'
            partial=dict(schema='rep1-triple-simple-preparation-v1',run_id='fixture',run_root=tmp,source_snapshot=tmp,source_commit='fixture',source_digest='fixture',resources=rt.resource_contract(),resource_proposal=identity(budget));atomic_json(params,partial);state={'domain_interrupt':True,'preflight_interrupt':True}
            def freeze_inputs(*args):return {}
            def freeze_domain(*args):
                folder=root/'frozen_domain';folder.mkdir(exist_ok=True);p=folder/'chunk.json'
                if not p.exists():atomic_json(p,dict(committed=True))
                if state['domain_interrupt']:state['domain_interrupt']=False;raise InterruptedError('owned fixture domain checkpoint')
                if not (folder/'complete.json').exists():atomic_json(folder/'complete.json',dict(status='succeeded',outputs={'chunk':identity(p)}))
            def preflight(cfg,fds):
                if state['preflight_interrupt']:state['preflight_interrupt']=False;raise InterruptedError('owned fixture after immutable config')
                p=root/'admission/preflight.json'
                if not p.exists():atomic_json(p,dict(config_digest=digest(cfg)))
            with patch.object(rt,'check_fds'),patch.object(ex,'verify_preparation'),patch.object(inputs,'freeze_inputs',side_effect=freeze_inputs),patch.object(frozen_domains,'freeze',side_effect=freeze_domain),patch.object(rt,'domain_cardinalities'),patch.object(ex,'environment',return_value={'fixture':'environment'}),patch.object(ex,'make_preflight',side_effect=preflight),patch.object(rt,'ResourceGuard',side_effect=lambda c:lambda *args:None),patch.object(rt,'forecast',return_value={'admitted':True}):
                with self.assertRaises(InterruptedError):rt.prepare_snapshot(params,(9,10,11))
                chunk=(root/'frozen_domain/chunk.json').read_bytes()
                with self.assertRaises(InterruptedError):rt.prepare_snapshot(params,(9,10,11))
                config=(root/'configs/run.json').read_bytes();domain=(root/'frozen_domain/complete.json').read_bytes()
                rt.prepare_snapshot(params,(9,10,11));self.assertEqual(config,(root/'configs/run.json').read_bytes());self.assertEqual(chunk,(root/'frozen_domain/chunk.json').read_bytes());self.assertEqual(domain,(root/'frozen_domain/complete.json').read_bytes());self.assertTrue((root/'admission/preflight.json').exists())
                atomic_json(root/'configs/run.json',{'corrupted':True})
                with self.assertRaises(ValueError):rt.prepare_snapshot(params,(9,10,11))

    def test_resume_preparation_rejects_forged_existing_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'rep1-triple-simple-boundary-20261017-r99';(root/'control').mkdir(parents=True);p=root/'control/preparation.json';atomic_json(p,dict(run_id=root.name,run_root=str(root),source_snapshot=str(root/'source_snapshot')))
            with patch.object(ex,'OUTPUT',root.parent),self.assertRaises(PermissionError):rt.resume_prepare(p)

if __name__=='__main__':unittest.main()
