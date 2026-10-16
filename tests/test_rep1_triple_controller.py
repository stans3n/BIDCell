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

if __name__=='__main__':unittest.main()
