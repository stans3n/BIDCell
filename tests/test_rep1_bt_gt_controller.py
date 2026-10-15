"""Controller correctness tests on isolated files; no formal scientific outputs."""
from pathlib import Path
import json
import os
import signal
import subprocess
import tempfile
import unittest
from datetime import datetime,timezone,timedelta
from unittest.mock import patch
from scripts.rep1_bt_gt_fusion import runtime as rt, execution as ex
from scripts.rep1_bt_gt_fusion.common import atomic_json,identity,digest,dependencies
from scripts.rep1_bt_gt_fusion.context_sync import commit_append_journal,context_block


class ControllerContracts(unittest.TestCase):
    def test_status_stale_is_read_only_and_fresh_dispatch_has_grace(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg=dict(run_root=tmp,run_id='fixture',source_commit='x',source_digest='x',input_digest='x',plan_sha256='x',source_snapshot=tmp,environment_digest='x')
            state=rt.state_new(cfg);state.update(status='RUNNING',supervisor=rt.process_identity(),heartbeat_utc=(datetime.now(timezone.utc)-timedelta(seconds=61)).isoformat(),
                      attempts=[dict(created_utc=(datetime.now(timezone.utc)-timedelta(seconds=1000)).isoformat())])
            path=Path(tmp)/'state.json';atomic_json(path,state);before=path.read_bytes()
            with patch.object(rt,'load_config',return_value=cfg):observed=rt.status(path)
            self.assertEqual(observed['observed_status'],'STALE');self.assertEqual(observed['disk_status'],'RUNNING')
            self.assertTrue(observed['needs_progress_inspection']);self.assertEqual(path.read_bytes(),before)
            self.assertEqual(list(Path(tmp).rglob('*')),[path])
            state.update(status='STARTING',supervisor=None,heartbeat_utc=None,attempts=[dict(created_utc=rt.utc())]);atomic_json(path,state)
            with patch.object(rt,'load_config',return_value=cfg):self.assertEqual(rt.status(path)['observed_status'],'STARTING')

    def test_process_start_ticks_and_zombie(self):
        own=rt.process_identity();self.assertTrue(rt.alive(own))
        self.assertFalse(rt.alive(dict(own,start_ticks=own['start_ticks']+1)))
        self.assertFalse(rt.alive(dict(own,boot_id='other-boot')))
        child=subprocess.Popen(['/bin/true']);os.waitid(os.P_PID,child.pid,os.WEXITED|os.WNOWAIT)
        self.assertFalse(rt.alive(dict(own,pid=child.pid)))
        child.wait()

    def test_atomic_json_interruption_preserves_old_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'state.json';atomic_json(p,{'cursor':7});before=p.read_bytes()
            with patch('scripts.rep1_bt_gt_fusion.common.os.replace',side_effect=OSError('fixture interruption')):
                with self.assertRaises(OSError):atomic_json(p,{'cursor':8})
            self.assertEqual(p.read_bytes(),before)
            atomic_json(p,{'cursor':8});self.assertEqual(json.loads(p.read_text())['cursor'],8)

    def test_source_independent_copies_and_drift(self):
        with tempfile.TemporaryDirectory(prefix='runtime-fixture-') as tmp:
            root=Path(tmp);m=ex.copy_snapshot(root,'fixture')
            cfg=dict(run_root=str(root),source_snapshot=m['snapshot_root'],source_commit='fixture',source_digest=m['source_digest'])
            ex.verify_snapshot(cfg,False)
            for r in m['files']:
                original=Path(r['original']);copy=Path(r['snapshot_path'])
                self.assertNotEqual((original.stat().st_dev,original.stat().st_ino),(copy.stat().st_dev,copy.stat().st_ino))
            p=Path(m['files'][0]['snapshot_path']);p.chmod(0o644);p.write_bytes(p.read_bytes()+b'\n');p.chmod(0o444)
            with self.assertRaises(ValueError):ex.verify_snapshot(cfg,False)

    def test_formal_loader_rejects_engineering_schema_and_path(self):
        with tempfile.TemporaryDirectory(prefix='runtime-fixture-') as tmp:
            p=Path(tmp)/'run.json';atomic_json(p,dict(schema='engineering',run_root=tmp,run_id='fake'))
            with self.assertRaises(ValueError):rt.load_config(p)

    def test_fake_admission_not_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'admission').mkdir()
            atomic_json(root/'admission/final.json',dict(status='ADMITTED',approved_for_formal_launch=True,
                         reviewer=dict(model='gpt-6-astra',reasoning_effort='max',implementation_author=True)))
            cfg=dict(run_root=tmp,source_commit='x',source_digest='x',input_digest='x',environment_digest='x',
                     plan_sha256='x',prompt_sha256='x',source_snapshot=str(root/'source_snapshot'))
            with self.assertRaises(ValueError):ex.verify_admission(cfg)

    def test_context_append_partial_install_resume_and_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);a=root/'a.md';b=root/'b.csv';a.write_bytes(b'old-history\n');b.write_bytes(b'header\r\nold\r\n')
            updates=[(a,b'new-block\n'+a.read_bytes()),(b,b.read_bytes()+b'new\r\n')]
            journal=root/'journal'
            real=__import__('scripts.rep1_bt_gt_fusion.context_sync',fromlist=['write_bytes']).write_bytes
            def interrupted(path,data):
                if Path(path)==b:raise OSError('fixture interruption after first target installed')
                return real(path,data)
            with patch('scripts.rep1_bt_gt_fusion.context_sync.write_bytes',side_effect=interrupted):
                with self.assertRaises(OSError):commit_append_journal(journal,updates,{'run':'fixture'})
            self.assertTrue(a.read_bytes().startswith(b'new-block'))
            proof=commit_append_journal(journal,[],{'run':'fixture'})
            self.assertEqual(a.read_bytes(),updates[0][1]);self.assertEqual(b.read_bytes(),updates[1][1])
            self.assertEqual(len(proof['outputs']),2)
            with self.assertRaises(ValueError):commit_append_journal(journal,[],{'run':'another'})
            b.write_bytes(b'other-edit')
            with self.assertRaises(ValueError):commit_append_journal(journal,[],{'run':'fixture'})

    def test_context_block_retains_unavailable_values_and_two_rows(self):
        from scripts.rep1_bt_gt_fusion import NEW,PARENTS
        names=list(NEW)+list(PARENTS.values());rows=[]
        for n in names:rows.append(dict(method=n,scclassify_positive_F1='NA',scclassify_negative_F1='0',fixed_positive_F1='.4',fixed_negative_F1='.03',capture_pct='62'))
        text=context_block(dict(run_root='/fixture',run_id='fixture'),rows,{'unique_mask_contents':29},
                           dict(selection_summary='Negative results retained.',local_RNA_explanation='Unresolved local cost.')).decode()
        self.assertIn('NA（原值不可用）',text)
        self.assertIn('| P2_BASE_GT |',text);self.assertIn('| V2_MORPH_GT |',text)
        self.assertLess(text.index('| V2_MORPH_GT |'),text.index('相对'))

    def test_checkpoint_witness_excludes_revalidated_work(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg=dict(run_root=tmp,run_id='fixture',source_digest='x',input_digest='x',plan_sha256='x')
            folder=Path(tmp)/'endpoints';folder.mkdir();data=folder/'data';data.write_text('scientific-output')
            p=folder/'commit.json';atomic_json(p,dict(status='succeeded',dependencies=dependencies(cfg,'fixture'),outputs={'data':identity(data)}))
            baseline=rt.checkpoint_baseline(cfg);self.assertIsNone(rt.new_checkpoint(cfg,baseline))
            data2=folder/'data2';data2.write_text('new-output')
            p2=folder/'commit2.json';atomic_json(p2,dict(status='succeeded',dependencies=dependencies(cfg,'fixture'),outputs={'data':identity(data2)}))
            self.assertEqual(rt.new_checkpoint(cfg,baseline)['path'],str(p2));self.assertIsNone(rt.new_checkpoint(cfg,baseline))


if __name__=='__main__':unittest.main()
