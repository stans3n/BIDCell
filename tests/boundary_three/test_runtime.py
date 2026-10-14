import errno
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import subprocess
import contextlib
import io
from scripts.boundary_three import runtime
from scripts.boundary_three.common import atomic_json,digest,identity,read,write_complete
from scripts.boundary_three.contracts import OUTPUT_ROOT,NODES
from scripts.boundary_three.settings import default_config


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='unit-runtime-',dir=OUTPUT_ROOT/'mini')
        self.root=Path(self.temp.name);self.cfg=default_config(self.root,self.root.name)
        self.cfg.update(source_digest='test-only',input_digest='test-only',budget_seconds=18*3600)
        evidence=self.root/'estimate.json';atomic_json(evidence,dict(status='fixture'))
        self.cfg['node_estimates']={n:dict(seconds=19800,retained_bytes=1024,atomic_extra_bytes=512,
            peak_ram_bytes=20*1024**3,peak_gpu_bytes=0,evidence=[identity(evidence)]) for n in NODES}
        self.cfg['node_estimates']['A1_CORE']['seconds']=3600
        self.cfg['node_estimates']['P2_BASE']['seconds']=22601.96
        self.cfg['node_estimates']['A2_CORE']['seconds']=85352
        runtime.save_state(self.cfg,runtime.new_state(self.cfg))
        self.compute_patch=patch.object(runtime,'semantic_lock',side_effect=lambda cfg:contextlib.nullcontext())
        self.compute_patch.start()
    def tearDown(self):self.compute_patch.stop();self.temp.cleanup()

    def test_real_budget_minimum_and_lease_disk_exclusion(self):
        forecast=runtime.budget_forecast(self.cfg)
        self.assertAlmostEqual(forecast['minimum_required_seconds'],59302.45)
        self.assertEqual(runtime.budget_forecast(self.cfg,8*3600)['status'],'NOT_ADMITTED_BUDGET')
        disk=runtime.disk_forecast(self.cfg)
        self.assertEqual(set(disk['time_ineligible_exclusions']),{'A2_CORE'})
        self.assertIn('A2_CORE',runtime.disk_forecast(self.cfg,lease_seconds=36*3600)['remaining_retained_bytes'])

    def test_process_identity_pid_reuse_and_boot(self):
        saved=runtime.process_identity();self.assertTrue(runtime.process_alive(saved))
        saved['start_ticks']+=1;self.assertFalse(runtime.process_alive(saved))
        saved=runtime.process_identity();saved['boot_id']='other-boot';self.assertFalse(runtime.process_alive(saved))

    def test_live_without_progress_and_incomplete_receipt(self):
        state=runtime.bound_state(self.cfg);state['supervisor']=runtime.process_identity();runtime.save_state(self.cfg,state)
        path=self.root/'configs/run.json';atomic_json(path,self.cfg)
        atomic_json(self.root/'A1_CORE/complete.json',dict(execution_status='RUNNING'))
        result=runtime.status(path)
        self.assertEqual(result['liveness'],'LIVE_NO_PROGRESS')
        self.assertEqual(result['nodes']['A1_CORE']['receipt_status']['status'],'INCOMPLETE_OR_WRONG_IDENTITY')

    def test_final_cli_refuses_fixture_and_null_freeze(self):
        cfg=dict(self.cfg);cfg['engineering_fixture']=True
        with self.assertRaisesRegex(ValueError,'fixture'):runtime.check_run_config(cfg)
        cfg=dict(self.cfg);cfg['source_digest']=None
        with self.assertRaisesRegex(ValueError,'digests'):runtime.check_run_config(cfg)

    def test_safe_stop_runtimeerror_is_controlled_but_other_runtimeerror_fails(self):
        SafeStop=type('SafeStop',(RuntimeError,),{'__module__':'scripts.boundary_three.a_core'})
        for exc,status in [(SafeStop('boundary'),'CONTROLLED_PAUSE'),(RuntimeError('OOM'),'FAILED_RUNTIME'),
                           (ValueError('hash drift'),'FAILED_INTEGRITY'),(InterruptedError('boundary'),'CONTROLLED_PAUSE')]:
            module=SimpleNamespace(run_node=lambda *a,e=exc:(_ for _ in ()).throw(e))
            result=self.root/'result.json'
            with patch.object(runtime,'node_module',return_value=module),patch.object(runtime,'verify_worker_authorization'),contextlib.redirect_stderr(io.StringIO()):
                runtime.node_worker(self.cfg,'A1_CORE',result)
            self.assertEqual(read(result)['status'],status)

    def test_stop_and_progress_never_mutate_config_identity(self):
        before=digest(self.cfg);runtime.request_stop(self.cfg,'USER','fixture request')
        self.assertEqual(runtime.stop_status(self.cfg),'PAUSED_USER')
        state=runtime.bound_state(self.cfg);state['current_node']='A1_CORE'
        atomic_json(self.root/'A1_CORE/progress.json',dict(updated_utc=runtime.utc(),stage='count',chunk=2))
        runtime._heartbeat(self.cfg,state)
        self.assertEqual(state['stage'],'count');self.assertEqual(digest(self.cfg),before)

    def test_reserve_monitor_does_not_add_peak_twice(self):
        measured=dict(ram_available_bytes=7*1024**3,gpu_free_bytes=2*1024**3,disk_free_bytes=80*1024**3)
        self.assertEqual(runtime.resource_reasons(self.cfg,measured),[])
        self.assertIn('available RAM below requirement',runtime.resource_reasons(self.cfg,measured,'A1_CORE'))

    def test_complete_check_needs_semantic_validator_and_valid_skip(self):
        state=runtime.bound_state(self.cfg);state['nodes']['A2_CORE']['execution_status']='SKIPPED_SCIENTIFIC_GATE';runtime.save_state(self.cfg,state)
        path=self.root/'configs/run.json';atomic_json(path,self.cfg)
        atomic_json(self.root/'A1_CORE/complete.json',dict(fake=True))
        module=SimpleNamespace(validate_node=lambda *a:(_ for _ in ()).throw(ValueError('missing own matrix')))
        with patch.object(runtime,'node_module',return_value=module):result=runtime.complete_check(path)
        self.assertFalse(result['minimum_three_complete']);self.assertFalse(result['all_enabled_applicable_complete'])
        self.assertEqual(result['nodes']['A1_CORE']['status'],'INVALID_OR_INCOMPLETE')
        self.assertEqual(result['nodes']['A2_CORE']['status'],'INVALID_SCIENTIFIC_SKIP')

    def test_duplicate_lock_refused_and_inherited_lock_retained(self):
        with runtime.lock_file(self.root/'queue.lock'):
            with self.assertRaisesRegex(RuntimeError,'lock held'):
                with runtime.lock_file(self.root/'queue.lock'):pass

    def test_internal_node_cannot_run_without_supervisor_locks(self):
        with patch.object(runtime,'validated_launch'):
            with self.assertRaisesRegex(ValueError,'inherited'):
                runtime.node_worker(self.cfg,'A1_CORE',self.root/'result.json')

    def test_orphan_inherited_open_description_keeps_queue_locked(self):
        with runtime.lock_file(self.root/'orphan.lock') as stream:
            child=subprocess.Popen([self.cfg['python'],'-c','import time;time.sleep(30)'],pass_fds=(stream.fileno(),))
        try:
            with self.assertRaisesRegex(RuntimeError,'lock held'):
                with runtime.lock_file(self.root/'orphan.lock'):pass
        finally:child.terminate();child.wait(timeout=10)
        with runtime.lock_file(self.root/'orphan.lock'):pass

    def _queue(self,science='NO_USEFUL_INCREMENT',failure=None,retry_errno=None):
        cfgpath=self.root/'configs/run.json';atomic_json(cfgpath,self.cfg)
        lease=dict(lease_id='unit-lease',created_utc=runtime.utc(),budget_seconds=self.cfg['budget_seconds'],resume=False,config_digest=digest(self.cfg))
        state=runtime.bound_state(self.cfg);state.update(active_lease='unit-lease',leases=[lease],execution_status='STARTING')
        runtime.save_state(self.cfg,state);atomic_json(self.root/'control/leases/unit-lease.json',lease)
        started=[]
        def worker(cfg,config,state,node,*args):
            started.append(node)
            if node==failure:return dict(status='FAILED_RUNTIME',reason='nonfinite/OOM fixture',elapsed_seconds=0,errno=retry_errno)
            artifact=self.root/node/'actual_fixture.json';atomic_json(artifact,dict(fixture=True))
            resultscience=science if node=='A1_CORE' else 'tradeoff_or_negative'
            write_complete(cfg,node,resultscience)
            return dict(status='COMPLETED',science_status=resultscience,elapsed_seconds=0,receipt=identity(self.root/node/'complete.json'))
        from scripts.boundary_three.common import validate_complete
        module=SimpleNamespace(validate_node=validate_complete)
        with patch.object(runtime,'validated_launch'),patch.object(runtime,'verify_source'), \
             patch('scripts.boundary_three.provenance.verify_inputs'),patch('scripts.boundary_three.provenance.environment_check'), \
             patch.object(runtime,'_wait_compute',side_effect=lambda *a:(self.root/'compute.lock').open('a')), \
             patch.object(runtime,'_wait_resources',return_value=True),patch.object(runtime,'_run_worker',side_effect=worker), \
             patch.object(runtime,'checkpoint_evidence',return_value=dict(status='verified_fixture_unit')), \
             patch.object(runtime,'node_module',return_value=module):
            runtime.run_queue(cfgpath,'unit-lease')
        return started,runtime.bound_state(self.cfg)

    def test_negative_science_continues_and_a2_skip_is_separate(self):
        started,state=self._queue()
        self.assertEqual(started,[n for n in NODES if n!='A2_CORE'])
        self.assertEqual(state['nodes']['A2_CORE']['execution_status'],'SKIPPED_SCIENTIFIC_GATE')
        self.assertEqual(state['execution_status'],'FINISHED')
        self.assertTrue(read(self.root/'complete-check.json')['all_enabled_applicable_complete'])

    def test_positive_a1_a2_full_chain_budget_deferred_others_continue(self):
        started,state=self._queue('GO_A2')
        self.assertEqual(started,[n for n in NODES if n!='A2_CORE'])
        self.assertEqual(state['nodes']['A2_CORE']['execution_status'],'DEFERRED_TIME')
        self.assertEqual(state['execution_status'],'FINISHED_WITH_DEFERRALS')
        self.assertFalse(read(self.root/'complete-check.json')['all_enabled_applicable_complete'])

    def test_engineering_failure_stops_without_negative_or_later_launch(self):
        started,state=self._queue(failure='P2_BASE')
        self.assertEqual(started,['A1_CORE','P2_BASE'])
        self.assertEqual(state['nodes']['P2_BASE']['execution_status'],'FAILED_RUNTIME')
        self.assertIsNone(state['nodes']['P2_BASE']['science_status'])
        self.assertEqual(state['nodes']['MORPH_NO_DAPI']['execution_status'],'PREPARED')

    def test_only_one_verified_transient_io_retry_and_then_failure(self):
        started,state=self._queue(failure='P2_BASE',retry_errno=errno.EIO)
        self.assertEqual(started,['A1_CORE','P2_BASE','P2_BASE'])
        self.assertEqual(state['nodes']['P2_BASE']['retries'],1)
        self.assertEqual(state['nodes']['P2_BASE']['execution_status'],'FAILED_RUNTIME')

    def test_same_execution_lease_cannot_renew_deadline(self):
        self._queue()
        with patch.object(runtime,'validated_launch'):
            with self.assertRaisesRegex(ValueError,'newly dispatched'):
                runtime.run_queue(self.root/'configs/run.json','unit-lease')

    def test_external_full_validation_does_not_overlap_queue(self):
        path=self.root/'configs/run.json';atomic_json(path,self.cfg)
        with runtime.lock_file(self.root/'control/queue.lock'):
            with self.assertRaisesRegex(RuntimeError,'lock held'):runtime.complete_check(path)

    def test_package_preserves_scientific_table_set_and_never_creates_a1_mask_row(self):
        self._queue()
        state=runtime.bound_state(self.cfg);node='P2_BASE';folder=self.root/node/'comparison'
        atomic_json(folder/'complete.json',dict(endpoint_ids=['BASE_MATCH','P2_BASE']))
        atomic_json(folder/'purity_scaling.json',dict(version='specific-set-v1',comparison_set=['BASE_MATCH','P2_BASE']))
        (folder/'endpoint_table.csv').write_text('method,scclassify_positive_F1,scclassify_negative_F1\nP2_BASE,0.012,0.0003\n')
        write_complete(self.cfg,node,'tradeoff_or_negative');state['nodes'][node]['receipt']=identity(self.root/node/'complete.json')
        runtime.package(self.cfg,state)
        with (self.root/'reports/endpoint_table.csv').open() as stream:rows=list(runtime.csv.DictReader(stream))
        actual=next(r for r in rows if r['method']=='P2_BASE')
        self.assertEqual(actual['scclassify_positive_F1'],'0.012')
        self.assertEqual(actual['purity_scaling_version'],'specific-set-v1')
        self.assertNotIn('A1_CORE',[r['method'] for r in rows])


if __name__=='__main__':unittest.main()
