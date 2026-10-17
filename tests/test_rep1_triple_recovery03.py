"""Actual accounting regression and generation/reviewer contracts."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import os,time,unittest
from scripts.rep1_triple_simple_boundary import runtime as rt,execution as ex
from scripts.rep1_triple_simple_boundary.common import atomic_json,identity


class Recovery03Resource(unittest.TestCase):
    def setUp(self):
        base=Path(os.environ['REP1_TRIPLE_RECOVERY03_FIXTURE_ROOT'])
        self.root=base/self._testMethodName;self.root.mkdir(parents=True,exist_ok=False)
        self.cfg=dict(run_root=str(self.root),resources=rt.resource_contract())
        self.required=sum(self.cfg['resources']['retained_bytes'].values())+self.cfg['resources']['shared_bytes']+self.cfg['resources']['temporary_peak_bytes']+self.cfg['resources']['reserve_bytes']
    def install(self):
        p=self.root/'endpoints/P2_SIMPLE/predictions/committed.tif';p.parent.mkdir(parents=True);p.write_bytes(b'x'*65536)
        return p.stat().st_blocks*512
    def test_cached_false_disk_failure_refreshes_actual_committed_allocation_once(self):
        available=[self.required+4096]
        with patch.object(rt.shutil,'disk_usage',side_effect=lambda p:SimpleNamespace(free=available[0])),patch.object(rt,'memory_available',return_value=40*rt.GIB),patch.object(rt,'pss',return_value={'pss_bytes':rt.GIB}),patch.object(rt,'forecast',wraps=rt.forecast) as forecast:
            guard=rt.ResourceGuard(self.cfg);first=guard(True);growth=self.install();available[0]-=growth
            guard.last=time.monotonic();forecast.reset_mock();fresh=guard()
            self.assertLess(available[0],first['required_free_bytes']);self.assertTrue(fresh['admitted'])
            self.assertEqual(first['required_free_bytes']-fresh['required_free_bytes'],growth)
            self.assertEqual(fresh['used_allocated_unique_bytes']['p2_replay'],growth)
            forecast.assert_called_once_with(self.cfg,False)
            atomic_json(self.root/'accounting.json',dict(cached=first,fresh=fresh,growth_bytes=growth))
    def test_true_fresh_disk_shortage_still_rejects_without_loop(self):
        available=[self.required+4096]
        with patch.object(rt.shutil,'disk_usage',side_effect=lambda p:SimpleNamespace(free=available[0])),patch.object(rt,'memory_available',return_value=40*rt.GIB),patch.object(rt,'pss',return_value={'pss_bytes':rt.GIB}),patch.object(rt,'forecast',wraps=rt.forecast) as forecast:
            guard=rt.ResourceGuard(self.cfg);guard(True);growth=self.install();available[0]-=growth+8192
            guard.last=time.monotonic();forecast.reset_mock()
            with self.assertRaisesRegex(RuntimeError,'RESOURCE_ADMISSION_FAILED'):guard()
            forecast.assert_called_once_with(self.cfg,False);self.assertFalse(guard.saved['admitted'])
            atomic_json(self.root/'true_shortage.json',guard.saved)
    def test_cached_memory_shortage_still_rejects_after_one_fresh_forecast(self):
        memory=[40*rt.GIB]
        with patch.object(rt.shutil,'disk_usage',return_value=SimpleNamespace(free=self.required+4096)),patch.object(rt,'memory_available',side_effect=lambda:memory[0]),patch.object(rt,'pss',return_value={'pss_bytes':rt.GIB}),patch.object(rt,'forecast',wraps=rt.forecast) as forecast:
            guard=rt.ResourceGuard(self.cfg);guard(True);guard.last=time.monotonic();memory[0]=5*rt.GIB;forecast.reset_mock()
            with self.assertRaisesRegex(RuntimeError,'RESOURCE_ADMISSION_FAILED'):guard()
            forecast.assert_called_once_with(self.cfg,False);self.assertFalse(guard.saved['admitted'])
    def test_stage_start_keeps_original_combined_RAM_semantics_no_retry(self):
        with patch.object(rt.shutil,'disk_usage',return_value=SimpleNamespace(free=self.required+4096)),patch.object(rt,'memory_available',return_value=26*rt.GIB),patch.object(rt,'pss',return_value={'pss_bytes':rt.GIB}),patch.object(rt,'forecast',wraps=rt.forecast) as forecast:
            guard=rt.ResourceGuard(self.cfg)
            with self.assertRaisesRegex(RuntimeError,'RESOURCE_ADMISSION_FAILED'):guard(True)
            forecast.assert_called_once_with(self.cfg,True)
            self.assertTrue(guard.saved['stage_start']);self.assertEqual(guard.saved['reserve_bytes'],14*rt.GIB)
    def test_cached_success_does_not_force_full_scan(self):
        with patch.object(rt.shutil,'disk_usage',return_value=SimpleNamespace(free=self.required+4096)),patch.object(rt,'memory_available',return_value=40*rt.GIB),patch.object(rt,'pss',return_value={'pss_bytes':rt.GIB}),patch.object(rt,'forecast',wraps=rt.forecast) as forecast:
            guard=rt.ResourceGuard(self.cfg);guard(True);guard.last=time.monotonic();forecast.reset_mock()
            self.assertTrue(guard()['admitted']);forecast.assert_not_called()
    def test_generation3_reviewer_is_exact_and_previous_reviewers_preserved(self):
        self.assertEqual(ex.reviewer({'continuation_request':{'p2_validation':True}})['agent_id'],'/root/recovery03_auditor')
        for generation,request in [(1,{'donor_validation':True}),(2,{'replay_validation':True})]:
            cfg={'continuation_request':request};self.assertEqual(ex.recovery_generation(cfg),generation)
            self.assertEqual(ex.reviewer(cfg)['agent_id'],'/root/independent_code_auditor')
        self.assertEqual(ex.audit_path(3),ex.RECOVERY3_AUDIT)
        budget=rt.resource_contract();self.assertEqual(budget['ram_allowance_bytes']+budget['ram_reserve_bytes'],28*rt.GIB)
        self.assertEqual(budget['temporary_peak_bytes'],4*rt.GIB);self.assertEqual(budget['retained_bytes']['analysis'],15*rt.GIB)


if __name__=='__main__':unittest.main()
