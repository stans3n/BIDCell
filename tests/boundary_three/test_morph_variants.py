import copy
import unittest
import tempfile
from pathlib import Path
import torch
from bidcell.model.model.morphology_variants import *

class MorphVariantsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2);cls.full=canonical_full()
    def test_reconstruction_and_mapping(self):
        self.assertEqual(tensor_sha(self.full.state_dict()),CANONICAL_SHA)
        before=torch.get_rng_state()
        for variant in ('no_dapi','no_geo','simple_fusion'):
            m=MorphologyVariant(variant,self.full);rows=initialization_map(m,self.full)
            self.assertTrue(torch.equal(before,torch.get_rng_state()))
            self.assertEqual(hasattr(m,'geometry'),VARIANTS[variant]['use_geometry'])
            self.assertEqual(hasattr(m,'image'),VARIANTS[variant]['use_dapi'])
            self.assertGreater(len(rows),20)
            if variant=='simple_fusion':
                self.assertFalse(torch.equal(m.blocks[0].context[0].weight,m.blocks[1].context[0].weight))
                self.assertEqual(m.blocks[0].context[3].groups,64)
    def test_zero_projection_and_active_learned_branches(self):
        h=torch.randn(2,256,12,12);g=torch.randn(2,16,48,48);d=torch.randn(1,3,48,48)
        for variant in ('no_dapi','no_geo','simple_fusion'):
            m=MorphologyVariant(variant,self.full);spec=VARIANTS[variant]
            args=(h,g if spec['use_geometry'] else None,d if spec['use_dapi'] else None)
            self.assertTrue(torch.equal(m(*args),h))
            opt=torch.optim.Adam(m.parameters(),lr=1e-5,weight_decay=1e-4)
            for _ in range(2):
                opt.zero_grad();(m(*args).square().mean()).backward();opt.step()
            self.assertFalse(torch.equal(m(*args),h))
            for name in ('geometry','image','memory','expression','blocks','output'):
                if hasattr(m,name):self.assertTrue(any(p.grad is not None and torch.count_nonzero(p.grad)>0 for p in getattr(m,name).parameters()),name)
    def test_variant_shape_and_checkpoint_rejection(self):
        a=MorphologyVariant('no_dapi',self.full);b=MorphologyVariant('no_geo',self.full)
        with self.assertRaises(RuntimeError):b.load_state_dict(a.state_dict(),strict=True)
        with self.assertRaises(ValueError):a(torch.randn(1,256,12,12),torch.randn(1,16,48,48),torch.randn(1,3,48,48))
    def test_checkpoint_metadata_and_pilot_rejection_precedes_restore(self):
        from scripts.boundary_three.morph import restore
        with tempfile.TemporaryDirectory(prefix='morph-contract-') as tmp:
            path=Path(tmp)/'state.pth'
            torch.save(dict(header={'morph_variant':'no_dapi'},pilot=True),path)
            with self.assertRaisesRegex(ValueError,'variant/source/input/pilot'):
                restore({},path,None,None,{'morph_variant':'no_geo'},pilot=True)
            with self.assertRaisesRegex(ValueError,'variant/source/input/pilot'):
                restore({},path,None,None,{'morph_variant':'no_dapi'},pilot=False)
    def test_cpu_uninterrupted_resume_model_optimizer_rng_exact(self):
        for variant in ('no_dapi','no_geo','simple_fusion'):
            m=MorphologyVariant(variant,self.full);o=torch.optim.Adam(m.parameters(),lr=1e-5,weight_decay=1e-4)
            h=torch.randn(2,256,12,12);g=torch.randn(2,16,48,48) if VARIANTS[variant]['use_geometry'] else None
            d=torch.randn(1,3,48,48) if VARIANTS[variant]['use_dapi'] else None
            def step():o.zero_grad();m(h,g,d).square().mean().backward();o.step()
            step();saved=copy.deepcopy((m.state_dict(),o.state_dict(),torch.get_rng_state()));step()
            expected=copy.deepcopy((m.state_dict(),o.state_dict(),torch.get_rng_state()))
            m=MorphologyVariant(variant,self.full);o=torch.optim.Adam(m.parameters(),lr=1e-5,weight_decay=1e-4)
            m.load_state_dict(saved[0]);o.load_state_dict(saved[1]);torch.set_rng_state(saved[2]);step()
            from scripts.paired_v2.common import state_equal
            self.assertTrue(state_equal(expected,(m.state_dict(),o.state_dict(),torch.get_rng_state())))

if __name__=='__main__':unittest.main()
