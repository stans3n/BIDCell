"""Independent native and exhaustive complete-nucleus oracle tests for P2."""
import random
import unittest
import numpy as np
from bidcell.model.postprocess_predictions import postprocess_connect,remove_islands
from scripts.boundary_three.p2_rules import connect,FullNucleusIndex
from scripts.boundary_three.p2_native import native_connect,tile_coordinates


class P2Tests(unittest.TestCase):
    def test_current_native_and_full_coordinate_oracle(self):
        rng = np.random.default_rng(9817)
        changed = ties = 0
        for seed in range(24):
            nuclei = np.zeros((44,53),np.uint32)
            for cid in (3,9,17):
                for _ in range(5):
                    nuclei[rng.integers(44),rng.integers(53)] = cid
            image = rng.choice([0,0,0,3,9,17],size=(24,29)).astype(np.uint32)
            origin = (7,11)
            nuc = nuclei[7:31,11:40]
            index = FullNucleusIndex(nuclei,max_pairs=7)
            random.seed(seed)
            expected_native = postprocess_connect(image,nuc)
            state = random.getstate()
            expected_islands = remove_islands(expected_native,nuc)
            native_exit = random.getstate()
            random.seed(seed)
            observed,observed_diag,observed_arrays = native_connect(image,nuc)
            self.assertTrue(np.array_equal(observed,expected_native))
            self.assertEqual(state,random.getstate())
            self.assertTrue(np.array_equal(remove_islands(observed,nuc),expected_islands))
            self.assertEqual(native_exit,random.getstate())
            random.seed(seed)
            disabled,diag,arrays,_ = connect(image,nuc,'B',index,origin)
            self.assertTrue(np.array_equal(disabled,expected_native))
            self.assertEqual(state,random.getstate())
            random.seed(seed)
            p2,diag,arrays,masks = connect(image,nuc,'P2',index,origin,capture=True)
            self.assertEqual(state,random.getstate())
            self.assertEqual(diag['candidate_hashes'],observed_diag['candidate_hashes'])
            self.assertEqual(diag['candidate_order'],observed_diag['candidate_order'])
            self.assertTrue(np.array_equal(arrays['proposal_count'],observed_arrays['proposal_count']))
            expected = np.zeros_like(image)
            for y,x in np.ndindex(image.shape):
                candidates = [cid for cid,m in masks.items() if m[y,x]]
                if candidates:
                    distances = {cid:int(((np.argwhere(nuclei==cid)-[y+origin[0],x+origin[1]])**2).sum(1).min()) for cid in candidates}
                    expected[y,x] = min(candidates,key=lambda cid:(distances[cid],cid))
                    ordered = sorted((d,cid) for cid,d in distances.items())
                    ties += len(ordered)>1 and ordered[0][0]==ordered[1][0]
            expected = np.where(nuc>0,nuc,expected)
            self.assertTrue(np.array_equal(p2,expected))
            self.assertLessEqual(index.max_observed_pairs,7)
            self.assertTrue(np.array_equal(p2>0,expected_native>0))
            self.assertFalse(np.any((p2!=expected_native)&~((nuc==0)&(arrays['proposal_count']>=2))))
            changed += diag['changed_vs_B']
        self.assertGreater(changed,0)
        self.assertGreater(ties,0)

    def test_native_no_background_drops_first_sorted_id(self):
        image = np.full((12,15),2,np.uint32)
        image[:,7:] = 9
        nuclei = np.zeros_like(image)
        nuclei[4,3],nuclei[4,10] = 2,9
        random.seed(17)
        expected = postprocess_connect(image,nuclei)
        state = random.getstate()
        for rule in ('B','P2'):
            random.seed(17)
            actual,diag,_,_ = connect(image,nuclei,rule,FullNucleusIndex(nuclei))
            self.assertEqual(diag['candidate_order'],[9])
            self.assertEqual(random.getstate(),state)
            self.assertTrue(np.array_equal(actual,expected))

    def test_exact_complete_multicomponent_distance_and_near_tie(self):
        n = np.zeros((300,400),np.uint32)
        n[3,2],n[3,4],n[299,399] = 7,2,7
        index = FullNucleusIndex(n,3)
        self.assertEqual(index.squared(7,[[3,3]])[0],1)
        self.assertEqual(index.squared(2,[[3,3]])[0],1)
        self.assertEqual(index.squared(7,[[298,399]])[0],1)
        self.assertNotEqual(index.squared(7,[[3,2]])[0],index.squared(2,[[3,2]])[0])
        self.assertEqual(index.squared(7,[[200,300]]).dtype,np.int64)
        self.assertLessEqual(index.max_observed_pairs,3)
        with self.assertRaisesRegex(ValueError,'missing complete nucleus'):
            connect(np.array([[0,99],[2,2]],np.uint32),n[:2,:2],'P2',index)
        with self.assertRaisesRegex(ValueError,'outside global grid'):
            index.squared(7,[[-1,3]])

    def test_native_tile_order_overlap(self):
        size,ys,xs,coords = tile_coordinates((140,190),96)
        self.assertEqual(size,96)
        self.assertEqual(ys,[0,44])
        self.assertEqual(xs,[0,94])
        self.assertEqual(coords,[(0,0),(0,94),(44,0),(44,94)])


if __name__ == '__main__':
    unittest.main()
