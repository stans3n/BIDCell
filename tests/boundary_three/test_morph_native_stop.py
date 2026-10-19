"""Exercise actual native tiles/splits, comparing original code with stop/resume."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
import numpy as np
import tifffile
from scripts.crgf.postprocess import postprocess

class NativeStopTest(unittest.TestCase):
    def test_original_final_exact_after_tile_and_split_stops(self):
        base=Path('/home/zwan0726/work/outputs/bidcell_boundary_three_20261015/mini/morph/unit_tmp')
        base.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='native-stop-',dir=base) as tmp:
            root=Path(tmp);shared=root/'shared';shared.mkdir()
            nuclei=np.zeros((48,48),np.uint32)
            nuclei[5:8,5:8]=1;nuclei[24:27,22:25]=2;nuclei[40:43,40:43]=3
            pre=np.zeros_like(nuclei);pre[:20,:24]=1;pre[10:36,16:35]=2;pre[28:,28:]=3
            tifffile.imwrite(shared/'nuclei.tif',nuclei)
            def cfg(name,workers=1):
                out=root/name;(out/'predictions/X').mkdir(parents=True);tifffile.imwrite(out/'predictions/X/pre.tif',pre)
                return SimpleNamespace(output_root=str(out),seed=1234,postprocess_size=16,cpu_workers=workers,
                    native=lambda:SimpleNamespace(files=SimpleNamespace(data_dir=str(shared))))
            original_path=Path('/home/zwan0726/work/projects/BIDCell_v2_paired_20261014/scripts/crgf/postprocess.py')
            spec=importlib.util.spec_from_file_location('scripts.crgf.original_native_stop_oracle',original_path)
            original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original)
            # Fork workers need the importable module registered in the parent.
            import sys
            sys.modules[spec.name]=original
            old=cfg('original');original.postprocess(old,'X')
            new=cfg('resumed',2);status={}
            new.progress_callback=lambda **values:status.update(values)
            new.stop_requested=lambda:status.get('stage')=='tiles' and status.get('completed',0)>=2
            with self.assertRaises(InterruptedError):postprocess(new,'X')
            out=Path(new.output_root)/'predictions/X';receipts=list((out/'postprocess').glob('tile_*.json'))
            self.assertEqual(len(receipts),2);self.assertFalse((out/'final.tif').exists());self.assertFalse((out/'postprocess/complete.json').exists())
            new.stop_requested=lambda:status.get('stage')=='splits' and status.get('completed',0)>0
            with self.assertRaises(InterruptedError):postprocess(new,'X')
            self.assertTrue(list((out/'postprocess/splits').glob('*.json')));self.assertFalse((out/'final.tif').exists())
            new.stop_requested=lambda:False;postprocess(new,'X')
            np.testing.assert_array_equal(tifffile.imread(out/'final.tif'),tifffile.imread(Path(old.output_root)/'predictions/X/final.tif'))
            self.assertTrue((out/'postprocess/complete.json').is_file())

if __name__=='__main__':unittest.main()
