"""Numerical adversarial checks for genuine full-index stage06 validation."""
import copy,os,random,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pandas as pd
from scripts.boundary_three.p2_rules import FullNucleusIndex
from scripts.rep1_triple_simple_boundary import nucleus_validation as nv,inputs,p2,runtime as rt
from scripts.rep1_triple_simple_boundary.common import (atomic_json,atomic_npz,atomic_tiff,
    dependencies,digest,identity,read)


class Recovery02Numerics(unittest.TestCase):
    def setUp(self):
        retained=os.environ.get('REP1_TRIPLE_RECOVERY02_FIXTURE_ROOT')
        self.tmp=None
        if retained:
            self.root=Path(retained)/self._testMethodName;self.root.mkdir(parents=True,exist_ok=False)
        else:self.tmp=tempfile.TemporaryDirectory(prefix='triple-recovery02-',dir='/tmp');self.root=Path(self.tmp.name)
    def tearDown(self):
        if self.tmp:self.tmp.cleanup()
    def raw(self):
        return np.array([[0,2,2,0,0],[0,2,0,7,0],[7,0,0,7,7],[0,0,0,0,0]],np.uint32)
    def fixture(self,name):
        root=self.root/name;root.mkdir();(root/'frozen_domain').mkdir()
        raw=self.raw();atomic_tiff(root/'nuclei.tif',raw)
        pd.DataFrame(dict(cell_id=[2,7])).to_csv(root/'frozen_domain/cells.csv',index=False)
        cfg=dict(run_id=name,run_root=str(root),source_digest='isolated_source',input_digest='isolated_input',
                 domain_digest='isolated_domain',plan_digest='isolated_plan')
        manifest=dict(endpoints={'MORPH_SIMPLE_FUSION':{'nuclei':str(root/'nuclei.tif')}})
        dep=dict(rule='P2',run=dependencies(cfg,'06'),full_nuclei=identity(root/'nuclei.tif'),
                 source={'p2_rules.py':identity(Path(nv.__file__).parents[1]/'boundary_three/p2_rules.py')})
        out=root/'endpoints/P2_SIMPLE/predictions/postprocess';out.mkdir(parents=True)
        events=[]
        with patch.object(nv,'EXPECTED_CELLS',2):
            receipt=nv.commit(FullNucleusIndex(raw),out,dep,nv.contract(cfg),events.append)
        return cfg,manifest,receipt,events
    def signed_payload(self,path,change):
        rec=read(path);dest=Path(rec['outputs']['numerical_payload']['path'])
        with np.load(dest) as z:arrays={k:z[k] for k in z.files}
        change(arrays);atomic_npz(dest,**arrays);rec['outputs']['numerical_payload']=identity(dest)
        rec['integrity']=digest({k:v for k,v in rec.items() if k!='integrity'});atomic_json(path,rec)
    def witness(self,cfg,manifest,path,stage='06'):
        with patch.object(nv,'EXPECTED_CELLS',2),patch.object(inputs,'load_inputs',return_value=manifest):
            return rt.witness(cfg,stage,path)
    def test_independent_counts_boxes_and_complete_coordinate_membership(self):
        raw=self.raw();index=FullNucleusIndex(raw);before=random.getstate()
        p=nv.actual_index_payload(index,np.array([2,7],np.int64))
        np.testing.assert_array_equal(p['counts'],[3,4])
        np.testing.assert_array_equal(p['bboxes'],[[0,1,2,3],[1,0,3,5]])
        np.testing.assert_array_equal(p['maximum_squared_distance'],[25])
        self.assertEqual(random.getstate(),before);self.assertEqual(index.max_observed_pairs,0)
        index.boxes[1]=(slice(0,1),slice(1,3))
        with self.assertRaisesRegex(ValueError,'bbox'):nv.actual_index_payload(index,np.array([2,7],np.int64))
        index=FullNucleusIndex(raw);original=index.coordinates
        index.coordinates=lambda cid:np.repeat(original(cid)[:1],len(original(cid)),axis=0)
        with self.assertRaisesRegex(ValueError,'duplicated'):nv.actual_index_payload(index,np.array([2,7],np.int64))
        with self.assertRaisesRegex(ValueError,'ID'):nv.raster_payload(raw,np.array([2],np.int64))
    def test_resigned_wrong_numerical_payloads_are_rejected_by_actual_witness(self):
        mutations={'count':lambda a:a['counts'].__setitem__(0,4),
            'bbox':lambda a:a['bboxes'].__setitem__((0,0),1),
            'shape':lambda a:a['shape'].__setitem__(0,5),
            'distance_range':lambda a:a['maximum_squared_distance'].__setitem__(0,24),
            'missing_id':lambda a:a.__setitem__('ids',a['ids'][:1]),
            'wrong_dtype':lambda a:a.__setitem__('counts',a['counts'].astype(np.uint64)),
            'extra_field':lambda a:a.__setitem__('invented',np.array([1],np.int64))}
        for name,mutation in mutations.items():
            cfg,manifest,path,_=self.fixture(name);self.witness(cfg,manifest,path)
            self.signed_payload(path,mutation)
            with self.assertRaisesRegex(ValueError,'global index numerical'):self.witness(cfg,manifest,path)
    def test_wrong_producer_stage_source_and_kind_cannot_bypass_numeric_witness(self):
        for name,field,value in [('run','run_id','another'),('config','config_digest','bad'),('stage','stage','P2_SIMPLE_POSTPROCESS'),('source','source_digest','bad')]:
            cfg,manifest,path,_=self.fixture(name);rec=read(path);rec['dependencies']['run'][field]=value
            rec['integrity']=digest({k:v for k,v in rec.items() if k!='integrity'});atomic_json(path,rec)
            with self.assertRaises(ValueError):self.witness(cfg,manifest,path)
        cfg,manifest,path,_=self.fixture('wrong_called_stage')
        with self.assertRaisesRegex(ValueError,'stage'):self.witness(cfg,manifest,path,'05')
        cfg,manifest,path,_=self.fixture('missing_kind');rec=read(path);del rec['kind']
        rec['integrity']=digest({k:v for k,v in rec.items() if k!='integrity'});atomic_json(path,rec)
        with self.assertRaisesRegex(ValueError,'receipt'):self.witness(cfg,manifest,path)
    def test_existing_validation_is_not_fresh_and_malformed_population_fails(self):
        cfg,manifest,path,events=self.fixture('resume');before=identity(path);self.assertEqual(len(events),1)
        dep=dict(rule='P2',run=dependencies(cfg,'06'),full_nuclei=identity(Path(cfg['run_root'])/'nuclei.tif'),
                 source={'p2_rules.py':identity(Path(nv.__file__).parents[1]/'boundary_three/p2_rules.py')})
        with patch.object(nv,'EXPECTED_CELLS',2):
            nv.commit(FullNucleusIndex(self.raw()),path.parent,dep,nv.contract(cfg),events.append)
            self.assertEqual(len(events),1);self.assertEqual(identity(path),before)
            csv=Path(cfg['run_root'])/'frozen_domain/cells.csv';pd.DataFrame(dict(cell_id=[2,2])).to_csv(csv,index=False)
            with self.assertRaisesRegex(ValueError,'duplicated'):nv.canonical_ids(nv.contract(cfg))
        with self.assertRaises(OverflowError):
            nv.raster_payload(SimpleNamespace(ndim=2,dtype=np.dtype('uint32'),shape=(1,3037000501)),np.array([1],np.int64))


if __name__=='__main__':unittest.main()
