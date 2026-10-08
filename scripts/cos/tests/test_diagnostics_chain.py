import tempfile,unittest
from pathlib import Path
import numpy as np
import pandas as pd
import tifffile
from scripts.crgf.common import identity,atomic_json,sha256
from scripts.cos import diagnostics as d

class DiagnosticChainTests(unittest.TestCase):
    def test_full_stage_and_pair_integer_identity(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);prepared=root/'prepared';prepared.mkdir();atomic_json(prepared/'markers.json',{'positive':{'T':['a']},'negative':{'T':['b']}});(prepared/'reference_genes.txt').write_text('a\nb\n')
            labels=root/'labels.csv';pd.DataFrame({'cell_id':[1,2,3],'evaluation_label':['T','T','unassigned'],'unmatched_reason':['','','classifier_rejected']}).to_csv(labels,index=False);anchors={a:identity(labels) for a in ['B1','E','G','R','GR']}
            shape=(4,4);nuc=np.array([[1,0,0,2],[0,0,0,0],[0,0,0,0],[3,0,0,0]],np.uint32);npth=root/'nuclei.tif';tifffile.imwrite(npth,nuc)
            coords=np.zeros(16,dtype=[('transcript_id','<u8'),('x','<i4'),('y','<i4'),('gene','<i4')]);coords['transcript_id']=np.arange(16,dtype=np.uint64)+2**54;coords['x']=np.tile(np.arange(4),4);coords['y']=np.repeat(np.arange(4),4);coords['gene']=np.arange(16)%2;cp=root/'coords.npy';np.save(cp,coords)
            specs=[]
            for arm in ['B1_CONT','B1_COS']:
                r=root/arm;pred=r/'predictions'/arm;pred.mkdir(parents=True);matrix=r/'matrix';matrix.mkdir();ev=r/'evaluation';(ev/'paper_aligned_v1').mkdir(parents=True);(ev/'engineering/structure').mkdir(parents=True)
                raw=np.array([[1,1,2,2],[1,1,2,2],[3,3,2,2],[3,3,0,0]],np.uint32)
                if arm.endswith('COS'):raw[1,1]=2;raw[2,2]=0
                final=np.where(nuc>0,nuc,raw).astype(np.uint32);entries={}
                for name in ['raw_shift0','raw_shift24','shift0','shift24','pre','tile_merge','split_cleanup','before_global_nuclear_restore','final']:
                    p=pred/(name+'.tif');tifffile.imwrite(p,raw if name!='final' else final);entries[name]=identity(p)
                atomic_json(pred/'endpoints.json',{'endpoints':entries});owner=final.ravel();np.save(matrix/'owners.npy',owner);counts=np.zeros((4,2),np.int64);np.add.at(counts,(owner,coords['gene']),1);np.save(matrix/'counts.npy',counts[1:])
                pd.DataFrame({'canonical_cell_id':[1,2,3]}).to_csv(matrix/'cell_mapping.csv',index=False);pd.DataFrame({'gene':['a','b']}).to_csv(matrix/'gene_mapping.csv',index=False)
                rec=dict(row_start=0,row_stop=16,rows=16,coordinate_file=str(cp),coordinate_sha256=sha256(cp),owner_file='owners.npy',sha256=sha256(matrix/'owners.npy'))
                atomic_json(matrix/'complete.json',dict(status='succeeded',validation_kind='independent_full_native_resize_lookup_and_conservation',outputs={'counts':identity(matrix/'counts.npy')},dependencies={'mask':identity(pred/'final.tif'),'transcripts':{'sha256':'synthetic'}},contract={'shape':[4,4]},receipts=[rec],totals={'eligible':16}))
                machine=r/'machine.json';atomic_json(machine,dict(final_mask=str(pred/'final.tif'),nuclei_mask=str(npth),baseline_final_mask=str(npth),pixel_um=1))
                pd.DataFrame({'cell_id':[1,2,3],'cell_type':['T','T','unassigned']}).to_csv(ev/'paper_aligned_v1/markers_per_cell.csv',index=False)
                pd.DataFrame({'cell_id':[1,2,3],'x_um':[0,3,0],'y_um':[0,0,3]}).to_csv(ev/'engineering/structure/annotation_nuclear.csv',index=False)
                spec=dict(machine=str(machine),matrix=str(matrix),evaluation=str(ev),base_variant='B1');specs.append(spec)
                result=d.stage_diagnostics(spec,anchors,prepared,ev/'diagnostics/stages');self.assertTrue(result['cell_gene_gain_loss_verified'])
                rows=pd.read_csv(ev/'diagnostics/stages/stage_effects.csv');self.assertTrue((rows.groupby(['operation','unit']).n.sum()==16).all())
            result=d.ownership_pair(specs[1],specs[0],root/'pair',anchors['B1'],prepared,'B1_COS-B1_CONT');self.assertTrue(result['integer_conservation'])
            ledger=pd.read_csv(root/'pair/cell_ledger.csv');self.assertEqual(int((ledger.transcript_gain-ledger.transcript_loss).sum()),-1)
            d.postprocess_increment(dict(zip(['B1_CONT','B1_COS'],specs)),anchors,[('B1_COS','B1_CONT')],[1,2,3],root/'increment')
            delta=pd.read_csv(root/'increment/postprocess_cos_increment.csv');self.assertTrue((delta.n_common_qc==3).all());self.assertTrue((delta.n_evaluable<=2).all())
            # Corrupt row identity: reject even when owners/counts still look plausible.
            path=Path(specs[1]['matrix'])/'complete.json';manifest=d.read(path);manifest['receipts'][0]['coordinate_sha256']='bad';atomic_json(path,manifest)
            with self.assertRaises(ValueError):d.ownership_pair(specs[1],specs[0],root/'badpair',anchors['B1'],prepared,'bad')

if __name__=='__main__':unittest.main()
