"""F01–F08 and scientific F10–F14 isolated fixtures; no formal endpoint."""
import json,os,random,shutil,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import tifffile
from scipy import sparse,ndimage as ndi
from scripts.rep1_triple_simple_boundary import common,contracts,inputs,p2,populations,boundary_analysis as ba
from scripts.rep1_triple_simple_boundary.common import *
from scripts.rep1_triple_simple_boundary.bt import PANELS,MARKERS,curve,nearest,canonical_bt,mask_moments
from scripts.rep1_triple_simple_boundary.gt import build_domain,score_domain,apply_gt,validate_topology
from scripts.rep1_triple_simple_boundary.matrix import build,validate_full_oracle,check_parent_conservation
from scripts.rep1_triple_simple_boundary.frozen_domains import *
from scripts.rep1_triple_simple_boundary.comparisons import pooled_interval,contrast_interval
from scripts.rep1_triple_simple_boundary.roi import select_rois
from scripts.boundary_three.core_refine import NuclearSupport,interface_band,fixed_domain,score_proposals,topology_oracle
from scripts.boundary_three.p2_rules import FullNucleusIndex,connect
from scripts.crgf.matrix import grid,validate_production

class ScienceFixtures(unittest.TestCase):
    def setUp(self):
        retained=os.environ.get('REP1_TRIPLE_SCIENCE_FIXTURE_ROOT');self.tmp=None
        if retained:self.root=Path(retained)/self._testMethodName;self.root.mkdir(parents=True,exist_ok=False)
        else:
            folder=Path('/home/zwan0726/work/tmp/rep1_triple_science');folder.mkdir(parents=True,exist_ok=True)
            self.tmp=tempfile.TemporaryDirectory(dir=folder);self.root=Path(self.tmp.name)
    def tearDown(self):
        if self.tmp:self.tmp.cleanup()
    def cfg(self):
        return dict(run_id='fixture',run_root=str(self.root),source_digest='s',input_digest='i',domain_digest='d',plan_digest='p',
                    compute_authorization=dict(mode='admitted_compute',allowed_stages=list(contracts.STAGE_ORDER)))

    def test_bt_boundaries_and_empty_rows(self):
        ids=np.arange(1,9)
        records=pd.DataFrame(dict(cell_id=np.r_[ids,100],x_um=np.r_[[0,10,20,30,40,50,100,101],0],y_um=np.zeros(9))).set_index('cell_id')
        values=pd.DataFrame(1,index=records.index,columns=MARKERS)
        values.loc[1,'CD4']=0
        table,dist=curve(records,ids,[100],values,PANELS[0])
        part=table[table.marker.eq('CD4')]
        self.assertEqual(part.n_eligible.tolist(),[2,1,1,1,1,1,1])
        self.assertEqual(part.n_positive.tolist(),[1,1,1,1,1,1,1])
        self.assertEqual(dist.bin_index.tolist(),[0,0,1,2,3,4,5,6])
        short,_=curve(records,ids[:7],[100],values,PANELS[0])
        self.assertTrue(short[short.bin_index.eq(6)].fraction.isna().all())
        self.assertTrue(short[short.bin_index.eq(6)].upper_um.eq(100).all())
        no_source,_=curve(records,[],[100],values,PANELS[0]);self.assertEqual(len(no_source),21)
        self.assertTrue(no_source.status.eq('empty_recipient_pool').all())
        no_target,_=curve(records,ids,[],values,PANELS[0]);self.assertEqual(len(no_target),21)
        self.assertTrue(no_target.fraction.isna().all());self.assertTrue(no_target.status.eq('empty_target_pool').all())
        with self.assertRaises(ValueError):curve(records,ids,[100],values.drop(columns='CD8B'),PANELS[0])
        with self.assertRaises(ValueError):curve(records,[1,1],[100],values,PANELS[0])
        with self.assertRaises(ValueError):curve(records,[999],[100],values,PANELS[0])

    def test_merged_reverse_hand_calculation_and_ties(self):
        rec=pd.DataFrame(dict(cell_id=[1,2,3,10,11],x_um=[0,0,10,-10,10],y_um=[0]*5)).set_index('cell_id')
        val=pd.DataFrame(0,index=rec.index,columns=MARKERS);val.loc[2,'CD79A']=1;val.loc[3,'CD79B']=3
        labels=pd.Series(['B Cells','CD4 T','CD8 T','B','CD4Tconv/Treg'],index=rec.index)
        rids=labels.index[labels.isin(PANELS[1]['recipient_types'])]
        tab,_=curve(rec,rids,[1],val,PANELS[1])
        self.assertEqual(tab[(tab.marker=='CD79A')&(tab.bin_index==0)].n_eligible.item(),2)
        self.assertEqual(tab[(tab.marker=='CD79A')&(tab.bin_index==0)].fraction.item(),.5)
        self.assertTrue(canonical_bt(['B','CD4Tconv/Treg','CD8T/CD8Tex']).isna().all())
        self.assertEqual(canonical_bt(['B','CD4Tconv/Treg','CD8T/CD8Tex'],'engineering').tolist(),['B Cells','CD4 T','CD8 T'])
        self.assertEqual(nearest(rec,[1],[11,10]).nearest_target_id.iloc[0],10)
        # Removing a QC recipient must not move another fixed nearest target.
        frozen=nearest(rec,[2,3],[1]);copy=rec.copy();copy.loc[1,'x_um']=1000
        fixed,_=curve(copy,[3],[1],val,PANELS[1],frozen)
        self.assertEqual(fixed[fixed.marker.eq('CD79A')&fixed.bin_index.eq(0)].n_eligible.item(),1)

    def _tiny_gt(self,boundary=10):
        shape=(18,22)
        nuclei=np.zeros(shape,np.uint32);nuclei[8:10,5:7]=1;nuclei[8:10,15:17]=2
        parent=np.ones(shape,np.uint32);parent[:,boundary:]=2
        parent[nuclei>0]=nuclei[nuclei>0]
        ids=np.array([1,2]);types=np.array([0,1])
        sup=NuclearSupport(nuclei,ids,types,['A','B'])
        records=[]
        for y in (6,7,8,9,10,11):
            for x in range(6,17):
                records.extend([(len(records)+1,x,y,(x+y)%3),(len(records)+2,x,y,1)])
        arr=np.array(records,dtype=[('transcript_id','<u8'),('x','<i4'),('y','<i4'),('gene','<i4')])
        path=self.root/f'coords_{boundary}.npy';atomic_npy(path,arr)
        return dict(parent='tiny_'+str(boundary),baseline=parent,nuclei=nuclei,ids=ids,types=types,support=sup,
                    genes=['A','B','C'],gene_indices=np.array([0,1]),prior=np.array([[.99,.01],[.01,.99]]),
                    eligible=np.ones(2,bool),donor=np.array([1,0]),bins=np.zeros(2,int),expected_molecules=len(arr),
                    coordinates=[dict(**identity(path),rows=len(arr),row_start=0,row_stop=len(arr))])

    def test_gt_parent_domains_exact_algorithm_and_resume(self):
        data=self._tiny_gt(10);other=self._tiny_gt(12)
        events=[]
        domain,expr,info=build_domain(data,self.root/'domain',{'fixture':'one'},events.append)
        second,_,_=build_domain(other,self.root/'domain2',{'fixture':'two'})
        self.assertNotEqual(domain['pixels'].tolist(),second['pixels'].tolist())
        pixels=np.load(self.root/'domain/query_pixels.npy')
        # Independently form expression and invoke the unmodified old definition.
        arr=np.load(data['coordinates'][0]['path']);flat=arr['y'].astype(np.int64)*data['nuclei'].shape[1]+arr['x']
        mat=np.zeros((len(pixels),2),np.int64)
        for p,g in zip(flat,arr['gene']):
            if p in pixels and g<2:mat[np.flatnonzero(pixels==p)[0],g]+=1
        off,rows,_=data['support'].candidate_csr(pixels)
        pos,eo,er,reasons=fixed_domain(data['baseline'],data['nuclei'],pixels,off,rows,data['ids'],data['eligible'],data['donor'],sparse.csr_matrix(mat))
        np.testing.assert_array_equal(domain['pixels'],pixels[pos]);np.testing.assert_array_equal(domain['offsets'],eo)
        np.testing.assert_array_equal(domain['candidate_rows'],er);self.assertEqual(info['abstentions'],reasons)
        np.testing.assert_array_equal(expr.toarray(),mat[pos])
        p1,s1=score_domain(data,domain,expr,self.root/'scores',{'fixture':'score'})
        self.assertGreater(len(p1),1)
        old,_,_=score_proposals(data['baseline'],domain['pixels'],domain['offsets'],domain['candidate_rows'],domain['geometry'],expr,data['ids'],data['prior'])
        np.testing.assert_array_equal(p1,old)
        full,summary=apply_gt(data['baseline'],data['nuclei'],p1,self.root/'full',{'fixture':'apply'},oracle=True)
        self.assertGreater(summary['accepted'],0)
        interrupted,_=apply_gt(data['baseline'],data['nuclei'],p1,self.root/'resume',{'fixture':'apply'},stop_after=1,oracle=True)
        self.assertIsNone(interrupted)
        recovered,summary2=apply_gt(data['baseline'],data['nuclei'],p1,self.root/'resume',{'fixture':'apply'},oracle=True)
        np.testing.assert_array_equal(full,recovered);self.assertEqual(summary,summary2)
        validate_topology(data['baseline'],full,data['nuclei'],data['ids'],p1,self.root/'topology',{'fixture':'topology'})
        self.assertTrue(events);self.assertTrue(all('cursor' in e for e in events))
        # Committed domain receipt detects corruption; it does not silently regenerate.
        (self.root/'domain/domain.json').write_text('{}')
        with self.assertRaises(ValueError):build_domain(data,self.root/'domain',{'fixture':'one'})

    def test_gt_eligibility_margin_tie_and_rejections(self):
        data=self._tiny_gt();data['eligible'][1]=False
        dom,expr,info=build_domain(data,self.root/'ineligible',{'fixture':'ineligible'})
        self.assertEqual(len(dom['pixels']),0)
        self.assertGreater(info['abstentions'].get('candidate_not_A1_evaluable',0),0)
        baseline=np.array([[1,1,2],[1,1,2],[1,1,2]],np.uint32);nuc=np.zeros_like(baseline);nuc[1,0]=1;nuc[1,2]=2
        pixels=np.array([4]);off=np.array([0,2]);rows=np.array([0,1]);counts=sparse.csr_matrix([[1,0]])
        props,_,_=score_proposals(baseline,pixels,off,rows,np.zeros(2),counts,np.array([1,2]),np.array([[.5,.5],[.5,.5]]))
        self.assertEqual(len(props),0)
        props,_,_=score_proposals(baseline,pixels,off,rows,np.array([0,.1]),counts,np.array([1,2]),np.array([[.5,.5],[.5,.5]]))
        self.assertEqual(len(props),0)
        self.assertEqual(topology_oracle(baseline,nuc,1,0,1,2),(False,'nuclear_pixel'))
        # Removing the centre bridge splits the donor.
        bridge=np.array([[1,0,1],[1,1,1],[0,2,2]],np.uint32);bn=np.zeros_like(bridge);bn[0,0]=1;bn[2,2]=2
        self.assertEqual(topology_oracle(bridge,bn,1,1,1,2),(False,'donor_components_increased'))

    def test_matrix_full_oracle_and_recovery(self):
        mask=np.array([[1,1,1,2,2,2,0],[1,1,1,2,2,2,0],[1,1,1,2,2,2,0],[0,0,0,2,2,2,0],[0,0,0,0,0,0,0]],np.uint32)
        tifffile.imwrite(self.root/'final.tif',mask)
        (self.root/'genes.txt').write_text('A\nB\nC\n');(self.root/'baseline.yaml').write_text('fixture: true\n')
        transcript=pd.DataFrame(dict(transcript_id=np.arange(1,13),x_location=[0,.10625,.31875,.53125,1,2.5,3,4.1,5,6.5,7.5,-.4],
            y_location=[0,0,0,0,1,1,2,3,3,4,4,0],feature_name=['A','B','C']*4,qv=[30]*12))
        transcript.to_csv(self.root/'transcripts.csv',index=False)
        cfg=dict(final_mask=str(self.root/'final.tif'),genes=str(self.root/'genes.txt'),baseline_config=str(self.root/'baseline.yaml'),
                 transcripts=str(self.root/'transcripts.csv'),scale_x=.2125,scale_y=.2125,pixel_um=1.,chunksize=5)
        coords=[];cursor=0
        for i,frame in enumerate(pd.read_csv(cfg['transcripts'],chunksize=5)):
            x,y,_=grid(frame,mask.shape,.2125,.2125)
            arr=np.empty(len(frame),dtype=[('transcript_id','<u8'),('x','<i4'),('y','<i4'),('gene','<i4')])
            arr['transcript_id']=frame.transcript_id;arr['x']=x;arr['y']=y;arr['gene']=pd.Index(['A','B','C']).get_indexer(frame.feature_name)
            p=self.root/f'coordinate_{i}.npy';atomic_npy(p,arr)
            coords.append(dict(**identity(p),rows=len(frame),row_start=cursor,row_stop=cursor+len(frame)));cursor+=len(frame)
        out=self.root/'matrix';events=[]
        class Interrupt(Exception):pass
        def stop(event):
            events.append(event)
            if event['stage']=='matrix_own_owner_and_counts' and event['cursor']==1:raise Interrupt()
        with self.assertRaises(Interrupt):build(cfg,coords,out,{'fixture':'matrix'},stop,expected_molecules=12)
        result=build(cfg,coords,out,{'fixture':'matrix'},events.append,expected_molecules=12)
        self.assertEqual(result['totals']['eligible'],12)
        self.assertEqual(result['validation_kind'],'independent_full_native_resize_lookup_and_conservation')
        # Run the unmodified legacy full 2D oracle on the same tiny generated artifacts.
        legacy=self.root/'legacy';legacy.mkdir()
        for name in ['counts.npy','expr_mat.csv','cell_mapping.csv','gene_mapping.csv',*[r['owner_file'] for r in result['receipts']]]:shutil.copyfile(out/name,legacy/name)
        old=dict(result);old['dependencies']=result['matrix_dependencies']
        validate_production(cfg,legacy,old)
        self.assertEqual(json.loads((legacy/'validation.json').read_text())['rows'],12)
        self.assertTrue(any(e['stage']=='matrix_independent_full_oracle' for e in events))
        with (out/'owners_0000.npy').open('ab') as f:f.write(b'corruption')
        with self.assertRaises(ValueError):build(cfg,coords,out,{'fixture':'matrix'},expected_molecules=12)

    def test_F03_dynamic_parent_and_assigned_are_not_endpoint_lookup(self):
        # Deliberately use a new, nonhistorical assigned count and endpoint name.
        def matrix(folder,counts):
            folder.mkdir();atomic_npy(folder/'counts.npy',np.asarray(counts,np.int64))
            atomic_csv(folder/'cell_mapping.csv',pd.DataFrame(dict(matrix_row=[0,1],mask_label=[7,19],canonical_cell_id=[7,19])))
            atomic_csv(folder/'gene_mapping.csv',pd.DataFrame(dict(gene_column=[0,1,2],gene=['A','CD8B','Z'])))
        parent=self.root/'parent';child=self.root/'child';matrix(parent,[[3,2,1],[5,1,0]]);matrix(child,[[2,0,1],[6,3,0]])
        total=12;spec=dict(name='previously_unknown_parent',matrix=str(parent))
        rec=dict(totals=dict(eligible=20,assigned=total,unassigned=8))
        out=check_parent_conservation(spec,rec,child,expected_cells=2,expected_genes=3,expected_molecules=20)
        self.assertEqual(out['assigned'],12)
        bad=dict(totals=dict(eligible=20,assigned=13,unassigned=7))
        with self.assertRaises(ValueError):check_parent_conservation(spec,bad,child,2,3,20)
        cfg=self.cfg();prod=self.root/'endpoints/P2_SIMPLE/production_complete.json';datum=self.root/'data';datum.write_text('done')
        write_receipt(prod,dependencies(cfg,'06'),{'datum':identity(datum)},endpoint='P2_BASE',parent='BASE_MATCH',stage='06')
        with self.assertRaisesRegex(ValueError,'dynamic P2_SIMPLE parent'):inputs.resolve_parent(cfg,{},'P2_SIMPLE_GT')
        with self.assertRaises(ValueError):inputs.resolve_parent(cfg,{},'P2_BASE_GT')

    def test_F04_uint32_full313_and_duplicate_independent_csv(self):
        mask=np.array([[0,7,2**32-2],[7,0,2**32-2]],np.uint32);tifffile.imwrite(self.root/'mask.tif',mask)
        genes=['CD8B']+['G'+str(i) for i in range(312)];(self.root/'genes').write_text('\n'.join(genes)+'\n');(self.root/'cfg').write_text('fixture')
        frame=pd.DataFrame(dict(transcript_id=[1,2,3,4,5,6],x_location=[1,2,0,-1,.5,2],y_location=[0,0,1,0,0,1],feature_name=['CD8B','G311','CD8B','G0','G1','G311'],qv=30))
        csv=self.root/'raw.csv';frame.to_csv(csv,index=False)
        machine=dict(final_mask=str(self.root/'mask.tif'),genes=str(self.root/'genes'),baseline_config=str(self.root/'cfg'),transcripts=str(csv),scale_x=1.,scale_y=1.,chunksize=6)
        x,y,_=grid(frame,mask.shape,1.,1.);arr=np.empty(6,dtype=[('transcript_id','<u8'),('x','<i4'),('y','<i4'),('gene','<i4')]);arr['transcript_id']=frame.transcript_id;arr['x']=x;arr['y']=y;arr['gene']=pd.Index(genes).get_indexer(frame.feature_name)
        cp=self.root/'coords.npy';atomic_npy(cp,arr);coords=[dict(**identity(cp),rows=6,row_start=0,row_stop=6)]
        rec=build(machine,coords,self.root/'matrix',{'fixture':'313'},expected_molecules=6)
        counts=np.load(self.root/'matrix/counts.npy');self.assertEqual(counts.shape,(2,313));self.assertEqual(counts[0,0],2);self.assertEqual(counts[1,-1],2)
        self.assertEqual(rec['totals']['assigned'],4);self.assertEqual(rec['totals']['out_of_bounds'],1)
        arr['transcript_id'][1]=1;atomic_npy(cp,arr);coords[0].update(identity(cp))
        with self.assertRaisesRegex(ValueError,'duplicate transcript'):build(machine,coords,self.root/'duplicate',{'fixture':'duplicate'},expected_molecules=6)
        arr['transcript_id'][1]=2;arr['gene'][0]=1;atomic_npy(cp,arr);coords[0].update(identity(cp))
        with self.assertRaisesRegex(ValueError,'gene association'):build(machine,coords,self.root/'gene_wrong',{'fixture':'wrong'},expected_molecules=6)

    def test_F05_domains_interface_unknown_density_and_recovery(self):
        b=np.zeros((9,11),np.uint32);b[3:6,2:5]=1;b[3:6,5:8]=2;n=np.zeros_like(b);n[4,2]=1;n[4,7]=2
        K=np.zeros_like(b);K[2:8,1:10]=2;reg=region_arrays(b,n,K,interface_band(b))
        self.assertTrue(np.any(reg['interface']&(b==0)));self.assertFalse(np.any(reg['interface']&(n>0)))
        self.assertTrue(np.array_equal(reg['nuclear'].astype(int)+reg['competition']+reg['noncompetitive'],np.ones(b.shape)))
        self.assertEqual(type_class([True,False,False],[True,True,False]).tolist(),[2,1,0])
        self.assertFalse(interface_band(np.ones((8,9),np.uint32)).any())
        a,pairs=build_interfaces(b,n,np.array([-2,0,-1]),self.root/'iface',{'case':'fixed'})
        self.assertTrue((pairs.type_class=='Unknown').all());self.assertTrue(np.array_equal(a['assignment']>=0,reg['interface']))
        self.assertTrue(a['assignment'][1,4]>=0);self.assertTrue(a['assignment'][0,4]<0)
        before={str(p):identity(p) for p in (self.root/'iface').rglob('*') if p.is_file()}
        build_interfaces(b,n,np.array([-2,0,-1]),self.root/'iface',{'case':'fixed'});self.assertEqual(before,{str(p):identity(p) for p in (self.root/'iface').rglob('*') if p.is_file()})
        (self.root/'iface/complete.json').unlink() # Fixture-only interrupted outer commit.
        bad=self.root/'iface/exposure_000000.npz';bad.write_bytes(b'corrupt committed exposure');old=bad.read_bytes()
        with self.assertRaisesRegex(ValueError,'identity drift'):build_interfaces(b,n,np.array([-2,0,-1]),self.root/'iface',{'case':'fixed'})
        self.assertEqual(bad.read_bytes(),old)
        multi=np.array([[1,2,0,0,0,3,4]]*3,np.uint32);empty=np.zeros_like(multi)
        several,_=build_interfaces(multi,empty,np.array([-2,0,0,1,1]),self.root/'several',{'case':'overlapping homotypic pairs'})
        self.assertTrue(several['multiple_known_types'][1,3]);self.assertFalse(several['known_heterotypic'][1,3]);self.assertFalse(several['any_unknown'][1,3])
        counts=np.arange(143).reshape(11,13)%5;y,x=np.mgrid[-2:3,-2:3];kernel=(y*y+x*x<=4).astype(float)
        expect=ndi.convolve(counts.astype(float),kernel,mode='constant')/ndi.convolve(np.ones(counts.shape),kernel,mode='constant')
        np.testing.assert_allclose(local_density(counts,2,3),expect,rtol=1e-7)
        edge,g=quartiles([1,2,2,3,4,np.nan]);self.assertEqual(g[-1],-1);self.assertEqual(g[1],np.searchsorted(edge,2,side='left'))

    def test_F06_populations_global_weights_factorial_and_pooled(self):
        names=contracts.METHODS;tables={}
        for index,name in enumerate(names):
            f=pd.DataFrame({'m':[0.,1.,2.,3.,4.]},index=[1,2,3,4,5]);tables[name]=dict(sc=f.copy(),fixed=f.copy(),qc_ids=np.array([1,2,3,4,5]))
        tables[names[0]]['sc'].loc[2,'m']=np.nan;tables[names[1]]['fixed']=tables[names[1]]['fixed'].drop(3)
        tables[names[7]]['sc'].loc[4,'m']=np.nan;tables[names[-1]]['qc_ids']=np.array([1,2,3,4])
        _,pair,_,_=populations.common_finite(tables,names[:2],[1,2,3,4,5],'m');self.assertEqual(pair.tolist(),[1,4,5])
        _,eight,_,_=populations.common_finite(tables,contracts.FACTORIAL8,[1,2,3,4,5],'m');self.assertEqual(eight.tolist(),[1,5])
        _,twelve,_,_=populations.common_finite(tables,names,[1,2,3,4,5],'m');self.assertEqual(twelve.tolist(),[1])
        xy=np.c_[np.arange(12)*1000,np.zeros(12)];blocks,bi,W=bootstrap_weights(xy,1000);b2,i2,W2=bootstrap_weights(xy,1000)
        np.testing.assert_array_equal(W,W2);self.assertTrue(np.all(W.sum(1)==12))
        n=np.array([1,9]+[0]*10);s=np.array([1,0]+[0]*10);r=pooled_interval(s,n,W,minimum_units=0)
        self.assertAlmostEqual(r['point'],.1);self.assertNotEqual(r['point'],.5);self.assertEqual(r['universe_blocks'],12);self.assertEqual(r['NA_reason'],'NA_insufficient_blocks')
        r=pooled_interval(np.arange(12),np.ones(12)*2,np.zeros((2000,12),int));self.assertEqual(r['NA_reason'],'NA_insufficient_valid_replicates')
        # A polynomial on the 2x2x2 cube fixes every edge and interaction sign.
        self.assertEqual(len(contracts.DAG['factorial_contrasts']),19)
        # Read coordinates from endpoint membership rather than name substring inference.
        bits={'BASE_MATCH':(0,0,0),'V2_MORPH':(1,0,0),'P2_BASE':(0,1,0),'A2_GT':(0,0,1),'P2_MORPH':(1,1,0),'V2_MORPH_GT':(1,0,1),'P2_BASE_GT':(0,1,1),'P2_MORPH_GT':(1,1,1)}
        value={m:1+2*x+3*y+5*z+7*x*y+11*x*z+13*y*z+17*x*y*z for m,(x,y,z) in bits.items()}
        results={c['id']:sum(k*value[m] for m,k in c['coefficient'].items()) for c in contracts.DAG['factorial_contrasts']}
        self.assertEqual(sum(len(c['coefficient'])==2 for c in contracts.DAG['factorial_contrasts']),12)
        self.assertEqual(sum(len(c['coefficient'])==4 for c in contracts.DAG['factorial_contrasts']),6)
        third=next(c for c in contracts.DAG['factorial_contrasts'] if len(c['coefficient'])==8);self.assertEqual(results[third['id']],17)
        for c in contracts.DAG['factorial_contrasts']:
            v={m:np.repeat(value[m],24) for m in c['coefficient']};r=contrast_interval(v,c['coefficient'],np.repeat(np.arange(12),2),W)
            self.assertEqual(r['point'],results[c['id']]);self.assertAlmostEqual(r['ci_low'],results[c['id']]);self.assertAlmostEqual(r['ci_high'],results[c['id']])

    def test_F08_complete_sparse_transfer_and_fixed_ROI(self):
        cells=pd.DataFrame(index=pd.Index([1,7,20],name='cell_id'))
        cells['original_type_index']=[0,-1,2]
        for dim in CELL_DIMENSIONS:cells[dim]=[0,1,2]
        for s in (500,1000,2000):cells['block_'+str(s)]=[0,1,2]
        source=np.array([1,1,7,20,0,7,0]);target=np.array([7,0,20,1,1,7,0]);gene=np.array([0,312,1,2,312,1,2])
        st=dict(region_bits=np.full(7,(1<<0)|(1<<1)|(1<<2),np.uint16),local_density=np.arange(7)%4,K=np.full(7,4),interface_id=np.arange(7))
        for s in (500,1000,2000):st['block_'+str(s)]=np.arange(7)%3
        keys,n=ba.sparse_transfer(source,target,gene,st,cells);self.assertEqual(int(n.sum()),5);self.assertIn(312,keys[:,0]);self.assertTrue((keys[:,1]==0).any());self.assertTrue((keys[:,2]==0).any());self.assertIn(-1,keys[:,3])
        delta=np.zeros((3,313),int)
        for row,count in zip(keys,n):
            if row[1]>0:delta[cells.index.get_loc(row[1]),row[0]]-=count
            if row[2]>0:delta[cells.index.get_loc(row[2]),row[0]]+=count
        hand=np.zeros_like(delta)
        for a,b,g in zip(source,target,gene):
            if a>0:hand[cells.index.get_loc(a),g]-=1
            if b>0:hand[cells.index.get_loc(b),g]+=1
        np.testing.assert_array_equal(delta,hand);self.assertEqual(int(delta.sum()),0)
        types=np.indices((12,16)).sum(0)%3;density=np.indices((12,16))[1]%4;reg={r:np.ones((12,16),bool) for r in ['competition','interface','nuclear','noncompetitive']}
        a=select_rois(reg,types,types,density);b=select_rois(reg,types,types,density)
        self.assertEqual(a,b);self.assertEqual(len(a['rows']),32);self.assertEqual(a['summary']['selected'],32)
        self.assertEqual(a['rows'][0]['seed_sequence'],[20261017,0,0,0]);self.assertEqual(a['rows'][24]['seed_sequence'],[20261017,2,3,0])
        reg['competition'][:]=False;c=select_rois(reg,types,types,density);self.assertEqual(c['summary']['empty'],12);self.assertTrue(all(r['reason'] for r in c['rows'][:12]))

    def test_F10_atomic_failure_preserves_bytes_and_initial_reuse(self):
        cfg=self.cfg();p=self.root/'value.npy';atomic_npy(p,np.array([1,2,3]));before=identity(p)
        with patch.object(common.os,'replace',side_effect=OSError('injected interruption')):
            with self.assertRaises(OSError):atomic_npy(p,np.array([4,5,6]))
        self.assertEqual(identity(p),before);self.assertTrue(list(self.root.glob('.value.npy.*')))
        source=self.root/'initial';source.mkdir();value=source/'value.csv';value.write_text('x\n1\n');binding={'population':'same','W':'same','coefficient':{'A':1,'B':-1}}
        write_receipt(source/'complete.json',dependencies(cfg,'04_part'),inventory(source),reuse_binding=binding)
        dest=self.root/'all12';reuse_partition(source,dest,dependencies(cfg,'13_part'),binding)
        self.assertEqual(os.stat(value).st_ino,os.stat(dest/'value.csv').st_ino)
        rec=check_receipt(dest/'complete.json');self.assertEqual(rec['reused_from'],identity(source/'complete.json'))
        with self.assertRaisesRegex(ValueError,'incompatible'):reuse_partition(source,self.root/'wrong',dependencies(cfg,'13_part'),dict(binding,W='changed'))
        value.write_text('corrupted');
        with self.assertRaisesRegex(ValueError,'identity drift'):reuse_partition(source,self.root/'bad',dependencies(cfg,'13_part'),binding)

    def test_F14_sampler_rejects_incomplete_measurement(self):
        from tests import rep1_triple_real_fixtures as real
        sampler=real.MemorySampler()
        with patch.object(real,'process_memory_tree',side_effect=RuntimeError('injected unexpected sampler failure')):
            with self.assertRaisesRegex(RuntimeError,'unexpected sampler'):
                with sampler:pass
        with self.assertRaisesRegex(RuntimeError,'measurement incomplete'):sampler.result()

    def test_F12_compute_admission_and_independent_publication_gate(self):
        cfg=self.cfg();ad=self.root/'admission.json';atomic_json(ad,dict(config_digest=digest(cfg),status='PASS_FOR_LAUNCH',open_blockers=0))
        payload=dict(authorization='admitted_compute',run_id=cfg['run_id'],config_digest=digest(cfg),stage='03',admission=identity(ad),production=None)
        with self.assertRaises(PermissionError):require_compute_authorization(cfg,payload,'03')
        production=self.root/'production.json';atomic_json(production,dict(status='succeeded',stage='02',endpoint='P2_MORPH_GT',dependencies=dependencies(cfg,'02')))
        payload['production']=identity(production);require_compute_authorization(cfg,payload,'03',production)
        with self.assertRaises(PermissionError):require_manual_resume(cfg,None)
        self.assertFalse((self.root/'control/manual_resume.json').exists())
        with self.assertRaises(PermissionError):require_compute_authorization(cfg,dict(payload,stage='07'),'03')
        with self.assertRaises(PermissionError):require_compute_authorization(dict(cfg,run_id='other'),payload,'03')

    def test_F06_F10_bootstrap_batch_and_retirement_recovery(self):
        from scripts.rep1_triple_simple_boundary import spatial_bootstrap as sb
        rng=np.random.default_rng(7);W=rng.multinomial(12,np.ones(12)/12,size=2000);n=np.full((12,3),4.);s=rng.normal(size=n.shape)
        names=['all','fixed1','fixed2'];deps={'ids':id_hash(np.arange(48))};out=self.root/'bootstrap'
        def interrupt(event):
            if event['cursor']==100:raise InterruptedError('after committed batch')
        with self.assertRaises(InterruptedError):sb.block_intervals(s,n,W,names,checkpoint=(out,deps),callback=interrupt)
        old=identity(out/'batch_0000.json');failed=out/'.uncommitted.tmp';failed.write_text('retained failure evidence')
        # A committed batch is reused, even if its atomic writer is made fatal.
        original=sb.atomic_npz
        def checked(path,**arrays):
            if Path(path).name=='batch_0000.npz':raise AssertionError('committed batch recomputed')
            return original(path,**arrays)
        with patch.object(sb,'atomic_npz',side_effect=checked):actual=sb.block_intervals(s,n,W,names,checkpoint=(out,deps))
        expected=sb.block_intervals(s,n,W,names);self.assertEqual(actual,expected);self.assertTrue(failed.exists())
        self.assertEqual(read_json_gzip(out/'intervals.json.gz'),expected)
        again=self.root/'deterministic.json.gz';atomic_json_gzip(again,expected)
        self.assertEqual(sha256(again),sha256(out/'intervals.json.gz'))
        retired=read(out/'retired_batches.json');self.assertEqual(retired['batches'][0]['receipt'],old)
        self.assertFalse(list(out.glob('batch_*.npz')));self.assertEqual(sb.block_intervals(s,n,W,names,checkpoint=(out,deps)),expected)
        bad=self.root/'corrupt';
        with self.assertRaises(InterruptedError):sb.block_intervals(s,n,W,names,checkpoint=(bad,deps),callback=interrupt)
        (bad/'batch_0000.npz').write_bytes(b'corrupt committed batch')
        with self.assertRaisesRegex(ValueError,'identity drift'):sb.block_intervals(s,n,W,names,checkpoint=(bad,deps))
        self.assertEqual((bad/'batch_0000.npz').read_bytes(),b'corrupt committed batch')
        retirement=self.root/'retirement';unlink=Path.unlink
        def fault(path,*args,**kw):
            if path==retirement/'batch_0000.npz':raise OSError('retirement interruption')
            return unlink(path,*args,**kw)
        with patch.object(Path,'unlink',fault):
            with self.assertRaises(OSError):sb.block_intervals(s,n,W,names,checkpoint=(retirement,deps))
        complete=identity(retirement/'complete.json');self.assertEqual(sb.block_intervals(s,n,W,names,checkpoint=(retirement,deps)),expected)
        self.assertEqual(identity(retirement/'complete.json'),complete);self.assertFalse(list(retirement.glob('batch_*.npz')))

    def test_F06_F08_sparse_projection_and_categorical_spatial_exact(self):
        from scripts.rep1_triple_simple_boundary.spatial_bootstrap import GroupProjector
        from scripts.rep1_triple_simple_boundary.local_statistics import spatial_masks,accumulate_spatial,kinds,_statistics
        rng=np.random.default_rng(4);ids=np.arange(1,25);cells=pd.DataFrame(index=pd.Index(ids,name='cell_id'))
        for dim in CELL_DIMENSIONS:cells[dim]=rng.integers(0,4,len(ids))
        cells['original_type_index']=np.arange(len(ids))%5-1
        for s in contracts.SCALES:cells['block_'+str(s)]=np.arange(len(ids))%12
        groups={'all':ids,'odd':ids[::2],'sparse':ids[:1],'empty':np.array([],np.int64)};projector=GroupProjector(cells,groups)
        use=ids[1::2];values=np.linspace(-1,1,len(use));s,n=projector.blocks(use,values,1000,12)
        for j,(_,group) in enumerate(groups.items()):
            keep=np.isin(use,group);bi=cells.loc[use[keep],'block_1000'].to_numpy(int)
            np.testing.assert_array_equal(n[:,j],np.bincount(bi,minlength=12));np.testing.assert_array_equal(s[:,j],np.bincount(bi,weights=values[keep],minlength=12))
        N=109;st=dict(BASE_owner=rng.integers(0,25,N),region_bits=rng.integers(0,1<<len(contracts.REGIONS),N),competition_type=rng.integers(-1,3,N),interface_type=rng.integers(-1,3,N),K=rng.integers(0,5,N),local_density=rng.integers(-1,4,N),single_unknown=rng.integers(0,2,N).astype(bool))
        for side in contracts.SCALES:st['block_'+str(side)]=rng.integers(-1,12,N)
        kind=kinds(rng.integers(0,4,N),rng.integers(0,4,N));masks=list(spatial_masks(st,cells));names=[k for k,v in masks]
        totals=np.zeros((len(names),5),np.int64);acc={s:np.zeros((12,len(names),5),np.int64) for s in contracts.SCALES}
        accumulate_spatial(st,cells,kind,names,totals,acc)
        for j,(name,mask) in enumerate(masks):
            np.testing.assert_array_equal(totals[j],np.bincount(kind[mask],minlength=5))
            for side in contracts.SCALES:
                good=mask&(st['block_'+str(side)]>=0);v=np.bincount(st['block_'+str(side)][good]*5+kind[good],minlength=60).reshape(12,5)
                np.testing.assert_array_equal(acc[side][:,j],v)
        v=np.array([10,8,3,2,5]);stats=_statistics(v);self.assertEqual(stats['capture_delta_pp'],(300,28));self.assertEqual(stats['owner_retention'],(8,13));self.assertEqual(stats['gross_transfer_rate'],(10,28))

    def test_F04_complete_coordinate_producer_admission(self):
        import hashlib,cv2
        from scripts.crgf import matrix as builder,common as bc
        root=self.root;mask=root/'mask.tif';atomic_tiff(mask,np.zeros((3,4),np.uint32));trans=root/'trans.csv';trans.write_text('original source');genes=root/'genes.txt';genes.write_text('\n'.join('G'+str(i) for i in range(313)))
        machine=dict(final_mask=str(mask),transcripts=str(trans),genes=str(genes),scale_x=2.,scale_y=3.)
        expected=dict(transcripts=sha256(trans),genes=sha256(genes),shape=[3,4],scale_x=2.,scale_y=3.,builder=sha256(builder.__file__),common=sha256(bc.__file__),numpy=np.__version__,opencv=cv2.__version__,pandas=pd.__version__)
        key=hashlib.sha256(json.dumps(expected,sort_keys=True).encode()).hexdigest();folder=root/key;folder.mkdir()
        a=np.array([(1,2,1,312),(2,-1,-1,0)],dtype=[('transcript_id','<u8'),('x','<i4'),('y','<i4'),('gene','<i4')]);path=folder/'chunk_0000.npy';atomic_npy(path,a)
        cache=folder/'complete.json';obj=dict(dependencies=expected,duplicate_ids=0,chunks=[dict(file=path.name,sha256=sha256(path),rows=2)]);atomic_json(cache,obj)
        coord=[dict(**identity(path),rows=2,row_start=0,row_stop=2)];receipt=dict(contract=dict(shape=[3,4],scale_x=2.,scale_y=3.,source_sha256=expected['builder']))
        self.assertEqual(inputs.validate_coordinate_cache(cache,coord,machine,receipt,2,1)['rows'],2)
        for field in ('numpy','opencv','pandas','builder','common','scale_x','scale_y','shape'):
            changed=dict(obj,dependencies=dict(expected,**{field:'changed'}));atomic_json(cache,changed)
            with self.assertRaisesRegex(ValueError,'producer identity'):inputs.validate_coordinate_cache(cache,coord,machine,receipt,2,1)
        atomic_json(cache,obj)
        with self.assertRaisesRegex(ValueError,'row span'):inputs.validate_coordinate_cache(cache,[dict(coord[0],row_start=1)],machine,receipt,2,1)

    def test_F13_matrix_alone_cannot_claim_full_evaluate(self):
        from scripts.rep1_triple_simple_boundary.coverage import validate_endpoint_products
        root=self.root/'matrix';root.mkdir();v=root/'validation.json';atomic_json(v,dict(integer_count_differences=0,owner_differences=0))
        atomic_json(root/'complete.json',dict(validation_kind='independent_full_native_resize_lookup_and_conservation',validation=identity(v)))
        spec=dict(name='P2_SIMPLE',evaluation=str(self.root/'missing_evaluation'),matrix=str(root),sc_scores=str(self.root/'no_scores'),fixed_scores=str(self.root/'no_fixed'))
        with self.assertRaises(FileNotFoundError):validate_endpoint_products(spec)

    def test_F13_actual_comparison_and_marker_keys_reject_truncation(self):
        from itertools import product
        from scripts.rep1_triple_simple_boundary.coverage import comparison_rows,bt_curve_rows
        groups=['all','Unknown','empty'];hashes={(m,p):'actual_population' for m in contracts.METRICS for p in ('sc','fixed')}
        rows=[dict(metric=m,profile=p,scale_um=s,stratum=g,population='pairwise',coefficient='{"A":1,"B":-1}',population_ids_sha256='actual_population',point=0.,ci_low=0.,ci_high=0.,n_units=24,NA_reason='',replicates=2000,valid_replicates=2000,contributing_blocks=12,universe_blocks=12) for m,p,s,g in product(contracts.METRICS,('sc','fixed'),contracts.SCALES,groups)]
        frame=pd.DataFrame(rows);comparison_rows(frame,groups,{'A':1,'B':-1},'pairwise',hashes)
        with self.assertRaisesRegex(ValueError,'actual unique key'):comparison_rows(frame.iloc[:-1],groups,{'A':1,'B':-1},'pairwise',hashes)
        dup=frame.copy();dup.iloc[-1]=dup.iloc[0]
        with self.assertRaisesRegex(ValueError,'actual unique key'):comparison_rows(dup,groups,{'A':1,'B':-1},'pairwise',hashes)
        changed=frame.copy();changed['coefficient']='{"A":-1,"B":1}'
        with self.assertRaisesRegex(ValueError,'coefficient'):comparison_rows(changed,groups,{'A':1,'B':-1},'pairwise',hashes)
        changed=frame.copy();changed['population_ids_sha256']='another_population'
        with self.assertRaisesRegex(ValueError,'population ID hash'):comparison_rows(changed,groups,{'A':1,'B':-1},'pairwise',hashes)
        points=pd.DataFrame(dict(cell_id=[1,2],x_um=[0.,1.],y_um=[0.,0.])).set_index('cell_id');values=pd.DataFrame(0,index=points.index,columns=MARKERS)
        tables=[]
        for panel in PANELS:
            table,_=curve(points,[1],[2],values,panel);table['population_ids_sha256']='fixture';tables.append(table)
        table=pd.concat(tables,ignore_index=True);bt_curve_rows(table,'six markers')
        with self.assertRaisesRegex(ValueError,'actual unique key'):bt_curve_rows(table[table.marker!='CD8B'],'missing marker')
        table.loc[41]=table.iloc[0]
        with self.assertRaisesRegex(ValueError,'actual unique key'):bt_curve_rows(table,'repeated bin')

    def test_F08_full_pair_artifacts_and_initial_core_reuse(self):
        cfg=self.cfg();frozen=self.root/'frozen_domain';frozen.mkdir();nuc=np.zeros((3,4),np.uint32);nuc[0,0]=1;nuc[0,2]=2
        base=np.array([[1,1,2,2],[1,0,2,0],[1,1,2,2]],np.uint32);target=base.copy();target[0,1]=2;target[1,1]=2;target[1,2]=0
        K=np.full(base.shape,2,np.uint8);regions=region_arrays(base,nuc,K,interface_band(base));bits=region_bits(regions);atomic_npy(frozen/'region_bits.npy',bits)
        local=np.arange(base.size,dtype=np.int8).reshape(base.shape)%4;atomic_npy(frozen/'local_density_group.npy',local)
        typ=np.full(base.shape,2,np.int8);typ[nuc>0]=-1
        atomic_npy(frozen/'competition_type.npy',typ);atomic_npy(frozen/'interface_type.npy',typ)
        historical=self.root/'historical';historical.mkdir();atomic_npy(historical/'support_counts.npy',K);atomic_npy(historical/'any_unknown.npy',np.ones(base.shape,bool))
        cells=pd.DataFrame(index=pd.Index([1,2],name='cell_id'));cells['original_type_index']=[0,-1];cells['original_type']=['A','Unknown']
        for dim in CELL_DIMENSIONS:cells[dim]=[0,1]
        for side in contracts.SCALES:
            cells['block_'+str(side)]=0;atomic_npz(frozen/f'bootstrap_weights_{side}.npz',blocks=np.array([[0,0]]),W=np.ones((2000,1),np.int16))
        atomic_csv(frozen/'cells.csv',cells,index=True);atomic_npz(frozen/'cell_groups.npz',all=np.array([1,2]),single=np.array([1]),empty=np.array([],np.int64))
        ia,pairs=build_interfaces(base,nuc,np.array([-2,0,-1]),frozen/'interfaces',{'fixture':'interfaces'})
        genes=['CD8B']+['G'+str(i) for i in range(312)];gene_path=self.root/'genes';gene_path.write_text('\n'.join(genes)+'\n');config=self.root/'baseline_config';config.write_text('fixture')
        yy,xx=np.indices(base.shape);raw=pd.DataFrame(dict(transcript_id=np.arange(1,13),x_location=xx.ravel(),y_location=yy.ravel(),feature_name=[genes[0] if i%2 else genes[-1] for i in range(12)],qv=30))
        csv=self.root/'raw.csv';atomic_csv(csv,raw);coord=np.empty(12,dtype=[('transcript_id','<u8'),('x','<i4'),('y','<i4'),('gene','<i4')]);coord['transcript_id']=raw.transcript_id;coord['x']=xx.ravel();coord['y']=yy.ravel();coord['gene']=pd.Index(genes).get_indexer(raw.feature_name)
        cp=self.root/'coordinates.npy';atomic_npy(cp,coord);coordinates=[dict(**identity(cp),rows=12,row_start=0,row_stop=12)]
        st=dict(region_bits=bits.ravel(),BASE_owner=base.ravel(),local_density=local.ravel(),competition_type=typ.ravel(),interface_type=typ.ravel(),K=K.ravel(),interface_id=ia['assignment'].ravel(),single_unknown=np.zeros(12,bool))
        for side in contracts.SCALES:st['block_'+str(side)]=np.zeros(12,np.int16)
        atomic_npz(frozen/'molecule_strata/chunk_0000.npz',**st);atomic_npy(frozen/'molecule_count.npy',np.ones(base.shape,np.int64))
        manifest=dict(historical_strata=str(historical),coordinates=coordinates,endpoints={});specs=[]
        for name,mask in [('BASE_MATCH',base),('P2_SIMPLE',target)]:
            path=self.root/name;path.mkdir();atomic_tiff(path/'final.tif',mask)
            machine=dict(final_mask=str(path/'final.tif'),genes=str(gene_path),baseline_config=str(config),transcripts=str(csv),scale_x=1.,scale_y=1.,chunksize=12)
            build(machine,coordinates,path/'matrix',{'fixture':name},expected_molecules=12)
            spec=dict(name=name,final=str(path/'final.tif'),matrix=str(path/'matrix'),evaluation=str(path/'evaluation'));specs.append(spec)
        manifest['endpoints']['BASE_MATCH']=specs[0];atomic_json(frozen/'complete.json',dict(fixture='bound synthetic domains'));cfg['domain_manifest']=str(frozen/'complete.json')
        out=self.root/'initial_pair';rec=ba.pair_transfers(cfg,manifest,specs[1],specs[0],out,include_structure=False)
        self.assertTrue(rec['matrix_gene_cell_delta_exact']);self.assertTrue(rec['all313_gene_ledger']);self.assertFalse(rec['structure_completed'])
        table=pd.read_csv(out/'cells.csv.gz');self.assertEqual(table.incoming_RNA.sum(),2);self.assertEqual(table.outgoing_RNA.sum(),2);self.assertEqual(table.net_RNA.sum(),0)
        regions_table=pd.read_csv(out/'regions.csv');whole=regions_table[regions_table.region=='all'].iloc[0];self.assertEqual(whole.cell_to_cell,1);self.assertEqual(whole.cell_to_background,1);self.assertEqual(whole.background_to_cell,1)
        stats=pd.read_csv(out/'spatial_strata/molecule/stratified_counts.csv.gz');allrow=stats[stats.stratum=='all'].iloc[0]
        self.assertEqual(allrow.cell_to_cell,1);self.assertEqual(allrow.cell_to_background,1);self.assertEqual(allrow.background_to_cell,1)
        after=ba.pair_transfers(cfg,manifest,specs[1],specs[0],self.root/'all12_pair',reuse_from=out,include_structure=False)
        self.assertEqual(after['reused_from'],identity(out/'complete.json'));self.assertEqual(sha256(out/'chunks/0000.npz'),sha256(self.root/'all12_pair/chunks/0000.npz'))
        from scripts.rep1_triple_simple_boundary.coverage import boundary_tables
        groups=dict(np.load(frozen/'cell_groups.npz'));boundary_tables(cfg,manifest,specs[1],specs[0],out,frozen,cells,groups,False)
        truncated=stats.iloc[:-1];ba.atomic_csv_gzip(out/'spatial_strata/molecule/stratified_counts.csv.gz',truncated)
        # A new file hash cannot make a missing actual stratum semantically complete.
        with self.assertRaisesRegex(ValueError,'actual unique key'):
            boundary_tables(cfg,manifest,specs[1],specs[0],out,frozen,cells,groups,False)



from scripts.rep1_triple_simple_boundary import gt
GT_FIXTURE_ENDPOINTS=('P2_MORPH_GT','SIMPLE_GT','P2_SIMPLE_GT')


class RefineRepairFixtures(unittest.TestCase):
    setUp=ScienceFixtures.setUp
    tearDown=ScienceFixtures.tearDown

    def _config(self,name):
            root=self.root/name
            return dict(run_id='isolated_'+name,run_root=str(root),source_digest='fixture_source',
                        source_commit='fixture_commit',input_digest='fixture_input',plan_digest='fixture_plan',domain_digest='fixture_domain',
                        input_manifest=str(root/'configs/inputs_manifest.json'))

    def _refine_inputs(self,name,boundary=10):
            cfg=self._config(name);source=self.root/(name+'_input');source.mkdir()
            nuclei=np.zeros((18,22),np.uint32);nuclei[8:10,5:7]=1;nuclei[8:10,15:17]=2
            parent=np.ones_like(nuclei);parent[:,boundary:]=2;parent[nuclei>0]=nuclei[nuclei>0]
            ids=np.array([1,2]);types=np.array([0,1]);records=[]
            for y in range(6,12):
                for x in range(6,17):
                    records.extend([(len(records)+1,x,y,(x+y)%3),(len(records)+2,x,y,1)])
            arr=np.array(records,dtype=[('transcript_id','<u8'),('x','<i4'),('y','<i4'),('gene','<i4')])
            coord=source/'coordinates.npy';atomic_npy(coord,arr)
            final=source/'parent.tif';tifffile.imwrite(final,parent)
            reference=source/'reference.json';atomic_json(reference,{'synthetic':'reference identity'})
            cache=source/'cache.json';atomic_json(cache,{'synthetic':'coordinate identity'})
            a1=source/'a1.json';atomic_json(a1,{'synthetic':'A1 eligibility loaded by fixture'})
            manifest=dict(a1={'fixture':str(a1)},reference_region=str(reference),coordinate_cache=str(cache),
                          files={str(p.resolve()):identity(p) for p in (final,reference,cache,a1)})
            data=dict(parent='P2_BASE' if boundary==10 else 'V2_MORPH',spec={'final':str(final),'parent_evidence':{'final':identity(final)}},
                      baseline=parent,nuclei=nuclei,ids=ids,types=types,
                      support=NuclearSupport(nuclei,ids,types,['A','B']),genes=['A','B','C'],
                      gene_indices=np.array([0,1]),prior=np.array([[.99,.01],[.01,.99]]),
                      eligible=np.ones(2,bool),donor=np.array([1,0]),bins=np.zeros(2,int),
                      expected_molecules=len(arr),coordinates=[dict(**identity(coord),rows=len(arr),row_start=0,row_stop=len(arr))])
            return cfg,manifest,data

    def _upstream(self,root):
            return {str(p.relative_to(root)):{**identity(p),'mtime_ns':p.stat().st_mtime_ns}
                    for group in ('domain','scores','recovery') for p in sorted((root/group).rglob('*')) if p.is_file()}

    def _assert_refine(self,cfg,data,endpoint,result):
            root=Path(cfg['run_root'])/'endpoints'/endpoint
            final=tifffile.imread(root/'final.tif')
            self.assertEqual(final.dtype,np.dtype(np.uint32))
            np.testing.assert_array_equal(final>0,data['baseline']>0)
            np.testing.assert_array_equal(final[data['nuclei']>0],data['nuclei'][data['nuclei']>0])
            np.testing.assert_array_equal(np.unique(final),data['ids'])
            oracle=data['baseline'].copy();accepted=0
            proposals=np.load(root/'gt/scores/proposals.npy')
            for p in proposals:
                y,x,source,target=map(int,(p['y'],p['x'],p['source'],p['target']))
                ok,_=topology_oracle(oracle,data['nuclei'],y,x,source,target)
                if ok:oracle[y,x]=target;accepted+=int(ok)
            self.assertGreater(accepted,0)
            np.testing.assert_array_equal(final,oracle)
            self.assertEqual(result['application']['cursor'],len(proposals))
            self.assertEqual(result['application']['accepted'],accepted)
            self.assertTrue(result['invariants']['nuclear_connections_preserved'])
            check_receipt(root/'gt/complete.json',dependencies(cfg,endpoint+'_GT'),cfg['run_root'])
            return root

    def test_atomic_tiff_exact_uint32_and_interrupted_install(self):
            path=self.root/'final.tif'
            array=np.arange(260*7,dtype=np.uint32).reshape(260,7);array[-1,-1]=2**32-1
            atomic_tiff(path,array)
            np.testing.assert_array_equal(tifffile.imread(path),array)
            self.assertEqual(tifffile.imread(path).dtype,np.dtype(np.uint32))
            before=identity(path)
            with patch.object(common.os,'replace',side_effect=OSError('fixture before replace')):
                with self.assertRaises(OSError):atomic_tiff(path,array+np.uint32(1))
            self.assertEqual(identity(path),before)
            real_write=tifffile.imwrite
            def partial(path,data):
                Path(path).write_bytes(b'partial TIFF');raise OSError('fixture write interruption')
            with patch.object(tifffile,'imwrite',side_effect=partial):
                with self.assertRaises(OSError):atomic_tiff(path,array)
            self.assertEqual(identity(path),before)
            def wrong_pixels(path,data):real_write(path,np.zeros_like(data))
            with patch.object(tifffile,'imwrite',side_effect=wrong_pixels):
                with self.assertRaisesRegex(ValueError,'pixels differ'):atomic_tiff(path,array)
            self.assertEqual(identity(path),before)
            with self.assertRaises(ValueError):atomic_tiff(path,array.astype(np.uint16))
            self.assertGreaterEqual(len(list(self.root.glob('.final.tif.*'))),3)
            atomic_json(self.root/'fixture_receipt.json',dict(status='passed',uint32_exact=True,
                before_replace_and_during_write_failures_preserve_destination=True,
                temporary_pixel_corruption_rejected=True,final=identity(path)))

    def test_refine_actual_final_tiff_and_receipt_for_both_parents(self):
            for endpoint,boundary in zip(GT_FIXTURE_ENDPOINTS,(10,12,11)):
                cfg,manifest,data=self._refine_inputs(endpoint,boundary)
                with patch.object(gt,'load_data',return_value=data):result=gt.refine(cfg,manifest,endpoint)
                root=self._assert_refine(cfg,data,endpoint,result)
                before={str(p):identity(p) for p in root.rglob('*') if p.is_file()}
                with patch.object(gt,'load_data',side_effect=AssertionError('completed refine must only verify')):
                    self.assertEqual(gt.refine(cfg,manifest,endpoint),result)
                self.assertEqual(before,{str(p):identity(p) for p in root.rglob('*') if p.is_file()})
            atomic_json(self.root/'fixture_receipt.json',dict(status='passed',both_parent_domains=True,
                actual_refine_final_TIFF_and_complete_receipt=True,full_image_topology_oracle_equal=True,
                completed_receipt_reuse_verified=True))

    def test_refine_resume_before_and_after_final_replace(self):
            for point in ('before_replace','after_replace'):
                cfg,manifest,data=self._refine_inputs(point)
                endpoint=GT_FIXTURE_ENDPOINTS[0];root=Path(cfg['run_root'])/'endpoints'/endpoint;final=root/'final.tif'
                real_replace=common.os.replace;real_sync=common.fsync_dir
                def replace(source,target):
                    if point=='before_replace' and Path(target)==final:raise OSError('fixture before final replace')
                    return real_replace(source,target)
                def sync(path):
                    if point=='after_replace' and Path(path)==root:raise OSError('fixture after final replace')
                    return real_sync(path)
                with patch.object(gt,'load_data',return_value=data),patch.object(common.os,'replace',side_effect=replace),patch.object(common,'fsync_dir',side_effect=sync):
                    with self.assertRaises(OSError):gt.refine(cfg,manifest,endpoint)
                self.assertEqual(final.exists(),point=='after_replace')
                self.assertFalse((root/'gt/complete.json').exists())
                state=read(root/'gt/recovery/state.json');proposals=np.load(root/'gt/scores/proposals.npy')
                self.assertEqual(state['cursor'],len(proposals))
                before=self._upstream(root/'gt')
                with patch.object(gt,'load_data',return_value=data),patch.object(gt,'score_proposals',side_effect=AssertionError('recomputed scores')),patch.object(gt.CompleteCellTopology,'check',side_effect=AssertionError('reapplied proposals')):
                    if point=='after_replace':
                        with patch.object(gt,'atomic_tiff',side_effect=AssertionError('rewrote installed TIFF')):
                            result=gt.refine(cfg,manifest,endpoint)
                    else:result=gt.refine(cfg,manifest,endpoint)
                self._assert_refine(cfg,data,endpoint,result)
                self.assertEqual(self._upstream(root/'gt'),before)
                atomic_json(self.root/(point+'_receipt.json'),dict(status='passed',interruption=point,
                    committed_cursor=state['cursor'],upstream_hashes_and_mtimes_unchanged=True,
                    no_score_or_topology_replay=True,complete=identity(root/'gt/complete.json'),upstream=before))

    def test_refine_rejects_installed_tiff_drift_after_interruption(self):
            cfg,manifest,data=self._refine_inputs('drift');root=Path(cfg['run_root'])/'endpoints'/GT_FIXTURE_ENDPOINTS[0]
            real_sync=common.fsync_dir
            def sync(path):
                if Path(path)==root:raise OSError('fixture after final replace')
                return real_sync(path)
            with patch.object(gt,'load_data',return_value=data),patch.object(common,'fsync_dir',side_effect=sync):
                with self.assertRaises(OSError):gt.refine(cfg,manifest,GT_FIXTURE_ENDPOINTS[0])
            final=root/'final.tif';bad=tifffile.imread(final);bad[0,0]=99;tifffile.imwrite(final,bad)
            with patch.object(gt,'load_data',return_value=data):
                with self.assertRaisesRegex(ValueError,'final mask differs'):gt.refine(cfg,manifest,GT_FIXTURE_ENDPOINTS[0])
            self.assertFalse((root/'gt/complete.json').exists())

from bidcell.model.postprocess_predictions import postprocess_connect,remove_islands
native_connect=p2.native_connect
tile_coordinates=p2.tile_coordinates

class P2ExactCompatibility(unittest.TestCase):

    def test_current_native_and_full_coordinate_oracle(self):
            rng = np.random.default_rng(9817)
            changed = ties = 0
            for seed in range(24):
                nuclei = np.zeros((44,53),np.uint32)
                for cid in (3,9,17):
                    for _ in range(5):
                        nuclei[rng.integers(44),rng.integers(53)] = cid
                nuclei[18,25] = 99  # Complete nucleus with no proposal in this pre.
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
                self.assertNotIn(99,masks)
                self.assertFalse(np.any((p2==99)&(nuc==0)))
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
            # Squared distances differ by exactly one at a scale where float32
            # comparison would turn the strict nearest candidate into a tie.
            tall = np.zeros((100001,2),np.uint32)
            tall[0,0],tall[0,1] = 7,2
            exact = FullNucleusIndex(tall,2)
            near = exact.squared(7,[[100000,0]])[0]
            farther = exact.squared(2,[[100000,0]])[0]
            self.assertEqual(int(near),10000000000)
            self.assertEqual(int(farther-near),1)

    def test_native_tile_order_overlap(self):
            size,ys,xs,coords = tile_coordinates((140,190),96)
            self.assertEqual(size,96)
            self.assertEqual(ys,[0,44])
            self.assertEqual(xs,[0,94])
            self.assertEqual(coords,[(0,0),(0,94),(44,0),(44,94)])

if __name__=='__main__':unittest.main()
