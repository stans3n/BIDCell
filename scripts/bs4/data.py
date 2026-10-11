"""Adapter retains unmasked augmented counts and aligned frozen inputs."""
import numpy as np
import torch
import tifffile
from bidcell.model.dataio.dataset_input import DataProcessing
from bidcell.model.dataio.candidate_geometry import CandidateGeometry


class OffsetGeometry(CandidateGeometry):
    def __init__(self,labels,origin):super().__init__(labels);self.origin=origin
    def maps(self,ids,row,col,*args):return super().maps(ids,row+self.origin[0],col+self.origin[1],*args)

class BS4Dataset(DataProcessing):
    def __init__(self,cfg,isTraining=True,**kwargs):
        kwargs.setdefault('structured',True);kwargs.setdefault('mask_unknown_markers',True);kwargs.setdefault('total_steps',None)
        if cfg.mini_full_nuclei and kwargs.get('geometry_provider') is None:
            if cfg.mini_origin is None:raise ValueError('mini geometry origin absent')
            kwargs['geometry_provider']=OffsetGeometry(tifffile.imread(cfg.mini_full_nuclei),cfg.mini_origin)
        super().__init__(cfg.native(),isTraining=isTraining,**kwargs)
        self.alpha=cfg.alpha;self.beta=cfg.beta
        p=self.pos_markers.loc[self.type_names,self.gene_names].to_numpy();q=self.neg_markers.loc[self.type_names,self.gene_names].to_numpy()
        for values in (p,q):
            if not np.isfinite(values).all() or (values<0).any():raise ValueError('marker tables must be nonnegative finite')
        self.exclusive=(p>0)&~(q>0);self.marker_intersection=(p>0)&(q>0)

    def __getitem__(self,index):
        sample=super().__getitem__(index);k=len(sample['cell_ids'])
        positive=np.zeros((k,len(self.gene_names)),np.float32)
        for i,(valid,ct) in enumerate(zip(sample['type_valid'],sample['type_indices'])):
            if valid:positive[i]=self.exclusive[int(ct)]
        sample['exclusive_positive']=torch.from_numpy(positive)
        sample['alpha']=self.alpha;sample['beta']=self.beta
        return sample

def dataset(cfg,isTraining=True,**kwargs):return BS4Dataset(cfg,isTraining,**kwargs)
