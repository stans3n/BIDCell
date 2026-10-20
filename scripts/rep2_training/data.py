"""Dataset-location adapter around the unchanged frozen native tuple loader."""
import random
import natsort
import numpy as np
from .common import ROOT, configure
from scripts.new_methods.data import NativeViews, IndexedIDs
from scripts.paired_v2.common import rng_state, restore_rng

class Views(NativeViews):
    def __init__(self, kind='BASE', full_pool=False):
        configure();self.full_pool=full_pool
        super().__init__(ROOT/'shared', auxiliary=kind=='V2_MORPH', morphology=kind=='V2_MORPH')
        if full_pool:
            files=natsort.natsorted((ROOT/'shared/data/expr_maps/expr_maps_input_patches_48x48_shift_0').glob('*.hdf5'))
            indices=random.Random(1234).sample(range(len(files)),k=int(.8*len(files)))
            self.fp_patches=[str(files[i]) for i in indices]
        if self.evidence is not None:
            original=self.evidence.aux
            def pure_aux(*args, **kwargs):
                saved=rng_state()
                try:return original(*args, **kwargs)
                finally:restore_rng(saved)
            self.evidence.aux=pure_aux

    def get_view(self, view, device='cpu'):
        if str(self.fp_patches[int(view['index'])])!=view['path']:raise ValueError('frozen path/index differs')
        batch=super().get_view(view,device)
        if self.full_pool:
            # Canonical B1 structured tuple makes only x contiguous; geometry
            # is unused by fusion=False and does not enter the forward/loss.
            batch['input']=batch['input'].contiguous()
        if len(batch['input'])!=view['K'] or batch['xy']!=(view['row'],view['col']):raise ValueError('frozen K/coordinates differ')
        return batch
