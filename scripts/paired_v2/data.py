"""Frozen native views; extra features are deterministic read-only sidecars."""
import numpy as np
import torch
from scripts.new_methods.data import NativeViews, transform_matrix
from .common import INPUT, rng_state, restore_rng


def relations(evidence, ids, flip, rot):
    ids = np.asarray(ids, dtype=np.int64)
    if len(np.unique(ids)) != len(ids) or np.any(ids <= 0):
        raise ValueError('candidate IDs must be unique positive canonical IDs')
    rows = [evidence.id_to_row[int(cid)] for cid in ids]  # absent IDs fail closed
    mu = evidence.centroids[rows]
    radii = np.asarray([max(np.sqrt(len(evidence.nucleus(int(cid))[0])/np.pi), .5) for cid in ids])
    delta = mu[None, :, :] - mu[:, None, :]
    den = radii[:, None] + radii[None, :]
    vector = (delta @ transform_matrix(flip, rot).T)/den[:, :, None]
    distance = np.linalg.norm(delta, axis=-1)/den
    ratio = np.log(radii[None, :]/radii[:, None])
    return np.concatenate((np.clip(vector, -4, 4)/4, np.minimum(distance, 4)[..., None]/4,
                           (np.clip(ratio, -2, 2)/2)[..., None]), -1).astype(np.float32)


class PairedViews(NativeViews):
    def __init__(self, kind, training=True, shift=0, diagnostic=False):
        morph = kind in ('V2_MORPH', 'V2_INTERACT')
        super().__init__(INPUT, training=training, shift=shift, auxiliary=morph or diagnostic, morphology=morph)
        self.kind = kind
        if self.evidence is not None:
            original = self.evidence.aux
            def pure_aux(native, raw, flip, rot, morphology):
                saved = rng_state()
                try:
                    aux = original(native, raw, flip, rot, morphology)
                    if kind == 'V2_INTERACT':
                        aux['relation'] = torch.from_numpy(relations(self.evidence, aux['candidate_ids'].numpy(), flip, rot))
                    return aux
                finally:
                    # Auxiliary deterministic transforms must not advance the
                    # native data RNG relative to the BASE training path.
                    restore_rng(saved)
            self.evidence.aux = pure_aux

    def get_view(self, view, device='cpu'):
        if 'path' in view and str(self.fp_patches[int(view['index'])]) != view['path']:
            raise ValueError('frozen patch/index mismatch')
        b = super().get_view(view, device)
        if 'K' in view and len(b['input']) != view['K']:
            raise ValueError('frozen candidate count mismatch')
        if 'row' in view and b['xy'] != (view['row'], view['col']):
            raise ValueError('frozen native coordinate mismatch')
        return b
