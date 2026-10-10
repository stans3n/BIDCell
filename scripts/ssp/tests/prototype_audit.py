"""Independent original processed-transcript rounding/count audit against map cache."""
import argparse,json,re,time
from pathlib import Path
import numpy as np
import pandas as pd
import tifffile
from scripts.crgf.common import identity,atomic_json,source_identity
from scripts.ssp.config import read
from scripts.ssp.prototype import PrototypeCache

def main(root):
    root=Path(root);cfg=read(root/'S/configs/run.json');native=cfg.native();inputs=json.loads((root/'configs/inputs.json').read_text());genes=inputs['gene_order'];cache=PrototypeCache(cfg.prepared_manifest,genes)
    labels=tifffile.imread(inputs['nuclei_mask']['path']);ids=cache.arrays['ids'];counts=np.zeros_like(cache.arrays['counts']);prior=np.zeros(len(genes),np.float64);union=np.zeros(labels.shape,bool)
    for v in json.loads((root/'configs/views.json').read_text())['views']:
        if v['k']:
            y,x=map(int,re.findall(r'\d+',Path(v['patch']).name)[:2]);union[y:y+48,x:x+48]=True
    names=native.transcripts;columns=[names.x_col,names.y_col,names.gene_col]+([names.counts_col] if names.counts_col else [])
    index=pd.Index(genes);rows=0;assigned=0.;start=time.monotonic();path=inputs['transcripts']['path']
    for block in pd.read_csv(path,usecols=columns,chunksize=500000):
        x=block[names.x_col].round().to_numpy(np.int64);y=block[names.y_col].round().to_numpy(np.int64);g=index.get_indexer(block[names.gene_col]);values=block[names.counts_col].to_numpy(np.float64) if names.counts_col else np.ones(len(x),np.float64)
        if (g<0).any() or (x<0).any() or (y<0).any() or (x>=labels.shape[1]).any() or (y>=labels.shape[0]).any():raise ValueError('raw transcript coordinate/gene contract')
        owner=labels[y,x];keep=owner>0;np.add.at(counts,(np.searchsorted(ids,owner[keep]),g[keep]),values[keep]);np.add.at(prior,g[union[y,x]],values[union[y,x]])
        rows+=len(x);assigned+=values[keep].sum()
    if not np.array_equal(counts,cache.arrays['counts']):raise AssertionError('input map vs original transcript nuclear integer counts mismatch')
    if not np.array_equal(prior,cache.arrays['train_union_gene_counts']):raise AssertionError('deduplicated train prior differs from original transcript')
    atomic_json(root/'tests/prototype_audit.json',dict(source=source_identity(['scripts/ssp/tests/prototype_audit.py','scripts/ssp/prototype.py']),status='passed',rows=rows,nuclear_counts=assigned,all_nuclei_all_genes_integer_equal=True,train_union_all_genes_equal=True,processed_transcripts=identity(path),cache=identity(cfg.prepared_manifest),coordinate_rule='processed inputs already scale/shift transformed; pandas ties-to-even round once to native 1um input map; no vendor IDs',wall_seconds=time.monotonic()-start))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();main(a.root)
