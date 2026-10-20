"""Fresh-process legacy/paired native ROI output entrypoint for replay tests."""
import argparse
import gc
import json
from pathlib import Path
import sys
from types import SimpleNamespace


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',choices=('legacy','paired'),required=True)
    p.add_argument('--run-id',required=True);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--shared',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    source=Path('/home/zwan0726/work/projects/BIDCell_new_methods_20261013') if a.source=='legacy' else Path(__file__).resolve().parents[2]
    sys.path.insert(0,str(source))
    import numpy as np
    import torch
    import tifffile
    from scripts.new_methods.common import read,identity,atomic_json,atomic_npy,atomic_tiff
    from scripts.new_methods.prediction import merge_native
    from scripts.crgf.postprocess import postprocess
    from bidcell.model.utils.utils import get_seg_mask
    inp=Path('/home/zwan0726/work/outputs/bidcell_new_methods_20261013');roi=read(a.shared/'roi.json')
    y0,x0,size=roi['row'],roi['col'],roi['size'];out=a.out;pred=out/'predictions'/a.run_id;pred.mkdir(parents=True,exist_ok=True)
    state=torch.load(a.checkpoint,map_location='cpu')
    if a.source=='legacy':
        from scripts.new_methods.common import make_model
        from scripts.new_methods.data import NativeViews
        model=make_model('V2_MORPH',inp/'prepared/shared_initial.pth')
        factory=lambda shift:NativeViews(inp,training=False,shift=shift,auxiliary=True,morphology=True)
        run=lambda b:model(b['input'],b['aux'])
    else:
        from scripts.paired_v2.common import make_model,forward,RUNS
        from scripts.paired_v2.data import PairedViews
        kind='V2_MORPH' if a.run_id=='V2_MORPH' else RUNS[a.run_id]
        model=make_model(kind);factory=lambda shift:PairedViews(kind,training=False,shift=shift)
        run=lambda b:forward(model,b,kind)
    model.load_state_dict(state['model'],strict=True);model.eval();del state
    maps=[];patches=[]
    for shift in (0,24):
        data=factory(shift);pool=read(inp/f'configs/predict_shift{shift}.json')
        chosen=[v for v in pool if y0<=v['row'] and v['row']+48<=y0+size and x0<=v['col'] and v['col']+48<=x0+size]
        if not chosen:raise ValueError('fixed ROI lacks one of the native shifts')
        whole=np.zeros((size,size),np.uint32);records=[]
        with torch.inference_mode():
            for v in chosen:
                b=data.get_view(v,'cuda');index=v['index']
                if len(b['input']):
                    z=run(b);array=z.cpu().numpy();mask=get_seg_mask(array,b['labels'].cpu().numpy()).astype(np.uint32)
                else:array=np.zeros((0,2,48,48),np.float32);mask=np.zeros((48,48),np.uint32)
                atomic_npy(pred/f'logits_{shift}_{index}.npy',array)
                records.append(dict(view=v,candidate_ids=b['candidate_ids'].tolist(),logits=identity(pred/f'logits_{shift}_{index}.npy')))
                y,x=v['row']-y0,v['col']-x0;whole[y:y+48,x:x+48]=mask
        atomic_tiff(pred/f'shift{shift}.tif',whole);maps.append(whole);patches.append(dict(shift=shift,records=records))
        if shift==24:shifted=[f"{v['row']-y0}_{v['col']-x0}.hdf5" for v in chosen]
        del data;gc.collect()
    pre,fill=merge_native(*maps,shifted);atomic_tiff(pred/'pre.tif',pre);atomic_tiff(pred/'fill.tif',fill)
    adapter=SimpleNamespace(output_root=str(out),seed=1234,postprocess_size=1024,cpu_workers=1,
        native=lambda:SimpleNamespace(files=SimpleNamespace(data_dir=str(a.shared))))
    postprocess(adapter,a.run_id)
    nuc=tifffile.imread(a.shared/'nuclei.tif');final=tifffile.imread(pred/'final.tif')
    assert np.array_equal(final[nuc>0],nuc[nuc>0]) and set(np.unique(final))<=set(np.unique(nuc))
    atomic_json(out/'prediction_complete.json',dict(status='passed',source_root=str(source),entrypoint=identity(__file__),
        checkpoint=identity(a.checkpoint),roi=roi,patches=patches,
        producer={str(f.relative_to(source)):identity(f) for f in [source/'scripts/new_methods/prediction.py',source/'bidcell/model/model/model.py',source/'scripts/crgf/postprocess.py']},
        outputs={name:identity(pred/name) for name in ('shift0.tif','shift24.tif','pre.tif','fill.tif','final.tif')}))


if __name__=='__main__':main()
