"""Unrounded directions alongside capture, reference and connectivity costs."""
from pathlib import Path
import json
import numpy as np,pandas as pd
from scripts.crgf.common import atomic_json

def write(methods,out):
    out=Path(out);rows=[]
    for arm in ('P1','P2','T1','T2'):
        a=methods[arm];b=methods['B']
        ma=json.loads((Path(a['matrix'])/'complete.json').read_text())['totals'];mb=json.loads((Path(b['matrix'])/'complete.json').read_text())['totals']
        ca=ma['assigned']/ma['eligible'];cb=mb['assigned']/mb['eligible']
        rows.append(dict(arm=arm,metric='capture_fraction',arm_value=ca,B_value=cb,delta=ca-cb,adverse=bool(ca<cb),n_arm=ma['eligible'],n_B=mb['eligible'],direction='higher capture; separate from marker objective'))
        for metric,path,column,direction in [('disconnected_cell_fraction','engineering/structure/components_8.csv','n_components_8','lower'),('reference_mean_correlation','paper_aligned_v1/type_correlations.csv','pearson','higher')]:
            if metric=='disconnected_cell_fraction':
                # Accepted structural per-cell components are in a normalized distribution table.
                x=pd.read_csv(Path(a['evaluation'])/path);y=pd.read_csv(Path(b['evaluation'])/path)
                av=float((x.n_components_8>1).mean());bv=float((y.n_components_8>1).mean());na=len(x);nb=len(y)
                rows.append(dict(arm=arm,metric=metric,arm_value=av,B_value=bv,delta=av-bv,adverse=bool(av>bv),n_arm=na,n_B=nb,direction=direction))
            else:
                x=pd.read_csv(Path(a['evaluation'])/path);y=pd.read_csv(Path(b['evaluation'])/path)
                joined=x.merge(y,on=['cell_type','definition'],suffixes=('_arm','_B'),validate='one_to_one')
                for _,r in joined.iterrows():
                    delta=r.pearson_arm-r.pearson_B;rows.append(dict(arm=arm,metric='reference_'+r.definition+':'+r.cell_type,arm_value=r.pearson_arm,B_value=r.pearson_B,delta=delta,adverse=bool(delta<0) if np.isfinite(delta) else None,n_arm=r.n_cells_arm,n_B=r.n_cells_B,direction=direction))
    pd.DataFrame(rows).to_csv(out/'tradeoffs.csv',index=False,na_rep='NA')
