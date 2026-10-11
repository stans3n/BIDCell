"""Lossless deterministic CSV compression with unambiguous legacy readers."""
from pathlib import Path
import gzip,io
import pandas as pd

def table_path(path):
    path=Path(path)
    candidates=[path,Path(str(path)+'.gz')] if path.suffix!='.gz' else [path]
    present=[p for p in candidates if p.exists()]
    if len(present)!=1:raise ValueError('expected exactly one table representation: '+str(path))
    return present[0]

def read_table(path,**kwargs):return pd.read_csv(table_path(path),**kwargs)

def write_table(frame,path,**kwargs):
    path=Path(path)
    if path.suffix!='.gz':path=Path(str(path)+'.gz')
    path.parent.mkdir(parents=True,exist_ok=True)
    # No filename/timestamp header: identical CSV content has identical compressed bytes.
    with path.open('wb') as raw:
        with gzip.GzipFile(fileobj=raw,mode='wb',filename='',mtime=0,compresslevel=6) as zipped:
            with io.TextIOWrapper(zipped,encoding='utf-8',newline='') as text:frame.to_csv(text,**kwargs)
    return path
