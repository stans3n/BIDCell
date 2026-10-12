"""Versioned engineering utilities. Never write into the historical baseline."""
import hashlib
import json
import os
import time
import threading
from pathlib import Path

VERSION = 'remote-prep-1.0'

def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''): h.update(b)
    return h.hexdigest()

def identity(path):
    p=Path(path).resolve(); s=p.stat()
    return dict(path=str(p), bytes=s.st_size, mtime_ns=s.st_mtime_ns, sha256=sha256(p))

def atomic_json(path, value):
    p=Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    tmp=p.with_name(p.name+'.tmp')
    with tmp.open('w') as f:
        json.dump(value, f, indent=2, allow_nan=False); f.flush(); os.fsync(f.fileno())
    os.replace(tmp,p)

def atomic_npy(path, value):
    import numpy as np
    p=Path(path); tmp=p.with_name(p.name+'.tmp')
    with tmp.open('wb') as f:
        np.save(f,value,allow_pickle=False); f.flush(); os.fsync(f.fileno())
    os.replace(tmp,p)

class Monitor:
    """One process (no workers used here), sampled RSS/PSS/swap and OS I/O."""
    def __init__(self,path): self.path=Path(path); self.stop=threading.Event(); self.samples=[]
    def __enter__(self):
        self.start=time.monotonic(); self.io=self.read('/proc/self/io')
        self.thread=threading.Thread(target=self.loop,daemon=True); self.thread.start(); return self
    def loop(self):
        while not self.stop.is_set():
            m=self.read('/proc/self/smaps_rollup'); v=self.read('/proc/meminfo')
            self.samples.append(dict(elapsed=time.monotonic()-self.start,rss=m.get('Rss',0)*1024,pss=m.get('Pss',0)*1024,process_swap=m.get('Swap',0)*1024,available=v.get('MemAvailable',0)*1024,system_swap=(v.get('SwapTotal',0)-v.get('SwapFree',0))*1024))
            self.stop.wait(0.5)
    @staticmethod
    def read(path):
        result={}
        for line in Path(path).read_text().splitlines():
            if ':' in line:
                k,v=line.split(':',1)
                try: result[k]=int(v.split()[0])
                except (ValueError,IndexError): pass
        return result
    def __exit__(self,*args):
        self.stop.set(); self.thread.join(); io=self.read('/proc/self/io')
        result=dict(wall_seconds=time.monotonic()-self.start,nominal_sample_seconds=.5,samples=self.samples,io_delta={k:io[k]-self.io[k] for k in io},scope='single process, sampled peaks may miss transients; OS cache uncontrolled',failed=args[0] is not None)
        atomic_json(self.path,result)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def atomic_torch(path, value):
    import torch
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+'.tmp')
    with tmp.open('wb') as stream:
        torch.save(value, stream); stream.flush(); os.fsync(stream.fileno())
    os.replace(tmp, path)


def atomic_tiff(path, value):
    import tifffile
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+'.tmp')
    tifffile.imwrite(tmp, value, photometric='minisblack')
    with tmp.open('rb') as stream: os.fsync(stream.fileno())
    os.replace(tmp, path)


def source_identity(paths):
    root = Path(__file__).resolve().parents[2]
    return {p: sha256(root/p) for p in paths}


def verify_outputs(manifest):
    for obj in manifest.get('outputs', {}).values():
        if sha256(obj['path']) != obj['sha256']:
            raise ValueError('artifact changed: '+obj['path'])


def process_identity(pid=None):
    import socket
    pid = pid or os.getpid()
    return dict(pid=pid, host=socket.gethostname(), boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                start_ticks=Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1].split()[19])
