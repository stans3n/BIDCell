"""The queue configuration is separate from scientific per-variant configs."""
import json
from pathlib import Path
from pydantic import BaseModel,ConfigDict,Field
class QueueConfig(BaseModel):
    model_config=ConfigDict(extra='forbid')
    run_id:str
    output_root:str
    methods:dict[str,dict[str,str]]
    paper_protocol:str
    prepared:str
    cpu_workers:int=4
    minimum_free_gb:float=12.
    stage_timeouts:dict[str,int]=Field(default_factory=lambda:dict(train=21600,predict=43200,postprocess=28800,matrix=7200,evaluate=43200,report=21600))
def read(path):
    cfg=QueueConfig.model_validate_json(Path(path).read_text())
    if set(cfg.methods)!=set(('E','G','R')):raise ValueError('E/G/R required')
    return cfg
