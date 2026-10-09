"""Frozen A/B/C scientific contract; architecture and supervision are independent."""
from typing import Literal
from pathlib import Path
from pydantic import model_validator, Field
from scripts.crgf.config import RunConfig
ARMS=('A','B','C')
REGION='outside-all-nuclei-search-no-positive-v1'
class CosConfig(RunConfig):
    arm_id: Literal['A','B','C']
    base_variant: Literal['B1']='B1'
    cos_enabled: Literal[True]=True
    queue_config: str
    variant: Literal['B1']='B1'
    method_version: Literal['COS-LOSS-ABC-v2.0']='COS-LOSS-ABC-v2.0'
    parent_sha256: str
    historical_updates: Literal[4000]=4000
    mini_test: bool=False
    anchor_weight: float=Field(ge=0,le=1000)
    pos_background_rho: Literal[1.0,0.5]
    pos_region_version: Literal['outside-all-nuclei-search-no-positive-v1']=REGION
    calibration_identity: str
    @model_validator(mode='after')
    def matched(self):
        if self.arm_id=='A' and self.anchor_weight!=0: raise ValueError('A anchor must be zero')
        if self.arm_id!='A' and self.anchor_weight<1: raise ValueError('B/C calibrated anchor must be at least one')
        if self.pos_background_rho != (0.5 if self.arm_id=='C' else 1.): raise ValueError('arm rho mismatch')
        if not self.mini_test and self.updates!=4000: raise ValueError('formal budget exactly 4000')
        if self.cpu_workers>2: raise ValueError('GPU torch threads must be <=2')
        return self

def read(path): return CosConfig.model_validate_json(Path(path).read_text())
