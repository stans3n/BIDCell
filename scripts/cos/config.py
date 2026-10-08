"""No implicit base variant or unrecognised scientific configuration."""
from typing import Literal
from pathlib import Path
from pydantic import model_validator
from scripts.crgf.config import RunConfig

ARMS=('B1_CONT','B1_COS','G_CONT','G_COS')
class CosConfig(RunConfig):
    arm_id: Literal['B1_CONT','B1_COS','G_CONT','G_COS']
    base_variant: Literal['B1','G']
    cos_enabled: bool
    queue_config: str
    variant: Literal['B1','G']
    method_version: Literal['COS-min-v1'] = 'COS-min-v1'
    parent_sha256: str
    historical_updates: Literal[4000] = 4000
    mini_test: bool = False
    @model_validator(mode='after')
    def matched(self):
        if self.arm_id != self.base_variant+('_COS' if self.cos_enabled else '_CONT') or self.variant != self.base_variant:
            raise ValueError('arm/base/COS mismatch')
        if not self.mini_test and self.updates != 4000: raise ValueError('formal budget is exactly 4000')
        return self

def read(path): return CosConfig.model_validate_json(Path(path).read_text())
