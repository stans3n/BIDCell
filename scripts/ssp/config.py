"""Fixed S/SP contract; historical COS configs retain their original schema."""
from typing import Literal
from pathlib import Path
from pydantic import model_validator
from scripts.crgf.config import RunConfig
ARMS=('S','SP')
class SSPConfig(RunConfig):
    arm_id: Literal['S','SP']
    base_variant: Literal['B1']='B1'
    variant: Literal['B1']='B1'
    cos_enabled: Literal[True]=True
    queue_config: str
    method_version: Literal['SSP-OVERNIGHT-v1.0']='SSP-OVERNIGHT-v1.0'
    parent_sha256: Literal['02ce88f2007eaa7a4f4de534b951a4ea0c75243bccf2804e4ea17c50d4de3b39']='02ce88f2007eaa7a4f4de534b951a4ea0c75243bccf2804e4ea17c50d4de3b39'
    historical_updates: Literal[4000]=4000
    mini_test: bool=False
    anchor_weight: Literal[34.61094491628047]=34.61094491628047
    pos_background_rho: Literal[1.0]=1.0
    pos_region_version: Literal['outside-all-nuclei-search-no-positive-v1']='outside-all-nuclei-search-no-positive-v1'
    calibration_identity: Literal['c1f77e69c03e0f1768898989f6190e448140a0f805f8a8ed28d22de1e75cf64a']='c1f77e69c03e0f1768898989f6190e448140a0f805f8a8ed28d22de1e75cf64a'
    mini_origin: tuple[int,int] | None=None
    mini_full_nuclei: str | None=None
    prepared_manifest: str
    teacher_version: Literal['S-5x5-exclusive-positive-v1']='S-5x5-exclusive-positive-v1'
    prototype_version: Literal['SP-full-nucleus-20mu-R16-v1']='SP-full-nucleus-20mu-R16-v1'
    @model_validator(mode='after')
    def matched(self):
        if not self.mini_test and self.updates!=4000: raise ValueError('formal budget exactly 4000')
        if self.cpu_workers>2 or self.betas!=(.9,.999): raise ValueError('thread/Adam contract')
        return self

def read(path): return SSPConfig.model_validate_json(Path(path).read_text())
