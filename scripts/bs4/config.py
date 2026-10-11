"""Fixed S/SP contract; historical COS configs retain their original schema."""
from typing import Literal
from pathlib import Path
from pydantic import model_validator
from scripts.crgf.config import RunConfig
ARMS=('P1','P2','T1','T2')
class BS4Config(RunConfig):
    arm_id: Literal['P1','P2','T1','T2','B']
    base_variant: Literal['B1']='B1'
    variant: Literal['B1']='B1'
    cos_enabled: Literal[True]=True
    queue_config: str
    method_version: Literal['BS4-OVERNIGHT-v2.0']='BS4-OVERNIGHT-v2.0'
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
    alpha: float=0.
    beta: float=1.
    post_rule: Literal['B','P1','P2']='B'
    reused_prediction_root: str | None=None
    @model_validator(mode='after')
    def matched(self):
        if not self.mini_test and self.updates!=4000: raise ValueError('formal budget exactly 4000')
        if self.cpu_workers>2 or self.betas!=(.9,.999): raise ValueError('thread/Adam contract')
        if self.arm_id=='T1' and (self.alpha,self.beta)!=(0.,1.):raise ValueError('T1 alpha/beta')
        if self.arm_id=='T2' and (self.alpha,self.beta)!=(.25,1.):raise ValueError('T2 alpha/beta')
        if self.arm_id in ('P1','P2') and self.post_rule!=self.arm_id:raise ValueError('P rule identity')
        if self.arm_id in ('T1','T2') and self.post_rule!='B':raise ValueError('T native rule')
        return self

def read(path): return BS4Config.model_validate_json(Path(path).read_text())
