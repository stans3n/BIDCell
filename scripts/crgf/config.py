"""Strict stage-1 schema, separate from backward-compatible native YAML."""
from pathlib import Path
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field
import json
from bidcell.config import load_config


class RunConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')
    variant: Literal["E", "G", "R", "GR", "B1"] | None = None
    run_id: str
    output_root: str
    baseline_config: str
    checkpoint: str
    seed: Literal[20261008] = 20261008
    updates: int = Field(default=4000, gt=0)
    checkpoint_every: int = Field(default=500, gt=0)
    early_checkpoint: int = 100
    learning_rate: Literal[0.00001] = .00001
    betas: tuple[float, float] = (.9, .999)
    eps: Literal[0.00000001] = 1e-8
    weight_decay: Literal[0.0001] = .0001
    schedule: Literal['constant'] = 'constant'
    precision: Literal['fp32'] = 'fp32'
    workers: Literal[0] = 0
    cpu_workers: int = Field(default=4, ge=1, le=4)
    batch_groups: Literal[1] = 1
    shifts: tuple[Literal[0], Literal[24]] = (0,24)
    patch_size: Literal[48] = 48
    postprocess_size: Literal[1024] = 1024
    distance_cap: Literal[4.0] = 4.
    hidden_dim: Literal[32] = 32
    residual_scale: Literal[0.1] = .1
    method_version: Literal['CRGF-1008-v1.1', 'CRGF-EGR-1.1'] = 'CRGF-1008-v1.1'
    stage_timeouts: dict[str,int] = Field(default_factory=lambda: dict(train=21600,predict=43200,postprocess=28800,matrix=3600,evaluate=14400,report=1800))
    minimum_free_gb: float = 8.

    def native(self):
        cfg = load_config(self.baseline_config)
        if cfg.model_params.patch_size != 48 or cfg.model_params.name != 'custom':
            raise ValueError('baseline architecture mismatch')
        if cfg.model_params.elongated != ['Endothelial','Fibroblasts','Myofibroblasts','SMC']:
            raise ValueError('elongated types changed')
        if self.betas != (.9,.999): raise ValueError('Adam betas changed')
        if Path(self.output_root).resolve().is_relative_to(Path(cfg.files.data_dir).resolve()):
            raise ValueError('output inside read-only input root')
        return cfg


def read(path):
    return RunConfig.model_validate(json.loads(Path(path).read_text()))
