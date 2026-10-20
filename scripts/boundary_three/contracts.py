"""Shared, lightweight contracts; no legacy runner or mutable global registry."""
from pathlib import Path

VERSION = 'BOUNDARY-THREE-20261015-v1.0'
WORK = Path('/home/zwan0726/work')
SOURCE_ROOT = WORK / 'projects/BIDCell_boundary_three_20261015'
OUTPUT_ROOT = WORK / 'outputs/bidcell_boundary_three_20261015'
PYTHON = '/home/zwan0726/miniforge3/envs/bidcell_local/bin/python'
RSCRIPT = '/home/zwan0726/work/envs/bidcell_egr_r/bin/Rscript'
HISTORICAL_INPUT = WORK / 'outputs/bidcell_new_methods_20261013'
PAIRED_INPUT = WORK / 'outputs/bidcell_v2_paired_20261014'
GPU_LOCK = WORK / 'outputs/cos_gpu.lock'
NODES = ('A1_CORE', 'P2_BASE', 'MORPH_NO_DAPI', 'P2_MORPH', 'A2_CORE',
         'MORPH_SIMPLE_FUSION', 'MORPH_NO_GEO')
MINIMUM_NODES = NODES[:3]
A2_ENDPOINTS = ('A2_NATIVE', 'A2_GT', 'A2_GTC', 'A2_SHUFFLED')
EXECUTION_STATUSES = ('PREPARED', 'ADMITTED', 'RUNNING', 'COMPLETED',
    'FAILED_INTEGRITY', 'FAILED_RUNTIME', 'PAUSED_BUDGET', 'PAUSED_RESOURCE',
    'SKIPPED_SCIENTIFIC_GATE', 'DEFERRED_NOT_ADMITTED', 'DEFERRED_TIME')
A1_SCIENCE_STATUSES = ('GO_A2', 'NO_USEFUL_INCREMENT', 'INCONCLUSIVE', 'INVALID')
RECEIPT_SCHEMA = 'boundary-three-receipt-v1'
PLAN_FILES = {'a_core':'BIDCell_A_Core_Loss_Validation_ChangePlan_20261015.md',
    'p2':'BIDCell_P2_Transfer_ChangePlan_20261015.md',
    'morph':'BIDCell_Morph_Ablation_ChangePlan_20261015.md',
    'execution':'BIDCell_Linear_Execution_Plan_20261015.md'}

def endpoint_registry():
    result = {}
    for name, root in [('BASE_MATCH', PAIRED_INPUT), ('V2_MORPH', HISTORICAL_INPUT)]:
        base = root / name
        result[name] = dict(endpoint_id=name, root=str(base),
            machine=str(base/'configs/machine.json'),
            pre=str(base/'predictions'/name/'pre.tif'),
            final=str(base/'predictions'/name/'final.tif'),
            checkpoint=str(base/'checkpoints/final.pth'),
            matrix=str(base/'matrix'/name), evaluation=str(base/'evaluation'/name))
    return result
