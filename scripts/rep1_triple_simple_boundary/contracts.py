"""Executable scope is the byte-bound, root-approved DAG, not an old runner."""
from pathlib import Path
import hashlib
import json

SCHEMA = 'rep1-triple-simple-boundary-science-v1'
CONFIG_SCHEMA = 'rep1-triple-simple-boundary-run-v1'
INPUT_SCHEMA = 'rep1-triple-simple-boundary-inputs-v1'
DAG_SHA256 = '9585ee48fa04961542aaeb4476eeb1e2f6c605b4a4a530eabbd94ea464a06be3'
INSPECTED_SHA256 = '06fc8b2521bdb38863dc7cdc4a8cd3758f2761ab8305412ed7d02db81c2c6995'
PLAN_MANIFEST_SHA256 = 'e7c1aa8ce8c87a7419f585e019ea99aa1a615ca477a8cae1d021e2d4fcb3250b'
PLAN_AUDIT_SHA256 = '4ec6026e72b52d80a11ad2f6415373b432bcc810894b7a56a823444ffe893743'

def approved_dag():
    raw = Path(__file__).with_name('approved_dag.json').read_bytes()
    if hashlib.sha256(raw).hexdigest() != DAG_SHA256:
        raise ValueError('approved scientific DAG source drift')
    return json.loads(raw)

DAG = approved_dag()
NEW = tuple(DAG['new4'])
EXISTING = tuple(DAG['reused8'])
METHODS = tuple(DAG['methods12'])
CUBE = dict(DAG['cube'])
FACTORIAL8 = tuple(CUBE.values())
PARENTS = {k:v['parent'] for k,v in DAG['endpoints'].items()}
GT_ENDPOINTS = tuple(k for k,v in DAG['endpoints'].items() if v['operation']=='GT_parent_final')
STAGE_ORDER = tuple(x['id'] for x in DAG['stages'])
STAGES = {x['id']:x for x in DAG['stages']}
PRODUCTION_STAGES = dict(zip(('02','06','08','10'),NEW))
EVALUATION_STAGES = dict(zip(('03','07','09','11'),NEW))
N_CELLS, N_GENES, ELIGIBLE = 107955, 313, 34442716
APPLICABLE_TYPES = [0,2,3,4,5,6,7,9,10,13]
METRICS = tuple(DAG['score_metrics'])
SCALES = (1000,500,2000)
REGIONS = ('all','competition','interface','nuclear','noncompetitive','interface_positive','competition_interface','competition_only','interface_only','oob')

def normalize_stage(stage):
    stage = str(stage)
    if stage in STAGES:
        return stage
    for key, value in STAGES.items():
        if stage == value['name']:
            return key
    raise ValueError('unsupported compute stage: '+stage)

def comparisons(methods=METHODS):
    return [(r['candidate'],r['control']) for r in DAG['comparisons']
            if r['candidate'] in methods and r['control'] in methods]

def comparison_id(a,b):
    return a+'__minus__'+b

def evaluation_pairs(name):
    if name not in NEW:raise ValueError('paired full evaluation is restricted to the four new endpoints')
    return list(dict.fromkeys(((name,PARENTS[name]),(name,'BASE_MATCH'))))
