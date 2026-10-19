import json
from pathlib import Path
import pytest
from scripts.boundary_three.common import *
from scripts.boundary_three.settings import default_config

def test_only_isolated_new_outputs():
    cfg=default_config()
    assert endpoint_registry()['V2_MORPH']['root'].startswith(str(HISTORICAL_INPUT))
    assert endpoint_spec(cfg,'P2_MORPH')['root'].startswith(cfg['output_root'])
    with pytest.raises(ValueError):output_path(cfg,'..','..','history')
    cfg['output_root']=str(HISTORICAL_INPUT)
    with pytest.raises(ValueError):load_run(cfg)

def test_changed_order_and_wrong_model_endpoint_are_rejected():
    cfg=default_config();cfg['node_order'][3]='MORPH_NO_DAPI'
    with pytest.raises(ValueError):load_run(cfg)
    cfg=default_config();cfg['endpoints']['V2_MORPH']['root']=str(OUTPUT_ROOT/'MORPH_NO_DAPI')
    with pytest.raises(ValueError):load_run(cfg)

def test_incomplete_and_hash_changed_receipts_rejected(tmp_path):
    # Use an isolated new-output run, and preserve any preexisting files.
    import uuid
    cfg=default_config(OUTPUT_ROOT/'mini'/'contract-tests'/uuid.uuid4().hex)
    root=output_path(cfg,'A1_CORE');root.mkdir(parents=True)
    artifact=root/'scores.csv';artifact.write_text('cell_id,score\n1,0\n')
    write_complete(cfg,'A1_CORE','NO_USEFUL_INCREMENT')
    assert validate_complete(cfg,'A1_CORE')['science_status']=='NO_USEFUL_INCREMENT'
    artifact.write_text('cell_id,score\n1,1\n')
    with pytest.raises(ValueError):validate_complete(cfg,'A1_CORE')
    saved=read(root/'complete.json');saved['execution_status']='FAILED_INTEGRITY';atomic_json(root/'complete.json',saved)
    with pytest.raises(ValueError):validate_complete(cfg,'A1_CORE')
