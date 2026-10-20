"""Real accepted receipt formats must retain complete tile ancestry."""
from pathlib import Path
import pytest
from scripts.boundary_three.common import read
from scripts.boundary_three.settings import default_config
from scripts.boundary_three.provenance import tile_receipt_identities
from scripts.boundary_three.evaluation import verify_legacy_receipt


def test_actual_historical_count_and_identity_list_formats():
    cfg=default_config()
    for name in ('BASE_MATCH','V2_MORPH'):
        folder=Path(cfg['endpoints'][name]['pre']).parent/'postprocess'
        completion=read(folder/'complete.json');manifest=read(folder/'manifest.json')
        assert type(completion['tiles']) is int
        assert tile_receipt_identities(completion)==[]
        tiles=tile_receipt_identities(manifest)
        assert len(tiles)==completion['tiles']==48
        for obj in tiles:
            receipt=verify_legacy_receipt(obj['path'])
            assert receipt['outputs']


def test_malformed_historical_tile_ancestry_rejected():
    for value in (-1,True,'48',[dict(path='unhashed.json')]):
        with pytest.raises(ValueError):tile_receipt_identities(dict(tiles=value))
