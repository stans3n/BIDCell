from pathlib import Path
import pytest
from scripts.boundary_three.common import identity
from scripts.boundary_three.external import verify_spatial_bindings


def fixture(tmp_path):
    author=tmp_path/'publish_spatialDiversity_metrics.R';author.write_text('pinned author\n')
    wrapper=tmp_path/'wrapper.R';wrapper.write_text('pinned wrapper\n')
    util=tmp_path/'utils.R';util.write_text('pinned utility\n')
    lib=tmp_path/'R_library/entropy/R/entropy.rdb';lib.parent.mkdir(parents=True);lib.write_bytes(b'pinned database')
    a=identity(author);w=identity(wrapper)
    authority=dict(spatial_diversity=dict(source=a['path'],sha256=a['sha256'],entropy_version='1.3.2'),extension=dict(spatial_wrapper=w))
    pipeline=dict(bindings=dict(spatial_author=a,spatial_wrapper=w,spatial_utils=identity(util),entropy_database=identity(lib)))
    return authority,pipeline,author,wrapper,lib


def test_author_and_wrapper_drift_rejected(tmp_path):
    a,p,author,wrapper,lib=fixture(tmp_path);verify_spatial_bindings(a,p)
    author.write_text('changed author\n')
    with pytest.raises(ValueError):verify_spatial_bindings(a,p)
    author.write_text('pinned author\n');wrapper.write_text('changed wrapper\n')
    with pytest.raises(ValueError):verify_spatial_bindings(a,p)


def test_external_library_changed_or_extra_file_rejected(tmp_path):
    a,p,author,wrapper,lib=fixture(tmp_path);verify_spatial_bindings(a,p)
    lib.write_bytes(b'changed database')
    with pytest.raises(ValueError):verify_spatial_bindings(a,p)
    lib.write_bytes(b'pinned database');(lib.parent/'foreign.rdb').write_bytes(b'extra')
    with pytest.raises(ValueError):verify_spatial_bindings(a,p)
