"""Pinned external author spatial scorer and immutable isolated R library."""
from pathlib import Path
import shutil
from .common import identity,read,sha256,check_identity,output_path
from .evaluation import check_legacy_identity


def verify_spatial_bindings(authority, pipeline):
    """Content identities come from frozen history, never today's accepted hash."""
    spatial=authority['spatial_diversity'];author=dict(path=spatial['source'],sha256=spatial['sha256'])
    wrapper=authority['extension']['spatial_wrapper'];bindings=pipeline['bindings']
    for obj in (author,wrapper,bindings['spatial_author'],bindings['spatial_wrapper'],bindings['spatial_utils']):
        check_legacy_identity(obj)
    if bindings['spatial_author']['sha256']!=author['sha256'] or bindings['spatial_wrapper']['sha256']!=wrapper['sha256']:
        raise ValueError('historical spatial producers disagree with authority')
    library=Path(author['path']).parent/'R_library'
    expected={}
    for key,obj in bindings.items():
        if key.startswith('entropy_'):
            check_legacy_identity(obj)
            relative=Path(obj['path']).relative_to(library).as_posix()
            expected[relative]=identity(obj['path'])
    actual={p.relative_to(library).as_posix() for p in library.rglob('*') if p.is_file()}
    if not expected or actual!=set(expected):raise ValueError('isolated R library file inventory drift')
    return dict(author=identity(author['path']),wrapper=identity(wrapper['path']),
        utils=identity(bindings['spatial_utils']['path']),library_root=str(library),library_files=expected,
        library_version=spatial['entropy_version'])


def spatial_bindings(cfg):
    authority=read(cfg['inputs']['authoritative_protocol'])
    # Current20 authority pins the accepted prior execution completion. Its
    # pipeline identity pins every entropy binary/database and utility file.
    historical=authority['paired_extension']['completion'];check_legacy_identity(historical)
    completion=read(historical['path']);check_legacy_identity(completion['pipeline_config'])
    pipeline=read(completion['pipeline_config']['path'])
    result=verify_spatial_bindings(authority,pipeline)
    result.update(historical_completion=identity(historical['path']),
        historical_pipeline_config=identity(completion['pipeline_config']['path']))
    return result


def snapshot_spatial_library(cfg,work,bound):
    work=Path(work).resolve();root=Path(cfg['output_root']).resolve()
    if root not in work.parents:raise ValueError('spatial snapshot outside isolated run')
    lib=work/'sources/R_library'
    if lib.is_symlink():
        if lib.resolve()!=Path(bound['library_root']).resolve():raise ValueError('unexpected old spatial library symlink')
        lib.unlink()  # Remove only this run's reference, never the original library.
    for relative,obj in bound['library_files'].items():
        check_identity(obj);target=lib/relative;target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():
            if sha256(target)!=obj['sha256']:raise ValueError('existing spatial library snapshot drift')
        else:shutil.copyfile(obj['path'],target)
        if sha256(target)!=obj['sha256']:raise ValueError('spatial library copy differs')
    actual={p.relative_to(lib).as_posix() for p in lib.rglob('*') if p.is_file()}
    if actual!=set(bound['library_files']):raise ValueError('spatial library snapshot contains foreign files')
    return lib
