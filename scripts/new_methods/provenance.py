"""Component producers permit reviewed report repairs without retraining."""
from .common import REPO,sha256


def component_source(stage):
    common=['scripts/new_methods/common.py','scripts/new_methods/provenance.py','scripts/crgf/common.py']
    native=[str(p.relative_to(REPO)) for p in (REPO/'bidcell').rglob('*.py')]
    core=common+native+['scripts/new_methods/data.py','scripts/new_methods/training.py','scripts/new_methods/inputs.py']
    if stage=='train':paths=core+['scripts/new_methods/diagnostics.py']
    elif stage=='predict':paths=core+['scripts/new_methods/prediction.py']
    elif stage=='postprocess':paths=common+['scripts/crgf/postprocess.py','bidcell/model/postprocess_predictions.py']
    elif stage=='matrix':paths=common+['scripts/crgf/matrix.py']
    elif stage=='evaluate':paths=common+['scripts/new_methods/evaluation.py','scripts/new_methods/accepted_profiles.py','scripts/new_methods/author_marker.py']+[str(p.relative_to(REPO)) for p in (REPO/'scripts/crgf').glob('*') if p.suffix in ('.py','.R')]
    elif stage=='report':paths=common+['scripts/new_methods/report.py','scripts/new_methods/author_marker.py']
    else:raise ValueError(stage)
    return {p:sha256(REPO/p) for p in sorted(set(paths))}
