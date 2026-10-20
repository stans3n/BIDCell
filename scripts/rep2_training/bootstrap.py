"""Materialize a local native configuration from the frozen baseline template."""
import json
from pathlib import Path
import yaml
from .common import ROOT, atomic_json, identity

def main():
    template = Path('/home/zwan0726/work/baseline/latest_bidcell_official_run/configs/xenium_breast_official_step4000.yaml')
    data = Path('/home/zwan0726/work/data/Xenium_FFPE_Human_Breast_Cancer_Rep2/outs')
    for name in ('configs', 'logs', 'receipts', 'progress', 'shared/data', 'shared/prepared', 'shared/configs', 'tests'):
        (ROOT/name).mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(template.read_text())
    cfg['files'].update(data_dir=str(ROOT/'shared/data'), fp_dapi=str(data/'morphology_mip.ome.tif'), fp_transcripts=str(data/'transcripts.csv.gz'))
    cfg['affine'].update(base_pix_x=.2125, base_pix_y=.2125)
    path = ROOT/'configs/native.yaml'
    content = yaml.safe_dump(cfg, sort_keys=False)
    if path.exists() and path.read_text()!=content: raise ValueError('native configuration already frozen differently')
    path.write_text(content)
    initial = Path('/home/zwan0726/work/outputs/bidcell_new_methods_20261013/prepared/shared_initial.pth')
    # Large raw input hashes are supplied by the independent preflight before launch.
    run = dict(version='REP2-TRAINING-ONLY-v1', seed=1234, shared_initial=str(initial),
               baseline_template=identity(template), checkpoint_every=100,
               input_identities={k:identity(cfg['files'][k]) for k in ('fp_ref','fp_pos_markers','fp_neg_markers')},
               initial_identity=identity(initial), gene_panel='/home/zwan0726/work/baseline/latest_bidcell_official_run/all_gene_names.txt',
               morphology_initialization_seed=20261013, morphology_seed_scope='frozen M-V2.2 module constructor fork_rng only; training/data seeds remain1234',
               preflight_identities={name:identity(ROOT/'manifests'/f'{name}.json') for name in ('raw_input_audit','source_audit','initialization_audit','reference_metadata_audit','deferred_reference_asset_audit','resource_admission')},
               authority_identity=identity(ROOT/'authority/ISBI2027_Authoritative_Writing_Pack_V4/REP2_FROZEN_MINIMAL_PROTOCOL_V4.md'),
               scope_identity=identity(ROOT/'manifests/execution_scope.json'), excluded_stages=['inference','P2','refinement','labels','scClassify','evaluation'],
               b1=dict(parent='R2_BIDCELL_NATIVE_MATCHED', additional_updates=4000, optimizer='new Adam; no parent moments',
                       canonical_source='/home/zwan0726/work/projects/BIDCell_crgf_1008_20261008T031528/scripts/crgf/training.py',
                       sequence='full native 80% pool; numpy RandomState(1234) permutation then flip/rotation',
                       loss='unchanged native six weights, including original unknown indexing'))
    run['input_identities']['gene_panel']=identity(run['gene_panel'])
    target = ROOT/'configs/run.json'
    if target.exists() and 'preflight_identities' in json.loads(target.read_text()) and json.loads(target.read_text())!=run:raise ValueError('run configuration differs')
    atomic_json(target, run)

if __name__ == '__main__': main()
