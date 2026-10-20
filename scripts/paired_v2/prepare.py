"""Verify immutable historical producers/inputs, then freeze complete step zero."""
import copy
import gc
import shutil
from .common import *


def verify_inputs(root=DEFAULT_ROOT, full=False):
    root = Path(root); expected = read(AUDIT)
    source = read(INPUT/'source/manifest.json')
    if source['source_digest'] != '6e69adee2175963621226287c4da2a8283c15502ab6ebaa4193c5b1f97d03211' or len(source['files']) != 55:
        raise ValueError('historical source manifest identity mismatch')
    for name, value in source['files'].items():
        if sha256(HISTORICAL_SOURCE/name) != value:
            raise ValueError('historical frozen source changed: '+name)
    for name, obj in expected['inputs'].items():
        if sha256(obj['path']) != obj['sha256']:
            raise ValueError('historical immutable input changed: '+name)
    prep = read(INPUT/'prepared/manifest.json')
    for name, value in prep['preparation_source'].items():
        if sha256(HISTORICAL_SOURCE/name) != value:
            raise ValueError('historical preparation producer changed: '+name)
    for obj in [*prep['configs'].values(), *prep['dependencies'].values(), *prep['outputs'].values()]:
        if sha256(obj['path']) != obj['sha256']:
            raise ValueError('prepared evidence/input changed: '+obj['path'])
    count = 0; hashes = {}
    if full:
        for item in prep['raw_maps']:
            obj = item['source']
            if sha256(obj['path']) != obj['sha256']:
                raise ValueError('raw expression changed')
        for shift in (0,24):
            for view in read(INPUT/f'configs/predict_shift{shift}.json'):
                obj = view['identity']
                if sha256(view['path']) != obj['sha256']:
                    raise ValueError('native patch changed: '+view['path'])
                hashes[view['path']] = obj['sha256']; count += 1
    visits = read(INPUT/'configs/views.json')['views']
    if (len(visits),sum(v['K']>0 for v in visits),sum(v['K']==0 for v in visits),max(v['K'] for v in visits)) != (3998,3625,373,40):
        raise ValueError('frozen training visit budget mismatch')
    rec = dict(status='passed',full_patch_hash_check=full,patch_files_checked=count,
               patch_digest=digest(hashes) if full else None,historical_source=identity(INPUT/'source/manifest.json'),
               dependencies=expected['inputs'],historical_source_files=55,
               training_visits=3998,updates=3625,empty_skips=373,contract=identity(CONTRACT))
    if full:
        path = root/'prepared/input_audit.json'
        if path.exists() and read(path) != rec:
            raise ValueError('immutable paired input audit differs')
        if not path.exists(): atomic_json(path,rec)
    else:
        saved=read(root/'prepared/input_audit.json')
        if not saved['full_patch_hash_check'] or saved['dependencies'] != rec['dependencies']:
            raise ValueError('missing full input audit')
    return rec


def prepare(root=DEFAULT_ROOT):
    from .training import setup, initial_header
    root=Path(root); root.mkdir(parents=True,exist_ok=True)
    frozen=root/'prepared';frozen.mkdir(exist_ok=True)
    for original, name in ((AUTHORITY,'authoritative_protocol.json'),(CONTRACT,'experiment_contract.json')):
        dest=frozen/name
        provenance=frozen/(name+'.source.json')
        if not dest.exists():
            shutil.copyfile(original,dest)
            atomic_json(provenance,dict(source=identity(original),snapshot=identity(dest)))
        else:
            record=read(provenance)
            if identity(dest)!=record['snapshot']:raise ValueError('frozen protocol snapshot changed')
    environment=check_environment(); verify_inputs(root,full=True)
    q=read(INPUT/'configs/queue.json'); machine_template=read(INPUT/'V2_MORPH/configs/machine.json')
    common_rng=None; shared=torch.load(INPUT/'prepared/shared_initial.pth',map_location='cpu')
    if shared['trained'] is not False or shared['updates']!=0 or shared['seed']!=1234:
        raise ValueError('shared backbone is not the frozen untrained initial')
    results={}
    for run_id,kind in RUNS.items():
        out=root/run_id; (out/'configs').mkdir(parents=True,exist_ok=True)
        cfg=dict(version=VERSION,run_id=run_id,model_kind=kind,morphology=kind!='BASE',interaction=kind=='V2_INTERACT',
            input_root=str(INPUT),runroot=str(root),output_root=str(out),updates=3625,visits=3998,empty_skips=373,
            config_path=str(out/'configs/run.json'),initial_snapshot=str(root/'prepared'/f'{run_id}_step_zero.pth'),
            machine=str(out/'configs/machine.json'),contract=identity(frozen/'experiment_contract.json'),authoritative_protocol=identity(frozen/'authoritative_protocol.json'))
        if Path(cfg['config_path']).exists() and read(cfg['config_path'])!=cfg:
            raise ValueError('existing paired configuration differs')
        if not Path(cfg['config_path']).exists(): atomic_json(cfg['config_path'],cfg)
        machine=dict(machine_template,final_mask=str(out/'predictions'/run_id/'final.tif'),
            pre_mask=str(out/'predictions'/run_id/'pre.tif'),final_matrix=str(out/'matrix'/run_id/'expr_mat.csv'),
            checkpoint=str(out/'checkpoints/final.pth'))
        if Path(cfg['machine']).exists() and read(cfg['machine'])!=machine:raise ValueError('machine differs')
        if not Path(cfg['machine']).exists(): atomic_json(cfg['machine'],machine)
        model,opt,data,diag=setup(cfg)
        state=model.state_dict(); base_state={k:v for k,v in state.items() if not k.startswith(('morphology.','interaction.'))}
        if not state_equal(base_state,shared['model']):raise ValueError('shared tensor/BN mismatch')
        morph_sha=tensor_sha(model.morphology.state_dict()) if hasattr(model,'morphology') else None
        if morph_sha is not None and morph_sha!=MORPHOLOGY_INITIAL_SHA:raise ValueError('Morph initial changed')
        initial=dict(header=initial_header(cfg),model=state,optimizer=opt.state_dict(),rng=rng_state(),
                     actual_updates=0,view_cursor=0,empty_skips=0,trained=False,
                     rng_scope='after model and native dataset/diagnostic sidecar construction; imgaug seeded1234',
                     shared_tensor_sha=tensor_sha(base_state),morphology_tensor_sha=morph_sha)
        if common_rng is None:common_rng=copy.deepcopy(initial['rng'])
        elif not state_equal(common_rng,initial['rng']):raise ValueError('paired dataset-constructed RNG mismatch')
        path=Path(cfg['initial_snapshot'])
        if path.exists():
            previous=torch.load(path,map_location='cpu')
            if not state_equal(initial,previous):raise ValueError('frozen step-zero changed; preserve and investigate')
            del previous
        else:atomic_torch(path,initial)
        results[run_id]=dict(checkpoint=identity(path),header=initial['header'],shared_tensor_sha=initial['shared_tensor_sha'],
                            morphology_tensor_sha=morph_sha,optimizer_empty=not initial['optimizer']['state'])
        del model,opt,data,diag,state,base_state,initial;gc.collect();torch.cuda.empty_cache()
    rec=dict(status='passed',environment=environment,arms=results,shared_rng_exact=True,
             source=producer('initial'),input_audit=identity(root/'prepared/input_audit.json'),
             frozen_before_pilots=True)
    dest=root/'prepared/step_zero_receipt.json'
    if dest.exists() and read(dest)!=rec:raise ValueError('step zero receipt drift')
    if not dest.exists():atomic_json(dest,rec)
    return rec
