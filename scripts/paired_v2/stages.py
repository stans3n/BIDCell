"""Callable stages and semantic acceptance, independent of process orchestration."""
from types import SimpleNamespace
import time
import numpy as np
import pandas as pd
import tifffile
from .common import *

STAGES=('train','predict','postprocess','matrix','evaluate')


def validate_layout(config):
    cfg=load_config(config);root=Path(cfg['runroot']).resolve();out=root/cfg['run_id']
    for forbidden in (INPUT,HISTORICAL_SOURCE):
        if root==forbidden or forbidden in root.parents:raise ValueError('historical root is read only')
    expected={'output_root':out,'config_path':out/'configs/run.json','machine':out/'configs/machine.json',
              'initial_snapshot':root/'prepared'/f"{cfg['run_id']}_step_zero.pth"}
    if any(Path(cfg[key]).resolve()!=value for key,value in expected.items()):
        raise ValueError('paired output/config/checkpoint layout differs')
    return cfg


def native_receipt(cfg,stage):
    out=Path(cfg['output_root']);run=cfg['run_id']
    return {'train':out/'checkpoints/complete.json','predict':out/'predictions'/run/'complete.json',
        'postprocess':out/'predictions'/run/'postprocess/complete.json','matrix':out/'matrix'/run/'complete.json',
        'evaluate':out/'evaluation'/run/'complete.json'}[stage]


def expected_native_dependencies(cfg,stage):
    """Reproduce each native producer's actual complete dependency schema."""
    cfg=validate_layout(cfg);out=Path(cfg['output_root']);machine=read(cfg['machine'])
    if stage=='train':
        from .training import checkpoint_header
        return checkpoint_header(cfg)
    if stage=='predict':
        return dict(checkpoint=identity(out/'checkpoints/final.pth'),producer=producer('predict'),
                    prepared=identity(INPUT/'prepared/manifest.json'),run_id=cfg['run_id'],model_kind=cfg['model_kind'])
    if stage=='postprocess':
        return dict(pre=identity(out/'predictions'/cfg['run_id']/'pre.tif'),nuclei=identity(machine['nuclei_mask']),
                    seed=1234,patch_size=1024,
                    source=source_identity(['scripts/crgf/postprocess.py','bidcell/model/postprocess_predictions.py']))
    if stage=='matrix':
        return dict(**{k:identity(machine[k]) for k in ('transcripts','genes','baseline_config')},mask=identity(machine['final_mask']))
    if stage=='evaluate':
        from .evaluation import evaluation_contract
        _,_,contract=evaluation_contract(cfg['runroot'])
        return dict(matrix=identity(native_receipt(cfg,'matrix')),machine=identity(cfg['machine']),**contract)
    raise ValueError(stage)


def _exact_outputs(rec,paths):
    expected={name:identity(path) for name,path in paths.items()}
    if rec.get('outputs')!=expected:raise ValueError('native output identity/path inventory differs')


def validate_prediction_chunk(path,base,start,stop,patches):
    path=Path(path);chunk=read(path)
    expected=dict(base=base,start=start,stop=stop,patches=patches)
    if chunk.get('status')!='succeeded' or chunk.get('dependencies')!=expected:
        raise ValueError('native prediction chunk complete dependency mismatch')
    _exact_outputs(chunk,dict(masks=path.with_suffix('.npy')))
    return identity(path)


def validate_postprocess_split(path,base,ids):
    path=Path(path);split=read(path)
    if split.get('status')!='succeeded' or split.get('dependencies')!=dict(base=base,ids=ids):
        raise ValueError('postprocess split complete dependency mismatch')
    _exact_outputs(split,dict(split=path.with_suffix('.npz')))
    return dict(receipt=identity(path),output=identity(path.with_suffix('.npz')))


def validate_native_lineage(config,stage,rec=None,check_upstream=True):
    """Validate provenance before any old native success can acquire a bound receipt.

    An unchanged complete.json alone is insufficient: check current upstream
    artifacts and the native chunk/cache manifests that actually produced them.
    This does not change scientific generation or the frozen training producer.
    """
    cfg=validate_layout(config);out=Path(cfg['output_root']);run=cfg['run_id'];machine=read(cfg['machine'])
    rec=read(native_receipt(cfg,stage)) if rec is None else rec
    if rec.get('status')!='succeeded':raise ValueError('native stage did not succeed: '+stage)
    expected=expected_native_dependencies(cfg,stage)
    if rec.get('dependencies')!=expected:raise ValueError('native '+stage+' dependencies differ from current inputs/producer')
    verify_outputs(rec)
    if stage=='train':
        _exact_outputs(rec,dict(checkpoint=out/'checkpoints/final.pth',sequence=out/'effective_sequence.jsonl'))
        if rec.get('pilot') or rec.get('run_id')!=run or rec.get('model_kind')!=cfg['model_kind']:
            raise ValueError('native train identity/pilot differs')
    elif stage=='predict':
        p=out/'predictions'/run
        _exact_outputs(rec,{name:p/name for name in ('pre.tif','shift0.tif','shift24.tif','fill.tif')})
        chunks=[]
        for shift in (0,24):
            pool=read(INPUT/f'configs/predict_shift{shift}.json')
            for start in range(0,len(pool),128):
                cp=p/f'shift{shift}_patches'/f'{start:06d}.json';stop=min(start+128,len(pool))
                chunks.append(validate_prediction_chunk(cp,expected,start,stop,[v['identity'] for v in pool[start:stop]]))
        if rec.get('chunks')!=chunks:raise ValueError('native prediction ordered chunk inventory differs')
    elif stage=='postprocess':
        p=out/'predictions'/run/'postprocess';m=read(p/'manifest.json')
        _exact_outputs(rec,dict(final_mask=out/'predictions'/run/'final.tif',manifest=p/'manifest.json'))
        if m.get('dependencies')!=expected:raise ValueError('postprocess manifest dependencies differ')
        if len(m['tiles'])!=len(m['coords']):raise ValueError('postprocess tile inventory incomplete')
        for obj,coords in zip(m['tiles'],m['coords']):
            y,x=coords;path=p/f'tile_{y}_{x}.json';tile=read(path)
            if obj!=identity(path) or tile.get('status')!='succeeded' or tile.get('dependencies')!=dict(base=expected,coords=coords):
                raise ValueError('postprocess tile complete dependency mismatch')
            _exact_outputs(tile,dict(tile=path.with_suffix('.tif')))
        splitbase=digest(dict(base=expected,tiles=m['tiles'],ids=m['split_ids']))
        expected_chunks=[]
        for start in range(0,len(m['split_ids']),128):
            ids=m['split_ids'][start:start+128];path=p/'splits'/f'{start:06d}.json'
            expected_chunks.append(dict(start=start,ids=ids,**validate_postprocess_split(path,splitbase,ids)))
        if m.get('split_chunks')!=expected_chunks:raise ValueError('postprocess ordered split inventory differs')
    elif stage=='matrix':
        import cv2
        from scripts.crgf.common import VERSION as MATRIX_VERSION
        p=out/'matrix'/run;shape=list(tifffile.imread(machine['final_mask']).shape)
        _exact_outputs(rec,{name:p/name for name in ('counts.npy','cell_mapping.csv','gene_mapping.csv','expr_mat.csv')})
        source=REPO/'scripts/crgf/matrix.py'
        contract=dict(version=MATRIX_VERSION,source_sha256=sha256(source),shape=shape,scale_x=machine['scale_x'],scale_y=machine['scale_y'],
            coordinate='processed coords / scale_pix -> pandas-compatible ties-to-even round -> native cv2 INTER_NEAREST axes; OOB unassigned',
            filter='already processed qv>=20 + configured prefixes + nonnegative; no refilter; vendor cell_id ignored',
            data_scope='own-mask Rep1 full field engineering',fit_scope='none',kind='final')
        if rec.get('kind')!='final' or rec.get('contract')!=contract:raise ValueError('matrix source/coordinate contract differs')
        cdeps=dict(transcripts=expected['transcripts']['sha256'],genes=expected['genes']['sha256'],shape=shape,
            scale_x=machine['scale_x'],scale_y=machine['scale_y'],builder=contract['source_sha256'],
            common=sha256(REPO/'scripts/crgf/common.py'),numpy=np.__version__,opencv=cv2.__version__,pandas=pd.__version__)
        cdir=Path(rec['coordinate_cache']);cache=read(cdir/'complete.json')
        key=hashlib.sha256(json.dumps(cdeps,sort_keys=True).encode()).hexdigest()
        if cdir.name!=key or cache.get('dependencies')!=cdeps or cache.get('duplicate_ids')!=0:
            raise ValueError('matrix coordinate cache input/scale/source/version mismatch')
        if len(rec['receipts'])!=len(cache['chunks']):raise ValueError('matrix coordinate/owner inventory mismatch')
        rows=0
        for i,(owner,coordinate) in enumerate(zip(rec['receipts'],cache['chunks'])):
            cp=cdir/f'chunk_{i:04d}.npy';op=p/f'owners_{i:04d}.npy';saved=read(op.with_suffix('.json'))
            csha=sha256(cp);osha=sha256(op);n=coordinate['rows']
            if coordinate['file']!=cp.name or coordinate['sha256']!=csha or owner['coordinate_file']!=str(cp) or owner['coordinate_sha256']!=csha:
                raise ValueError('matrix coordinate chunk identity differs')
            if owner['chunk']!=i or owner['owner_file']!=op.name or owner['sha256']!=osha or (owner['row_start'],owner['row_stop'],owner['rows'])!=(rows,rows+n,n):
                raise ValueError('matrix owner chunk identity/row coverage differs')
            if saved!={'dependencies':dict(mask=expected['mask']['sha256'],coordinates=csha,source=contract['source_sha256']),'sha256':osha}:
                raise ValueError('matrix owner receipt dependency mismatch')
            rows+=n
        if rows!=rec['totals']['eligible']:raise ValueError('matrix complete coordinate coverage differs')
        validation=read(p/'validation.json')
        if rec.get('validation')!=identity(p/'validation.json') or rec.get('validation_kind')!='independent_full_native_resize_lookup_and_conservation':
            raise ValueError('matrix independent validation receipt differs')
        if validation.get('source')!=identity(source) or validation.get('mask')!=expected['mask'] or validation.get('outputs')!=rec['outputs'] or validation.get('conservation')!=rec['totals']:
            raise ValueError('matrix validation provenance differs')
        if validation.get('numpy')!=np.__version__ or validation.get('opencv')!=cv2.__version__:
            raise ValueError('matrix validation environment differs')
    else:
        p=out/'evaluation'/run
        # Actual generation records the whole current evaluation tree, including
        # profiles_complete. Compare its paths as well as hashes, not just fields.
        if rec.get('outputs')!=file_tree(p):raise ValueError('evaluation output identity inventory differs')
        profiles=read(p/'profiles_complete.json')
        if profiles.get('status')!='succeeded' or profiles.get('dependencies')!=expected:
            raise ValueError('evaluation profiles matrix/machine/science dependencies differ')
        verify_outputs(profiles)
    if check_upstream and stage!='train':
        # Check the predecessor's real artifacts and native scientific inputs.
        # The formal gateway also requires its bound semantic acceptance below.
        previous=STAGES[STAGES.index(stage)-1]
        validate_native_lineage(cfg,previous,check_upstream=False)
    return expected


def dependencies(cfg,stage):
    cfg=validate_layout(cfg)
    rec=dict(run_id=cfg['run_id'],model_kind=cfg['model_kind'],config=identity(cfg['config_path']),
             machine=identity(cfg['machine']),producer=producer(stage),input_audit=identity(Path(cfg['runroot'])/'prepared/input_audit.json'))
    if stage=='train':
        from .training import checkpoint_header
        rec['training_header']=checkpoint_header(cfg)
    else:
        previous=STAGES[STAGES.index(stage)-1]
        rec['upstream']=identity(native_receipt(cfg,previous))
    if stage=='evaluate':rec['authoritative_protocol']=cfg['authoritative_protocol']
    return rec


def postprocess(config):
    from scripts.crgf.postprocess import postprocess as native
    cfg=validate_layout(config);machine=read(cfg['machine'])
    adapter=SimpleNamespace(output_root=cfg['output_root'],seed=1234,postprocess_size=1024,cpu_workers=4,
                           native=lambda:SimpleNamespace(files=SimpleNamespace(data_dir=str(Path(machine['nuclei_mask']).parent))))
    native(adapter,cfg['run_id']);return read(native_receipt(cfg,'postprocess'))


def matrix(config):
    from scripts.crgf.matrix import build
    cfg=validate_layout(config);out=Path(cfg['output_root'])/'matrix'/cfg['run_id']
    if (out/'complete.json').exists():
        return validate_stage(cfg,'matrix')
    return build(read(cfg['machine']),out,cache=Path(cfg['runroot'])/'coordinate_cache')


def repair_uncommitted_training_log(config,output=None):
    """Discard only log bytes beyond an exact durable checkpoint prefix.

    A killed write may leave a partial final JSON line. The checkpoint contains
    the complete committed sequence; a damaged committed prefix remains fatal.
    This gateway recovery does not alter checkpoint tensors or scientific code.
    """
    from .training import checkpoint_header,validate_cursor,rewrite_sequence
    cfg=validate_layout(config);out=Path(output or cfg['output_root']).resolve()
    tests=Path(cfg['runroot']).resolve()/'tests'
    if out!=Path(cfg['output_root']).resolve() and tests not in out.parents:
        raise ValueError('training log recovery output outside this arm/tests')
    checkpoint=out/'checkpoints/latest.pth';log=out/'effective_sequence.jsonl'
    if not checkpoint.exists():return dict(status='no_durable_checkpoint')
    state=torch.load(checkpoint,map_location='cpu')
    if state['header']!=checkpoint_header(cfg):raise ValueError('log recovery checkpoint header differs')
    if state['sequence_digest']!=digest(state['effective_sequence']):raise ValueError('log recovery checkpoint sequence differs')
    validate_cursor(state);rows=state['effective_sequence']
    if log.exists():
        lines=log.read_text().splitlines()
        if len(lines)<len(rows):raise ValueError('durable training log prefix was truncated')
        for index,row in enumerate(rows):
            if json.loads(lines[index])!=row:raise ValueError('durable training log prefix changed')
        if len(lines)==len(rows):return dict(status='unchanged',committed_updates=len(rows))
    else:lines=[]
    before=identity(log) if log.exists() else None
    rewrite_sequence(log,rows)
    result=dict(status='recovered_uncommitted_tail',committed_updates=len(rows),checkpoint=identity(checkpoint),
                previous_log=before,recovered_log=identity(log),uncommitted_lines_removed=max(0,len(lines)-len(rows)))
    atomic_json(out/f'log_recovery_{time.time_ns()}.json',result)
    return result


def validate_stage(config,stage,require_bound=False):
    cfg=validate_layout(config)
    if stage not in STAGES:raise ValueError(stage)
    path=native_receipt(cfg,stage);rec=read(path)
    validate_native_lineage(cfg,stage,rec)
    out=Path(cfg['output_root']);run=cfg['run_id'];machine=read(cfg['machine'])
    semantic={}
    if stage=='train':
        from .training import checkpoint_header,validate_cursor
        state=torch.load(out/'checkpoints/final.pth',map_location='cpu')
        if state['header']!=checkpoint_header(cfg) or state['pilot']:raise ValueError('training header/pilot mismatch')
        validate_cursor(state)
        if (state['actual_updates'],state['view_cursor'],state['empty_skips'])!=(3625,3998,373):raise ValueError('incomplete training budget')
        log=[json.loads(line) for line in (out/'effective_sequence.jsonl').read_text().splitlines()]
        if log!=state['effective_sequence']:raise ValueError('final effective sequence differs from checkpoint')
        steps=sorted({int(s['step']) for s in state['optimizer']['state'].values()})
        if steps!=[3625] or any(not torch.isfinite(t).all() for t in state['model'].values()):raise ValueError('invalid Adam/model endpoint')
        expected=make_model(cfg['model_kind'],'cpu');expected.load_state_dict(state['model'],strict=True)
        for boundary in (100,1000,2000,3625):
            diag=read(out/'diagnostics'/f'probe_{boundary:04d}.json')
            if diag['status']!='passed' or not diag['exact_rng_preserved'] or diag['producer']!=producer('train'):raise ValueError('training diagnostic missing/changed')
        semantic=dict(updates=3625,visits=3998,empty_skips=373,adam_steps=steps,sequence_digest=state['sequence_digest'])
    elif stage=='predict':
        if rec['shifts']!=[0,24]:raise ValueError('missing prediction shift')
        total=0
        for shift in (0,24):
            pool=read(INPUT/f'configs/predict_shift{shift}.json')
            if rec['shift_counts'][str(shift)]!=len(pool):raise ValueError('prediction coverage count mismatch')
            for start in range(0,len(pool),128):
                cp=out/'predictions'/run/f'shift{shift}_patches'/f'{start:06d}.json';chunk=read(cp)
                verify_outputs(chunk);stop=min(start+128,len(pool))
                if chunk['dependencies']['start']!=start or chunk['dependencies']['stop']!=stop or chunk['dependencies']['patches']!=[v['identity'] for v in pool[start:stop]]:
                    raise ValueError('prediction patch order/coverage differs')
                if chunk['dependencies']['base']['producer']!=producer('predict'):raise ValueError('prediction producer differs')
                a=np.load(cp.with_suffix('.npy'),mmap_mode='r')
                if a.shape!=(stop-start,48,48) or a.dtype!=np.uint32 or len(chunk['candidate_ids'])!=stop-start:raise ValueError('prediction chunk dimensions/IDs differ')
                total+=stop-start
        for name in ('pre.tif','shift0.tif','shift24.tif','fill.tif'):
            if list(tifffile.imread(out/'predictions'/run/name).shape)!=read(INPUT/'prepared/manifest.json')['shape']:raise ValueError('prediction image shape mismatch')
        semantic=dict(covered_patches=total,shifts=[0,24],max_K=rec['max_K'])
    elif stage=='postprocess':
        m=read(out/'predictions'/run/'postprocess/manifest.json');dep=m['dependencies']
        if dep['seed']!=1234 or dep['patch_size']!=1024 or dep['pre']['sha256']!=sha256(machine['pre_mask']):raise ValueError('native postprocess settings/input changed')
        final=tifffile.imread(machine['final_mask']);nuc=tifffile.imread(machine['nuclei_mask'])
        if final.shape!=nuc.shape or not np.array_equal(final[nuc>0],nuc[nuc>0]) or not np.array_equal(np.unique(final[final>0]),np.unique(nuc[nuc>0])):raise ValueError('final nuclear/ID preservation failed')
        h,w=nuc.shape;size=min(1024,h,w);coords=[[y,x] for y in list(range(0,h-size,size))+[h-size] for x in list(range(0,w-size,size))+[w-size]]
        if m['coords']!=coords or m['split_ids']!=sorted(set(m['split_ids'])):raise ValueError('postprocess tile/split order drift')
        if [c['ids'] for c in m['split_chunks']]!=[m['split_ids'][s:s+128] for s in range(0,len(m['split_ids']),128)]:raise ValueError('postprocess split chunks drift')
        for obj in m['tiles']:
            if sha256(obj['path'])!=obj['sha256']:raise ValueError('postprocess tile receipt changed')
            verify_outputs(read(obj['path']))
        for c in m['split_chunks']:
            if sha256(c['receipt']['path'])!=c['receipt']['sha256']:raise ValueError('split receipt changed')
            verify_outputs(read(c['receipt']['path']))
        semantic=dict(nuclear_preservation_fraction=1.,new_ids=0,tiles=len(coords),split_ids=len(m['split_ids']))
    elif stage=='matrix':
        p=out/'matrix'/run;validation=read(p/'validation.json')
        if sha256(p/'validation.json')!=rec['validation']['sha256'] or validation['status']!='succeeded' or validation['owner_differences']!=0 or validation['integer_count_differences']!=0:raise ValueError('full matrix acceptance failed')
        counts=np.load(p/'counts.npy',mmap_mode='r');ids=pd.read_csv(p/'cell_mapping.csv').canonical_cell_id.to_numpy(np.int64)
        genes=pd.read_csv(p/'gene_mapping.csv').gene.tolist();nuc_ids=np.load(INPUT/'prepared/nucleus_metadata.npz')['ids']
        if counts.shape!=(107955,313) or not np.array_equal(ids,nuc_ids) or genes!=Path(machine['genes']).read_text().splitlines() or not np.issubdtype(counts.dtype,np.integer) or (counts<0).any():raise ValueError('integer matrix shape/IDs/genes/values invalid')
        totals=rec['totals']
        if totals['eligible']!=34442716 or totals['assigned']+totals['unassigned']!=34442716 or int(counts.sum())!=totals['assigned']:raise ValueError('native molecule conservation failed')
        if rec['dependencies']['mask']['sha256']!=sha256(machine['final_mask']):raise ValueError('matrix final mask lineage differs')
        for chunk in rec['receipts']:
            if sha256(p/chunk['owner_file'])!=chunk['sha256'] or sha256(chunk['coordinate_file'])!=chunk['coordinate_sha256']:raise ValueError('matrix owner/coordinate artifact changed')
        semantic=dict(shape=list(counts.shape),totals=totals,independent_native_owner_differences=0)
    else:
        from .evaluation import spec_for,validate_evaluation,evaluation_contract
        _,_,contract=evaluation_contract(cfg['runroot'])
        if any(rec['dependencies'][k]!=v for k,v in contract.items()):raise ValueError('evaluation science/producer changed')
        semantic=validate_evaluation(spec_for(cfg))
    result=dict(status='passed',stage=stage,dependencies=dependencies(cfg,stage),native_receipt=identity(path),semantic=semantic)
    bound=out/'stages'/f'{stage}.json'
    if require_bound:
        saved=read(bound)
        if saved!=result:raise ValueError('bound stage receipt changed')
    return result


def run_stage(config,stage):
    cfg=validate_layout(config);bound=Path(cfg['output_root'])/'stages'/f'{stage}.json'
    if bound.exists():return _bind_completed_stage(cfg,stage)
    environment=check_environment(require_cuda=stage in ('train','predict'))
    frozen=read(Path(cfg['runroot'])/'prepared/step_zero_receipt.json')['environment']
    for key in ('host','python','torch','numpy','scipy','opencv','imgaug','cuda','cudnn',
                'threads','interop_threads','tf32','cudnn_benchmark','cudnn_deterministic','environment'):
        if environment[key]!=frozen[key]:raise ValueError('admitted numerical environment changed: '+key)
    if stage in ('train','predict') and (environment['gpu'],environment['total_vram'])!=(frozen['gpu'],frozen['total_vram']):
        raise ValueError('admitted CUDA device changed')
    if stage in ('train','predict'):
        from .prepare import verify_inputs
        verify_inputs(Path(cfg['runroot']),full=True)
    index=STAGES.index(stage)
    if index:validate_stage(cfg,STAGES[index-1],require_bound=True)
    # The orchestration freezes E0 count strata between matrix and evaluate.
    # Stages never automatically advance to the next stage.
    if not native_receipt(cfg,stage).exists() or read(native_receipt(cfg,stage)).get('status')!='succeeded':
        if stage=='train':
            from .training import train
            gate=read(Path(cfg['runroot'])/'admission'/f"{cfg['run_id']}.json")
            if gate.get('status')!='admitted' or gate.get('train_producer')!=producer('train'):
                raise ValueError('this arm has no current independently admitted train producer')
            repair_uncommitted_training_log(cfg)
            train(cfg)
        elif stage=='predict':
            from .prediction import predict
            predict(cfg)
        elif stage=='postprocess':postprocess(cfg)
        elif stage=='matrix':matrix(cfg)
        else:
            from .evaluation import evaluate
            evaluate(cfg)
    return _bind_completed_stage(cfg,stage)


def _bind_completed_stage(config,stage):
    """Gateway commit point; a previous bound receipt is never silently replaced."""
    cfg=validate_layout(config);bound=Path(cfg['output_root'])/'stages'/f'{stage}.json'
    exists=bound.exists();result=validate_stage(cfg,stage,require_bound=exists)
    if not exists:atomic_json(bound,result)
    return result
