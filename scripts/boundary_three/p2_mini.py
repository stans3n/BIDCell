"""Real P2 admission. Development artifacts never become formal endpoints."""
from pathlib import Path
import random
import resource
import time
from types import SimpleNamespace
import numpy as np
import pandas as pd
import tifffile
from bidcell.model import postprocess_predictions as native
from scripts.crgf.common import atomic_tiff
from scripts.crgf.matrix import grid
from scripts.crgf.postprocess import postprocess as native_postprocess
from .common import (load_run,read,identity,sha256,digest,atomic_json,compute_lock,
                     endpoint_spec,inventory,check_identity)
from .contracts import PAIRED_INPUT
from .p2_native import process,validate_process,native_connect,STAGES,isolated,rng_hash
from .p2_diagnostics import transfer_diagnostics
from .p2_transfer import BASES,copy_exact,matrix_only,assert_matrix_equal


def verify_roi_population(cfg, original, roi, transcripts):
    """Compare each real ROI molecule ID/gene/pixel to full production cache."""
    local = pd.read_csv(transcripts)
    size,y0,x0 = roi['size'],roi['row'],roi['col']
    lx,ly,valid = grid(local,(size,size),original['scale_x'],original['scale_y'])
    if not valid.all():
        raise ValueError('ROI invalid local molecule')
    genes = Path(original['genes']).read_text().splitlines()
    local_gc = pd.Index(genes).get_indexer(local.feature_name)
    actual = pd.DataFrame(dict(transcript_id=local.transcript_id,x=lx+x0,y=ly+y0,gene=local_gc)).set_index('transcript_id').sort_index()
    source = read(Path(endpoint_spec(cfg,'BASE_MATCH')['matrix'])/'complete.json')
    pieces = []
    for rec in source['receipts']:
        if sha256(rec['coordinate_file']) != rec['coordinate_sha256']:
            raise ValueError('source coordinate identity drift')
        a = np.load(rec['coordinate_file'],mmap_mode='r')
        keep = (a['x'] >= x0) & (a['x'] < x0+size) & (a['y'] >= y0) & (a['y'] < y0+size)
        pieces.append(pd.DataFrame.from_records(a[keep]).set_index('transcript_id')[['x','y','gene']])
    expected = pd.concat(pieces).sort_index()
    if len(expected) != len(actual) or not expected.index.equals(actual.index) or not np.array_equal(expected.to_numpy(),actual.to_numpy()):
        raise ValueError('ROI molecule ID/gene/native-pixel association changed')
    return dict(molecules=len(actual),unique_ids=actual.index.is_unique,global_native_coordinates_exact=True,
                full_coordinate_source=identity(Path(endpoint_spec(cfg,'BASE_MATCH')['matrix'])/'complete.json'))


def historical_tile_replay(cfg,base,root):
    """A true production 1024 tile, exact same seed, checked against saved tile."""
    e = endpoint_spec(cfg,base)
    m = read(e['machine'])
    manifest = read(Path(e['pre']).parent/'postprocess/manifest.json')
    # Predeclared tile overlaps the known real ROI; it is not chosen by score.
    y,x = 0,2048
    if [y,x] not in manifest['coords']:
        raise ValueError('required admission tile absent')
    size = 1024
    image = tifffile.imread(e['pre'])[y:y+size,x:x+size]
    nuclei = tifffile.imread(m['nuclei_mask'])[y:y+size,x:x+size]
    seed = 1234+y*1000003+x
    random.seed(seed)
    connected,diagnostics,arrays = native_connect(image,nuclei)
    captured_enter = random.getstate()
    output = native.remove_islands(connected,nuclei)
    captured_exit = rng_hash()
    expected_path = Path(e['pre']).parent/'postprocess'/f'tile_{y}_{x}.tif'
    expected_receipt = read(expected_path.with_suffix('.json'))
    old = next(iter(expected_receipt['outputs'].values()))
    if sha256(expected_path) != old['sha256'] or not np.array_equal(output,tifffile.imread(expected_path)):
        raise ValueError('production native disabled tile replay differs: '+base)
    # Separately call the uninstrumented current native to test observation.
    random.seed(seed)
    plain = native.postprocess_connect(image,nuclei)
    if not np.array_equal(plain,connected) or random.getstate() != captured_enter:
        raise ValueError('native observer changed candidate output or RNG')
    plain = native.remove_islands(plain,nuclei)
    if not np.array_equal(plain,output) or rng_hash() != captured_exit:
        raise ValueError('native observer changed downstream RNG')
    root.mkdir(parents=True,exist_ok=True)
    atomic_tiff(root/'postconnect.tif',connected)
    atomic_tiff(root/'postislands.tif',output)
    rec = dict(status='passed',base=base,coords=[y,x],size=size,seed=seed,
        candidate_count=len(diagnostics['candidate_order']),diagnostics=diagnostics,
        exact_to_historical_production_tile=True,uninstrumented_native_exact=True,RNG_exact=True,
        historical_tile=identity(expected_path),historical_pre=identity(e['pre']),
        outputs={name:identity(root/(name+'.tif')) for name in ('postconnect','postislands')})
    atomic_json(root/'complete.json',rec)
    return rec


def recovery_oracle(cfg,root,pre,nuc,full,origin):
    """Real 144 ROI, overlapping 96 tiles, tile and split interruption."""
    baseline = root/'native'
    process(pre,nuc,full,baseline,'B',workers=1,patch_size=96,origin=origin)
    fullpath = root/'uninterrupted'
    process(pre,nuc,full,fullpath,'P2',native_pred=baseline,workers=1,patch_size=96,origin=origin)
    result = {}
    for stage in ('tiles','splits'):
        target = root/('interrupted_'+stage)
        try:
            process(pre,nuc,full,target,'P2',native_pred=baseline,workers=1,patch_size=96,
                    origin=origin,interrupt_after=(stage,1))
        except InterruptedError:
            pass
        else:
            raise ValueError('real interruption boundary was not exercised: '+stage)
        if (target/'postprocess/complete.json').exists():
            raise ValueError('interrupted work claimed complete')
        process(pre,nuc,full,target,'P2',native_pred=baseline,workers=1,patch_size=96,origin=origin)
        validate_process(target)
        comparisons = {name:sha256(target/(name+'.tif')) == sha256(fullpath/(name+'.tif')) for name in STAGES}
        if not all(comparisons.values()):
            raise ValueError('recovery changed native stage outputs')
        result[stage] = dict(interruption_exercised=True,all_stage_sha256_equal=comparisons)
    atomic_json(root/'recovery.json',result)
    return result


def validate_mini(config,output):
    cfg = load_run(config)
    output = isolated(output)
    if output != Path(cfg['output_root']).resolve():
        raise ValueError('P2 mini must use its own config output_root')
    output.mkdir(parents=True,exist_ok=True)
    started = time.monotonic()
    deps = dict(config=digest(cfg),sources={p.name:identity(p) for p in Path(__file__).parent.glob('p2_*.py')})
    if (output/'complete.json').exists():
        rec = read(output/'complete.json')
        if rec['dependencies'] != deps:
            raise ValueError('completed P2 mini source/config drift; use new attempt')
        for obj in rec['outputs'].values():
            check_identity(obj)
        return rec
    with compute_lock():
        roi_source = PAIRED_INPUT/'tests/roi/shared'
        roi = read(roi_source/'roi.json')
        shared = output/'shared'
        shared.mkdir(exist_ok=True)
        for name in ('roi.json','nuclei.tif','transcripts_processed.csv'):
            copy_exact(roi_source/name,shared/name)
        base_machine = read(endpoint_spec(cfg,'BASE_MATCH')['machine'])
        full_nuclei = base_machine['nuclei_mask']
        y,x,size = roi['row'],roi['col'],roi['size']
        if not np.array_equal(tifffile.imread(shared/'nuclei.tif'),tifffile.imread(full_nuclei)[y:y+size,x:x+size]):
            raise ValueError('real ROI nucleus origin differs')
        population = verify_roi_population(cfg,base_machine,roi,shared/'transcripts_processed.csv')
        evidence = {}
        for node,base in BASES.items():
            print('P2 mini starting',node,flush=True)
            base_e = endpoint_spec(cfg,base)
            original = read(base_e['machine'])
            replay = historical_tile_replay(cfg,base,output/'historical_tile_replay'/base)
            root = output/node
            pred = root/'predictions'/node
            pred.mkdir(parents=True,exist_ok=True)
            crop = tifffile.imread(base_e['pre'])[y:y+size,x:x+size]
            atomic_tiff(pred/'pre.tif',crop)
            native_pred = root/'native_replay/predictions'/base
            native_pred.mkdir(parents=True,exist_ok=True)
            copy_exact(pred/'pre.tif',native_pred/'pre.tif')
            process(pred/'pre.tif',shared/'nuclei.tif',full_nuclei,native_pred,'B',workers=1,origin=(y,x))
            # The copied current native stage is a separate uninstrumented oracle.
            plain_root = root/'plain_native'
            plain_pred = plain_root/'predictions'/base
            plain_pred.mkdir(parents=True,exist_ok=True)
            copy_exact(pred/'pre.tif',plain_pred/'pre.tif')
            adapter = SimpleNamespace(output_root=str(plain_root),seed=1234,postprocess_size=1024,cpu_workers=1,
                                      native=lambda:SimpleNamespace(files=SimpleNamespace(data_dir=str(shared))))
            native_postprocess(adapter,base)
            if not np.array_equal(tifffile.imread(plain_pred/'final.tif'),tifffile.imread(native_pred/'final.tif')):
                raise ValueError('real ROI disabled adapter differs from full current native stage')
            spec = dict(root=str(root),machine=str(root/'configs/machine.json'),
                        final=str(pred/'final.tif'),matrix=str(root/'matrix'/node),
                        evaluation=str(root/'evaluation'/node),cost_root=str(root))
            machine = dict(original,pre_mask=str(pred/'pre.tif'),final_mask=spec['final'],
                           final_matrix=str(Path(spec['matrix'])/'expr_mat.csv'),
                           nuclei_mask=str(shared/'nuclei.tif'),transcripts=str(shared/'transcripts_processed.csv'),
                           baseline_final_mask=str(native_pred/'final.tif'))
            atomic_json(spec['machine'],machine)
            native_matrix = root/'native_replay/matrix'/base
            native_machine = dict(machine,final_mask=str(native_pred/'final.tif'),final_matrix=str(native_matrix/'expr_mat.csv'))
            matrix_only(native_machine,native_matrix,cfg)
            plain_matrix = plain_root/'matrix'/base
            plain_machine = dict(native_machine,final_mask=str(plain_pred/'final.tif'),final_matrix=str(plain_matrix/'expr_mat.csv'))
            matrix_only(plain_machine,plain_matrix,cfg)
            matrix_replay = assert_matrix_equal(native_matrix,plain_matrix)
            process(pred/'pre.tif',shared/'nuclei.tif',full_nuclei,pred,'P2',native_pred=native_pred,workers=1,origin=(y,x))
            validate_process(pred)
            from .evaluation import build_and_evaluate
            evaluated = build_and_evaluate(cfg,node,spec=spec)
            diagnostics = transfer_diagnostics(cfg,node,pred,native_pred,machine,spec['matrix'],origin=(y,x))
            recovery = recovery_oracle(cfg,root/'recovery',pred/'pre.tif',shared/'nuclei.tif',full_nuclei,(y,x))
            evidence[node] = dict(historical_tile_replay=replay,ROI_native_final_exact=True,
                                  ROI_native_matrix_replay=matrix_replay,evaluation=evaluated,
                                  diagnostics=diagnostics,recovery=recovery)
            print('P2 mini complete',node,flush=True)
        result = dict(status='passed',dependencies=deps,scope='real development ROI; no full-slide or efficacy claim',
            full_native_replay_still_required_at_formal_node=True,real_population=population,roi=roi,
            max_K=max(e['historical_tile_replay']['candidate_count'] for e in evidence.values()),
            evidence=evidence,wall_seconds=time.monotonic()-started,
            peak_parent_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            peak_child_rss_bytes=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss*1024,
            outputs=inventory(output))
        atomic_json(output/'complete.json',result)
        return result
