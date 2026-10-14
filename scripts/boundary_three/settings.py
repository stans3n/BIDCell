"""Preparation defaults; budget is frozen from measured complete-chain estimates."""
from .contracts import *

def default_config(output_root=None,run_id='boundary-three-20261015-r01'):
    root = str(output_root or OUTPUT_ROOT/run_id)
    return dict(version=VERSION,run_id=run_id,source_root=str(SOURCE_ROOT),output_root=root,
        python=PYTHON,rscript=RSCRIPT,node_order=list(NODES),endpoints=endpoint_registry(),
        inputs=dict(historical_input_root=str(HISTORICAL_INPUT),
            shared_initial=str(HISTORICAL_INPUT/'prepared/shared_initial.pth'),
            prepared_manifest=str(HISTORICAL_INPUT/'prepared/manifest.json'),
            nucleus_metadata=str(HISTORICAL_INPUT/'prepared/nucleus_metadata.npz'),
            views=str(HISTORICAL_INPUT/'configs/views.json'),
            predict_shift0=str(HISTORICAL_INPUT/'configs/predict_shift0.json'),
            predict_shift24=str(HISTORICAL_INPUT/'configs/predict_shift24.json'),
            reference_region=str(HISTORICAL_INPUT/'configs/reference_region.json'),
            evaluation_queue=str(HISTORICAL_INPUT/'configs/queue.json'),
            authoritative_protocol=str(WORK/'outputs/bidcell_historical_dual_20261013/protocol.json'),
            frozen_strata=str(PAIRED_INPUT/'frozen_strata'),frozen_counts=str(PAIRED_INPUT/'frozen_counts'),
            pairing_audit=str(WORK/'outputs/bidcell_v2_fuse_decision_20261014/historical_pairing_audit.json'),
            historical_reuse=str(PAIRED_INPUT/'reviews/historical_endpoint_reuse.json'),
            environment_lock=str(PAIRED_INPUT/'reviews/environment_lock.json'),
            paper_R_environment=str(PAIRED_INPUT/'tests/paper_R_runtime_lock.json')),
        enabled_nodes={n:True for n in NODES},
        seeds=dict(training=1234,canonical_morph=20261013,simple_fusion=2026101501,
            a1_split=20261015,a1_shuffles=list(range(2026101500,2026101520)),
            a1_bootstrap=20261015,evaluation_bootstrap=20261013),
        resources=dict(cpu_workers=4,worker_threads=1,gpu_lock=str(GPU_LOCK),
            min_gpu_free_bytes=1024**3,min_ram_available_bytes=6*1024**3,
            disk_reserve_bytes=10*1024**3,resource_wait_seconds=600),
        budget_seconds=8*3600,node_estimates={},source_digest=None,input_digest=None,
        no_ai_dependency=True,auto_ai_wakeup=False,nightly_formal_sync=False)
