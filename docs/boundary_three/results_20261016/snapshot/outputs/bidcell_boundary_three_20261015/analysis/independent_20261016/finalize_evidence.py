"""Add recorded cost and historical absolute-level context, without new gates."""
import json
import torch
import analyze as a

d=json.loads((a.OUT/'main_evidence.json').read_text())
a.identities.update(json.loads((a.OUT/'source_identities.json').read_text()));a.checks.extend(json.loads((a.OUT/'checks.json').read_text()))
p=a.RUN/'prepared/morph/canonical_full_seed20261013.pth'
canonical=torch.load(a.source(p),map_location='cpu')
full=a.Path('/home/zwan0726/work/outputs/bidcell_new_methods_20261013/V2_MORPH')
tr=a.readj(full/'checkpoints/complete.json');pr=a.readj(full/'predictions/V2_MORPH/complete.json')
d['timing']['V2_MORPH_historical']=dict(added_parameters_from_canonical_state=sum(v.numel() for v in canonical['model'].values()),
    training={k:tr[k] for k in ['actual_updates','view_cursor','empty_skips','wall_seconds','peak_vram_bytes']},prediction={k:pr[k] for k in ['wall_seconds','peak_vram_bytes','shifts']},
    timing_scope='Actual historical receipt values, not a new paired runtime benchmark')
hist=a.readc('/home/zwan0726/work/outputs/bidcell_historical_dual_20261013/main_table.csv',encoding='utf-8-sig')
b1=hist.loc[hist.method.eq('B1')].iloc[0]
cols=['scclassify_positive_F1','scclassify_negative_F1','fixed_positive_F1','fixed_negative_F1','capture_pct']
d['historical_absolute_level_context']=dict(method='B1',values={k:float(b1[k]) for k in cols},
    meaning='Illustrates that new matched-control increments do not establish an overall absolute F1 record across the historical 28-method publication cohort; B1 is historical, not the matched mechanistic control.')
d['frozen_scoring_protocol_identity']='SC-PRIMARY-FIXED-v1.2-20methods-20261014'
d['current_assessment_publication_identity']='DUAL-F1-v1.0-28methods-20261016'
for e in d['endpoints'].values():
    a.check('historical B1 higher absolute four F1 directions than '+e['method'],
        float(b1.scclassify_positive_F1)>e['metrics']['scclassify_positive_F1']['mean'] and float(b1.scclassify_negative_F1)<e['metrics']['scclassify_negative_F1']['mean'] and
        float(b1.fixed_positive_F1)>e['metrics']['fixed_positive_F1']['mean'] and float(b1.fixed_negative_F1)<e['metrics']['fixed_negative_F1']['mean'])
for name in ['analyze.py','fixed_profile_analysis.py','b0_references.py','finalize_evidence.py','write_report.py']:a.source(a.OUT/name)
a.savej('main_evidence.json',d);a.savej('checks.json',a.checks);a.savej('source_identities.json',a.identities)
print('Finalized independent evidence; checks',len(a.checks))
