"""Bind the implementation, scientific fixtures, reused sources and evidence."""
import argparse,subprocess
from pathlib import Path
from scripts.rep1_triple_simple_boundary.common import identity,atomic_json,digest
from scripts.rep1_triple_simple_boundary.inputs import source_dependencies

def run(out):
    repo=Path(__file__).resolve().parents[1]
    if subprocess.check_output(['git','status','--porcelain'],cwd=repo,text=True).strip():raise ValueError('source manifest requires a clean committed repository')
    tracked=subprocess.check_output(['git','ls-files'],cwd=repo,text=True).splitlines()
    fixture=[repo/n for n in tracked if n.startswith(('tests/rep1_triple','tests/test_rep1_triple','docs/rep1_triple')) or n=='scripts/rep1_triple_simple_boundary/README.md']
    source=source_dependencies();records=[dict(**identity(p),snapshot_relative=str(p.relative_to(repo))) for p in fixture]
    obj=dict(schema='rep1-triple-simple-science-implementation-source-v1',source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
        branch=subprocess.check_output(['git','branch','--show-current'],cwd=repo,text=True).strip(),source_dependencies=source,fixture_and_documentation=records,
        sources_digest=digest(source),scope='scientific implementation only; supervisor source snapshot and admission remain separate')
    atomic_json(out,obj);print(identity(out))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();run(a.output)
