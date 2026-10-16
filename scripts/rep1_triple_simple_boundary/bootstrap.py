"""-I -S -B entry: explicit frozen project/stdlib/purelib, no .pth or user site."""
import os
from pathlib import Path
import sys
import sysconfig
ROOT=Path(__file__).resolve().parents[2]
sys.dont_write_bytecode=True
sys.path[:]=[str(ROOT)]+[p for p in sys.path if p and 'site-packages' not in p]+[sysconfig.get_paths()['purelib']]
os.chdir(ROOT)
os.environ['PYTHONDONTWRITEBYTECODE']='1'
if (ROOT.parent/'source_manifest.json').is_file():
    from scripts.rep1_triple_simple_boundary.execution import install_guard
    install_guard(ROOT)
elif __name__=='__main__' and (len(sys.argv)<2 or sys.argv[1]!='prepare'):
    raise SystemExit('Formal commands require a frozen source_snapshot; live checkout permits prepare only')
if __name__=='__main__':
    from scripts.rep1_triple_simple_boundary.cli import main
    main()
