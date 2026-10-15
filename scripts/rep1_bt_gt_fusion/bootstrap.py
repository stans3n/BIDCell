"""Entry point run with -I -S -B: no cwd, user site, .pth or editable hooks."""
import os
from pathlib import Path
import sys
import sysconfig

ROOT = Path(__file__).resolve().parents[2]
sys.dont_write_bytecode = True
sys.path[:] = [str(ROOT)] + [p for p in sys.path if p and 'site-packages' not in p]
sys.path.append(sysconfig.get_paths()['purelib'])
os.chdir(ROOT)
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'

if __name__ == '__main__':
    from scripts.rep1_bt_gt_fusion.cli import main
    main()
