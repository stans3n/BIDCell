"""Author-rule recovery population regression; explicit isolated output required."""
import argparse
from scripts.bs4.tests.author_stage import run
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True);a=p.parse_args();run(a.root,a.output)
