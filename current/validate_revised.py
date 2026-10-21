"""Small-graph release checks only; no paper-scale experiments."""
from pathlib import Path
import argparse
import subprocess
import sys

ROOT=Path(__file__).resolve().parent
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--with-gurobi',action='store_true')
    a=p.parse_args()
    jobs=[('TPCA','validate_comparison.py',[] if a.with_gurobi else ['--skip-gurobi']),
          ('TPCA','validate_ablation.py',[]),('TPCA','validate_progressive.py',[]),
          ('DGCA','validate_bgcr.py',[]),('DGCA','validate_strategy.py',[])]
    for folder,script,args in jobs:
        subprocess.run([sys.executable,script]+args,cwd=ROOT/folder,check=True)
