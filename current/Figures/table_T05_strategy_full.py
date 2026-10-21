"""Rebuild this asset from saved data; no solver is run."""
from pathlib import Path
import runpy
import sys

if __name__=='__main__':
    source=Path(__file__).resolve().parent/'make_all.py'
    sys.argv=[str(source),'--only','T05_strategy_full']
    runpy.run_path(str(source),run_name='__main__')
