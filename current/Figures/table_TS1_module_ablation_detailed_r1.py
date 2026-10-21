from pathlib import Path
import runpy
import sys

root = Path(__file__).resolve().parent
sys.argv = [str(root / 'make_all.py'), '--only', 'TS1_module_ablation_detailed_r1']
runpy.run_path(str(root / 'make_all.py'), run_name='__main__')
