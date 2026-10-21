"""Compile the unchanged revised native sources; no benchmark runs."""
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parent
if __name__=='__main__':
    for folder,script in [('TPCA','setup_compact.py'),('TPCA','setup_ablation.py'),
                          ('TPCA','setup_progressive.py'),('DGCA','setup.py')]:
        subprocess.run([sys.executable,script,'build_ext','--inplace'],cwd=ROOT/folder,check=True)
