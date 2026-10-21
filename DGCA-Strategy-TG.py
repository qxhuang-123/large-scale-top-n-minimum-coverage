"""Current manuscript entry point; see current/README_CN.md."""
from pathlib import Path
import runpy

if __name__ == '__main__':
    runpy.run_path(str(Path(__file__).resolve().parent / 'current' / 'DGCA-Strategy-TG.py'), run_name='__main__')
