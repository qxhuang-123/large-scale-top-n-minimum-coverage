import sys,runpy
from pathlib import Path
if __name__ == '__main__':
    folder=Path(__file__).resolve().parent/'TPCA'
    sys.path.insert(0,str(folder))
    runpy.run_path(str(folder/'run_progressive_gap.py'),run_name='__main__')
