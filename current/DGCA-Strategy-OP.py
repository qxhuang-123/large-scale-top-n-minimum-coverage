import runpy
from pathlib import Path
if __name__ == '__main__':
    task=runpy.run_path(str(Path(__file__).resolve().parent/'DGCA-Strategies.py'))
    task['main']('OP')
