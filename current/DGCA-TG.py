"""DGCA: TG, same graph and integer scores as TPCA."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent/'DGCA'))
from run_bgcr import main
if __name__ == '__main__': main('TG')
