"""TPCA: YELP, common rounded graph, one independent run."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent/'TPCA'))
from run_comparison import main
if __name__ == '__main__':
    if '--repeats' not in sys.argv: sys.argv += ['--repeats','1']
    if '--percentages' not in sys.argv and '--target' not in sys.argv: sys.argv += ['--percentages','100']
    main('YELP', 'TPCAR')
