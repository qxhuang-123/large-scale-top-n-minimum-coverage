"""Shared gate for original and added VG/TG ablation jobs."""
import json
from pathlib import Path
SESSION=Path(__file__).resolve().parent/'outputs/second_batch_ablation_20261018'
def read(path):
    try:return json.loads(path.read_text(encoding='utf-8-sig'))
    except (OSError,ValueError):return {}
def complete():
    if not read(SESSION/'progress.json').get('complete'):return False
    return not read(SESSION/'scope.json').get('extension_required') or bool(read(SESSION/'extension_progress.json').get('complete'))

def complete_for_bgcr():
    if not complete():return False
    extra=SESSION.parent/'progressive_gap_20261018'
    return not read(extra/'scope.json').get('required') or bool(read(extra/'progress.json').get('complete'))
