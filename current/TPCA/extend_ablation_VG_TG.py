"""Wait for the original queue, then reuse its results and add VG/TG."""
import json,subprocess,sys,time
from pathlib import Path
from bench_common import ROOT,write_json
from ablation_completion import SESSION,read

def main():
    out=SESSION/'extension_progress.json'
    if read(out).get('complete'):return
    try:
        while not read(SESSION/'progress.json').get('complete'):
            if (ROOT/'execution/stop.request').exists():
                write_json(out,dict(complete=False,state='STOP_REQUESTED'));return
            write_json(out,dict(complete=False,state='WAITING_FOR_ORIGINAL_THREE_DATASETS',additional_max_jobs=24,updated_unix=time.time()))
            time.sleep(5)
        if (ROOT/'execution/stop.request').exists():
            write_json(out,dict(complete=False,state='STOP_REQUESTED'));return
        write_json(out,dict(complete=False,state='ADDING_VG_TG_REUSING_EXISTING_RESULTS',updated_unix=time.time()))
        subprocess.run([sys.executable,'-u',str(ROOT/'run_ablation.py')],cwd=str(ROOT),check=True)
        progress=read(SESSION/'progress.json')
        if not progress.get('complete'):raise RuntimeError('Extended queue did not finish; inspect progress.json')
        results=read(SESSION/'all_results.json')
        datasets={r['job']['dataset'] for r in results}
        if datasets!={'OP','YELP','SO','VG','TG'}:raise RuntimeError('Missing dataset in extended results')
        write_json(out,dict(complete=True,state='COMPLETE',recorded_jobs=len(results),planned_max_jobs=60,updated_unix=time.time()))
    except BaseException as exc:
        write_json(out,dict(complete=False,state='ERROR',error=str(exc),updated_unix=time.time()));raise
if __name__=='__main__':main()
