"""Wait for all representative comparisons, export data, figures and tables."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent;BENCH=ROOT.parent/'TPCA'
sys.path.insert(0,str(BENCH))
from resilient_json import write_json
from strategy_supervisor import read_json
SESSION=ROOT/'outputs/BGCR_strategy_comparison_20261019';OUT=SESSION/'paper_report'
VARIANTS=['R','A','B','C','D']


def fmt(x,n=3):return '--' if x is None or pd.isna(x) else ('%.'+str(n)+'f')%x
def sd(values):return values.std(ddof=1) if len(values)>1 else np.nan


def export_tables(df):
    folder=OUT/'tables';folder.mkdir(exist_ok=True);tables=[];summaries=[]
    for ds in ['OP','YELP','SO']:
        quality=[];cost=[];work=[]
        for variant in VARIANTS:
            g=df[(df.dataset==ds)&(df.variant==variant)];good=g[g.status.isin(['FEASIBLE','TARGET_UNREACHED'])]
            n=len(good);K=good.K.dropna();gap=good.gap_same_K_pct.dropna()
            total=good.total.dropna();core=good.core.dropna();build=good.build.dropna()
            summaries.append(dict(dataset=ds,variant=variant,runs_recorded=len(g),completed=n,
                coverage_mean=K.mean(),coverage_sd=sd(K),gap_mean=gap.mean(),gap_sd=sd(gap),
                core_median=core.median(),core_mean=core.mean(),core_sd=sd(core),
                build_median=build.median(),total_median=total.median(),total_mean=total.mean(),total_sd=sd(total),
                peak_gib=g.peak_gib.max()))
            quality.append([variant,str(n),fmt(K.mean(),2),fmt(sd(K),2),fmt(K.min(),0),fmt(K.max(),0),fmt(gap.mean(),5),fmt(sd(gap),5)])
            cost.append([variant,str(n),fmt(build.median()),fmt(core.median()),fmt(total.median()),fmt(total.mean()),fmt(sd(total)),fmt(g.peak_gib.max())])
            cols=['donor_slot_checks','column_edges','item_queries','heap_pops','terminal_edge_checks']
            work.append([variant]+[('--' if not len(good[c].dropna()) else '%.3g'%good[c].median()) for c in cols])
        for id,title,headers,rows,note in [
            ('T13_quality_'+ds,'Selection-rule quality on '+ds,['Var.','n','Mean K','SD K','Min K','Max K','Mean gap %','SD gap %'],quality,
             'R uses 10 fixed random seeds; A/B/C/D use 3 deterministic process repeats. SD is sample SD, not SEM or a confidence interval. Gap compares each run with exact TPCAR at that run\'s actual attained coverage. Unreached is not a global infeasibility certificate.'),
            ('T14_cost_'+ds,'Selection-rule computational cost on '+ds,['Var.','n','Build med.','Core med.','Total med.','Total mean','SD total','Peak GiB'],cost,
             'All times are seconds. Each setting starts a fresh process. R has seed variation as well as machine timing variation; deterministic repeats only measure timing variation. Total includes real method-specific input construction and validation. Peak is the largest observed OS working set. OS cache is not cleared.'),
            ('T15_work_'+ds,'Selection-rule work counters on '+ds,['Var.','Slot checks','Column edges','Item queries','Heap pops','Terminal checks'],work,
             'Counters are medians of completed runs. A/B use identical decisions with and without source aggregation. B/C share source aggregation and the same item query but differ in fixed scan versus current-minimum selection. C/D have identical dynamic decisions but different data structures; their runtime difference is not solely a terminal-aggregation effect.')]:
            tables.append(dict(id=id,title=title,headers=headers,rows=rows,note=note))
            pd.DataFrame(rows,columns=headers).to_csv(folder/(id+'.csv'),index=False,encoding='utf-8-sig')
            def esc(x):return str(x).replace('_',r'\_').replace('%',r'\%').replace('&',r'\&')
            lines=[r'\begin{table*}[t]',r'\centering\footnotesize\setlength{\tabcolsep}{3pt}',r'\caption{'+esc(title)+'}',
                r'\begin{tabular}{'+'l'*len(headers)+'}',r'\toprule',' & '.join(map(esc,headers))+r' \\',r'\midrule']
            lines+=[' & '.join(map(esc,row))+r' \\' for row in rows]
            lines += [r'\bottomrule',r'\end{tabular}',r'\par\smallskip',r'\begin{minipage}{.98\textwidth}\footnotesize',esc(note),r'\end{minipage}',r'\end{table*}']
            (folder/(id+'.tex')).write_text('\n'.join(lines),encoding='utf8')
    write_json(folder/'table_catalog.json',tables)
    pd.DataFrame(summaries).to_csv(OUT/'data/summary.csv',index=False,encoding='utf-8-sig')
    html='<meta charset="utf-8"><style>body{font-family:Georgia;margin:30px}table{border-collapse:collapse}th,td{padding:5px}thead{border-top:2px solid;border-bottom:1px solid}tbody{border-bottom:2px solid}</style>'
    for t in tables:html+='<h2>'+t['title']+'</h2>'+pd.DataFrame(t['rows'],columns=t['headers']).to_html(index=False)+'<p>'+t['note']+'</p>'
    (folder/'all_tables.html').write_text(html,encoding='utf8')


def generate():
    from run_strategy_comparison import summarize
    results=read_json(SESSION/'all_results.json');df=pd.DataFrame(summarize(results))
    assert not df.duplicated(['dataset','variant','repeat','seed']).any()
    for ds in ['OP','YELP','SO']:
        for variant in VARIANTS:
            g=df[(df.dataset==ds)&(df.variant==variant)]
            assert len(g)>0
            if g.status.isin(['FEASIBLE','TARGET_UNREACHED']).all():assert len(g)==(10 if variant=='R' else 3)
    for sub in ['code','data','figures','tables','qa']:(OUT/sub).mkdir(parents=True,exist_ok=True)
    df.to_csv(OUT/'data/comparison.csv',index=False,encoding='utf-8-sig')
    for name in ['protocol.json','checks.json','all_results.json']:
        shutil.copy2(SESSION/name,OUT/'data'/name)
    for name in ['strategy_20261019','strategy_pipeline_20261019']:
        shutil.copy2(ROOT/'validation'/name/'validation.json',OUT/'data'/('validation_'+name+'.json'))
    shutil.copy2(ROOT/'README_STRATEGY_20261019.md',OUT/'README.md')
    for p in [ROOT/'strategy_figures.py',ROOT/'strategy_figure_qa.py',ROOT/'strategy_pdf.py']:
        shutil.copy2(p,OUT/'code'/p.name)
    for number in [21,22,23,24]:
        (OUT/'code'/('figure_F%d.py'%number)).write_text('from strategy_figures import draw\n\nif __name__=="__main__":\n    draw('+str(number)+')\n',encoding='utf8')
    write_json(OUT/'data/figure_catalog.json',[
        dict(id='F21',name='F21_strategy_total_runtime',metric='Build + core seconds'),
        dict(id='F22',name='F22_strategy_core_runtime',metric='Core seconds'),
        dict(id='F23',name='F23_strategy_coverage_deficit',metric='Kmax - K'),
        dict(id='F24',name='F24_strategy_score_gap',metric='Exact score gap at the same actual coverage (%)')])
    (OUT/'requirements_plotting.txt').write_text('numpy=='+np.__version__+'\npandas=='+pd.__version__+'\nmatplotlib\nPillow\n',encoding='utf8')
    export_tables(df)
    # Reuse the already completed ID baseline only as a separately identified historical control.
    historical=[]
    for p in (ROOT/'outputs/BGCR_matched_TPCAR_20261018').glob('*_id_source_N10_*/result.json'):
        r=read_json(p);historical.append(r)
    write_json(OUT/'data/historical_ID_reference.json',historical)
    subprocess.run([sys.executable,str(ROOT/'strategy_figures.py'),'--root',str(OUT)],check=True)
    runtime=Path('C:/Users/24qxh/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe')
    subprocess.run([str(runtime),'-X','utf8',str(ROOT/'strategy_pdf.py'),'--root',str(OUT)],check=True)
    write_json(OUT/'progress.json',dict(complete=True,state='GENERATED_PENDING_VISUAL_REVIEW',runs=len(df),updated_unix=time.time()))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--wait',action='store_true');args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    while not read_json(SESSION/'progress.json').get('complete'):
        if not args.wait:raise RuntimeError('Strategy comparison is not complete')
        write_json(OUT/'progress.json',dict(complete=False,state='WAITING_FOR_STRATEGY_COMPARISON',updated_unix=time.time()));time.sleep(5)
    try:generate()
    except BaseException as exc:
        write_json(OUT/'progress.json',dict(complete=False,state='ERROR',error=str(exc),updated_unix=time.time()));raise


if __name__=='__main__':main()
