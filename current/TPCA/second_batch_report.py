from ablation_completion import complete as ablation_complete
"""After the two authorized batches finish, draw figures and booktabs tables.

This script reads saved results only. It never starts an optimization experiment.
Run with --wait to queue reporting after the active N/ablation batches.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'paper_figures_first_batch'))
from make_paper_figures import style, finish, axes_style, PAPER_WIDTH, QA
import matplotlib.pyplot as plt

OUT = ROOT/'outputs/second_batch_figures_20261018'
NROOT = ROOT/'outputs/second_batch_N_exact_20261018'
AROOT = ROOT/'outputs/second_batch_ablation_20261018'
DATASETS = ['OP','YELP','VG','TG','SO']
MODES = ['state_source','state_terminal','state_source_only','state_source_noheap']
SHORT = dict(zip(MODES,['Full','-Src','-Term','-Heap']))
COLORS = ['#3B75B9','#D96800','#159D80','#8064A2']
MARKERS = ['o','s','^','D']


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8')) if Path(path).exists() else {}


def save_json(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,default=lambda x:x.item()),encoding='utf8')


def n_results():
    paths=list((ROOT/'outputs/first_batch_final_20261018').glob('*_TPCAR_N10_*_r1/result.json'))
    paths=[p for p in paths if read(p)['job']['percentage']==100]
    paths+=list(NROOT.glob('N*/*/result.json'))
    rows=[]
    for path in paths:
        r=read(path);j=r['job'];a=r.get('audit',{})
        meta=read(Path(j['graph_folder'])/'graph.json')
        count=meta['n_users']*j['N'];q=a.get('integer_objective')
        known=(a.get('coverage')==j['D'] or r.get('maximum_coverage_certified') is True)
        rows.append(dict(Dataset=j['dataset'],N=j['N'],K0=j['initial_coverage'],
            Request=j['D'],Kmax=a.get('coverage') if known else np.nan,
            TopN_APS=j['initial_integer_objective']/(1e6*count),
            TPCAR_APS=q/(1e6*count) if q is not None else np.nan,
            Loss_pct=(j['initial_integer_objective']-q)/j['initial_integer_objective']*100 if q is not None else np.nan,
            Integer_objective=str(q) if q is not None else '--',Status=r['status'],
            graph_id=j['graph_id'],result_file=str(path)))
    df=pd.DataFrame(rows).sort_values(['Dataset','N'])
    assert len(df)==25 and not df.duplicated(['Dataset','N']).any()
    assert df.groupby('Dataset').graph_id.nunique().eq(1).all()
    return df,paths


def ablation_results():
    paths=list(AROOT.glob('*/result.json'));records=[]
    for path in paths:
        r=read(path);j=r['job'];a=r.get('audit',{});d=r.get('native_diagnostics',{})
        records.append(dict(Dataset=j['dataset'],Mode=j['native_mode'],Repeat=j['repeat'],
            Status=r['status'],Objective=str(a.get('integer_objective','--')),Coverage=a.get('coverage'),
            Build=r.get('build_seconds',np.nan),Solve=r.get('solve_seconds',np.nan),
            Total=r.get('optimization_total_seconds',np.nan),Peak_GiB=r.get('peak_working_set_bytes',0)/1024**3,
            Augmentations=d.get('augmentations',np.nan),Repairs=d.get('local_repairs',np.nan),
            Affected_total=d.get('affected_total',np.nan),Affected_max=d.get('affected_max',np.nan),
            Scans=d.get('adjacency_scans',np.nan),Heap_checks=d.get('heap_top_checks',np.nan),
            Boundary_scans=d.get('boundary_scan_checks',np.nan),result_file=str(path)))
    df=pd.DataFrame(records)
    for ds in DATASETS:
        good=df[(df.Dataset==ds)&df.Status.isin(['OPTIMAL','INFEASIBLE'])]
        assert good.Objective.nunique()<=1 and good.Status.nunique()<=1
    return df,paths


def plot_N(df,figdir,coverage=False):
    fig,axs=plt.subplots(2,3,figsize=(PAPER_WIDTH,4.65))
    fig.subplots_adjust(left=.10,right=.985,bottom=.12,top=.93,wspace=.48,hspace=.57)
    for i,(ds,ax) in enumerate(zip(DATASETS,axs.flat)):
        g=df[df.Dataset==ds].sort_values('N')
        for name,color,marker,ls in [('Top-N','#3B75B9','o','--'),('TPCAR','#D96800','s','-')]:
            key=('K0' if name=='Top-N' else 'Kmax') if coverage else ('TopN_APS' if name=='Top-N' else 'TPCAR_APS')
            ax.plot(g.N,g[key],color=color,marker=marker,markerfacecolor='white',ls=ls,label=name)
        ax.set_title('(%s) %s'%(chr(97+i),'Yelp' if ds=='YELP' else ds),loc='left',pad=7)
        ax.set_xticks([10,15,20,25,30]);ax.set_xlabel('Recommendation length N')
        ax.set_ylabel('Distinct covered items' if coverage else 'Average predicted score')
        ax.ticklabel_format(axis='y',style='plain',useOffset=False)
        axes_style(ax)
    note=axs.flat[-1];note.axis('off')
    note.legend(*axs.flat[0].get_legend_handles_labels(),loc='upper left',frameon=False)
    note.text(.03,.53,'Fixed candidate graph\nSame rounded integer scores\nTarget = candidate item count\nOne deterministic result per N\nTPCAR reports exact boundary\nwhen the target is infeasible',
              transform=note.transAxes,va='top',fontsize=7.5,linespacing=1.7)
    finish(fig,'N02_coverage' if coverage else 'N01_predicted_scores',figdir)


def plot_ablation(df,figdir,memory=False):
    fig,axs=plt.subplots(2,3,figsize=(PAPER_WIDTH,4.65));axs=axs.ravel();axs[-1].axis('off')
    fig.subplots_adjust(left=.10,right=.985,bottom=.12,top=.90,wspace=.48,hspace=.57)
    metric='Peak_GiB' if memory else 'Total'
    for i,(ds,ax) in enumerate(zip(DATASETS,axs)):
        for k,mode in enumerate(MODES):
            g=df[(df.Dataset==ds)&(df.Mode==mode)&df.Status.isin(['OPTIMAL','INFEASIBLE'])]
            values=g[metric].dropna().to_numpy()
            if len(values):
                med=np.median(values)
                ax.errorbar(k,med,yerr=[[med-values.min()],[values.max()-med]],
                    marker=MARKERS[k],color=COLORS[k],markerfacecolor='white',ls='none',capsize=2)
                ax.scatter(k+np.linspace(-.06,.06,len(values)),values,s=9,color=COLORS[k],alpha=.5)
            else:
                ax.text(k,.05,'Limited',rotation=90,va='bottom',ha='center',fontsize=7,
                        transform=ax.get_xaxis_transform(),color='#666666')
        ax.set_xticks(range(4));ax.set_xticklabels([SHORT[m] for m in MODES],fontsize=7.5)
        ax.set_xlim(-.4,3.4);ax.set_title('(%s) %s'%(chr(97+i),'Yelp*' if ds=='YELP' else ds),loc='left')
        ax.set_ylabel('Peak working set (GiB)' if memory else 'Build + solve (s, log scale)')
        if memory:ax.set_ylim(bottom=0)
        else:ax.set_yscale('log')
        axes_style(ax)
    fig.text(.5,.97,'Full: complete new TPCAR; -Src / -Term / -Heap: one module disabled',
             ha='center',va='top',fontsize=8)
    finish(fig,'A02_peak_memory' if memory else 'A01_total_runtime',figdir)


def tables(ndf,adf,folder):
    summaries=[]
    work=[]
    for ds in DATASETS:
        for mode in MODES:
            g=adf[(adf.Dataset==ds)&(adf.Mode==mode)]
            summaries.append(dict(Dataset=ds,Mode=SHORT[mode],Runs=len(g),
                Build=g.Build.median(),Solve=g.Solve.median(),Total=g.Total.median(),
                Total_min=g.Total.min(),Total_max=g.Total.max(),Peak_GiB=g.Peak_GiB.max(),
                Status='/'.join(sorted(g.Status.unique()))))
            work.append(dict(Dataset=ds,Mode=SHORT[mode],Scans=g.Scans.median(),
                Repairs=g.Repairs.median(),Affected_total=g.Affected_total.median(),
                Affected_max=g.Affected_max.max(),Heap_checks=g.Heap_checks.median(),Boundary_scans=g.Boundary_scans.median()))
    frames={'N_exact':ndf,'A_time_memory':pd.DataFrame(summaries),'A_work_counters':pd.DataFrame(work)}
    select={'N_exact':['Dataset','N','K0','Kmax','TopN_APS','TPCAR_APS','Loss_pct','Status'],
            'A_time_memory':['Dataset','Mode','Runs','Build','Solve','Total','Peak_GiB','Status'],
            'A_work_counters':['Dataset','Mode','Scans','Repairs','Affected_total','Heap_checks','Boundary_scans']}
    labels={'K0':'$K^0$','Kmax':'$K_{\\max}$','TopN_APS':'Top-$N$ APS','TPCAR_APS':'TPCAR APS',
        'Loss_pct':r'Loss (\%)','Build':'Build (s)','Solve':'Solve (s)','Total':'Total (s)',
        'Peak_GiB':'Peak (GiB)','Affected_total':'Repair nodes','Heap_checks':'Heap checks','Boundary_scans':'Column scans'}
    chunks=[]
    for name,frame in frames.items():
        frame.to_csv(folder/(name+'.csv'),index=False,encoding='utf-8-sig',float_format='%.6f')
        cols=select[name];lines=[r'\begin{table*}[t]',r'\centering\scriptsize',r'\setlength{\tabcolsep}{3pt}',
            r'\caption{'+{'N_exact':'Exact coverage and score outcomes across recommendation lengths.',
              'A_time_memory':'Ablation of the new TPCAR implementation.',
              'A_work_counters':'Search and repair work in the new TPCAR ablation.'}[name]+'}',
            r'\begin{tabular}{'+'l'*len(cols)+'}',r'\toprule',
            ' & '.join(labels.get(c,c.replace('_',r'\_')) for c in cols)+r' \\',r'\midrule']
        for _,r in frame.iterrows():
            values=[]
            for c in cols:
                value=r[c]
                if pd.isna(value):s='--'
                elif isinstance(value,(float,np.floating)):s=('%.'+('6' if 'APS' in c else '3')+'f')%value
                else:s=str(value)
                s=s.replace('SYSTEM_MEMORY_GUARD','Mem. guard').replace('INFEASIBLE','Infeas.').replace('OPTIMAL','Optimal')
                values.append(s.replace('_',r'\_'))
            lines.append(' & '.join(values)+r' \\')
        note=(r'APS uses the common rounded integer scores. An infeasible highest request returns the attainable boundary plan. Each new $N$ has one deterministic run; $N=10$ reuses the verified first-batch result.' if name=='N_exact' else
              r'Completed settings use three independent processes; memory-limited settings skip remaining repeats. Times are separate medians; component medians need not sum to total. Peak memory is the maximum OS peak working set. On Yelp all variants attempt the same infeasible highest request and return a boundary plan. Full, -Src, -Term, and -Heap retain exact augmentation and local tree repair.')
        lines.extend([r'\bottomrule',r'\end{tabular}',r'\par\smallskip',r'\begin{minipage}{.98\textwidth}\footnotesize',r'\textit{Note:} '+note,r'\end{minipage}',r'\end{table*}'])
        (folder/(name+'.tex')).write_text('\n'.join(lines),encoding='utf8')
        chunks.append('<h2>'+name+'</h2>'+frame.to_html(index=False,float_format=lambda x:f'{x:.6f}',na_rep='—'))
    (folder/'all_tables.html').write_text('<!doctype html><meta charset="utf-8"><style>body{font-family:Georgia;margin:30px}table{border-collapse:collapse;font-size:13px;margin-bottom:35px}th,td{padding:6px 8px;border:0}thead{border-top:2px solid;border-bottom:1px solid}tbody{border-bottom:2px solid}</style>'+''.join(chunks),encoding='utf8')
    preview=r'\documentclass[a4paper,10pt,twocolumn]{article}'+'\n'+r'\usepackage[hmargin=18.1mm,vmargin=20mm]{geometry}\usepackage{booktabs,graphicx}\begin{document}'+'\n'
    for name in frames:preview+=r'\input{tables/'+name+'}\n'+r'\clearpage'+'\n'
    preview+=r'\end{document}'
    (OUT/'tables_preview.tex').write_text(preview,encoding='utf8')
    if shutil.which('pdflatex'):
        subprocess.run(['pdflatex','-interaction=batchmode','-halt-on-error','tables_preview.tex'],cwd=str(OUT),check=True,stdout=subprocess.DEVNULL)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--wait',action='store_true');args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    while not (read(NROOT/'queue_progress.json').get('complete') and ablation_complete()):
        if not args.wait:raise RuntimeError('Both authorized batches must finish before final reporting.')
        save_json(OUT/'progress.json',dict(complete=False,state='WAITING_FOR_TWO_AUTHORIZED_BATCHES',updated_unix=time.time()))
        time.sleep(5)
    try:
        figdir=OUT/'figures';tabdir=OUT/'tables';figdir.mkdir(exist_ok=True);tabdir.mkdir(exist_ok=True)
        style();ndf,npaths=n_results();adf,apaths=ablation_results()
        plot_N(ndf,figdir);plot_N(ndf,figdir,True)
        plot_ablation(adf,figdir);plot_ablation(adf,figdir,True)
        adf.to_csv(tabdir/'ablation_individual_runs.csv',index=False,encoding='utf-8-sig')
        tables(ndf,adf,tabdir)
        save_json(OUT/'validation.json',dict(N_results=len(ndf),ablation_runs=len(adf),figure_QA=QA,
            source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in npaths+apaths},
            interpretation='Different N results are deterministic single runs. Ablation uses 3 repeats when completed. No BGCR, old TPCAR, or unrun Gurobi rows are included.'))
        save_json(OUT/'progress.json',dict(complete=True,N_results=len(ndf),ablation_runs=len(adf),updated_unix=time.time()))
        print('Second-batch figures and booktabs tables:',OUT,flush=True)
    except BaseException as exc:
        save_json(OUT/'progress.json',dict(complete=False,state='ERROR',error=str(exc),updated_unix=time.time()))
        raise


if __name__=='__main__':main()
