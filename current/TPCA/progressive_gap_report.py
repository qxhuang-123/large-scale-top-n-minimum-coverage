"""Read-only progressive ablation figures and booktabs tables; never solve."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter

ROOT=Path(__file__).resolve().parent
SESSION=ROOT/'outputs/progressive_gap_20261018'
OUT=SESSION/'paper_report'
sys.path.insert(0,str(ROOT/'paper_figures_first_batch'))
from make_paper_figures import style, axes_style, finish, PAPER_WIDTH, QA
from bench_common import write_json
from resilient_json import write_json
from progressive_supervisor import read_json

VERSIONS=['V1','V2','V3','V4']
COLORS=['#8064A2','#D96800','#159D80','#3B75B9']
MARKERS=['D','s','^','o']
STYLES=[':', '--', '-.', '-']


def figure_pair():
    fig,axs=plt.subplots(1,2,figsize=(PAPER_WIDTH,2.85))
    fig.subplots_adjust(left=.09,right=.985,bottom=.21,top=.78,wspace=.40)
    return fig,axs


def legend(fig, versions=VERSIONS):
    handles=[Line2D([],[],color=COLORS[VERSIONS.index(v)],marker=MARKERS[VERSIONS.index(v)],
        ls=STYLES[VERSIONS.index(v)],markerfacecolor='white',label=v) for v in versions]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.99),ncol=len(versions),frameon=False)


def trajectory_figure():
    """Existing high-target runs only; not independent low-target measurements."""
    old=ROOT/'outputs/second_batch_ablation_20261018';fig,axs=figure_pair();data=[];sources=[]
    for index,(ds,ax) in enumerate(zip(['OP','YELP'],axs)):
        for version,mode in [('V3','state_source_only'),('V4','state_source')]:
            C=2367 if ds=='OP' else 7017
            folder=old/('%s_%s_N10_D%d_r1'%(ds,mode,C))
            result=read_json(folder/'result.json');trace=read_json(folder/'trajectory.json')
            if result.get('status') not in ['OPTIMAL','INFEASIBLE']:continue
            K0=result['job']['initial_coverage'];Kmax=result['audit']['coverage']
            points=[(int(r['coverage'])-K0,float(r['time_sec'])) for r in trace
                    if r.get('trajectory_kind')=='exact_coverage_augmentation']
            assert points and all(t>=0 for _,t in points)
            assert all(a[0]<=b[0] and a[1]<=b[1] for a,b in zip(points,points[1:]))
            k=VERSIONS.index(version);x,y=zip(*points)
            ax.plot(x,y,color=COLORS[k],ls=STYLES[k],marker=MARKERS[k],markerfacecolor='white',
                markevery=max(1,len(points)//7),ms=3.5,lw=1.1)
            for gap,seconds in points:data.append(dict(dataset=ds,version=version,gap=gap,
                gap_percentage=100*gap/(Kmax-K0),cumulative_core_seconds=seconds,
                interpretation='Prefix of one high-target run; not independent target runtime'))
            sources.append(str(folder))
        ax.set_title('(%s) %s'%(chr(97+index),'Yelp' if ds=='YELP' else ds),loc='left')
        ax.set_xlabel(r'Attained coverage increase $K-K^0$');ax.set_ylabel('Cumulative core time (s)')
        ax.set_xlim(left=0);ax.set_ylim(bottom=0);axes_style(ax)
    legend(fig,['V3','V4'])
    OUT.mkdir(parents=True,exist_ok=True);figdir=OUT/'existing_trajectories';figdir.mkdir(exist_ok=True)
    finish(fig,'T01_existing_high_target_trajectories',figdir)
    pd.DataFrame(data).to_csv(figdir/'trajectory_points.csv',index=False,encoding='utf-8-sig')
    write_json(figdir/'interpretation.json',dict(sources=sources,
        meaning='Actual prefixes of previously completed high-target runs, one run per curve. Includes native initialization; excludes dedicated Python construction and the final no-path verification. Not independently started low-gap runs.',
        figure_QA=QA,visual_review='Pending manual image inspection'))
    return figdir


def independent_figure(df, metric, name, ylabel):
    fig,axs=figure_pair()
    for index,(ds,ax) in enumerate(zip(['OP','YELP'],axs)):
        for k,version in enumerate(VERSIONS):
            subset=df[(df.dataset==ds)&(df.version==version)].sort_values('gap_percentage')
            if subset.empty:continue
            good=subset.status.eq('OPTIMAL')
            values=pd.to_numeric(subset[metric],errors='coerce').where(good)
            ax.plot(subset.gap_percentage,values,color=COLORS[k],ls=STYLES[k],marker=MARKERS[k],
                markerfacecolor='white',lw=1.1,ms=4)
            for _,row in subset[~good].iterrows():
                # Axis-relative crosses identify a limited setting; not a measured runtime.
                ax.plot(row.gap_percentage,.04+.055*k,marker='x',color=COLORS[k],
                    ls='none',transform=ax.get_xaxis_transform())
        ax.set_title('(%s) %s'%(chr(97+index),'Yelp' if ds=='YELP' else ds),loc='left')
        ax.set_xscale('log');ax.set_xticks([1,10,50,100]);ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.set_xlim(.8,125);ax.set_xlabel(r'Requested gap fraction $\rho$ (%)');ax.set_ylabel(ylabel)
        if metric!='peak_working_set_gib':ax.set_yscale('log')
        else:ax.set_ylim(bottom=0)
        axes_style(ax)
    legend(fig);folder=OUT/'independent_targets';folder.mkdir(exist_ok=True)
    finish(fig,name,folder)


def tables(df):
    folder=OUT/'tables';folder.mkdir(exist_ok=True)
    df.to_csv(folder/'independent_runs.csv',index=False,encoding='utf-8-sig')
    statuses={'OPTIMAL':'Optimal','TIME_LIMIT':'Time limit','SYSTEM_MEMORY_GUARD':'Mem. guard',
              'PRIVATE_MEMORY_LIMIT':'Mem. limit','MEMORY_ERROR':'Mem. error','PROCESS_WALL_GUARD':'Wall limit'}
    docs=[];html=[]
    for ds in ['OP','YELP']:
        frame=df[df.dataset==ds].copy();frame['version']=pd.Categorical(frame.version,VERSIONS,ordered=True)
        frame=frame.sort_values(['gap_percentage','version'])
        columns=['version','gap_percentage','gap','D','coverage','build_seconds','solve_seconds','total_seconds','peak_working_set_gib','status']
        frame[columns].to_csv(folder/(ds+'_three_line.csv'),index=False,encoding='utf-8-sig')
        lines=[r'\begin{table*}[t]',r'\centering\footnotesize\setlength{\tabcolsep}{3pt}',
            r'\caption{Independent positive-gap ablation on '+('Yelp' if ds=='YELP' else ds)+r'. Each setting uses one process.}',
            r'\begin{tabular}{llrrrrrrrl}',r'\toprule',
            r'Var. & $\rho$ (\%) & $\Delta$ & $D$ & $K$ & Build (s) & Solve (s) & Total (s) & Peak (GiB) & Status \\',r'\midrule']
        for _,row in frame.iterrows():
            values=[]
            for c in columns:
                value=row[c]
                if pd.isna(value):s='--'
                elif c=='status':s=statuses.get(value,str(value).replace('_',r'\_'))
                elif c in ['build_seconds','solve_seconds','total_seconds','peak_working_set_gib']:s='%.3f'%value
                elif c in ['gap_percentage','gap','D','coverage']:s=str(int(value))
                else:s=str(value)
                values.append(s)
            lines.append(' & '.join(values)+r' \\')
        lines += [r'\bottomrule',r'\end{tabular}',r'\par\smallskip',
            r'\begin{minipage}{.98\textwidth}\footnotesize',
            r'\textit{Note:} $D=K^0+\lceil\rho(K_{\max}-K^0)\rceil$, with $\rho$ expressed as a fraction. Build and solve times are separate; total is their sum. Peak is the OS peak working set. One run per setting; no repeat-based uncertainty is estimated. Resource-limited settings have no completed-solve time. All completed variants are checked against the existing exact integer objective at the same $D$.',
            r'\end{minipage}',r'\end{table*}']
        (folder/(ds+'_three_line.tex')).write_text('\n'.join(lines),encoding='utf8')
        docs.append(ds+'_three_line');html.append('<h2>'+ds+'</h2>'+frame[columns].to_html(index=False))
    (folder/'all_tables.html').write_text('<!doctype html><meta charset="utf-8"><style>body{font-family:Georgia;margin:25px}table{border-collapse:collapse;font-size:13px}th,td{padding:5px}thead{border-top:2px solid;border-bottom:1px solid}tbody{border-bottom:2px solid}</style>'+''.join(html),encoding='utf8')
    tex=r'\documentclass[a4paper,10pt,twocolumn]{article}'+'\n'+r'\usepackage[hmargin=18.1mm,vmargin=20mm]{geometry}\usepackage{booktabs}\begin{document}'+'\n'
    tex+='\n'.join(r'\input{tables/'+name+r'}\clearpage' for name in docs)+'\n'+r'\end{document}'
    (OUT/'tables_preview.tex').write_text(tex,encoding='utf8')
    if shutil.which('pdflatex'):
        subprocess.run(['pdflatex','-interaction=batchmode','-halt-on-error','tables_preview.tex'],cwd=str(OUT),check=True,stdout=subprocess.DEVNULL)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--wait',action='store_true')
    parser.add_argument('--trajectory-only',action='store_true');args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True);style()
    if args.trajectory_only:
        print(trajectory_figure());return
    while not read_json(SESSION/'progress.json').get('complete'):
        if not args.wait:raise RuntimeError('Independent-gap batch is not complete')
        write_json(OUT/'progress.json',dict(complete=False,state='WAITING_FOR_INDEPENDENT_GAP_RUNS',updated_unix=time.time()))
        time.sleep(5)
    try:
        from run_progressive_gap import summarize
        rows=summarize(read_json(SESSION/'all_results.json'));df=pd.DataFrame(rows)
        assert len(df)==32 and not df.duplicated(['dataset','version','gap_percentage']).any()
        # Descriptive profile accompanies every exported data figure.
        profile=dict(rows=len(df),repeats_per_setting=1,status_counts=df.status.value_counts().to_dict(),
            missing_completed_times=int(df.solve_seconds.isna().sum()),
            numeric_summary=df[['build_seconds','solve_seconds','total_seconds','peak_working_set_gib']].describe().to_dict(),
            no_monotone_speedup_assumed=True)
        write_json(OUT/'data_profile.json',profile)
        independent_figure(df,'total_seconds','P01_independent_total_runtime','Build + solve (s, log scale)')
        independent_figure(df,'solve_seconds','P02_independent_core_runtime','Core solve time (s, log scale)')
        independent_figure(df,'peak_working_set_gib','P03_independent_peak_memory','Peak working set (GiB)')
        tables(df)
        write_json(OUT/'validation.json',dict(figure_QA=QA,rows=len(df),
            note='Independent per-target single runs; no V0 and no assumed speedup ordering. Existing trajectories are separate artifacts.',
            visual_review='Pending manual image and table inspection'))
        write_json(OUT/'progress.json',dict(complete=True,state='GENERATED_PENDING_VISUAL_REVIEW',updated_unix=time.time()))
        print('Progressive figures and tables generated:',OUT)
    except BaseException as exc:
        write_json(OUT/'progress.json',dict(complete=False,state='ERROR',error=str(exc)));raise


if __name__=='__main__':main()
