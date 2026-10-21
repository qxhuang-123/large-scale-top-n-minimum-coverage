"""Read the completed matched-input batch; create comparison figures/booktabs tables."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent;BENCH=ROOT.parent/'TPCA'
SESSION=ROOT/'outputs/BGCR_matched_TPCAR_20261018';OUT=SESSION/'paper_report'
sys.path.insert(0,str(BENCH/'paper_figures_first_batch'))
from make_paper_figures import style,finish,axes_style,PAPER_WIDTH,QA
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

DS=['OP','YELP','VG','TG','SO']
STYLE={'TPCAR':('#3B75B9','o','-'),'MCF':('#D96800','s','--'),
    'GUROBI-LP':('#159D80','^','-.'),'GUROBI-MIP':('#8064A2','D',':'),
    'BGCR-D':('#DFA800','P','-')}
MODE=['dynamic_user_heap','dynamic_user_heap_nosource','dynamic_user_heap_noterminal','dynamic_user_scan']
SHORT=dict(zip(MODE,['Full','-Src','-Term','-Heap']))


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8')) if Path(path).exists() else {}


def write(path,data):
    Path(path).write_text(json.dumps(data,indent=2,ensure_ascii=False,default=lambda x:x.item()),encoding='utf8')


def cost_figures(joint,folder,memory=False):
    fig,axs=plt.subplots(2,3,figsize=(PAPER_WIDTH,4.65))
    fig.subplots_adjust(left=.09,right=.98,bottom=.12,top=.93,hspace=.55,wspace=.43)
    metric='peak_working_set_gib' if memory else 'optimization_total_seconds'
    for i,(ds,ax) in enumerate(zip(DS,axs.flat)):
        for method,(color,marker,ls) in STYLE.items():
            g=joint[(joint.dataset==ds)&(joint.method==method)]
            good=g[g.status.isin(['OPTIMAL','FEASIBLE'])]
            if len(good):
                a=good.groupby('percentage')[metric].agg(['median','min','max'])
                ax.errorbar(a.index,a['median'],yerr=np.array([a['median']-a['min'],a['max']-a['median']]),
                    color=color,marker=marker,markerfacecolor='white',ls=ls,markersize=4,capsize=2,lw=1.1)
                for p,r in good.groupby('percentage'):
                    ax.scatter(p+np.linspace(-1.3,1.3,len(r)),r[metric],color=color,alpha=.4,s=8)
            for status,symbol in [('TARGET_UNREACHED','x'),('INFEASIBLE','+')]:
                bad=g[g.status==status]
                if len(bad):
                    a=bad.groupby('percentage')[metric].median()
                    ax.scatter(a.index,a,marker=symbol,color=color,s=30,zorder=5)
        ax.set_title('(%s) %s'%(chr(97+i),'Yelp' if ds=='YELP' else ds),loc='left')
        ax.set_xlim(-4,104);ax.set_xticks([0,20,40,60,80,100]);ax.set_xlabel(r'Coverage requirement $\rho$ (%)')
        ax.set_ylabel('Peak working set (GiB)' if memory else 'Build + solve (s, log scale)')
        if memory:ax.set_ylim(bottom=0)
        else:ax.set_yscale('log')
        if ds=='SO':ax.text(.04,.96,'MCF: memory guard',transform=ax.transAxes,va='top',fontsize=7,color='#666666')
        axes_style(ax)
    note=axs.flat[-1];note.axis('off')
    note.legend(handles=[Line2D([],[],color=c,marker=m,ls=l,markerfacecolor='white',label=n) for n,(c,m,l) in STYLE.items()],loc='upper left',frameon=False,fontsize=8)
    note.text(.03,.30,'Points: runs; lines: medians\nWhiskers: min-max (n=3)\nx: BGCR target unreached\n+: exact target infeasible\nMemory guards: not plotted',transform=note.transAxes,va='top',fontsize=7,linespacing=1.5)
    finish(fig,'joint02_peak_memory' if memory else 'joint01_total_runtime',folder)


def N_figures(ndata,folder,coverage=False):
    fig,axs=plt.subplots(2,3,figsize=(PAPER_WIDTH,4.65))
    fig.subplots_adjust(left=.10,right=.985,bottom=.12,top=.93,wspace=.48,hspace=.57)
    curves=([('Kmax_TPCAR','TPCAR boundary','#3B75B9','o'),('K_BGCR','BGCR terminal','#DFA800','P')]
        if coverage else [('APS_exact_same_K','TPCAR at BGCR coverage','#3B75B9','o'),('APS_BGCR','BGCR','#DFA800','P')])
    for i,(ds,ax) in enumerate(zip(DS,axs.flat)):
        g=ndata[ndata.Dataset==ds].sort_values('N')
        for key,label,color,marker in curves:
            ax.plot(g.N,g[key],label=label,color=color,marker=marker,markerfacecolor='white',ls='--' if key.startswith('Kmax') or key.startswith('APS_exact') else '-')
        ax.set_title('(%s) %s'%(chr(97+i),'Yelp' if ds=='YELP' else ds),loc='left')
        ax.set_xlabel('Recommendation length N');ax.set_xticks([10,15,20,25,30])
        ax.set_ylabel('Distinct covered items' if coverage else 'Average predicted score')
        ax.ticklabel_format(axis='y',style='plain',useOffset=False);axes_style(ax)
    note=axs.flat[-1];note.axis('off');note.legend(*axs.flat[0].get_legend_handles_labels(),loc='upper left',frameon=False,fontsize=7.5)
    note.text(.03,.52,'Fixed shared candidate graph\nCommon rounded integer scores\nOne result per N\nAPS compared at equal coverage\nBGCR terminal is not Kmax',transform=note.transAxes,va='top',fontsize=7.5,linespacing=1.6)
    finish(fig,'N02_terminal_coverage' if coverage else 'N01_matched_coverage_APS',folder)


def ablation_figure(bg,folder):
    fig,axs=plt.subplots(2,3,figsize=(PAPER_WIDTH,4.65));axs=axs.ravel();axs[-1].axis('off');fig.subplots_adjust(left=.10,right=.985,bottom=.12,top=.90,wspace=.48,hspace=.57)
    for i,(ds,ax) in enumerate(zip(DS,axs)):
        for k,mode in enumerate(MODE):
            g=bg[(bg.dataset==ds)&(bg.N==10)&(bg.percentage==100)&(bg.native_mode==mode)&bg.status.isin(['FEASIBLE','TARGET_UNREACHED'])]
            v=g.optimization_total_seconds.dropna().to_numpy()
            if len(v):
                med=np.median(v);color=list(STYLE.values())[k][0]
                ax.errorbar(k,med,yerr=[[med-v.min()],[v.max()-med]],color=color,marker='o',markerfacecolor='white',ls='none',capsize=2)
                ax.scatter(k+np.linspace(-.05,.05,len(v)),v,color=color,s=9,alpha=.5)
            else:ax.text(k,.05,'Limited',rotation=90,transform=ax.get_xaxis_transform(),ha='center',va='bottom',fontsize=7)
        ax.set_title('(%s) %s'%(chr(97+i),'Yelp' if ds=='YELP' else ds),loc='left')
        ax.set_xticks(range(4));ax.set_xticklabels([SHORT[m] for m in MODE],fontsize=7.5)
        ax.set_yscale('log');ax.set_ylabel('Build + solve (s, log scale)');axes_style(ax)
    fig.text(.5,.97,'Full BGCR-D; -Src / -Term / -Heap: one module disabled',ha='center',va='top',fontsize=8)
    finish(fig,'A01_BGCR_module_ablation',folder)


def export_table(name,frame,columns,caption,folder):
    frame.to_csv(folder/(name+'.csv'),index=False,encoding='utf-8-sig',float_format='%.6f')
    lines=[]
    for dataset,g in frame.groupby('Dataset',sort=False):
        cols=['Dataset']+[c for c in columns if c!='Dataset']
        labels={'Mode':'Variant','Coverage':'$K$','K_BGCR':'BGCR $K$','Kmax_TPCAR':'Exact $K_{\\max}$',
                'APS_BGCR':'BGCR APS','APS_exact_same_K':'Exact APS at same $K$',
                'Gap_pct':'Gap (\\%)','Total':'Total (s)','Core':'Solve (s)','Peak_GiB':'Peak (GiB)','rho':'$\\rho$ (\\%)'}
        lines += [r'\begin{table*}[t]',r'\centering\scriptsize',r'\setlength{\tabcolsep}{3pt}',
            r'\caption{'+caption+': '+dataset+'.}',r'\begin{tabular}{'+'l'*len(cols)+'}',r'\toprule',
            ' & '.join(labels.get(c,c.replace('_',r'\_')) for c in cols)+r' \\',r'\midrule']
        for _,r in g.iterrows():
            vals=[]
            for c in cols:
                v=r[c]
                if pd.isna(v):s='--'
                elif isinstance(v,(float,np.floating)):s=('%.'+('6' if 'APS' in c or 'Gap' in c else '3')+'f')%v
                else:s=str(v)
                s=s.replace('TARGET_UNREACHED','Unreached').replace('SYSTEM_MEMORY_GUARD','Mem. guard').replace('INFEASIBLE','Infeas.').replace('OPTIMAL','Optimal')
                vals.append(s.replace('_',r'\_').replace('%',r'\%'))
            lines.append(' & '.join(vals)+r' \\')
        lines += [r'\bottomrule',r'\end{tabular}',r'\par\smallskip',r'\begin{minipage}{.98\textwidth}\footnotesize',
            r'\textit{Note:} BGCR is heuristic. Failure to reach a target is not a global infeasibility certificate. Score gaps use the common integer objective at the same attained coverage. Times are medians of completed independent runs; total includes method-specific construction and validation. Peak memory is the maximum OS peak working set. Resource-limited settings are identified separately.',
            r'\end{minipage}',r'\end{table*}','']
    (folder/(name+'.tex')).write_text('\n'.join(lines),encoding='utf8')


def generate():
    folder=OUT/'figures';tables=OUT/'tables';folder.mkdir(parents=True,exist_ok=True);tables.mkdir(exist_ok=True)
    bg=pd.read_csv(SESSION/'comparison.csv',dtype={'integer_objective':'string','exact_integer_objective_same_coverage':'string'})
    exact=pd.read_csv(BENCH/'outputs/first_batch_final_20261018/comparison.csv',dtype={'integer_objective':'string'})
    main=bg[(bg.native_mode=='dynamic_user_heap')&(bg.N==10)]
    assert not main.duplicated(['dataset','D','repeat']).any()
    assert main.groupby('dataset').graph_id.nunique().eq(1).all()
    joint=pd.concat([exact,main],ignore_index=True)
    for ds,g in joint.groupby('dataset'):assert g.graph_id.nunique()==1
    joint.to_csv(tables/'all_individual_main_comparisons.csv',index=False,encoding='utf-8-sig')
    records=read(SESSION/'all_results.json');meta={}
    for r in records:
        ds=r['job']['dataset']
        if ds not in meta:meta[ds]=read(Path(r['job']['graph_folder'])/'graph.json')
    ndata=[]
    ng=bg[(bg.native_mode=='dynamic_user_heap')&(bg.percentage==100)&(bg.repeat==1)]
    for _,r in ng.iterrows():
        ref_exact=r.exact_integer_objective_same_coverage
        if pd.isna(ref_exact):aps=np.nan
        else:aps=int(ref_exact)/(1e6*meta[r.dataset]['n_users']*r.N)
        case=next(x for x in records if x['job']['dataset']==r.dataset and x['job']['N']==r.N and x['job']['native_mode']=='dynamic_user_heap' and x['job']['repeat']==1 and x['job']['percentage']==100)
        ndata.append(dict(Dataset=r.dataset,N=int(r.N),K_BGCR=r.coverage,
            Kmax_TPCAR=(case.get('tpcar_reference') or {}).get('exact_boundary'),APS_BGCR=r.rounded_APS,
            APS_exact_same_K=aps,Gap_pct=r.exact_gap_same_coverage_pct,Status=r.status))
    ndata=pd.DataFrame(ndata);assert len(ndata)==25
    costs=[]
    for (ds,m,N,p,status),g in bg.groupby(['dataset','native_mode','N','percentage','status']):
        costs.append(dict(Dataset=ds,Mode=m,N=N,rho=p,Coverage=g.coverage.max(),Runs=len(g),Core=g.solve_seconds.median(),
            Total=g.optimization_total_seconds.median(),Peak_GiB=g.peak_working_set_gib.max(),
            Gap_pct=g.exact_gap_same_coverage_pct.max(),Status=status))
    costs=pd.DataFrame(costs)
    ablation=costs[(costs.N==10)&(costs.rho==100)&costs.Mode.isin(MODE)&costs.Dataset.isin(DS)].copy()
    ablation['Mode']=ablation.Mode.map(SHORT)
    baselines=costs[(costs.N==10)&(costs.rho==100)&costs.Mode.isin(['dynamic_user_heap','fixed_source','id_source'])&costs.Dataset.isin(['OP','YELP','SO'])].copy()
    baselines['Mode']=baselines.Mode.map({'dynamic_user_heap':'Dynamic D','fixed_source':'Fixed loss order','id_source':'Unsorted ID scan'})
    export_table('N_results',ndata,['N','K_BGCR','Kmax_TPCAR','APS_BGCR','APS_exact_same_K','Gap_pct','Status'],'BGCR across recommendation lengths',tables)
    export_table('main_BGCR',costs[costs.Mode=='dynamic_user_heap'],['N','rho','Coverage','Core','Total','Peak_GiB','Status'],'BGCR-D matched-input results',tables)
    export_table('module_ablation',ablation,['Mode','Runs','Core','Total','Peak_GiB','Status'],'BGCR-D module ablation',tables)
    export_table('direct_baselines',baselines,['Mode','Coverage','Core','Total','Gap_pct','Status'],'Direct-replacement ordering controls',tables)
    html='<meta charset="utf-8"><style>body{font-family:Georgia;margin:30px}table{border-collapse:collapse;font-size:13px;margin-bottom:30px}td,th{padding:6px 8px;border:0}thead{border-top:2px solid;border-bottom:1px solid}tbody{border-bottom:2px solid}</style>'
    for name,df in [('N_results',ndata),('module_ablation',ablation),('direct_baselines',baselines),('all_BGCR_costs',costs)]:html+='<h2>'+name+'</h2>'+df.to_html(index=False,float_format=lambda x:f'{x:.6f}',na_rep='—')
    (tables/'all_tables.html').write_text(html,encoding='utf8')
    style();cost_figures(joint,folder);cost_figures(joint,folder,True);N_figures(ndata,folder);N_figures(ndata,folder,True);ablation_figure(bg,folder)
    write(OUT/'validation.json',dict(figure_QA=QA,matched_main_runs=len(main),N_result_settings=len(ndata),
        semantics='Same graph/N/D/integer score; BGCR target-unreached is not global infeasibility; matched-coverage gaps use exact TPCAR trajectories.'))
    write(OUT/'progress.json',dict(complete=True,updated_unix=time.time()))


def main():
    p=argparse.ArgumentParser();p.add_argument('--wait',action='store_true');args=p.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    while not read(SESSION/'progress.json').get('complete'):
        if not args.wait:raise RuntimeError('BGCR batch is not complete')
        write(OUT/'progress.json',dict(complete=False,state='WAITING_FOR_BGCR_BATCH',updated_unix=time.time()));time.sleep(5)
    try:generate()
    except BaseException as exc:
        write(OUT/'progress.json',dict(complete=False,state='ERROR',error=str(exc)));raise
    print('Joint BGCR/TPCAR figures and booktabs tables:',OUT,flush=True)


if __name__=='__main__':main()
