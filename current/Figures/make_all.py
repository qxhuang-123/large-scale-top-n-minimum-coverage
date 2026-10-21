"""Frozen-data publication previews. No solver imports or solver execution.

Python 3.9+, numpy/pandas/matplotlib/Pillow. All measurements use repeat=1;
random scan uses seed 2026101901. Main totals and cumulative native times
remain distinct. Run --only T05_strategy_full to rebuild the requested table.
"""
from pathlib import Path
from contextlib import nullcontext
import argparse
import csv
import hashlib
import json
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, NullLocator, MaxNLocator
from PIL import Image, ImageOps, ImageDraw

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'code'))
import table_layouts as t
import figure_layouts as f

DS=['OP','YELP','VG','TG','SO']
VAR=['R','A','B','C','D']
STYLES={'MCF':('#D55E00','s','--'),'TPCAR':('#0072B2','o','-'),
        'BGCR-D':('#009E73','^','-.'),'GUROBI-LP':('#9C65A8','D',':'),
        'GUROBI-MIP':('#555555','P','--')}
VARSTYLE={'R':('#777777','o'),'A':('#D55E00','s'),'B':('#B58A00','D'),
          'C':('#0072B2','v'),'D':('#009E73','^')}
ASSETS=[]


def save_frame(name,frame):
    frame.to_csv(ROOT/'data'/name,index=False,encoding='utf-8-sig')


def load():
    a,m,n,s,q=f.read_data()
    m=m[m['repeat']==1].copy();s=s[s['repeat']==1].copy()
    assert len(m)==102 and len(s)==25
    assert set(s.dataset)==set(DS)
    assert s.groupby('dataset').variant.nunique().eq(5).all()
    assert set(s.loc[s.variant=='R','seed'])=={2026101901}
    assert not m.duplicated(['dataset','method','percentage']).any()
    assert not s.duplicated(['dataset','variant']).any()
    s['score_loss_pct']=100*(s.initial_q-s.q)/s.initial_q
    for _,r in s.iterrows():
        exact=q[(q.dataset==r.dataset)&(q.method=='TPCA')&(q.K==r.K)]
        assert len(exact)==1
        eq=int(exact.q.iloc[0]);assert int(r.q)<=eq
        assert np.isclose(100*(eq-int(r.q))/eq,r.gap_same_K_pct,atol=1e-11)
    modules=pd.read_csv(ROOT/'data/module_ablation_all_110_runs.csv')
    modules=modules[modules['repeat']==1].copy()
    assert len(modules)==40
    for (algo,ds),g in modules.groupby(['paper_algorithm','dataset']):
        complete=g[g.status.isin(['OPTIMAL','INFEASIBLE','FEASIBLE','TARGET_UNREACHED'])]
        assert complete.empty or complete.K.nunique()==complete.q.nunique()==1
    c=pd.read_csv(ROOT/'data/cumulative_checkpoints_r1.csv')
    assert len(c)==60 and (c['repeat']==1).all()
    save_frame('main_selected_r1.csv',m)
    save_frame('strategy_selected_r1.csv',s)
    save_frame('modules_selected_r1.csv',modules)
    save_frame('quality_selected_r1.csv',q)
    (ROOT/'qa/selection_audit.json').write_text(json.dumps(dict(
        passed=True,main_rows=102,strategy_rows=25,module_rows=40,
        uniform_repeat=1,random_seed=2026101901,random_uncertainty_estimated=False,
        independent_time_scope='method construction + Python core call',
        cumulative_scope='first attainment within fixed maximum-target native history',
        source_repeats_retained=True,new_measurements_included=10,
        new_solver_runs_by_plotter=0),indent=2),encoding='utf-8')
    return a,m,n,s,q,modules,c


def finish(fig,name,title,caption,kind='figures'):
    if kind=='figures':f.finish(fig,name,title,caption)
    else:t.finish(fig,name,title,caption)
    ASSETS.append(dict(name=name,kind=kind,title=title,caption=caption))


def runtime_table(m):
    name='T04_runtime_all_targets_r1';rows=[];tex=[];blocks=[]
    for ds in DS:
        block=[];methods=['GUROBI-MIP','GUROBI-LP','MCF','TPCAR','BGCR-D'] if ds=='OP' else ['MCF','TPCAR','BGCR-D']
        if tex:tex.append(None)
        for j,method in enumerate(methods):
            vals=[]
            for pct in [0,20,40,60,80,100]:
                h=m[(m.dataset==ds)&(m.method==method)&(m.percentage==pct)]
                assert len(h)==1;r=h.iloc[0]
                if r.status=='SYSTEM_MEMORY_GUARD':value='MG'
                else:
                    value=f'{r.total:.3f}'
                    if r.status=='INFEASIBLE':value+=r'$^{\dagger}$'
                    if r.status=='TARGET_UNREACHED':value+=r'$^{\ddagger}$'
                vals.append(value)
            rows.append([t.label(ds),t.label(method)]+vals);block.append([t.label(method)]+vals)
            tex.append([r'\multirow{'+str(len(methods))+r'}{*}{'+t.label(ds)+'}' if j==0 else '',t.label(method)]+vals)
        blocks.append(block)
    t.save_csv(name,['Dataset','Algorithm']+[str(x)+'%' for x in [0,20,40,60,80,100]],rows)
    group=[r'\multirow{2}{*}{Dataset} & \multirow{2}{*}{Algorithm} & \multicolumn{6}{c}{Time (s) at coverage target $\rho$} \\',r'\cmidrule(l){3-8}']
    note=r'First independent run at each target; time includes method construction and the core call. '
    note+=r'MG: available-memory guard. $\dagger$: infeasible target; TPCA also returns the exact boundary plan. '
    note+=r'$\ddagger$: DGCA terminates below the target. Gurobi was evaluated only on OP. '
    note+=r'Independent SO timings vary with execution conditions; all recorded values are retained. These are not cumulative trajectory times.'
    t.save_tex(name,r'Independent build-plus-solve times ($N=10$, first run).',
        ['','']+[str(x)+r'\%' for x in [0,20,40,60,80,100]],tex,'llrrrrrr',note,group=group)
    fig,ax=t.canvas(4.8);xs=[.012,.125,.392,.510,.628,.746,.864,.992]
    t.rule(ax,.985,.95)
    t.cell(ax,xs[0],.930,'Dataset','left',9.3);t.cell(ax,xs[1],.930,'Algorithm','left',9.3)
    t.cell(ax,.670,.957,r'Time (s) at coverage target $\rho$','center',10);t.rule(ax,.925,.45,.31,1)
    for j,pct in enumerate([0,20,40,60,80,100]):t.cell(ax,xs[j+2],.892,f'{pct}%',size=9.3)
    t.rule(ax,.862);y=.818;step=.038
    for ds,block in zip(DS,blocks):
        t.cell(ax,xs[0],y-step*(len(block)-1)/2,t.label(ds),'left',9.5)
        for row in block:
            t.cell(ax,xs[1],y,row[0],'left',9.3)
            for j,value in enumerate(row[1:]):t.cell(ax,xs[j+2],y,value,size=9.15)
            y-=step
        y-=.026
    t.rule(ax,.015,.95)
    finish(fig,name,'Table 4. Independent runtime comparison - first run',
        'All methods use repeat 1 at each independently started target (N = 10). Time is method construction + core call. '
        'MG: memory guard. Dagger: infeasibility diagnosis; TPCA also returns a boundary plan. Double dagger: DGCA stops below the target. '
        'Gurobi is available only on OP. The SO timing reversal is retained; trajectory checkpoints are reported separately.',kind='tables')


def strategy_table(s,name='T05_strategy_full',table_number=5):
    rows=[];tex=[]
    labels={'R':'R: Random scan','A':'A: Fixed / slot scan','B':'B: Fixed / donor agg.',
            'C':'C: Dynamic / item heap','D':'D: DGCA'}
    for di,ds in enumerate(DS):
        if di:tex.append(None)
        for j,v in enumerate(VAR):
            r=s[(s.dataset==ds)&(s.variant==v)].iloc[0]
            vals=[labels[v],f'{int(r.K):,}',f'{r.score_loss_pct:.4f}',f'{r.core:.3f}',f'{r.total:.3f}']
            rows.append([t.label(ds)]+vals)
            tex.append([r'\multirow{5}{*}{'+t.label(ds)+'}' if j==0 else '',labels[v]]+vals[1:])
    t.save_csv(name,['Dataset','Variant','K','Loss (%)','Core (s)','Total (s)'],rows)
    group=[r'\multirow{2}{*}{Dataset} & \multirow{2}{*}{Variant} & \multicolumn{2}{c}{Terminal solution} & \multicolumn{2}{c}{Time (s)} \\',r'\cmidrule(lr){3-4}\cmidrule(l){5-6}']
    note=r'$N=10$, requested target $D=|I|$. All entries use the first run; R uses seed 2026101901. '
    note+=r'Loss is relative to the initial Top-$N$ integer-score sum. Core includes native initialization; Total also includes method construction. '
    note+=r'R/A/B scan items once; A/B use identical initial-loss order. C/D choose the current globally minimum-loss feasible direct replacement. '
    note+=r'A/B and C/D have identical exchange sequences. R/A/B/C require a reverse index; D uses forward input. '
    note+=r'Yelp, VG, TG, and SO terminate below the requested target. No uncertainty estimate is inferred from a single run.'
    t.save_tex(name,'Complete direct-replacement strategy comparison (first run).',
        ['','',r'$K$',r'Loss (\%)','Core','Total'],tex,'llrrrr',note,group=group)
    fig,ax=t.canvas(6.40);xs=[.012,.145,.650,.785,.895,.995]
    t.rule(ax,.985,.95)
    t.cell(ax,xs[0],.927,'Dataset','left',9.5);t.cell(ax,xs[1],.927,'Variant','left',9.5)
    t.cell(ax,.675,.955,'Terminal solution','center',10);t.rule(ax,.925,.45,.535,.800)
    t.cell(ax,.933,.955,'Time (s)','center',10);t.rule(ax,.925,.45,.835,1.)
    for x,h in zip(xs[2:],['$K$','Loss (%)','Core','Total']):t.cell(ax,x,.886,h,size=9.4)
    t.rule(ax,.850)
    for di,ds in enumerate(DS):
        y=.811-di*.159
        t.cell(ax,xs[0],y-.060,t.label(ds),'left',9.7)
        for j,row in enumerate(rows[di*5:di*5+5]):
            t.cell(ax,xs[1],y-j*.030,row[1],'left',9.3,bold=j==4)
            for x,value in zip(xs[2:],row[2:]):t.cell(ax,x,y-j*.030,value,size=9.4)
    t.rule(ax,.014,.95)
    finish(fig,name,f'Table {table_number}. Selection-strategy controls on five datasets - first run',
        'All five controls are evaluated on OP, Yelp, VG, TG, and SO; N = 10, requested D = |I|. All variants use repeat 1; random scan uses seed 2026101901. '
        'Loss is relative to the initial Top-N score. Core includes native initialization; Total adds method construction. '
        'A/B share the fixed initial-loss order; C/D implement the same dynamic minimum-loss rule. '
        'Yelp, VG, TG, and SO stop below D. The extension uses the same candidate and initialization interface and unchanged native binary. '
        'These single-run observations carry no error bars.',kind='tables')


def module_table(data,name='TS1_module_ablation_detailed_r1',table_number='S1'):
    rows=[];tex=[];blocks=[]
    maps={'TPCA':['state_source','state_terminal','state_source_only','state_source_noheap'],
          'DGCA':['dynamic_user_heap','dynamic_user_heap_nosource','dynamic_user_heap_noterminal','dynamic_user_scan']}
    for di,ds in enumerate(DS):
        if di:tex.append(None)
        block=[]
        for j,v in enumerate(['Full','-Src','-Term','-Heap']):
            vals=[]
            for algo in ['TPCA','DGCA']:
                r=data[(data.paper_algorithm==algo)&(data.dataset==ds)&(data['mode']==maps[algo][j])].iloc[0]
                if r.status in ['SYSTEM_MEMORY_GUARD','TIME_LIMIT']:
                    vals+=['MG' if r.status=='SYSTEM_MEMORY_GUARD' else 'TL']*3
                else:vals += [f'{r.core:.3f}',f'{r.total:.3f}',f'{r.peak_gib:.3f}']
            rows.append([t.label(ds),v]+vals);block.append([v]+vals)
            tex.append([r'\multirow{4}{*}{'+t.label(ds)+'}' if j==0 else '',v]+vals)
        blocks.append(block)
    t.save_csv(name,['Dataset','Variant','TPCA Core (s)','TPCA Total (s)','TPCA Peak (GiB)',
        'DGCA Core (s)','DGCA Total (s)','DGCA Peak (GiB)'],rows)
    group=[r'\multirow{2}{*}{Dataset} & \multirow{2}{*}{Variant} & \multicolumn{3}{c}{TPCA} & \multicolumn{3}{c}{DGCA} \\',r'\cmidrule(lr){3-5}\cmidrule(l){6-8}']
    note=r'First run of each matched module configuration, $N=10$, $D=|I|$. Total is method construction plus core time; Peak is that run\textquotesingle s peak working set. '
    note+=r'MG: memory guard; TL: 1,800-second limit. No completed metrics are assigned to stopped attempts. '
    note+=r'-Src/-Term remove donor/terminal aggregation. -Heap replaces TPCA boundary heaps by scans or DGCA user-heap selection by user scans. '
    note+=r'All TPCA variants retain local tree repair. Completed variants agree in coverage and integer objective within each algorithm and dataset. '
    note+=r'SO TPCA stopped attempts are not replaced with results from another batch.'
    t.save_tex(name,'Implementation-module comparison on five graphs (first run).',
        ['','','Core (s)','Total (s)','Peak (GiB)','Core (s)','Total (s)','Peak (GiB)'],tex,'llrrrrrr',note,group=group)
    fig,ax=t.canvas(5.30);xs=[.012,.148,.370,.480,.590,.740,.860,.995]
    t.rule(ax,.985,.95)
    t.cell(ax,xs[0],.927,'Dataset','left',9.3);t.cell(ax,xs[1],.927,'Variant','left',9.3)
    for title,left,right in [('TPCA',.285,.610),('DGCA',.655,1.)]:
        t.cell(ax,(left+right)/2,.955,title,'center',10);t.rule(ax,.925,.45,left,right)
    for j,h in enumerate(['Core (s)','Total (s)','Peak (GiB)']*2,2):t.cell(ax,xs[j],.886,h,size=8.8)
    t.rule(ax,.855)
    for i,(ds,block) in enumerate(zip(DS,blocks)):
        y=.820-i*.167;t.cell(ax,xs[0],y-.054,t.label(ds),'left',9.5)
        for j,row in enumerate(block):
            t.cell(ax,xs[1],y-j*.036,row[0],'left',9.1,bold=j==0)
            for c,v in enumerate(row[1:],2):t.cell(ax,xs[c],y-j*.036,v,size=9.1)
    t.rule(ax,.012,.95)
    finish(fig,name,f'Table {table_number}. Five-dataset module ablation - first run',
        'Each configuration uses repeat 1 from its matched ablation batch. Core, construction-plus-core and peak working set '
        'are from that same run. MG/TL are stopped attempts, not completed times. All TPCA variants retain local repair; '
        'DGCA variants retain dynamic minimum-loss selection. Completed variants preserve each algorithm\'s output quality. '
        'SO TPCA ablation remains resource-limited; no successful main-comparison run is substituted.',kind='tables')


def module_core_table(data):
    """Compact five-row main-text ablation table; full metrics remain in Table S1."""
    name='T06_module_ablation_r1';rows=[];tex=[]
    maps={'TPCA':['state_source','state_terminal','state_source_only','state_source_noheap'],
          'DGCA':['dynamic_user_heap','dynamic_user_heap_nosource',
                  'dynamic_user_heap_noterminal','dynamic_user_scan']}
    for ds in DS:
        vals=[]
        for algo in ['TPCA','DGCA']:
            for mode in maps[algo]:
                r=data[(data.paper_algorithm==algo)&(data.dataset==ds)&
                       (data['mode']==mode)].iloc[0]
                if r.status=='SYSTEM_MEMORY_GUARD':value='MG'
                elif r.status=='TIME_LIMIT':value='TL'
                else:value=f'{r.core:.3f}'
                vals.append(value)
        rows.append([t.label(ds)]+vals);tex.append([t.label(ds)]+vals)
    headers=['Dataset']+[f'TPCA {v}' for v in ['Full','-Src','-Term','-Heap']]+[
        f'DGCA {v}' for v in ['Full','-Src','-Term','-Heap']]
    t.save_csv(name,headers,rows)
    group=[r'\multirow{2}{*}{Dataset} & \multicolumn{4}{c}{TPCA core time (s)} & \multicolumn{4}{c}{DGCA core time (s)} \\',
           r'\cmidrule(lr){2-5}\cmidrule(l){6-9}']
    note=(r'First run of each matched configuration, $N=10$, $D=|I|$. '
          r'-Src/-Term remove donor/terminal aggregation; -Heap replaces heap selection by scans. '
          r'MG: memory guard; TL: 1,800-second limit. All TPCA variants retain local tree repair. '
          r'Completed variants preserve terminal coverage and integer objective within each algorithm and dataset. '
          r'Construction-plus-core time and peak working set are reported in Table S1.')
    t.save_tex(name,'Core-time module ablation on five datasets (first run).',
        ['']+['Full','-Src','-Term','-Heap']*2,tex,'lrrrrrrrr',note,group=group)
    fig,ax=t.canvas(2.35);xs=[.015,.240,.335,.430,.525,.655,.750,.845,.995]
    t.rule(ax,.985,.95)
    t.cell(ax,xs[0],.865,'Dataset','left',9.5)
    t.cell(ax,.383,.945,'TPCA core time (s)','center',10)
    t.cell(ax,.822,.945,'DGCA core time (s)','center',10)
    t.rule(ax,.900,.45,.185,.550);t.rule(ax,.900,.45,.610,1.)
    for i,h in enumerate(['Full','-Src','-Term','-Heap']):
        t.cell(ax,xs[i+1],.825,h,size=9.2,bold=(i==0))
        t.cell(ax,xs[i+5],.825,h,size=9.2,bold=(i==0))
    t.rule(ax,.785)
    for i,row in enumerate(rows):
        y=.700-i*.125;t.cell(ax,xs[0],y,row[0],'left',9.6)
        for j,value in enumerate(row[1:]):t.cell(ax,xs[j+1],y,value,size=9.4)
    t.rule(ax,.055,.95)
    finish(fig,name,'Table 6. Core-time module ablation on five datasets - first run',
        'Matched first-run configurations at N = 10 and D = |I|. Full is compared with removal of donor aggregation (-Src), '
        'terminal aggregation (-Term), and heap selection (-Heap). MG denotes the memory guard and TL the 1,800-second limit. '
        'Completed variants preserve terminal coverage and integer objective within each algorithm and dataset. '
        'All TPCA variants retain local tree repair; complete total-time and peak-memory results are retained in the supplement.',kind='tables')


def module_ablation_figure(data):
    """Matched first-run module effects for TPCA and DGCA.

    Ratios use core time within the same algorithm/dataset batch.  This avoids
    allowing the large differences between graph sizes to dominate the plot.
    No categorical points are connected, because the three removals are
    separate counterfactual configurations rather than a progression.
    """
    name='F05_balanced_module_ablation_r1'
    maps={
        'TPCA':{'Full':'state_source','-Src':'state_terminal',
                '-Term':'state_source_only','-Heap':'state_source_noheap'},
        'DGCA':{'Full':'dynamic_user_heap','-Src':'dynamic_user_heap_nosource',
                '-Term':'dynamic_user_heap_noterminal','-Heap':'dynamic_user_scan'},
    }
    variants=['-Src','-Term','-Heap']
    colors={'OP':'#0072B2','YELP':'#D55E00','VG':'#009E73',
            'TG':'#CC79A7','SO':'#555555'}
    markers={'OP':'o','YELP':'s','VG':'^','TG':'D','SO':'P'}
    offsets=dict(zip(DS,np.linspace(-.18,.18,len(DS))))
    fig,axs=plt.subplots(1,2,figsize=(f.WIDTH,2.85),sharey=True)
    fig.subplots_adjust(left=.105,right=.985,bottom=.245,top=.81,wspace=.20)
    rows=[]
    for ai,algo in enumerate(['TPCA','DGCA']):
        ax=axs[ai]
        ax.set_title(f'({chr(97+ai)}) {algo}',loc='left',pad=7)
        for ds in DS:
            full=data[(data.paper_algorithm==algo)&(data.dataset==ds)&
                      (data['mode']==maps[algo]['Full'])].iloc[0]
            full_ok=pd.notna(full.core) and full.status not in ['SYSTEM_MEMORY_GUARD','TIME_LIMIT']
            for xi,v in enumerate(variants):
                r=data[(data.paper_algorithm==algo)&(data.dataset==ds)&
                       (data['mode']==maps[algo][v])].iloc[0]
                if full_ok and pd.notna(r.core) and r.status not in ['SYSTEM_MEMORY_GUARD','TIME_LIMIT']:
                    ratio=float(r.core)/float(full.core)
                    ax.scatter(xi+offsets[ds],ratio,s=30,marker=markers[ds],
                               facecolor='white',edgecolor=colors[ds],linewidth=.95,zorder=3)
                    display=f'{ratio:.4f}'
                else:
                    ratio=np.nan
                    display='MG' if (full.status=='SYSTEM_MEMORY_GUARD' or r.status=='SYSTEM_MEMORY_GUARD') else 'TL'
                    if algo=='TPCA' and ds=='VG' and v=='-Heap':
                        ax.scatter(xi+offsets[ds],190,s=34,marker='v',
                                   facecolor='white',edgecolor=colors[ds],linewidth=.95,zorder=3)
                        ax.annotate('TL',(xi+offsets[ds],190),xytext=(0,5),
                                    textcoords='offset points',ha='center',va='bottom',
                                    fontsize=7,color=colors[ds])
                rows.append(dict(algorithm=algo,dataset=ds,variant=v,
                                 core_ratio_to_full=ratio,status=r.status,
                                 full_status=full.status,display=display,repeat=1))
        ax.axhline(1,color='#888888',lw=.75,ls='--',zorder=1)
        ax.set_yscale('log');ax.set_xlim(-.48,2.48);ax.set_ylim(.7,260)
        ax.set_xticks(range(3),variants);ax.set_xlabel('Removed module')
        ax.yaxis.set_major_locator(LogLocator(base=10,numticks=4))
        ax.yaxis.set_minor_locator(NullLocator());f.axis_style(ax)
    axs[0].set_ylabel('Core-time ratio to Full')
    axs[0].text(.02,.96,'SO Full: MG',transform=axs[0].transAxes,
                ha='left',va='top',fontsize=7.4,color=colors['SO'])
    handles=[Line2D([],[],color=colors[d],marker=markers[d],ls='None',
                    markerfacecolor='white',label=t.label(d)) for d in DS]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.53,.985),
               ncol=5,frameon=False,handletextpad=.4,columnspacing=1.35)
    save_frame(name+'_points.csv',pd.DataFrame(rows))
    finish(fig,name,'Figure 5. Matched module ablation for TPCA and DGCA - first run',
        'Core-time ratio of each removal to Full within the same algorithm and dataset; values above 1 indicate that removing the module is slower. '
        'Each point is repeat 1 from the matched ablation batch. The categories are separate removals and are not connected. '
        'TPCA on SO was stopped by the memory guard in every configuration; TPCA -Heap on VG reached the 1,800-second limit. '
        'Completed variants preserve each algorithm\'s terminal coverage and integer objective within a dataset. '
        'All TPCA variants retain local tree repair, so this figure does not isolate that mechanism.')


def cumulative_table(c):
    name='T07_cumulative_native_r1';rows=[];tex=[]
    terminal={}
    for ds in DS:
        if tex:tex.append(None)
        for j,algo in enumerate(['TPCA','DGCA']):
            g=c[(c.dataset==ds)&(c.method==algo)].set_index('coverage_target_pct')
            vals=[]
            for pct in [20,40,60,80,100]:
                r=g.loc[pct]
                value='--' if pd.isna(r.cumulative_native_s) else f'{r.cumulative_native_s:.3f}'
                if r.state=='initial_solution_already_sufficient':value+=r'$^{\dagger}$'
                vals.append(value)
            trace=read_native(ds,algo);K=trace[-1]['K'];terminal[(ds,algo)]=K
            vals.append(f'{K:,}')
            rows.append([t.label(ds),algo]+vals)
            tex.append([r'\multirow{2}{*}{'+t.label(ds)+'}' if j==0 else '',algo]+vals)
    t.save_csv(name,['Dataset','Algorithm','20%','40%','60%','80%','100%','Terminal K'],rows)
    group=[r'\multirow{2}{*}{Dataset} & \multirow{2}{*}{Algorithm} & \multicolumn{5}{c}{Cumulative native time (s)} & \multirow{2}{*}{Terminal $K$} \\',r'\cmidrule(lr){3-7}']
    note=r'First attainment along the 100\%-target run, repeat 1. Includes native initialization, excludes external construction/audit/export. '
    note+=r'$\dagger$: initial Top-$N$ coverage already satisfies the target; the entry is its long-run initialization checkpoint. '
    note+=r'--: target not attained. Final termination time may include work after the last successful augmentation. '
    note+=r'These values are not independent-target total times and are not compared with MCF/Gurobi totals.'
    t.save_tex(name,'Cumulative first-attainment times along full-target runs.',
        ['','']+[str(x)+r'\%' for x in [20,40,60,80,100]]+[''],tex,'llrrrrrr',note,group=group)
    fig,ax=t.canvas(3.65);xs=[.012,.140,.360,.475,.590,.705,.820,.995]
    t.rule(ax,.985,.95)
    t.cell(ax,xs[0],.928,'Dataset','left',9.4);t.cell(ax,xs[1],.928,'Algorithm','left',9.4)
    t.cell(ax,.570,.955,'Cumulative native time (s)','center',10);t.rule(ax,.925,.45,.275,.845)
    t.cell(ax,xs[-1],.928,'Terminal $K$',size=9.4)
    for j,pct in enumerate([20,40,60,80,100]):t.cell(ax,xs[j+2],.886,f'{pct}%',size=9.3)
    t.rule(ax,.850)
    for di,ds in enumerate(DS):
        y=.785-di*.161;t.cell(ax,xs[0],y-.029,t.label(ds),'left',9.5)
        for j,row in enumerate(rows[di*2:di*2+2]):
            t.cell(ax,xs[1],y-j*.058,row[1],'left',9.3)
            for x,value in zip(xs[2:],row[2:]):t.cell(ax,x,y-j*.058,value,size=9.3)
    t.rule(ax,.016,.95)
    finish(fig,name,'Table 7. Cumulative native time - fixed first run',
        'First attainment within each algorithm\'s 100%-target run. Includes native initialization, excludes method construction '
        'and external checks/export. Dagger: initial plan already satisfies the target. --: target not attained. '
        'Terminal K is actual coverage. These checkpoints are distinct from the independent-target totals in Table 4.',kind='tables')


def metric_figure(m,memory=False):
    name='F03_peak_memory_r1' if memory else 'F02_independent_runtime_r1'
    fig,axs=f.five_panels();points=[]
    order=['TPCAR','BGCR-D','MCF','GUROBI-LP','GUROBI-MIP']
    for ds,ax in zip(DS,axs):
        for method in order:
            g=m[(m.dataset==ds)&(m.method==method)].sort_values('percentage')
            if g.empty:continue
            col,marker,ls=STYLES[method];metric='peak_gib' if memory else 'total'
            complete=g[g.status.isin(['OPTIMAL','FEASIBLE','TARGET_UNREACHED'])]
            ax.plot(complete.percentage,complete[metric],color=col,marker=marker,ls=ls,
                markerfacecolor='white',markeredgewidth=.8,lw=1.1,ms=3.6)
            diagnosis=g[g.status=='INFEASIBLE']
            if len(diagnosis):ax.scatter(diagnosis.percentage,diagnosis[metric],marker='D',facecolors='white',edgecolors=col,s=22,zorder=4)
            bad=g[g.status=='SYSTEM_MEMORY_GUARD']
            if len(bad) and memory:ax.scatter(bad.percentage,bad.peak_gib,marker='x',color=col,s=24,lw=1)
            for _,r in g.iterrows():points.append(dict(dataset=ds,method=method,pct=r.percentage,
                value=r[metric],status=r.status,repeat=1))
        ax.set_xlim(-5,105);ax.set_xticks([0,20,40,60,80,100])
        ax.set_xlabel(r'Coverage target $\rho$ (%)',labelpad=3)
        if memory:
            ax.set_ylim(bottom=0);ax.yaxis.set_major_locator(MaxNLocator(4,min_n_ticks=3))
            ax.set_ylabel('Peak working set (GiB)',labelpad=4)
        else:
            ax.set_yscale('log');ax.yaxis.set_major_locator(LogLocator(base=10,numticks=4));ax.yaxis.set_minor_locator(NullLocator())
            ax.set_ylabel('Build + solve (s)',labelpad=4)
            if ds=='SO':ax.text(.96,.05,'MCF: memory guard',transform=ax.transAxes,ha='right',va='bottom',fontsize=7.2,color=STYLES['MCF'][0])
    handles=[Line2D([],[],color=STYLES[x][0],marker=STYLES[x][1],ls=STYLES[x][2],markerfacecolor='white',label=t.label(x)) for x in order]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.52,1),ncol=5,
        frameon=False,fontsize=8,handlelength=1.8,columnspacing=1.2)
    save_frame(name+'_points.csv',pd.DataFrame(points))
    if memory:caption='Peak working set in the first independent run at each target. Crosses mark memory-guard stops, not completed-solve requirements. Diamonds mark Yelp infeasibility diagnoses. Resident mapped pages are included. Gurobi is evaluated only on OP.'
    else:caption='First independent run at every target, with uniform method construction + core-call timing. Panel-specific log y axes. Diamonds mark Yelp infeasibility diagnoses; TPCA additionally returns the boundary plan. DGCA may stop below the request. Gurobi is evaluated only on OP. Observed nonmonotone SO timing is retained.'
    finish(fig,name,'Figure 3. Peak working set - first run' if memory else 'Figure 2. Independent runtime - first run',caption)


def read_native(ds,algo):
    path=ROOT/'data/raw_native'/f'{ds}_{algo}_r1_trajectory.csv'
    with path.open(encoding='utf-8-sig',newline='') as stream:
        rows=list(csv.DictReader(stream))
    result=[]
    for r in rows:
        if r.get('trajectory_kind')=='final_audited_endpoint':continue
        result.append(dict(K=int(r.get('coverage') or r['diversity']),time=float(r['time_sec'])))
    assert all(a['time']<=b['time'] and a['K']+1==b['K'] for a,b in zip(result,result[1:]))
    return result


def cumulative_figure(m):
    fig,axs=f.five_panels();points=[]
    for ds,ax in zip(DS,axs):
        C=int(m[(m.dataset==ds)&(m.percentage==100)].D.iloc[0])
        for algo,method in [('TPCA','TPCAR'),('DGCA','BGCR-D')]:
            trace=read_native(ds,algo);x=np.array([r['K']/C*100 for r in trace]);y=np.array([r['time'] for r in trace])
            col,marker,ls=STYLES[method]
            ax.plot(x,y,color=col,ls=ls,lw=1.2)
            ax.scatter([x[-1]],[y[-1]],color=col,marker=marker,s=22,zorder=4)
            for pct in [20,40,60,80,100]:
                D=int(np.ceil(C*pct/100));r=next((r for r in trace if r['K']>=D),None)
                if r and D>trace[0]['K']:ax.plot(r['K']/C*100,r['time'],marker=marker,ms=3.2,color=col,markerfacecolor='white')
            points += [dict(dataset=ds,algorithm=algo,K=r['K'],attained_coverage_pct=r['K']/C*100,cumulative_native_s=r['time'],repeat=1) for r in trace]
        ax.set_xlim(0,103);ax.set_xticks([0,20,40,60,80,100]);ax.set_yscale('log')
        ax.yaxis.set_major_locator(LogLocator(base=10,numticks=4));ax.yaxis.set_minor_locator(NullLocator())
        ax.set_xlabel('Attained coverage (%)',labelpad=3);ax.set_ylabel('Cumulative native time (s)',labelpad=4)
    handles=[Line2D([],[],color=STYLES[m][0],marker=STYLES[m][1],ls=STYLES[m][2],markerfacecolor='white',label=t.label(m)) for m in ['TPCAR','BGCR-D']]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.52,.995),ncol=2,frameon=False,handlelength=2.5,columnspacing=3)
    save_frame('F06_cumulative_native_r1_points.csv',pd.DataFrame(points))
    finish(fig,'F06_cumulative_native_r1','Figure 6. Cumulative native time - first full-target run',
        'Each curve follows a single 100%-target run, repeat 1, and ends at its actual terminal coverage. '
        'Attained coverage is 100 K / |I|. Includes native initialization, excludes method construction and external checks/export. '
        'Filled endpoints indicate the last attained coverage; subsequent termination work is not included. '
        'Panel-specific log y axes. This figure is separate from independent-target runtime comparisons.')


def strategy_figure(s):
    fig,axs=plt.subplots(2,3,figsize=(f.WIDTH,4.25),sharex=True)
    fig.subplots_adjust(left=.095,right=.985,bottom=.13,top=.90,hspace=.37,wspace=.38)
    for di,ds in enumerate(['OP','YELP','SO']):
        g=s[s.dataset==ds].set_index('variant').loc[VAR]
        axs[0,di].set_title(f'({chr(97+di)}) {t.label(ds)}',loc='left',pad=7)
        for j,v in enumerate(VAR):
            col,marker=VARSTYLE[v]
            axs[0,di].scatter(j,g.loc[v,'gap_same_K_pct'],color=col,marker=marker,s=28,zorder=3)
            axs[1,di].scatter(j,g.loc[v,'core'],color=col,marker=marker,s=28,zorder=3)
        for ax in axs[:,di]:
            ax.set_xlim(-.5,4.5);ax.set_xticks(range(5),VAR);f.axis_style(ax)
        axs[0,di].set_ylim(0,max(g.gap_same_K_pct)*1.22)
        axs[0,di].yaxis.set_major_locator(MaxNLocator(4,min_n_ticks=3))
        axs[0,di].ticklabel_format(axis='y',style='plain',useOffset=False)
        axs[1,di].set_yscale('log');axs[1,di].yaxis.set_major_locator(LogLocator(base=10,numticks=4));axs[1,di].yaxis.set_minor_locator(NullLocator())
        axs[1,di].set_xlabel('Variant')
    axs[0,0].set_ylabel('Gap to exact at same $K$ (%)',labelpad=5)
    axs[1,0].set_ylabel('Core time (s)',labelpad=5)
    save_frame('F07_strategy_r1_points.csv',s[['dataset','variant','seed','K','gap_same_K_pct','core','total']])
    finish(fig,'F07_strategy_r1','Figure 7. Full strategy controls - first run',
        'One point per variant and dataset, repeat 1; random scan uses seed 2026101901. Top: score shortfall from exact TPCA '
        'at each variant\'s own terminal coverage K. Bottom: core time including native initialization, on log axes. '
        'Categories are not connected. R is random one-pass scanning; A/B share initial-loss order; '
        'C/D share current-minimum direct-replacement selection. Final coverage and total time are listed in Table 5.')


def contact_sheet(filename='Preview_Contact_Sheet.png'):
    images=[]
    for a in ASSETS:
        path=ROOT/a['kind']/(a['name']+'.png')
        with Image.open(path) as im:
            tile=Image.new('RGB',(760,590),'white');small=ImageOps.contain(im.convert('RGB'),(720,540))
            tile.paste(small,((760-small.width)//2,35+(540-small.height)//2))
            ImageDraw.Draw(tile).text((15,12),a['name'],fill='black');images.append(tile)
    sheet=Image.new('RGB',(1520,590*((len(images)+1)//2)),(230,230,230))
    for i,im in enumerate(images):sheet.paste(im,((i%2)*760,(i//2)*590))
    sheet.save(ROOT/filename)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--only');ap.add_argument('--lean',action='store_true');args=ap.parse_args()
    args.lean=True  # This package publishes only the agreed concise main-text selection.
    for x in ['tables','figures','qa']:(ROOT/x).mkdir(exist_ok=True)
    f.style();a,m,n,s,q,modules,c=load()
    jobs={
        'T05_strategy_full':lambda:strategy_table(s),
        'T04_runtime_all_targets_r1':lambda:runtime_table(m),
        'F02_independent_runtime_r1':lambda:metric_figure(m),
        'F03_peak_memory_r1':lambda:metric_figure(m,True),
        'T06_module_ablation_r1':lambda:module_core_table(modules),
        'TS1_module_ablation_detailed_r1':lambda:module_table(modules),
        'F04_coverage_quality':lambda:f.quality_figure(q),
        'T02_instances_single_column':lambda:t.instance_table(m),
        'T03_exact_across_N':lambda:t.n_table(n),
    }
    if args.only and args.only not in jobs:ap.error('Unknown asset; choose '+', '.join(jobs))
    lean=['T02_instances_single_column','T03_exact_across_N','F04_coverage_quality',
          'T04_runtime_all_targets_r1','F02_independent_runtime_r1','F03_peak_memory_r1',
          'T05_strategy_full','T06_module_ablation_r1']
    output='Lean_Main_Preview.pdf'
    context=PdfPages(ROOT/output,metadata=f.PDFMETA) if not args.only else nullcontext(None)
    with context as book:
        f.BOOK=t.BOOK=book
        order=[args.only] if args.only else lean
        for key in order:
            job=jobs[key]
            if args.only and args.only!=key:continue
            print('Rendering',key,flush=True);job()
        f.BOOK=t.BOOK=None
    # Quality/N/instance functions append their metadata to the shared helpers.
    for kind,source in [('figures',f.ASSETS),('tables',t.ASSETS)]:
        for item in source:
            if not any(x['name']==item['name'] for x in ASSETS):ASSETS.append(dict(item,kind=kind))
    suffix='_'+args.only if args.only else ('_lean' if args.lean else '')
    (ROOT/'qa'/('layout_checks'+suffix+'.json')).write_text(json.dumps(dict(figures=f.QA,tables=t.QA),indent=2),encoding='utf-8')
    (ROOT/'qa'/('captions'+suffix+'.json')).write_text(json.dumps(ASSETS,indent=2),encoding='utf-8')
    if not args.only:
        contact_sheet('Lean_Main_Contact_Sheet.png' if args.lean else 'Preview_Contact_Sheet.png')
    print('Completed',len(ASSETS),'assets; no solvers run.')


if __name__=='__main__':main()
