"""Rebuild compact manuscript tables from frozen results; no solvers are called."""


from pathlib import Path


from datetime import datetime, timezone


import csv


import hashlib


import json


import textwrap


import warnings


import numpy as np


import pandas as pd


import matplotlib


matplotlib.use('Agg')


import matplotlib.pyplot as plt


from matplotlib.backends.backend_pdf import PdfPages


from matplotlib.transforms import Bbox


from PIL import Image, ImageOps, ImageDraw


ROOT = Path(__file__).resolve().parents[1]


DS = ['OP', 'YELP', 'VG', 'TG', 'SO']


RHO = [0, 20, 40, 60, 80, 100]


WIDTH = 173.8/25.4


COLUMN = (173.8-6.35)/2/25.4


META = {'Creator': 'TPCA DGCA saved-data table refinement',
        'CreationDate': datetime(2026, 10, 21, tzinfo=timezone.utc),
        'ModDate': datetime(2026, 10, 21, tzinfo=timezone.utc)}


BOOK = None


QA = {}


ASSETS = []


def label(x):
    return {'YELP': 'Yelp', 'TPCAR': 'TPCA', 'BGCR-D': 'DGCA',
            'GUROBI-MIP': 'Gurobi-MIP', 'GUROBI-LP': 'Gurobi-LP'}.get(x, x)


def rule(ax, y, width=.65, x0=0, x1=1):
    ax.plot([x0,x1], [y,y], color='#222222', lw=width, clip_on=False)


def cell(ax, x, y, text, ha='right', size=9.2, bold=False):
    ax.text(x, y, text, ha=ha, va='center', fontsize=size,
            fontweight='bold' if bold else 'normal')


def canvas(height, width=WIDTH):
    fig = plt.figure(figsize=(width, height))
    ax = fig.add_axes([.035,.06,.93,.88])
    ax.set_xlim(0,1);ax.set_ylim(0,1);ax.axis('off')
    return fig, ax


def save_csv(name, headers, rows):
    with (ROOT/'tables'/(name+'.csv')).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f);w.writerow(headers);w.writerows(rows)


def save_tex(name, caption, headers, rows, align, note, single=False, group=None):
    env='table' if single else 'table*'
    width=r'\linewidth' if single else r'\textwidth'
    lines=[r'\begin{'+env+'}[t]', r'\centering', r'\caption{'+caption+'}',
           r'\label{tab:'+name+'}', r'\small',
           r'\setlength{\tabcolsep}{'+('2.5' if single else '4')+'pt}',
           r'\begin{tabular*}{'+width+r'}{@{\extracolsep{\fill}}'+align+'@{}}',r'\toprule']
    if group:lines+=group
    lines+=[' & '.join(headers)+r' \\',r'\midrule']
    for row in rows:
        if row is None:lines.append(r'\addlinespace[4pt]')
        else:lines.append(' & '.join(map(str,row))+r' \\')
    lines += [r'\bottomrule',r'\end{tabular*}',r'\par\smallskip',
              r'\begin{minipage}{.98'+width+r'}\footnotesize',r'\textit{Note:} '+note,
              r'\end{minipage}',r'\end{'+env+'}']
    (ROOT/'tables'/(name+'.tex')).write_text('\n'.join(lines)+'\n',encoding='utf8')


def finish(fig, name, title, caption, kind='tables'):
    fig.canvas.draw();ren=fig.canvas.get_renderer();w,h=fig.canvas.get_width_height()
    outside=[]
    overlaps=[]
    for ax in fig.axes:
        for t in ax.texts:
            b=t.get_window_extent(ren)
            if b.x0 < -1 or b.x1>w+1 or b.y0 < -1 or b.y1>h+1:outside.append(t.get_text())
        texts=[v for v in ax.texts if v.get_visible() and v.get_text()]
        for i,first in enumerate(texts):
            a=first.get_window_extent(ren)
            for second in texts[i+1:]:
                b=second.get_window_extent(ren)
                if a.overlaps(b):overlaps.append([first.get_text(),second.get_text()])
    assert not outside,(name,outside)
    assert not overlaps,(name,'overlapping table text',overlaps)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        for ext in ['png','pdf','svg']:
            kw={'dpi':600} if ext=='png' else {'metadata':META} if ext=='pdf' else {'metadata':{'Date':'2026-10-21'}}
            fig.savefig(ROOT/kind/(name+'.'+ext),**kw)
    glyph=[str(x.message) for x in caught if 'Glyph' in str(x.message)]
    assert not glyph,glyph
    QA[name]={'outside_canvas':outside,'table_text_overlaps':overlaps,'missing_glyphs':glyph,'size_inches':list(fig.get_size_inches()),'dpi':600}
    ASSETS.append({'name':name,'title':title,'caption':caption})
    with Image.open(ROOT/kind/(name+'.png')) as im:
        im.convert('L').save(ROOT/'qa'/(name+'_gray.png'))
    if BOOK is not None:
        width=fig.get_figwidth();height=fig.get_figheight()
        wrapped=textwrap.fill(caption,width=int(115*width/WIDTH))
        extra=.4+.14*len(wrapped.splitlines())
        title1=fig.text(.04,-.15/height,title,fontsize=9 if width<4 else 10,
                        va='top',fontweight='bold')
        cap=fig.text(.04,-.36/height,wrapped,fontsize=8,va='top',linespacing=1.25)
        BOOK.savefig(fig,bbox_inches=Bbox.from_extents(0,-extra,width,height))
        title1.remove();cap.remove()
    plt.close(fig)


def instance_table(m):
    name='T02_instances_single_column';rows=[];tex=[]
    for ds in DS:
        r=m[m.dataset==ds].iloc[0]
        exponent=int(np.floor(np.log10(r.E)));mantissa=r.E/10**exponent
        values=[label(ds),f'{int(r.U):,}',f'{int(r.I):,}',f'{int(r.C):,}',
                rf'${mantissa:.3f}\times10^{{{exponent}}}$']
        rows.append(values)
    headers=['Dataset','Users','Items','Cand. items','Cand. edges']
    save_csv(name,headers,rows)
    save_tex(name,r'Statistics of the experimental instances at 40\% retention.',headers,rows,'lrrrr',
             r'Items counts the supplied catalogue before retention; Cand. items counts items incident to retained edges. '
             r'The latter is $|I|$ in the optimization model.',single=True)
    fig,ax=canvas(1.60,COLUMN)
    x=[.004,.282,.470,.712,.995]
    rule(ax,.98,.85)
    for j,h in enumerate(headers):cell(ax,x[j],.87,h,'left' if j==0 else 'right',8.1)
    rule(ax,.76,.55)
    for i,row in enumerate(rows):
        for j,value in enumerate(row):cell(ax,x[j],.65-i*.14,value,'left' if j==0 else 'right',8.5)
    rule(ax,.015,.85)
    finish(fig,name,'Table 2. Instance statistics',
           'Statistics at the 40% candidate-retention setting. Items denotes the original catalogue; '
           'Cand. items denotes the retained item union used by the model.')


def n_table(n):
    name='T03_exact_across_N';headers=['Dataset','Metric']+[f'N={v}' for v in [10,15,20,25,30]]
    raw=[];tex=[];blocks=[]
    metrics=[('K0',r'$K^0$',r'$K^0$'),
             ('Kmax',r'$K_{\max}$',r'$K_{\max}$'),
             ('Exact_loss_pct','Loss (%)',r'Loss (\%)')]
    for di,ds in enumerate(DS):
        g=n[n.dataset==ds].set_index('N');block=[]
        if di:tex.append(None)
        for j,(field,display,texdisplay) in enumerate(metrics):
            vals=[f'{g.loc[v,field]:.4f}' if field=='Exact_loss_pct' else f'{int(g.loc[v,field]):,}' for v in [10,15,20,25,30]]
            raw.append([label(ds),display]+vals)
            tex.append([r'\multirow{3}{*}{'+label(ds)+'}' if j==0 else '',texdisplay]+vals)
            block.append([display]+vals)
        blocks.append(block)
    save_csv(name,headers,raw)
    groups=[r'\multirow{2}{*}{Dataset} & \multirow{2}{*}{Metric} & \multicolumn{5}{c}{List length $N$} \\',
            r'\cmidrule(l){3-7}']
    save_tex(name,'Exact coverage and score loss across list lengths.',
             ['','']+[str(v) for v in [10,15,20,25,30]],tex,'llrrrrr',
             r'$K^0$ is initial Top-$N$ coverage. Score loss is measured at $K_{\max}$ relative to Top-$N$. '
             r'DGCA results at the same list lengths are supplied separately.',group=groups)
    fig,ax=canvas(4.28);x=[.012,.185,.415,.560,.705,.850,.995]
    rule(ax,.985,.95)
    cell(ax,x[0],.927,'Dataset','left',9.6)
    cell(ax,x[1],.927,'Metric','left',9.6)
    cell(ax,.687,.956,r'List length $N$','center',10)
    rule(ax,.925,.45,.32,1)
    for j,value in enumerate([10,15,20,25,30]):cell(ax,x[j+2],.890,str(value),size=9.6)
    rule(ax,.855)
    for di,(ds,block) in enumerate(zip(DS,blocks)):
        start=.807-di*.164;cell(ax,x[0],start-.047,label(ds),'left',9.6)
        for j,row in enumerate(block):
            y=start-j*.047;cell(ax,x[1],y,row[0],'left',9.4)
            for k,val in enumerate(row[1:]):cell(ax,x[k+2],y,val,'right',9.2)
    rule(ax,.015,.95)
    finish(fig,name,'Table 3. Exact results across list lengths',
           r'Exact coverage and predicted-score loss across recommendation-list lengths. '
           r'$K^0$ denotes initial Top-N coverage; loss is measured at the exact boundary '
           r'$K_{\max}$ relative to Top-N.')


