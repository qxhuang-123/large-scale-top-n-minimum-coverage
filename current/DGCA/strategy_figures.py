"""Portable R/A/B/C/D plots from a saved comparison.csv; no solvers run."""
from pathlib import Path
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from PIL import Image
from strategy_figure_qa import audit_layout

HERE=Path(__file__).resolve().parent
PACKAGE=HERE.parent if HERE.name=='code' else HERE/'outputs/BGCR_strategy_comparison_20261019/paper_report'
VARIANTS=['R','A','B','C','D'];COLORS=['#777777','#8064A2','#D96800','#159D80','#3B75B9'];MARKERS=['s','D','s','^','o']
NAMES={21:'F21_strategy_total_runtime',22:'F22_strategy_core_runtime',23:'F23_strategy_coverage_deficit',24:'F24_strategy_score_gap'}


def style():
    plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif'],
        'mathtext.fontset':'stix','font.size':9,'axes.labelsize':9,'axes.titlesize':10,
        'xtick.labelsize':8,'ytick.labelsize':8,'axes.spines.top':False,'axes.spines.right':False,
        'axes.grid':True,'grid.color':'#d8d8d8','grid.linewidth':.65,'axes.linewidth':.7,
        'pdf.fonttype':42,'svg.fonttype':'none'})


def draw(number,root=PACKAGE):
    root=Path(root);df=pd.read_csv(root/'data/comparison.csv');style()
    specs={21:('total','Build + solve (s, log scale)',True),
           22:('core','Core time (s, log scale)',True),
           23:('deficit',r'Coverage deficit $K_{\max}-K$',False),
           24:('gap_same_K_pct','Score gap at the same K (%)',False)}
    metric,ylabel,log=specs[number]
    fig,axes=plt.subplots(1,3,figsize=(173.8/25.4,2.95))
    fig.subplots_adjust(left=.10,right=.985,bottom=.25,top=.80,wspace=.48)
    for i,(ds,ax) in enumerate(zip(['OP','YELP','SO'],axes)):
        upper=0
        for k,variant in enumerate(VARIANTS):
            g=df[(df.dataset==ds)&(df.variant==variant)]
            good=g[g.status.isin(['FEASIBLE','TARGET_UNREACHED'])]
            values=(good.Kmax-good.K) if metric=='deficit' else good[metric]
            values=values.dropna().to_numpy(dtype=float)
            if len(values):
                median=np.median(values);upper=max(upper,float(values.max()))
                ax.errorbar(k,median,yerr=[[median-values.min()],[values.max()-median]],
                    marker=MARKERS[k],color=COLORS[k],mfc='white',ls='none',capsize=2,ms=4)
                ax.scatter(k+np.linspace(-.10,.10,len(values)),values,color=COLORS[k],s=9,alpha=.5)
            elif len(g):
                ax.text(k,.05,'Limited',rotation=90,ha='center',va='bottom',fontsize=7,transform=ax.get_xaxis_transform())
        ax.set_title('(%s) %s'%(chr(97+i),'Yelp' if ds=='YELP' else ds),loc='left')
        ax.set_xticks(range(5));ax.set_xticklabels(VARIANTS);ax.set_xlim(-.45,4.45)
        ax.set_xlabel('Variant');ax.set_ylabel(ylabel);ax.xaxis.grid(False)
        if log:ax.set_yscale('log')
        elif metric=='deficit':
            ax.set_ylim(-max(.1,upper*.035),max(1,upper*1.15));ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            if upper==0:ax.text(.5,.65,r'All runs reach $K_{\max}$',transform=ax.transAxes,ha='center',fontsize=8,color='#555555')
        else:ax.set_ylim(0,max(1e-8,upper*1.15))
    fig.text(.5,.97,'R: random scan; A / B: fixed order; C / D: dynamic minimum loss',ha='center',va='top',fontsize=8)
    attainment=[]
    for ds in ['OP','YELP','SO']:
        g=df[df.dataset==ds]
        attainment.append(('%s: %d/%d'%('Yelp' if ds=='YELP' else ds,int((g.K>=g.D).sum()),len(g))))
    fig.text(.5,.025,'Target reached (all runs): '+'; '.join(attainment),ha='center',va='bottom',fontsize=7)
    folder=root/'figures';qa=root/'qa';folder.mkdir(exist_ok=True);qa.mkdir(exist_ok=True)
    name=NAMES[number];fig.canvas.draw()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always');warnings.filterwarnings('ignore',category=DeprecationWarning)
        issues=audit_layout(fig)
        fig.savefig(folder/(name+'.png'),dpi=600);fig.savefig(qa/(name+'_preview.png'),dpi=150)
    if any(str(a)=='FAIL' for a,b in issues):raise RuntimeError(issues)
    for ext in ['pdf','svg']:fig.savefig(folder/(name+'.'+ext))
    with Image.open(folder/(name+'.png')) as im:im.convert('L').save(folder/(name+'_gray.png'),dpi=(600,600))
    (qa/(name+'.json')).write_text(json.dumps(dict(layout_issues=[(str(a),str(b)) for a,b in issues],
        warnings=[str(w.message) for w in caught],dpi=600,
        interpretation='Points are individual runs; symbol is median; whisker is min--max. R uses 10 seeds; deterministic variants use 3 process repeats. Mean and sample SD are in the tables.'),indent=2),encoding='utf8')
    plt.close(fig);print(name,flush=True)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=PACKAGE);p.add_argument('--figure',type=int,choices=[21,22,23,24]);a=p.parse_args()
    for number in [a.figure] if a.figure else [21,22,23,24]:draw(number,a.root)
