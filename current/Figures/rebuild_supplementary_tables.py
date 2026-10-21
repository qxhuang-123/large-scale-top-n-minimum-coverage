"""Rebuild grouped Tables S2/S3 from frozen records; never run a solver."""
from pathlib import Path
import argparse,csv,json,os,shutil,subprocess,hashlib,re

ROOT=Path(__file__).resolve().parent
DS=['OP','YELP','VG','TG','SO']
METHODS=['GUROBI-MIP','GUROBI-LP','MCF','TPCA','DGCA']
NAMES={'GUROBI-MIP':'Gurobi','GUROBI-LP':'Gurobi-LP','MCF':'MCF','TPCA':'TPCA','DGCA':'DGCA'}

def load(path):return list(csv.DictReader(path.open(encoding='utf-8-sig',newline='')))
def write(path,s):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(s,encoding='utf-8')
def outcsv(path,rows):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def s2():
    raw=load(ROOT/'data/progressive_32_runs.csv')
    assert len(raw)==32 and {r['repeat'] for r in raw}=={'1'}
    index={(r['dataset'],int(float(r['gap_percentage'])),r['version']):r for r in raw}
    assert len(index)==32
    body=[];wide=[]
    for ds in ['OP','YELP']:
        for n,p in enumerate([1,10,50,100]):
            group=[index[(ds,p,v)] for v in ['V1','V2','V3','V4']]
            assert len({int(float(r['D'])) for r in group})==1
            assert all(int(float(r['D']))==int(float(r['K'])) for r in group)
            assert len({int(float(r['q'])) for r in group})==1
            target=int(float(group[0]['D']))
            record=dict(dataset=ds,gap_percentage=p,D=target,K=target)
            for mi,metric in enumerate(['core','total','peak_gib']):
                metriclabel={'core':'Core (s)','total':'Total (s)','peak_gib':'Peak (GiB)'}[metric]
                cells=[r'\multirow{12}{*}{'+('Yelp' if ds=='YELP' else ds)+'}' if n==0 and mi==0 else '',
                    r'\multirow{3}{*}{'+str(p)+'}' if mi==0 else '',
                    r'\multirow{3}{*}{'+f'{target:,}'+'}' if mi==0 else '',metriclabel]
                for r in group:
                    cells.append(f'{float(r[metric]):.3f}')
                    record[metric+'_'+r['version']]=r[metric]
                body.append(' & '.join(cells)+r' \\')
            if n<3:body.append(r'\addlinespace[1pt]')
            wide.append(record)
        if ds=='OP':body.append(r'\specialrule{0.35pt}{1.8pt}{1.8pt}')
    note=r'\textit{Note:} $N=10$; $\eta$ is the fraction of the available coverage gap. All runs attain $D$, so $D=K$ is reported once. V1 retains local repair; V2 adds the boundary heap, V3 donor aggregation, and V4 terminal aggregation. Total includes construction. One run per setting; exact objectives agree across versions.'
    tex=r'''\begin{table*}[t]
\centering
\caption{Progressive TPCA comparison.}\label{tab:progressive-comparison}
\fontsize{8}{9}\selectfont
\renewcommand{\arraystretch}{.98}
\setlength{\tabcolsep}{4pt}
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lrrlrrrr@{}}
\toprule
\multirow{2}{*}{Dataset} & \multirow{2}{*}{$\eta$ (\%)} & \multirow{2}{*}{$D=K$} & \multirow{2}{*}{Metric} & \multicolumn{4}{c}{Version}\\
\cmidrule(l){5-8}
 & & & & V1 & V2 & V3 & V4\\
\midrule
'''+ '\n'.join(body)+r'''
\bottomrule
\end{tabular*}
\par\smallskip
\begin{minipage}{.98\textwidth}\footnotesize
'''+note+r'''
\end{minipage}
\end{table*}
'''
    outcsv(ROOT/'tables/TS2_progressive_grouped_values.csv',wide)
    return tex,dict(source_rows=32,display_rows=24,numeric_metrics_preserved=96,all_D_equal_K=True)

def number(row,run,pct):
    key=f'r{run}_total_s' if run else 'median_total_s'
    value=row[key]
    try:return f'{float(value):.3f}'
    except (TypeError,ValueError):
        status=row.get(f'r{run}_status','') if run else ''
        if 'MEMORY' in status:return 'MG'
        if status=='TIME_LIMIT':return 'TL'
        return '--'

def s3():
    raw=load(ROOT/'tables/TS3_repeated_runtime_records.csv')
    assert len(raw)==102
    index={(r['dataset'],r['method'],int(r['coverage_target_pct'])):r for r in raw}
    assert len(index)==102
    body=[];wide=[]
    for ds in DS:
        methods=METHODS if ds=='OP' else METHODS[2:]
        for mi,method in enumerate(methods):
            for ri,run in enumerate([1,2,3,0]):
                cells=[r'\multirow{'+str(len(methods)*4)+r'}{*}{'+('Yelp' if ds=='YELP' else ds)+'}' if mi==0 and ri==0 else '',
                    r'\multirow{4}{*}{'+NAMES[method]+'}' if ri==0 else '',str(run) if run else r'\textbf{Median}']
                record=dict(dataset=ds,method=method,run=str(run) if run else 'Median')
                for pct in [0,20,40,60,80,100]:
                    value=number(index[(ds,method,pct)],run,pct)
                    record[f'rho_{pct}']=value
                    cells.append(r'\textbf{'+value+'}' if run==0 else value)
                body.append(' & '.join(cells)+r' \\');wide.append(record)
            if mi!=len(methods)-1:body.append(r'\addlinespace[1pt]')
        if ds!='SO':body.append(r'\specialrule{0.35pt}{1.8pt}{1.8pt}')
    tex=r'''\begin{table*}[!t]
\centering
\caption{Repeated total optimization times.}\label{tab:repeat-times}
\fontsize{8}{9}\selectfont
\renewcommand{\arraystretch}{.92}
\setlength{\tabcolsep}{3.0pt}
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lllrrrrrr@{}}
\toprule
\multirow{2}{*}{Dataset} & \multirow{2}{*}{Algorithm} & \multirow{2}{*}{Run} & \multicolumn{6}{c}{Total time (s) at coverage target $\rho$}\\
\cmidrule(l){4-9}
 & & & 0\% & 20\% & 40\% & 60\% & 80\% & 100\%\\
\midrule
'''+ '\n'.join(body)+r'''
\bottomrule
\end{tabular*}
\par\smallskip
\begin{minipage}{.98\textwidth}\footnotesize
\textit{Note:} Total includes construction and solution. Main comparisons use Run 1; bold rows report medians of completed runs. MG denotes a memory-guard stop; -- denotes an unattempted run or an unavailable median. Full statuses and the separate construction/core times are retained in the CSV.
\end{minipage}
\end{table*}
'''
    outcsv(ROOT/'tables/TS3_repeated_grouped_values.csv',wide)
    return tex,dict(source_configurations=102,display_rows=68,recorded_total_time_fields=306,median_fields=102,source_values_unchanged=True)

def standalone(tex,number):
    tex=re.sub(r'\\begin\{table\*\}\[[^]]*\]','',tex).replace(r'\end{table*}','')
    tex=tex.replace(r'\caption{',r'\captionof{table}{')
    return r'''\documentclass[border=6pt]{standalone}
\pdfinfoomitdate=1
\pdftrailerid{}
\pdfsuppressptexinfo=15
\usepackage{stix,booktabs,multirow,caption,graphicx}
\captionsetup{font={small,sf},labelfont=bf,labelsep=newline,justification=raggedright,singlelinecheck=false}
\setlength{\textwidth}{174mm}
\renewcommand{\thetable}{S\arabic{table}}
\begin{document}
\begin{minipage}{174mm}
'''+r'\setcounter{table}{'+str(number-1)+'}'+'\n'+tex+r'''
\end{minipage}
\end{document}
'''

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--only',choices=['S2','S3'])
    p.add_argument('--compile',action='store_true')
    p.add_argument('--sync-paper',type=Path)
    p.add_argument('--dpi',type=int,default=600)
    args=p.parse_args()
    (ROOT/'tables').mkdir(exist_ok=True);(ROOT/'qa').mkdir(exist_ok=True)
    audits={}
    for tag,func,filename,n in [('S2',s2,'TS2_progressive',2),('S3',s3,'TS3_repeated_times',3)]:
        if args.only and args.only!=tag:continue
        tex,audit=func();stem=filename+'_grouped'
        write(ROOT/'tables'/(stem+'.tex'),tex)
        if args.sync_paper:write(args.sync_paper/'tables'/(filename+'.tex'),tex)
        if args.compile:
            work=ROOT/'qa'/(stem+'_build');work.mkdir(exist_ok=True)
            write(work/(stem+'.tex'),standalone(tex,n))
            exe=shutil.which('pdflatex') or 'E:/texlive/2025/bin/windows/pdflatex.exe'
            env=os.environ.copy();env['SOURCE_DATE_EPOCH']='1792588800';env['FORCE_SOURCE_DATE']='1'
            for _ in range(2):
                result=subprocess.run([exe,'-interaction=nonstopmode','-halt-on-error',stem+'.tex'],cwd=work,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
                if result.returncode:raise RuntimeError(result.stdout.decode('utf-8',errors='replace'))
            shutil.copy2(work/(stem+'.pdf'),ROOT/'tables'/(stem+'.pdf'))
            poppler=shutil.which('pdftoppm') or 'C:/Users/24qxh/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/poppler/Library/bin/pdftoppm.exe'
            subprocess.run([poppler,'-r',str(args.dpi),'-singlefile','-png',str(ROOT/'tables'/(stem+'.pdf')),str(ROOT/'tables'/stem)],check=True)
            subprocess.run([poppler,'-r','145','-singlefile','-png',str(ROOT/'tables'/(stem+'.pdf')),str(ROOT/'qa'/(stem+'_preview'))],check=True)
        audits[tag]=audit
    write(ROOT/'qa/supplementary_grouped_tables_audit.json',json.dumps(dict(passed=True,tables=audits,solver_calls=0),indent=2))
    print(json.dumps(audits,indent=2))

if __name__=='__main__':main()
