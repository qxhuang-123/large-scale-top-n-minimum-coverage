from pathlib import Path
import argparse,json,subprocess,sys,time
ROOT=Path(__file__).resolve().parent
ENGINE=ROOT/'公共代码'
def tpca(dataset):
 for part in ['CommonInput','TPCA']:sys.path.insert(0,str(ENGINE/part))
 from run_comparison import main
 main(dataset,'TPCAR')
def dgca(dataset):
 p=argparse.ArgumentParser();p.add_argument('--n',type=int,default=10);p.add_argument('--repeats',type=int,default=1);p.add_argument('--percentages',type=int,nargs='+',default=[100]);a=p.parse_args()
 if a.repeats<1 or a.n<1 or any(v<0 or v>100 for v in a.percentages):p.error('invalid parameters')
 folder=json.loads((ROOT/'数据路径.json').read_text(encoding='utf8'))[dataset]
 folder=Path(folder);folder=folder if folder.is_absolute() else ROOT/folder
 session=ROOT/'运行结果'/('DGCA_'+dataset+'_'+time.strftime('%Y%m%d_%H%M%S'))
 for pct in a.percentages:
  for repeat in range(1,a.repeats+1):
   subprocess.run([sys.executable,str(ENGINE/'DGCA/latest/run_shared.py'),'--folder',str(folder),'--N',str(a.n),'--percentage',str(pct),'--mode','grouped','--out',str(session/('p%d_r%d'%(pct,repeat)))],check=True)
