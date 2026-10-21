"""Text/bounds audit and raster previews; human visual sign-off stays separate."""
from pathlib import Path
import argparse
import json
import subprocess
import time
from PIL import Image, ImageOps, ImageDraw
from pypdf import PdfReader
import pdfplumber

ROOT=Path(__file__).resolve().parent/'outputs/BGCR_strategy_comparison_20261019/paper_report'
OLD=Path('E:/Users/24qxh/Desktop/TRMNL_BGCR_Experiments_20261019/Experiment_Report_20261019.pdf')
POPPLER=Path('C:/Users/24qxh/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/poppler/Library/bin/pdftoppm.exe')


def main(root=ROOT,wait=False):
    root=Path(root)
    while True:
        p=root/'progress.json'
        progress=json.loads(p.read_text(encoding='utf8')) if p.exists() else {}
        if progress.get('complete'):break
        if progress.get('state')=='ERROR':raise RuntimeError(progress)
        if not wait:raise RuntimeError('Reports not generated')
        time.sleep(5)
    supplementary=root/'BGCR_Strategy_Comparison_20261019.pdf';combined=root/'Experiment_Report_20261019_v2.pdf'
    new=PdfReader(str(supplementary));old=PdfReader(str(OLD));merged=PdfReader(str(combined))
    assert len(merged.pages)==1+len(old.pages)+len(new.pages)
    unchanged=[]
    for k,p in enumerate(old.pages):
        assert p.get_contents().get_data()==merged.pages[k+1].get_contents().get_data()
        unchanged.append(k+1)
    bad=[];text=[]
    with pdfplumber.open(supplementary) as pdf:
        for i,p in enumerate(pdf.pages,1):
            text.append(p.extract_text() or '')
            for c in p.chars:
                if c['x0']<-1 or c['x1']>p.width+1 or c['top']<-1 or c['bottom']>p.height+1:
                    bad.append(dict(page=i,char=c['text'],bounds=[c['x0'],c['x1'],c['top'],c['bottom']]))
    joined='\n'.join(text)
    figures=['F21','F22','F23','F24']
    tables=['T%d_%s_%s'%(num,stem,ds) for ds in ['OP','YELP','SO'] for num,stem in [(13,'quality'),(14,'cost'),(15,'work')]]
    assert not bad,bad
    assert all(id in joined for id in figures+tables)
    assert '\ufffd' not in joined
    qa=root/'qa';pages=qa/'supplement_pages';pages.mkdir(exist_ok=True)
    subprocess.run([str(POPPLER),'-scale-to','1600','-png',str(supplementary),str(pages/'page')],check=True)
    subprocess.run([str(POPPLER),'-f','1','-singlefile','-scale-to','1600','-png',str(combined),str(qa/'updated_cover')],check=True)
    previews=sorted(pages.glob('page-*.png'));thumbs=[]
    for p in previews:
        with Image.open(p) as im:
            im=im.convert('RGB');im.thumbnail((420,600));canvas=Image.new('RGB',(440,630),'#dddddd')
            canvas.paste(im,((440-im.width)//2,20));ImageDraw.Draw(canvas).text((10,610),p.stem,fill='black');thumbs.append(canvas)
    sheet=Image.new('RGB',(440*3,630*((len(thumbs)+2)//3)),'white')
    for k,im in enumerate(thumbs):sheet.paste(im,((k%3)*440,(k//3)*630))
    sheet.save(qa/'supplement_contact_sheet.png')
    result=dict(passed=True,supplement_pages=len(new.pages),combined_pages=len(merged.pages),
        preserved_original_pages=len(old.pages),unchanged_original_content_streams=len(unchanged),out_of_bounds_chars=bad,
        figures_found=figures,tables_found=tables,rendered_new_pages=len(previews)+1,
        visual_review='PENDING; this script does not sign off visual quality')
    (qa/'pdf_text_audit.json').write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=ROOT);parser.add_argument('--wait',action='store_true');args=parser.parse_args();main(args.root,args.wait)
