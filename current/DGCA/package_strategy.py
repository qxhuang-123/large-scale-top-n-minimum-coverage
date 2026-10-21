"""Package the visually reviewed report and trace evidence; never runs solvers."""
from pathlib import Path
import csv
import hashlib
import json
import shutil
import zipfile

ROOT=Path(__file__).resolve().parent
SESSION=ROOT/'outputs/BGCR_strategy_comparison_20261019'
OUT=SESSION/'paper_report'
DESKTOP=Path('E:/Users/24qxh/Desktop')
TARGET=DESKTOP/'TRMNL_BGCR_Strategy_20261019'
OLD=DESKTOP/'TRMNL_BGCR_Experiments_20261019'


def main():
    qa=json.loads((OUT/'qa/visual_review.json').read_text(encoding='utf8'))
    assert qa['passed'],'Visual review must pass before packaging'
    rows=list(csv.DictReader((OUT/'data/comparison.csv').open(encoding='utf-8-sig',newline='')))
    assert len(rows)==66
    TARGET.mkdir(exist_ok=True)
    for path in OUT.iterdir():
        if path.name=='qa' or path.name=='progress.json':continue
        destination=TARGET/path.name
        if path.is_dir():shutil.copytree(path,destination,dirs_exist_ok=True)
        else:shutil.copy2(path,destination)
    (TARGET/'qa').mkdir(exist_ok=True)
    for name in ['visual_review.json','pdf_text_audit.json']:
        shutil.copy2(OUT/'qa'/name,TARGET/'qa'/name)
    # Preserve evidence without multi-GB candidate arrays or recommendation CSVs.
    for row in rows:
        source=Path(row['result_folder']);folder=TARGET/'run_evidence'/source.name;folder.mkdir(parents=True,exist_ok=True)
        for name in ['job.json','result.json','events.npy','trajectory.csv','memory_trace.csv']:
            if (source/name).exists():shutil.copy2(source/name,folder/name)
    code=TARGET/'algorithm_code';code.mkdir(exist_ok=True)
    for name in ['run_strategy_comparison.py','strategy_worker.py','strategy_supervisor.py','validate_strategy.py',
        'strategy_report.py','strategy_figures.py','strategy_figure_qa.py','strategy_pdf.py',
        'audit_strategy_results.py','review_strategy_artifacts.py','package_strategy.py',
        'BGCR-STRATEGY-OP.py','BGCR-STRATEGY-YELP.py','BGCR-STRATEGY-SO.py','README_STRATEGY_20261019.md']:
        shutil.copy2(ROOT/name,code/name)
    (code/'native').mkdir(exist_ok=True)
    shutil.copy2(ROOT/'native/bgcr_d_bench.cpp',code/'native/bgcr_d_bench.cpp')
    for p in ROOT.glob('bgcr_d_bench*.pyd'):shutil.copy2(p,code/p.name)
    common=code/'TRMNL_BENCH_helpers';common.mkdir(exist_ok=True)
    for name in ['bench_common.py','bench_memory.py','resilient_json.py']:
        shutil.copy2(ROOT.parent/'TPCA'/name,common/name)
    for p in (ROOT.parent/'CommonInput').glob('prepare_cached_datasets.py'):shutil.copy2(p,common/p.name)
    configurations=TARGET/'pycharm_run_configurations';configurations.mkdir(exist_ok=True)
    for pattern in ['BGCR-STRATEGY-*.run.xml','BGCR-FIG-F2*.run.xml']:
        for p in (ROOT.parent/'.run').glob(pattern):shutil.copy2(p,configurations/p.name)
    (code/'READ_ME_FIRST.txt').write_text(
        'Algorithm source snapshot only. The working project is E:/PythonProject; large shared candidate arrays are not included.\n'
        'Restore the documented project layout and use its Python 3.9 interpreter to run solvers.\n'
        'The separate code/ directory redraws figures directly from the included data/ files, without any solver.\n',encoding='utf8')
    provenance=dict(passed=True,new_formal_runs=len(rows),previous_formal_runs=479,total_formal_runs=479+len(rows),
        new_figures=4,new_tables=9,total_figures=24,total_tables=57,qa=qa)
    provenance['files']={str(p.relative_to(TARGET)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in TARGET.rglob('*') if p.is_file() and p.name!='manifest.json'}
    (TARGET/'manifest.json').write_text(json.dumps(provenance,indent=2,ensure_ascii=False),encoding='utf8')
    archive=DESKTOP/(TARGET.name+'.zip')
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in TARGET.rglob('*'):
            if p.is_file():z.write(p,str(Path(TARGET.name)/p.relative_to(TARGET)))
    # Keep the original report and its manifest intact; put the extension alongside it.
    extension=OLD/'strategy_comparison'
    shutil.copytree(TARGET,extension,dirs_exist_ok=True)
    shutil.copy2(TARGET/'Experiment_Report_20261019_v2.pdf',OLD/'Experiment_Report_20261019_v2.pdf')
    (OLD/'README_v2.txt').write_text(
        'Updated material: Experiment_Report_20261019_v2.pdf.\n'
        'The original 479-run report is preserved. The strategy_comparison/ extension adds 66 runs, 4 figures and 9 tables.\n'
        'Total: 545 runs, 24 figures, 57 tables. Original manifest.json describes only the original package;\n'
        'the extension has its own manifest.json and source provenance.\n',encoding='utf8')
    print(str(TARGET));print(str(archive));print('66 new runs; 545 combined; 24 figures; 57 tables')


if __name__=='__main__':main()
