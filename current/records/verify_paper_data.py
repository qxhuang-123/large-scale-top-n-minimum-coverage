"""Verify the paper CSV summaries using only Python's standard library."""
import csv,json,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def rows(name):
 with (ROOT/name).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def check():
 main=rows('main_paper_runs.csv'); table=rows('Table4_TableS1_paper.csv')
 assert len(main)==294 and len(table)==102
 for line in table:
  group=[r for r in main if r['dataset']==line['dataset'] and r['method']==line['method'] and int(r['percentage'])==int(line['target_pct'])]
  assert len(group)==int(line['repeats'])
  if line['method']=='DGCA':assert len(group)==3 and len({r['record_id'] for r in group})==3
  for column in ['build_s','solve_s','total_s']:
   try:display=float(line[column])
   except ValueError:continue
   values=[float(r[column]) for r in group if r[column]!='']
   assert abs(statistics.mean(values)-display)<=.000501,(line,column)
  if line['total_sd_s']:
   assert abs(statistics.stdev(float(r['total_s']) for r in group)-float(line['total_sd_s']))<1e-9
 dgca=rows('DGCA_paper_runs.csv');assert len(dgca)==90
 assert all('warmup' not in r['record_id'].lower() for r in dgca)
 strategy=rows('strategy_paper_runs.csv')
 assert len(strategy)==60
 for line in rows('Table6_paper.csv'):
  group=[r for r in strategy if r['dataset']==line['dataset'] and r['method']==line['method']]
  assert len(group)==3
  assert abs(statistics.mean(float(r['total_s']) for r in group)-float(line['total_s']))<=.000501
  assert all(r['coverage']==line['coverage'] for r in group)
 mem=rows('Figure2_memory.csv');assert len(mem)==102
 assert len({(r['dataset'],r['method'],r['pct']) for r in mem})==102
 for line in mem:
  if line['method']!='DGCA':continue
  run=next(r for r in dgca if r['dataset']==line['dataset'] and int(r['percentage'])==int(line['pct']) and r['reported_repeat']=='1')
  assert abs(float(line['value'])-float(run['peak_working_set_gib']))<1e-12
 assert len(rows('Table2_instances.csv'))==5
 assert len(rows('Table3_coverage_score.csv'))==25
 assert len(rows('Table5_tpca_component.csv'))==6
 assert all(r['repeats']=='1' for r in rows('Table5_tpca_component.csv'))
 print('PASS: main and strategy means/SDs, three DGCA repetitions, updated memory, no warmup CSV entries, table sizes.')
if __name__=='__main__':check()
