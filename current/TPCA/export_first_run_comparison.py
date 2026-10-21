"""Export repeat=1 consistently; keep cumulative checkpoints a separate scope."""
import csv
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
OUT=Path('E:/Users/24qxh/Desktop/CIE_TPACE/First_Run_Comparison_20261021')
DATASETS=['OP','YELP','VG','TG','SO']
METHODS=['TPCA','DGCA','MCF','GUROBI-LP','GUROBI-MIP']


def write_csv(path,rows):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    OUT.mkdir(exist_ok=True)
    sources=[];rows=[];initial={}
    for base in [ROOT/'outputs/first_batch_final_20261018',ROOT.parent/'BGCR_D/outputs/BGCR_matched_TPCAR_20261018']:
        for p in base.glob('*/result.json'):
            r=json.loads(p.read_text(encoding='utf-8'));j=r.get('job',{})
            if j.get('N')!=10 or j.get('repeat')!=1:continue
            if j.get('method')=='BGCR-D' and j.get('native_mode')!='dynamic_user_heap':continue
            method={'TPCAR':'TPCA','BGCR-D':'DGCA'}.get(j['method'],j['method'])
            if method not in METHODS:continue
            key=j['dataset'];protocol=(j['graph_id'],j['initial_coverage'],j['initial_integer_objective'])
            assert initial.setdefault(key,protocol)==protocol
            a=r.get('audit') or {}
            rows.append(dict(dataset=key,method=method,N=10,coverage_target_pct=j['percentage'],D=j['D'],repeat=1,
                status=r['status'],coverage=a.get('coverage'),integer_objective=a.get('integer_objective'),
                build_s=r.get('build_seconds'),core_s=r.get('solve_seconds'),total_s=r.get('optimization_total_seconds'),
                peak_working_set_gib=r.get('peak_working_set_bytes',0)/2**30,
                graph_id=j['graph_id'],source=str(p),timing_scope='independent target: method construction plus Python core call'))
            sources.append(dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    rows.sort(key=lambda r:(DATASETS.index(r['dataset']),METHODS.index(r['method']),r['coverage_target_pct']))
    assert len(rows)==102
    assert len({(r['dataset'],r['method'],r['D']) for r in rows})==102
    comparisons=[]
    for dataset in DATASETS:
        for pct in [0,20,40,60,80,100]:
            group=[r for r in rows if r['dataset']==dataset and r['coverage_target_pct']==pct and r['status']=='OPTIMAL']
            objs={r['integer_objective'] for r in group if r['method']!='DGCA'}
            assert len(objs)<=1,(dataset,pct,objs)
            comparisons.append(dict(dataset=dataset,pct=pct,optimal_exact_methods=[r['method'] for r in group],integer_objective_agrees=True))
    write_csv(OUT/'main_comparison_r1_102_configs.csv',rows)
    wide=[]
    for dataset in DATASETS:
        for method in METHODS:
            values=[r for r in rows if r['dataset']==dataset and r['method']==method]
            if not values:continue
            row=dict(dataset=dataset,method=method)
            for r in values:row['rho_%d_pct'%r['coverage_target_pct']]=r['total_s'] if r['total_s'] is not None else r['status']
            wide.append(row)
    write_csv(OUT/'independent_total_time_r1_wide.csv',wide)
    source=Path('E:/Users/24qxh/Desktop/CIE_TPACE/Ablation_Results_Restored_20261021/data/module_ablation_all_110_runs.csv')
    with source.open(encoding='utf-8-sig',newline='') as f: ablation=[r for r in csv.DictReader(f) if r['repeat']=='1']
    assert len(ablation)==40
    write_csv(OUT/'module_ablation_r1_40_configs.csv',ablation)
    sources.append(dict(path=str(source),sha256=hashlib.sha256(source.read_bytes()).hexdigest()))
    (OUT/'source_sha256.json').write_text(json.dumps(sources,ensure_ascii=False,indent=2),encoding='utf-8')
    (OUT/'checks.json').write_text(json.dumps(dict(passed=True,main_configs=102,module_configs=40,selection='repeat=1 uniformly; retain all failures and original repeats',exact_objective_checks=comparisons),indent=2),encoding='utf-8')
    notes='''# 第一次运行的配套数据与论文口径

本次只重新提取已有记录，不运行MCF、Gurobi或V0，不覆盖已有中位数版本。

1. 累计时间图：采用此前已提取的TPCA/DGCA各数据集100%目标repeat=1轨迹。每个点是同一次运行首次达到相应覆盖量的累计原生时间。
2. 主比较：本目录102个设置全部统一取repeat=1，时间包含方法构建和Python核心调用。MCF和Gurobi已配套提取，无需为了更改统计方式重跑。
3. 两种计时不能混表。MCF/Gurobi没有相同的逐覆盖轨迹；不能让它们的独立总时间与TPCA/DGCA累计原生时间计算加速倍数。
4. 所有失败、资源停止与不可行状态原样保留。Yelp的MCF在100%诊断不可行，其时间不能与TPCA返回最大覆盖方案的时间当作等价输出任务比较。Gurobi本批只有OP记录，不补造其他数据集数据。
5. 40个模块配置同样固定取repeat=1。TPCA消融批次的SO Full为内存保护停止，不能从主比较另拿一个成功Full填入；所有配置的批次和状态均保留。
6. 累计时间不减是轨迹定义的性质，不证明独立计时更准确。SO独立目标repeat=1的80%仍为340.056841秒，100%为199.467136秒，不因改取第一次就消失。
7. V0可暂不新增正式实验。现有证据支持边界查询堆及两端聚合在所测配置中的作用；不单独量化树复用相对逐轮完整Dijkstra的收益，不声称每项模块在每个数据集都加速。
8. 模块作用优先用同一消融批次的输出一致性、核心时间和操作计数共同说明；构建加核心时间另报，避免把构建波动当成核心机制加速。
9. 三次原始记录继续保留，建议在补充材料提供三次范围/原始点。正文采用单次测量必须明确说明，没有显著性检验，不据此声称统计显著或稳定的精确加速倍数。

复现脚本：E:/PythonProject/TRMNL_BENCH/export_first_run_comparison.py
'''
    (OUT/'说明_首次运行与V0取舍.md').write_text(notes,encoding='utf-8')
    print(json.dumps(dict(output=str(OUT),main_configs=len(rows),module_configs=len(ablation),all_checks_passed=True),ensure_ascii=False))


if __name__=='__main__':main()
