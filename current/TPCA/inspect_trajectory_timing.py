"""Reconstruct fixed-repeat cumulative checkpoints; never rerun a solver."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent
DEFAULT_OUT = Path('E:/Users/24qxh/Desktop/CIE_TPACE/Trajectory_Timing_V0_Check_20261021')
DATASETS = ['OP', 'YELP', 'VG', 'TG', 'SO']
METHODS = [('TPCA', ROOT/'outputs/first_batch_final_20261018', 'TPCAR'),
           ('DGCA', ROOT.parent/'BGCR_D/outputs/BGCR_matched_TPCAR_20261018', 'dynamic_user_heap')]


def read_trace(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    # The Python wrapper adds a separately timed audited endpoint for TPCA.
    # Keep it out of first-attainment and native-history comparisons.
    return [dict(K=int(r.get('coverage') or r['diversity']),
                 q=int(r['integer_objective']), seconds=float(r['time_sec']))
            for r in rows if r.get('trajectory_kind') != 'final_audited_endpoint']


def write_csv(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--output', type=Path, default=DEFAULT_OUT)
    args = ap.parse_args(); out=args.output; out.mkdir(parents=True, exist_ok=True)
    raw=out/'raw'; raw.mkdir(exist_ok=True)
    checkpoints=[]; endpoints=[]; repeat_endpoints=[]; manifest=[]; checks=[]; graphs={}
    # Repeat 1 is chosen uniformly before considering any timing values.
    for dataset in DATASETS:
        for method, base, mode in METHODS:
            paths=list(base.glob('%s_%s_N10_D*_r1/result.json'%(dataset,mode)))
            candidates=[(p,json.loads(p.read_text(encoding='utf-8'))) for p in paths]
            p,r=max(candidates,key=lambda pr:pr[1]['job']['D'])
            job=r['job']; C=job['D']; K0=job['initial_coverage']; q0=job['initial_integer_objective']
            previous=graphs.setdefault(dataset, (job['graph_id'], K0, q0))
            assert previous==(job['graph_id'],K0,q0)
            trace=read_trace(p.parent/'trajectory.csv'); by_K={}
            assert trace and trace[0]['K']==K0 and trace[0]['q']==q0
            assert all(a['seconds']<=b['seconds'] for a,b in zip(trace,trace[1:]))
            assert all(b['K']==a['K']+1 for a,b in zip(trace,trace[1:]))
            assert all(b['q']<=a['q'] for a,b in zip(trace,trace[1:]))
            assert (trace[-1]['K'],trace[-1]['q'])==(r['audit']['coverage'],r['audit']['integer_objective'])
            for row in trace: by_K.setdefault(row['K'],row)
            for name in ['result.json','trajectory.csv']:
                src=p.parent/name; dest=raw/('%s_%s_r1_%s'%(dataset,method,name))
                shutil.copyfile(src,dest)
                manifest.append(dict(source=str(src),copy=str(dest.relative_to(out)),sha256=hashlib.sha256(src.read_bytes()).hexdigest()))
            native=r.get('native_diagnostics',{})
            native_end=native.get('termination_seconds',native.get('core_seconds'))
            endpoint=dict(dataset=dataset,method=method,repeat=1,source_target_D=C,
                initial_K=K0,terminal_K=r['audit']['coverage'],status=r['status'],
                last_attainment_native_s=trace[-1]['seconds'],native_termination_s=native_end,
                python_core_s=r['solve_seconds'],build_s=r['build_seconds'],
                build_plus_core_s=r['optimization_total_seconds'],peak_working_set_gib=r['peak_working_set_bytes']/2**30,
                integer_objective=r['audit']['integer_objective'],
                score_loss_pct=(q0-r['audit']['integer_objective'])/q0*100,
                source=str(p))
            endpoints.append(endpoint)
            for pct in [0,20,40,60,80,100]:
                D=math.ceil(C*pct/100); reached=by_K.get(max(D,K0))
                state='initial_solution_already_sufficient' if D<=K0 else ('reached' if reached else 'not_reached')
                checkpoints.append(dict(dataset=dataset,method=method,repeat=1,coverage_target_pct=pct,
                    target_D=D,checkpoint_K=reached['K'] if reached else None,
                    cumulative_native_s=reached['seconds'] if reached else None,
                    build_plus_checkpoint_s=r['build_seconds']+reached['seconds'] if reached else None,
                    integer_objective=reached['q'] if reached else None,state=state,
                    scope='checkpoint within the fixed 100%-target run; not an independent-target runtime',
                    source=str(p.parent/'trajectory.csv')))
                # Check that the independent lower-target run follows the same
                # objective/coverage prefix. This does NOT equate timing scopes.
                lower=base/('%s_%s_N10_D%d_r1'%(dataset,mode,D))/'result.json'
                if lower.exists():
                    lr=json.loads(lower.read_text(encoding='utf-8'))
                    assert lr['job']['graph_id']==job['graph_id']
                    lt=read_trace(lower.parent/'trajectory.csv')
                    assert all(x['K'] in by_K and x['q']==by_K[x['K']]['q'] for x in lt)
                    checks.append(dict(dataset=dataset,method=method,pct=pct,verified_prefix_rows=len(lt),
                        lower_run_status=lr['status'],graph_and_integer_objective_prefix_equal=True))
            for repeat in [1,2,3]:
                rp=base/('%s_%s_N10_D%d_r%d'%(dataset,mode,C,repeat))/'result.json'
                rr=json.loads(rp.read_text(encoding='utf-8')); tt=read_trace(rp.parent/'trajectory.csv')
                assert rr['job']['graph_id']==job['graph_id']
                assert [(x['K'],x['q']) for x in tt]==[(x['K'],x['q']) for x in trace]
                repeat_endpoints.append(dict(dataset=dataset,method=method,repeat=repeat,
                    terminal_K=rr['audit']['coverage'],last_attainment_native_s=tt[-1]['seconds'],
                    python_core_s=rr['solve_seconds'],build_s=rr['build_seconds'],
                    build_plus_core_s=rr['optimization_total_seconds'],source=str(rp)))
    write_csv(out/'cumulative_checkpoints_r1.csv',checkpoints)
    write_csv(out/'terminal_results_r1.csv',endpoints)
    write_csv(out/'all_three_repeats_terminal.csv',repeat_endpoints)
    (out/'audit.json').write_text(json.dumps(dict(passed=True,repeat_selection='repeat=1 for all 10 method/dataset combinations, without runtime-based selection',
        native_trajectory_count=10,checkpoint_rows=len(checkpoints),independent_prefix_checks=checks,
        repeats_have_identical_coverage_and_integer_objective_trajectories=True),ensure_ascii=False,indent=2),encoding='utf-8')
    (out/'SOURCE_SHA256.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    lines=['# 固定第一次运行的累计时间核查','',
      '来源：2026-10-18主比较批次，N=10；TPCA为state_source，DGCA为dynamic_user_heap。统一取100%请求目标的repeat=1，不按快慢挑选。',
      '', '## 同一次运行的累计原生时间（秒）','',
      '| 数据集 | 算法 | 20% | 40% | 60% | 80% | 100% | 实际终止覆盖 |',
      '|---|---|---:|---:|---:|---:|---:|---:|']
    for ep in endpoints:
        vals=[]
        for row in [x for x in checkpoints if x['dataset']==ep['dataset'] and x['method']==ep['method'] and x['coverage_target_pct']]:
            val='—' if row['cumulative_native_s'] is None else '%.3f'%row['cumulative_native_s']
            if row['state']=='initial_solution_already_sufficient':val+='†'
            vals.append(val)
        lines.append('| %s | %s | %s | %s |'%(ep['dataset'],ep['method'],' | '.join(vals),format(ep['terminal_K'],',')))
    lines += ['', '† 初始Top-N覆盖已经超过该目标；这里是长运行的初始化轨迹点，不是单独启动该目标的实测耗时。',
      '— 表示该次运行没有达到目标，不以最终终止时间代填。Yelp的100%目标本身不可行；DGCA其他未达目标只表明直接替换邻域耗尽。',
      '', '## 全程计时和最终结果（同为repeat=1）','',
      '| 数据集 | 算法 | 首次到达最终覆盖（s） | Python核心调用（s） | 构建＋核心（s） | 峰值工作集（GiB） | 得分损失（%） |',
      '|---|---|---:|---:|---:|---:|---:|']
    for e in endpoints:
        lines.append('| %s | %s | %.3f | %.3f | %.3f | %.3f | %.4f |'%(e['dataset'],e['method'],e['last_attainment_native_s'],e['python_core_s'],e['build_plus_core_s'],e['peak_working_set_gib'],e['score_loss_pct']))
    lines += ['', '## 使用边界','',
      '- 这是一套沿途累计时间数据，可以作为轨迹图或累计时间表；不是另一批独立目标计时。',
      '- 首次到达最终覆盖、原生终止、Python调用结束三种时间分别保留；不可行验证、转换和返回开销可能位于首次到达之后。',
      '- 原生累计时间含该求解器内部初始化，不含Python方法构建、共享数据载入、外部审计和文件导出。',
      '- build_plus_checkpoint_s为同一运行构建时间加轨迹截点，仅表示预处理加沿途累计耗时，不表示另一次独立求解。',
      '- 现有三次测量仍保留。若正文采用固定第一次运行，必须统一声明单次测量；不能只替换SO异常格，也不能与其他算法的三次中位数混用。',
      '- MCF/Gurobi没有相同的逐覆盖原生轨迹，不把它们的独立目标总时间混入累计原生时间表。',
      '- 同一输入不代表TPCA与DGCA的最优值相同：TPCA是精确求解，DGCA是直接替换贪心，可能得分更低或提前终止。',
      '', '复现：使用当前Python环境运行 E:/PythonProject/TRMNL_BENCH/inspect_trajectory_timing.py；仅读取旧记录，不运行求解器。']
    (out/'核查说明与完整数据表.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(output=str(out),checkpoints=len(checkpoints),independent_prefix_checks=len(checks),all_passed=True)))


if __name__=='__main__':main()
