"""Export every saved main-run timing and its median from frozen input only."""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent


def repeated_times():
    data=pd.read_csv(ROOT/'data/main_comparison.csv')
    rows=[]
    for (ds,method,pct),group in data.groupby(['dataset','method','percentage'],sort=False):
        row=dict(dataset=ds,method={'TPCAR':'TPCA','BGCR-D':'DGCA'}.get(method,method),
                 coverage_target_pct=int(pct),recorded_runs=len(group))
        complete=group[group.status.isin(['OPTIMAL','INFEASIBLE','FEASIBLE','TARGET_UNREACHED'])]
        for repeat in [1,2,3]:
            selected=group[group['repeat']==repeat]
            for metric in ['build','core','total']:
                row[f'r{repeat}_{metric}_s']=np.nan if selected.empty or selected.iloc[0].status=='SYSTEM_MEMORY_GUARD' else selected.iloc[0][metric]
            row[f'r{repeat}_status']='NOT_ATTEMPTED' if selected.empty else selected.iloc[0].status
        row['completed_or_diagnosed_runs']=len(complete)
        for metric in ['build','core','total']:
            row[f'median_{metric}_s']=complete[metric].median() if len(complete) else np.nan
        row['min_total_s']=complete.total.min() if len(complete) else np.nan
        row['max_total_s']=complete.total.max() if len(complete) else np.nan
        rows.append(row)
    frame=pd.DataFrame(rows)
    assert len(frame)==102
    (ROOT/'tables').mkdir(exist_ok=True)
    frame.to_csv(ROOT/'tables/TS3_repeated_runtime_records.csv',index=False,encoding='utf-8-sig')
    return frame


if __name__=='__main__':
    print('Repeat-runtime configurations:',len(repeated_times()))
