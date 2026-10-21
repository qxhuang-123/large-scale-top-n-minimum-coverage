"""Rebuild frozen-data assets in isolation; compare every exported byte."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT=Path(__file__).resolve().parent


def main():
    qa=(ROOT/'qa').resolve()
    checks=[]
    with tempfile.TemporaryDirectory(prefix='reproduction_',dir=str(qa)) as name:
        folder=Path(name).resolve()
        assert folder.parent==qa and folder.name.startswith('reproduction_')
        for part in ['data','code']:shutil.copytree(ROOT/part,folder/part)
        shutil.copyfile(ROOT/'make_all.py',folder/'make_all.py')
        if (ROOT/'tables/TS3_repeated_runtime_records.csv').exists():
            shutil.copyfile(ROOT/'make_repeat_runtime_table.py',folder/'make_repeat_runtime_table.py')
        if (ROOT/'rebuild_supplementary_tables.py').exists():
            shutil.copyfile(ROOT/'rebuild_supplementary_tables.py',folder/'rebuild_supplementary_tables.py')
        log=folder/'rebuild.log'
        commands=[[sys.executable,str(folder/'make_all.py')]]
        if (ROOT/'tables/TS1_module_ablation_detailed_r1.pdf').exists():
            commands.append([sys.executable,str(folder/'make_all.py'),'--only','TS1_module_ablation_detailed_r1'])
        if (ROOT/'tables/TS3_repeated_runtime_records.csv').exists():
            commands.append([sys.executable,str(folder/'make_repeat_runtime_table.py')])
        if (ROOT/'tables/TS2_progressive_grouped.pdf').exists():
            commands.append([sys.executable,str(folder/'rebuild_supplementary_tables.py'),'--compile'])
        with log.open('w',encoding='utf-8') as stream:
            for command in commands:
                result=subprocess.run(command,cwd=str(folder),stdout=stream,stderr=subprocess.STDOUT)
                if result.returncode:raise RuntimeError(log.read_text(encoding='utf-8'))
        paths=[p for part in ['tables','figures'] for p in sorted((ROOT/part).iterdir())
               if p.suffix in ['.pdf','.png','.svg','.csv','.tex']]
        paths.append(ROOT/'Lean_Main_Preview.pdf')
        for source in paths:
            target=folder/source.relative_to(ROOT)
            digest=hashlib.sha256(source.read_bytes()).hexdigest()
            assert target.exists() and hashlib.sha256(target.read_bytes()).hexdigest()==digest,str(source)
            checks.append(dict(path=str(source.relative_to(ROOT)),sha256=digest,identical=True))
    (qa/'reproduction_check.json').write_text(json.dumps(dict(passed=True,files=len(checks),
        scope='Byte-identical same-environment exported figures/tables; no solver rerun',checks=checks),indent=2),encoding='utf-8')
    print('Reproduction passed:',len(checks),'identical files.')


if __name__=='__main__':main()
