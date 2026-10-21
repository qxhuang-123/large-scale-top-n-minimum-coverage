"""Atomic JSON output resilient to transient Windows reader/AV file locks."""
import json
import os
from pathlib import Path
import time
import uuid


def write_json(path, value):
    import numpy as np
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    def encode(v):
        if isinstance(v,np.generic):return v.item()
        if isinstance(v,np.ndarray):return v.tolist()
        if isinstance(v,Path):return str(v)
        raise TypeError('Unsupported JSON value: '+type(v).__name__)
    text=json.dumps(value,ensure_ascii=False,indent=2,default=encode)
    pending=path.with_name(path.name+'.'+uuid.uuid4().hex+'.pending')
    deadline=time.monotonic()+30
    delay=.05
    try:
        while True:
            try:
                pending.write_text(text,encoding='utf8')
                os.replace(pending,path)
                return
            except PermissionError:
                if time.monotonic()>=deadline:raise
                time.sleep(delay);delay=min(1,delay*1.5)
    finally:
        if pending.exists():
            try:pending.unlink()
            except PermissionError:pass
