#!/usr/bin/env python3
"""Write the final pilot report when its controller finishes; no experiment calls."""
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PILOT=ROOT/'results_sweeps/sigmetrics_recipe_pilot_20260928'


def main():
    started=time.monotonic()
    initial=json.loads((PILOT/'state.json').read_text())
    pid=initial.get('pid')
    proc=Path('/proc')/str(pid)/'stat'
    start_ticks=proc.read_text().rsplit(')',1)[1].split()[19] if proc.exists() else None
    reason='controller_finished'
    while True:
        state=json.loads((PILOT/'state.json').read_text())
        if state['status']!='running':break
        alive=proc.exists() and proc.read_text().rsplit(')',1)[1].split()[19]==start_ticks
        if not alive:reason='controller_exited_before_terminal_state';break
        if time.monotonic()-started>43200:reason='monitor_limit_reached';break
        temp=PILOT/'completion_monitor.json.tmp'
        temp.write_text(json.dumps({'status':'watching','pid':os.getpid(),'controller_pid':pid,
             'updated_at':datetime.now(timezone.utc).isoformat()},indent=2)+'\n')
        temp.replace(PILOT/'completion_monitor.json')
        time.sleep(30)
    report=subprocess.run([sys.executable,str(ROOT/'scripts/report_sigmetrics_recipe_pilot.py')],cwd=ROOT,capture_output=True,text=True)
    snapshot=subprocess.run([sys.executable,str(ROOT/'scripts/snapshot_sigmetrics_progress.py')],cwd=ROOT,capture_output=True,text=True)
    result={'status':'finished','reason':reason,'finished_at':datetime.now(timezone.utc).isoformat(),
            'report_returncode':report.returncode,'report_directory':report.stdout.strip(),
            'snapshot_returncode':snapshot.returncode,'snapshot_directory':snapshot.stdout.strip()}
    (PILOT/'completion_monitor.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()
