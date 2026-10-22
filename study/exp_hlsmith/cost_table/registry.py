import glob, os, json, re
ROOT='/scratch/asa582/workspaces/c2hls_claude_code'
def runs():
    out=[]
    def add(fam, d, excluded=None):
        out.append(dict(family=fam, dir=d, tag=os.path.basename(d.rstrip('/')), excluded=excluded))
    # F1
    for d in sorted(glob.glob(ROOT+'/study/agentic_ablation/runs/*/')):
        n=os.path.basename(d.rstrip('/'))
        if n.startswith('_'): add('F1',d,'skipped: _-prefixed dir'); continue
        add('F1',d)
    for d in sorted(glob.glob(ROOT+'/runs_controlled/*/')):
        n=os.path.basename(d.rstrip('/'))
        if n.startswith('_'): add('F2-runs_controlled',d,'skipped: _-prefixed (superseded/timeout)'); continue
        if 'smoke' in n: add('F2-runs_controlled',d,'excluded: smoke test, not r<n>'); continue
        add('F2-runs_controlled',d)
    for d in sorted(glob.glob(ROOT+'/study/s90/runs/*/')):
        n=os.path.basename(d.rstrip('/'))
        if n.startswith('_'): add('F2-s90',d,'skipped: _-prefixed'); continue
        add('F2-s90',d)
    for d in sorted(glob.glob(ROOT+'/study/dsp5065/runs/*/')):
        n=os.path.basename(d.rstrip('/'))
        if n.startswith('_'): add('F2-dsp5065',d,'skipped: _-prefixed (invalid/timeout)'); continue
        add('F2-dsp5065',d)
    for t in ['mm1024_base_r1','mm1024_know_r1','mm1024_team_r1']:
        add('F3',ROOT+'/study/agentfw/runs/'+t)
    for t in ['mm1024_hlsm_r1','_aborted_pilot','smoke_base']:
        add('F3',ROOT+'/study/agentfw/runs/'+t,'not in requested set (excluded)')
    for d in sorted(glob.glob(ROOT+'/study/lud/runs/*/')):
        n=os.path.basename(d.rstrip('/'))
        if n.startswith('_'): add('F4-lud',d,'skipped: _-prefixed'); continue
        add('F4-lud',d)
    return out
def bench_dir(r):
    d=r['dir']
    if r['family']=='F3': return d
    subs=[s for s in glob.glob(d+'*/') if os.path.isdir(s) and not os.path.basename(s.rstrip('/')).startswith('_')]
    ctrl=[s for s in subs if os.path.exists(s+'ctrl_llm_calls.jsonl') or os.path.exists(s+'cc_run_summary.json') or os.path.exists(s+'cc_run_plan.json')]
    return (ctrl or subs or [None])[0]
if __name__=='__main__':
    R=runs()
    from collections import Counter
    print(Counter((r['family'], r['excluded'] is None) for r in R))
    for r in R:
        b=bench_dir(r)
        if r['excluded'] is None: print(r['family'], r['tag'], b and os.path.basename(b.rstrip('/')))
