import json, statistics, re, sys
sys.argv=['x','none']
exec(open('build.py').read().split("fams={}")[0])
from collections import defaultdict
def sub(r):
    if r['family']=='F1': return 'F1 ablation ('+('agentic' if r['mode']=='agentic-HLSmith' else 'plain')+')'
    if r['family']=='F2-runs_controlled': return 'F2a runs_controlled (old bench)'
    if r['family']=='F2-s90': return 'F2b s90'
    if r['family']=='F2-dsp5065': return 'F2c dsp5065'
    if r['family']=='F3': return 'F3 free agents'
    if r['family']=='F4-lud': return 'F4 LUD'
rs=[r for r in rows if not r.get('excluded') and r['family']!='F3' or (r['family']=='F3' and r['tag'].startswith('mm1024_') and r['tag'] in ('mm1024_base_r1','mm1024_know_r1','mm1024_team_r1'))]
rs=[r for r in rs if not (r.get('excluded') and r['excluded'].startswith('skipped'))]
groups=defaultdict(list)
for r in rs:
    v='DeepSeek' if r.get('vendor')=='DeepSeek' else ('Claude Code agent' if r['mode']=='agentic-HLSmith' else ('free' if r['family']=='F3' else 'Claude'))
    if r['family']=='F3': v='Claude Code (free)'
    rg = rung(r['tag']) if r['family']!='F3' else r['variant'] if False else r['tag'].split('_')[1]
    groups[(sub(r), rg, v)].append(r)
def med(xs):
    xs=[x for x in xs if x is not None]
    return (statistics.median(xs), len(xs)) if xs else (None,0)
def fm(x):
    if x is None: return 'n/a'
    if isinstance(x, tuple): return 'n/a' if x[0] is None else (f'{x[0]:,.2f}' if x[0]<1e4 else f'{x[0]:,.0f}') + f' (n={x[1]})'
    return str(x)
out=['| family | rung | vendor / mode | runs | median LLM calls | median input tok | median output tok | median cost USD | median wall s | median worst latency cyc (complete) | median DSP (complete) |','|---|---|---|---|---|---|---|---|---|---|---|']
for k in sorted(groups):
    g=groups[k]
    lc=[ (r.get('agent_llm_calls') if r['mode']=='agentic-HLSmith' else (r.get('llm_calls') if r['family']=='F3' else r.get('ctrl_calls'))) for r in g]
    inp=[(r.get('ctrl_in') if r['family']!='F3' else ((r.get('in_tokens_cum') or 0)+(r.get('cache_read_cum') or 0)+(r.get('cache_create_cum') or 0))) for r in g]
    outp=[(r.get('ctrl_out') if r['family']!='F3' else r.get('out_tokens_cum')) for r in g]
    cost=[(r.get('cost_usd') if r['family']=='F3' else r.get('cost_usd_shim')) for r in g]
    wall=[r.get('wallclock_s') for r in g]
    comp=[r for r in g if r.get('llm_calls_ms') is not None or r['family']=='F3']
    lat=[r.get('lat_worst') for r in comp]; dsp=[r.get('dsp') for r in comp]
    out.append(f"| {k[0]} | {k[1]} | {k[2]} | {len(g)} | {fm(med(lc))} | {fm(med(inp))} | {fm(med(outp))} | {fm(med(cost))} | {fm(med(wall))} | {fm(med(lat))} | {fm(med(dsp))} |")
print('\n'.join(out))
