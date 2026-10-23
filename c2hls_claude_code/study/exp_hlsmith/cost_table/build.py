import json, re, statistics, os, glob
rows=json.load(open('rows.json'))
FREE_MODEL={}
def model_of(r):
    if r['family']=='F3':
        p=f"/scratch/hpc-prf-llmfpga/asa582/projects/workspaces/c2hls_claude_code/study/agentfw/runs/{r['tag']}/transcript.jsonl"
        if os.path.exists(p):
            for l in open(p):
                try: e=json.loads(l)
                except: continue
                if e.get('type')=='system' and e.get('model'): return e['model']
        return 'n/a'
    return r.get('model') or 'n/a'
def rung(tag):
    for pat in [r'_ag_(.+?)_r\d', r'_pl_(zsk?)_(?:cc|ds)_r\d', r'ctl_(?:cc|ds)_(.+?)_r\d', r'_lud_(?:s90)?.*?_(?:cc|ds)_r\d', r'_s90(dsp)?_(?:cc|ds)_r\d', r'_st_([a-z]\d+)_(?:cc|ds|ag)', r'_st_r\d+_(?:cc|ds)_r\d']:
        pass
    m=re.search(r'_ag_(.+?)_r\d',tag)
    if m: return m.group(1)
    m=re.search(r'_pl_(zsk?)_(?:cc|ds)_r\d',tag)
    if m: return m.group(1)
    m=re.search(r'ctl_(?:cc|ds)_(.+?)_r\d',tag)
    if m: return m.group(1)
    m=re.search(r'^(noskills|skills|zero_shot)_lud_(.+?)_(?:cc|ds)_r\d',tag)
    if m: return m.group(2) if m.group(1)!='zero_shot' else 'zs'
    m=re.search(r'^skills_lud_s90_(?:cc|ds)_r\d',tag)
    if m: return 's90 (lud)'
    m=re.search(r'^skills_s90(dsp)?_(?:cc|ds)_r\d',tag)
    if m: return 's90'+('dsp' if m.group(1) else '')
    m=re.search(r'^skills_st_([sra]\d+)[a-z]*_(?:cc|ds|ag)',tag)
    if m: return m.group(1)
    return tag
def vendor(tag):
    if '_ag_' in tag: return 'Claude Code (agent)'
    if re.search(r'_ds_|_ds\b|_ds_r',tag): return 'DeepSeek'
    if re.search(r'_cc_|_cc\b',tag): return 'Claude'
    return 'Claude'
def f(x):
    if x is None or x=='' : return 'n/a'
    if isinstance(x,float):
        return f'{x:,.2f}' if x<1e6 else f'{x:,.0f}'
    if isinstance(x,int): return f'{x:,}'
    return str(x)
def cost_str(r):
    if r['family']=='F3':
        return ('%.4f (T)' % r['cost_usd']) if r.get('cost_usd') is not None else 'n/a'
    if r.get('cost_usd_shim') is not None:
        return '%.4f (S)' % r['cost_usd_shim']
    return 'n/a'
def row(r):
    tag=r['tag']
    fam=r['family']
    if r['family']=='F3':
        lc=r.get('llm_calls'); 
        inp=(r.get('in_tokens_cum') or 0)+(r.get('cache_read_cum') or 0)+(r.get('cache_create_cum') or 0)
        out=r.get('out_tokens_cum'); think=r.get('thinking_cum'); w=r.get('wallclock_s')
        csim=r.get('submit_csim'); csyn=r.get('submit_csynth')
        lat={'base':404451,'know':658279}.get(r['variant'] if False else tag.split('_')[1]) 
        dsp={'base':8552,'know':8960}.get(tag.split('_')[1])
        if tag=='mm1024_team_r1': lat=None; dsp=None
        status='session ended (usage limit, resumed)' if tag!='mm1024_team_r1' else 'no finished design'
        if tag=='mm1024_team_r1': status='no finished design (quota full)'
        else: status='completed'
        return [tag, model_of(r), 'free-agent', status, lc, inp, out, think, cost_str(r), w, csim, csyn, lat, dsp]
    ag = r.get('mode')=='agentic-HLSmith'
    status = 'complete' if r.get('llm_calls_ms') is not None else 'incomplete (ctrl only)'
    if ag:
        lc=r.get('agent_llm_calls'); think=r.get('agent_mu_think'); csim=r.get('checks_csim'); csyn=r.get('checks_csynth')
    else:
        lc=r.get('ctrl_calls'); think=r.get('ctrl_reason'); 
        if r.get('llm_calls_ms') is not None and r.get('csim_status'):
            csim=sum(1 for v in r['csim_status'].values() if v in('passed','failed'))
            csyn=r.get('gen_hist_n')
        else: csim=None; csyn=None
    inp=r.get('ctrl_in'); out=r.get('ctrl_out')
    w=r.get('wallclock_s')
    return [tag, model_of(r) if r.get('model') is None else r['model'], r['mode'], status, lc, inp, out, think, cost_str(r), w, csim, csyn, r.get('lat_worst'), r.get('dsp')]
HDR=['run tag','model','mode','status','LLM calls','input tok (incl cache)','output tok','reasoning/thinking tok','cost USD (src)','wall s','csim','csynth','worst latency cyc','DSP']
def table(rs):
    out=['| '+' | '.join(HDR)+' |','|'+'|'.join(['---']*len(HDR))+'|']
    for r in rs:
        cells=row(r)
        out.append('| '+' | '.join(f(c) for c in cells)+' |')
    return '\n'.join(out)
fams={}
for r in rows:
    if r.get('excluded') and r['excluded'].startswith('skipped'): continue
    if r.get('excluded') and r['family']!='F3': continue
    if r['family']=='F3' and r['tag'] not in ('mm1024_base_r1','mm1024_know_r1','mm1024_team_r1'): continue
    fams.setdefault(r['family'],[]).append(r)
import sys
mode=sys.argv[1] if len(sys.argv)>1 else 'tables'
if mode=='tables':
    for k,v in fams.items():
        print(f'\n### {k} ({len(v)} rows)\n'); print(table(v))
