import json, glob, os, sys, re
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from registry import runs, bench_dir, ROOT
from shimmatch import load_shim, ctrl_calls
import datetime

def jl(path):
    out=[]
    if not os.path.exists(path): return None
    for l in open(path):
        l=l.strip()
        if not l: continue
        try: out.append(json.loads(l))
        except Exception: out.append({'_bad':l[:80]})
    return out
def jsn(path):
    if not os.path.exists(path): return None
    try: return json.load(open(path))
    except Exception as e: return {'_err':str(e)[:120]}

SHIM=load_shim()
def match_shim(cc):
    """match a ctrl call line to a shim call by tokens and end time (t+elapsed)"""
    if not cc or 'prompt_tokens' not in cc: return None
    try:
        end=datetime.datetime.fromisoformat(cc['t']) + datetime.timedelta(seconds=float(cc.get('elapsed_s',0)))
    except Exception: return None
    best=None
    for s in SHIM:
        if s.get('in_tokens')==cc['prompt_tokens'] and s.get('out_tokens')==cc['completion_tokens']:
            dt=abs((s['_end']-end).total_seconds())
            if dt<=5 and (best is None or dt<best[0]): best=(dt,s)
    return best[1] if best else None

def transcript_stats(paths):
    """paths: list of stream-json transcripts for one session. returns dict"""
    msg_ids=set(); results=[]; ntool=0
    for p in paths:
        if not os.path.exists(p): continue
        for l in open(p):
            l=l.strip()
            if not l: continue
            try: e=json.loads(l)
            except Exception: continue
            if e.get('type')=='assistant':
                m=e.get('message',{})
                if m.get('id'): msg_ids.add(m['id'])
                for c in m.get('content',[]) or []:
                    if isinstance(c,dict) and c.get('type')=='tool_use': ntool+=1
            if e.get('type')=='result': results.append(e)
    return dict(msg_ids=len(msg_ids), tool_uses=ntool, results=results)

def main():
    rows=[]
    for r in runs():
        rec=dict(family=r['family'], tag=r['tag'], excluded=r['excluded'])
        if r['excluded'] and r['excluded'].startswith('skipped'): 
            rows.append(rec); continue
        rows.append(process(r, rec))
    return rows

def process(r, rec):
    d=r['dir']; fam=r['family']
    b=bench_dir(r)
    rec['bench']=os.path.basename(b.rstrip('/')) if b else None
    tag=r['tag']
    if re.search(r'(?:^|_)ag(?:_|$)', tag):
        rec['mode']='agentic-HLSmith'
    elif fam=='F3':
        rec['mode']='free-agent'
    else:
        rec['mode']='plain'
    rec['vendor']='DeepSeek' if re.search(r'_ds_|_ds(_|$)', tag) else ('Claude' if re.search(r'_cc_|_cc(_|$)', tag) else None)
    if fam=='F3':
        return process_free(d, rec)
    if b is None:
        rec['status']='no bench dir'; return rec
    plan=jsn(b+'cc_run_plan.json') or {}
    summ=jsn(b+'cc_run_summary.json') or {}
    ms=jsn(b+'%s_multistep_results.json' % rec['bench']) or None
    if ms is None:
        cand=glob.glob(b+'*_multistep_results.json')
        ms=jsn(cand[0]) if cand else None
    ctrl=jl(b+'ctrl_llm_calls.jsonl')
    rec['model']=plan.get('model') or (ms or {}).get('run',{}).get('model')
    rec['wallclock_s']=summ.get('wallclock_s')
    rec['cc_status']=summ.get('status')
    rec['ctrl_path']=b+'ctrl_llm_calls.jsonl' if ctrl is not None else None
    cl=[c for c in (ctrl or []) if 'prompt_tokens' in c]
    rec['ctrl_calls']=len(cl)
    rec['ctrl_calls_ok']=sum(1 for c in cl if c.get('outcome')=='ok')
    rec['ctrl_in']=sum(c.get('prompt_tokens') or 0 for c in cl)
    rec['ctrl_out']=sum(c.get('completion_tokens') or 0 for c in cl)
    rec['ctrl_reason']=sum(c.get('reasoning_tokens') or 0 for c in cl) if any(c.get('reasoning_tokens') is not None for c in cl) else None
    rec['ctrl_elapsed']=round(sum(c.get('elapsed_s') or 0 for c in cl),1)
    rec['ctrl_models']=sorted(set(c.get('model') for c in cl))
    # shim matches
    shim_hits=[match_shim(c) for c in cl]
    hits=[s for s in shim_hits if s]
    rec['shim_matched']=f"{len(hits)}/{len(cl)}"
    rec['shim_dirs']=sorted(set(os.path.basename(h['_dir']) for h in hits))
    rec['cost_usd_shim']=round(sum(h.get('cost_usd') or 0 for h in hits),6) if hits and all(h.get('cost_usd') is not None for h in hits) else None
    rec['shim_mode']=sorted(set(str(h.get('mode')) for h in hits)) or None
    rec['shim_calls']=[(os.path.basename(h['_dir']), h['n'], h.get('mode'), h['t']) for h in hits]
    if ms:
        u=ms.get('llm_usage') or {}
        rec['llm_calls_ms']=u.get('calls'); rec['in_ms']=u.get('input_tokens'); rec['out_ms']=u.get('output_tokens')
        rec['reason_ms']=u.get('reasoning_tokens'); rec['cache_read_ms']=u.get('cache_read_input_tokens')
        fr=ms.get('final_report') or {}
        rec['lat_worst']=fr.get('latency_cycles_worst'); rec['dsp']=fr.get('dsp')
        rec['csim_ms']=(ms.get('csim') or {}).get('status') if isinstance(ms.get('csim'),dict) else None
        rec['csim_ran']=(ms.get('csim') or {}).get('ran') if isinstance(ms.get('csim'),dict) else None
        rec['csim_status']=ms.get('csim_status')
        rec['steps_n']=len(ms.get('steps') or [])
        rec['gen_hist_n']=len(ms.get('generated_step_history') or [])
        rec['opt_hist_n']=len(ms.get('optimization_history') or [])
        rec['ms_success']=ms.get('success')
        rec['ms_phase']=ms.get('phase')
        rec['ms_err']=(ms.get('steps') or [{}])[0].get('error') if ms.get('steps') else None
    else:
        rec['ms']='missing'
    # ag runs: transcript stats from shim call dirs
    if rec['mode']=='agentic-HLSmith' or any(str(h.get('mode'))=='agent' for h in hits):
        rec.update(agent_stats(hits))
    return rec

def agent_stats(hits):
    out={}
    tot_cost=0; turns=0; dur=0; thinking=0; msg=0; checks_c=0; checks_s=0; tool=0; res_ok=[]
    for h in hits:
        cdir=h['_dir']+'/call_%05d' % h['n']
        tp=cdir+'/agent_transcript.jsonl'
        st=transcript_stats([tp])
        out['agent_dirs']=out.get('agent_dirs',[])+[os.path.basename(cdir)]
        out['agent_llm_calls']=(out.get('agent_llm_calls') or 0)+st['msg_ids']
        out['agent_tool_uses']=(out.get('agent_tool_uses') or 0)+st['tool_uses']
        for res in st['results']:
            out['agent_num_turns']=res.get('num_turns')
            out['agent_duration_ms']=res.get('duration_ms')
            out['agent_cost_result']=res.get('total_cost_usd')
            mu=res.get('modelUsage')
            mu=res.get('modelUsage') or {}
            m=next(iter(mu.values()),{}) if isinstance(mu,dict) and mu else {}
            out['agent_mu_in']=m.get('inputTokens'); out['agent_mu_out']=m.get('outputTokens'); out['agent_mu_think']=m.get('thinkingTokens'); out['agent_mu_cost']=m.get('costUSD'); out['agent_mu_cache_read']=m.get('cacheReadInputTokens'); out['agent_mu_cache_create']=m.get('cacheCreationInputTokens')
            u=res.get('usage') or {}
            out['agent_thinking']=(u.get('output_tokens_details') or {}).get('thinking_tokens') if isinstance(u,dict) else None
        ck=cdir+'/workspace/checks'
        if os.path.isdir(ck):
            out['checks_csim']=(out.get('checks_csim') or 0)+len([f for f in os.listdir(ck) if re.match(r'^\d+_csim\.txt$',f)])
            out['checks_csynth']=(out.get('checks_csynth') or 0)+len([f for f in os.listdir(ck) if re.match(r'^\d+_csynth\.txt$',f)])
        out['agent_tool_calls_file']=os.path.exists(cdir+'/agent_tool_calls.json')
    return out

def process_free(d, rec):
    b=d.rstrip('/')+'/'
    files=[b+'transcript.jsonl']+sorted(glob.glob(b+'transcript_resume*.jsonl'))
    st=transcript_stats(files)
    rec['llm_calls']=st['msg_ids']
    rec['tool_uses']=st['tool_uses']
    res=st['results']
    last=res[-1] if res else {}
    mu=last.get('modelUsage') or {}
    m=next(iter(mu.values()),{}) if isinstance(mu,dict) and mu else {}
    rec['in_tokens_cum']=m.get('inputTokens')
    rec['cache_read_cum']=m.get('cacheReadInputTokens')
    rec['cache_create_cum']=m.get('cacheCreationInputTokens')
    rec['out_tokens_cum']=m.get('outputTokens')
    rec['thinking_cum']=m.get('thinkingTokens')
    rec['cost_usd']=last.get('total_cost_usd')
    rec['cost_usd_model']=m.get('costUSD')
    rec['results_n']=len(res)
    rec['result_costs']=[r.get('total_cost_usd') for r in res]
    rec['result_is_error']=[str(r.get('is_error')) for r in res]
    rec['result_num_turns']=[r.get('num_turns') for r in res]
    rec['result_subtype']=[r.get('subtype') for r in res]
    rec['result_terminal']=[r.get('terminal_reason') for r in res]
    rec['session_duration_ms_sum']=sum(int(r.get('duration_ms') or 0) for r in res)
    rec['wall_started']=(open(b+'started_at').read().strip() if os.path.exists(b+'started_at') else None)
    rec['wall_ended']=(open(b+'ended_at').read().strip() if os.path.exists(b+'ended_at') else None)
    if rec['wall_started'] and rec['wall_ended']:
        s=datetime.datetime.fromisoformat(rec['wall_started'].replace('Z','+00:00'))
        e=datetime.datetime.fromisoformat(rec['wall_ended'].replace('Z','+00:00'))
        rec['wallclock_s']=(e-s).total_seconds()
    log=jl(b+'exp/log.jsonl') or []
    subs=[x for x in log if x.get('action')=='submit']
    rec['submit_csim']=sum(1 for x in subs if x.get('mode')=='csim')
    rec['submit_csynth']=sum(1 for x in subs if x.get('mode')=='csynth')
    rec['submit_other']=sum(1 for x in subs if x.get('mode') not in ('csim','csynth'))
    rec['exp_dirs']=len([x for x in os.listdir(b+'exp') if os.path.isdir(b+'exp/'+x)]) if os.path.isdir(b+'exp') else None
    rec['exp_post_dirs']=len(os.listdir(b+'exp_post')) if os.path.isdir(b+'exp_post') else None
    rec['variant']=(open(b+'variant.txt').read().strip() if os.path.exists(b+'variant.txt') else None)
    rec['exit_code']=(open(b+'exit_code').read().strip() if os.path.exists(b+'exit_code') else None)
    return rec

if __name__=='__main__':
    rows=main()
    json.dump(rows, open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'rows.json'),'w'), indent=1, default=str)
    print(len(rows))
