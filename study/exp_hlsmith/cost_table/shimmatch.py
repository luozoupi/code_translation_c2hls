import json, glob, os, datetime
ROOT='/scratch/asa582/workspaces/c2hls_claude_code'
def ts(s):
    return datetime.datetime.fromisoformat(s.replace('Z','+00:00').split('+')[0])
def load_shim():
    rows=[]
    for cj in glob.glob(ROOT+'/logs/shim*/calls.jsonl'):
        d=os.path.dirname(cj)
        for l in open(cj):
            l=l.strip()
            if not l: continue
            try: r=json.loads(l)
            except Exception: continue
            r['_dir']=d; r['_end']=ts(r['t']); rows.append(r)
    return rows
def ctrl_calls(path):
    out=[]
    for l in open(path):
        l=l.strip()
        if not l: continue
        try: r=json.loads(l)
        except Exception: continue
        if 'prompt_tokens' in r: out.append(r)
    return out
if __name__=='__main__':
    import sys
    rows=load_shim()
    print('shim rows',len(rows))
    from collections import Counter
    print(Counter(os.path.basename(r['_dir']) for r in rows))
