"""lsim_fix7.json (fork patches 0005-0007) vs cosim for the rows of lsim_vs_cosim_otus_20261027_frp.csv that have a cosim.
  compare_fix7.py OUT.csv      (run from this folder; mapping as in compare_soda.py)
"""
import csv,json,math,re,sys
from pathlib import Path
sys.path.insert(0,'.')
from compare_soda import nargs, DSE
LS=Path('/scratch/hpc-prf-llmfpga/asa582/projects/tmp/c2hls/lsim/soda')
rows=list(csv.DictReader(open('lsim_vs_cosim_otus_20261027_frp.csv')))
out=[]
for r in rows:
    if r['cosim'] in ('','None'): continue
    pat=r['set']=='designs_patched'; name=('pat_' if pat else '')+r['design']
    d=DSE/r['set']/r['design']; cy=int(r['cosim'])
    def get(tag):
        f=LS/name/f'lsim_{tag}.json'
        if not f.exists(): return None
        j=json.loads(f.read_text()); return j.get('end'), j.get('rc'), j.get('error') or j.get('err')
    n=nargs(d); P,K=6+n,27+12*n
    e,rc,err=get('fix7') or (None,None,'missing')
    m=P*math.ceil((e-14)/P)+K if e else None
    out.append(dict(set=r['set'],design=r['design'],cosim=cy,fix7_end=e,fix7_mapped=m,
        diff=None if m is None else m-cy,err_pct=None if m is None else round(100*(m-cy)/cy,3),
        P=P,prev_mapped=r['lsim_mapped'],prev_err_pct=r['err_mapped_pct'],rc=rc,note=err))
w=csv.DictWriter(open(sys.argv[1],'w'),list(out[0]));w.writeheader();w.writerows(out)
for o in out: print(o['set'][:9],o['design'],o['cosim'],o['fix7_mapped'],o['diff'],'P',o['P'],'prev',o['prev_err_pct'],o['rc'],o['note'] or '')
