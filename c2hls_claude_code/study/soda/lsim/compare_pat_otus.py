import json,re,math
from pathlib import Path
L=Path("/scratch/hpc-prf-llmfpga/asa582/projects/tmp/c2hls/lsim/soda")
D=Path("/scratch/hpc-prf-llmfpga/asa582/projects/workspaces/c2hls_claude_code/study/soda/dse")
F=D.parent/"inv3d"/"frp"   # lsim_frp.py results (frp_in) for the 3D designs that have a cosim
def end_of(j,key="end"):
    if not j.exists(): return None
    try: return json.load(open(j)).get(key)
    except Exception: return "bad"
RX=re.compile(r"\|\s*Verilog\|\s*(\w+)\|\s*(\d+)\|\s*(\d+)\|\s*(\d+)\|")
def cosim(d):
    for f in sorted(d.glob("cosim/*cosim.rpt")):
        m=RX.search(f.read_text(errors="replace"))
        if m and m.group(1)=="Pass": return int(m.group(4))
print("set,design,cosim,lsim_end,lsim_mapped,err_end_pct,err_mapped_pct,lsim_source,lsim_end_nofrp")
for kind,pre in (("designs_patched","pat_"),("designs","")):
  for d in sorted((D/kind).iterdir()):
    c=cosim(d)
    e0=end_of(L/(pre+d.name)/"lsim_v2023.json")   # no FRP rule
    # FRP-fixed end: fork rerun (lsim_frp.json), else lsim_frp.py on the same trace; FRP only occurs in heat3d/jacobi3d
    e,src=e0,"v2023"
    if pre and isinstance(end_of(L/(pre+d.name)/"lsim_frp.json"),int): e,src=end_of(L/(pre+d.name)/"lsim_frp.json"),"lsim_frp"
    elif pre and isinstance(end_of(F/(d.name+".json"),"frp_in"),int): e,src=end_of(F/(d.name+".json"),"frp_in"),"inv3d_frp_in"
    if c or isinstance(e,int):
        n=len(re.findall(r"#pragma HLS interface s_axilite port = (?!return)\w+",(d/"kernel.cpp").read_text())) if (d/"kernel.cpp").exists() else None
        mp=None
        if isinstance(e,int) and n is not None:
            P=6+n;K=27+12*n;mp=P*math.ceil((e-14)/P)+K
        err=f"{100*(e-c)/c:+.3f}" if c and isinstance(e,int) else ""
        errm=f"{100*(mp-c)/c:+.3f}" if c and mp else ""
        print(kind,d.name,c,e,mp,err,errm,src if isinstance(e,int) else "",e0 if e0!=e else "",sep=",")
