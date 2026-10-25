import json,re,math
from pathlib import Path
L=Path("/scratch/hpc-prf-llmfpga/asa582/projects/tmp/c2hls/lsim/soda")
D=Path("/scratch/hpc-prf-llmfpga/asa582/projects/workspaces/c2hls_claude_code/study/soda/dse")
RX=re.compile(r"\|\s*Verilog\|\s*(\w+)\|\s*(\d+)\|\s*(\d+)\|\s*(\d+)\|")
def cosim(d):
    for f in sorted(d.glob("cosim/*cosim.rpt")):
        m=RX.search(f.read_text(errors="replace"))
        if m and m.group(1)=="Pass": return int(m.group(4))
for kind,pre in (("designs_patched","pat_"),("designs","")):
  for d in sorted((D/kind).iterdir()):
    c=cosim(d)
    j=L/(pre+d.name)/"lsim_v2023.json"
    e=None
    if j.exists():
        try: e=json.load(open(j)).get("end")
        except Exception: e="bad"
    if c or isinstance(e,int):
        n=len(re.findall(r"#pragma HLS interface s_axilite port = (?!return)\w+",(d/"kernel.cpp").read_text())) if (d/"kernel.cpp").exists() else None
        mp=None
        if isinstance(e,int) and n is not None:
            P=6+n;K=27+12*n;mp=P*math.ceil((e-14)/P)+K
        err=f"{100*(e-c)/c:+.3f}" if c and isinstance(e,int) else ""
        errm=f"{100*(mp-c)/c:+.3f}" if c and mp else ""
        print(kind,d.name,c,e,mp,err,errm,sep=",")
