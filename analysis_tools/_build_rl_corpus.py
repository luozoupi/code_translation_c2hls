"""Build a cosim-verified SFT + DPO corpus from ALL enhanced-framework agent
trajectories (Opus + Sonnet [+ GPT-5.5 when it lands]), across BOTH the
flash/one-shot arms and the multistep arms (per-step kernels).

Selection: ALL cosim-verified kernels, deduplicated by (bench, kernel content hash).
Reward:    cosim latency_cycles (lower = better), gated on csim PASS + cosim PASS,
           vitis flow only (csynth never used as reward -- gameable).

Outputs (rl_dataset/):
  sft.jsonl         one chat example per distinct verified kernel {messages, meta}
  sft.train.jsonl   90% deterministic split
  sft.val.jsonl     10% deterministic split (by kernel hash -> identical kernels never cross)
  dpo.jsonl         preference pairs per bench (chosen=faster, rejected=>=10% slower distinct)
  manifest.json     stats + provenance

Self-contained: prompts (C code + header + target) are baked into each record, so
the JSONL ships to a training node with no Vitis/Argo/benchmarks dependency.

Run from anywhere: python3 analysis_tools/_build_rl_corpus.py
"""
import json, glob, os, hashlib
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
PART = "xcu280-fsvh2892-2L-e"
OUTDIR = "rl_dataset"
os.makedirs(OUTDIR, exist_ok=True)

SYSTEM = ("You are an expert in Xilinx Vitis HLS. Convert the given plain C/C++ "
          "kernel into synthesizable, high-performance Vitis HLS for the AMD Alveo "
          "U280 (xcu280-fsvh2892-2L-e, vitis kernel flow, 3.33 ns). Add HLS "
          "INTERFACE pragmas and performance pragmas (PIPELINE/UNROLL/ARRAY_PARTITION, "
          "on-chip buffering) while preserving the algorithm and the exact top-level "
          "signature the testbench expects. Return only the HLS C++ in one code block.")

_prompt_cache = {}
def make_user_prompt(bench):
    if bench in _prompt_cache:
        return _prompt_cache[bench]
    bd = f"benchmarks/{bench}"
    try:
        meta = json.load(open(f"{bd}/metadata.json"))
        plain = open(f"{bd}/plain.cpp").read()
        hdrn = meta.get("header_file") or "kernel.h"
        hdr = open(f"{bd}/{hdrn}").read() if os.path.exists(f"{bd}/{hdrn}") else ""
        top = meta.get("hls_top") or meta.get("kernel_top") or "workload"
        p = (f"Target: {PART}, top function `{top}`.\n\n"
             f"Header `{hdrn}`:\n```c\n{hdr}\n```\n\n"
             f"Plain C kernel to convert:\n```c\n{plain}\n```")
    except Exception:
        p = None
    _prompt_cache[bench] = p
    return p

def _cyc(co):
    if not co or not co.get("passed"):
        return None
    m = co.get("measured") or {}
    return m.get("latency_cycles_avg") or m.get("latency_cycles_min") or co.get("kernel_runtime_cycles")

def _verified(csim, co):
    return bool((csim or {}).get("passed") and (co or {}).get("passed") and _cyc(co))

# bench -> hlshash -> {hls, cosim(min), sources:set, model, arm, step}
by_bench = defaultdict(dict)
mined = 0

def add(bench, hls, cyc, model, arm, step=None):
    global mined
    if not hls or not hls.strip():
        return
    h = hashlib.sha1(hls.encode("utf-8", "ignore")).hexdigest()[:12]
    src = arm + (f"#{step}" if step else "")
    rec = by_bench[bench].get(h)
    if rec is None:
        by_bench[bench][h] = {"hls": hls, "cosim": int(cyc), "hlshash": h,
                              "sources": {src}, "model": model, "arm": arm, "step": step}
        mined += 1
    else:
        rec["sources"].add(src)
        if int(cyc) < rec["cosim"]:
            rec["cosim"] = int(cyc)

for rf in glob.glob("results_matrix_u280_ENH_*/hlsfactory_*/**/*_results.json", recursive=True):
    parts = rf.replace("\\", "/").split("/")
    campaign, bench, cfg = parts[0], parts[1], parts[-2]
    cell = os.path.dirname(rf)
    model = cfg.split("__")[0] if "__" in cfg else "?"
    arm = f"{campaign}/{cfg}"
    try:
        d = json.load(open(rf, encoding="utf-8", errors="ignore"))
    except Exception:
        continue
    if rf.endswith("_multistep_results.json"):
        for i, s in enumerate(d.get("steps") or []):
            if not _verified(s.get("csim"), s.get("cosim")):
                continue
            name = s.get("step_name") or f"step{i}"
            cpp = os.path.join(cell, "steps", f"{i}_{name}.cpp")
            if not os.path.exists(cpp):
                g = (glob.glob(os.path.join(cell, "steps", f"*_{name}.cpp"))
                     or glob.glob(os.path.join(cell, "steps", f"{i}_*.cpp")))
                cpp = g[0] if g else None
            if not cpp or not os.path.exists(cpp):
                continue
            add(bench, open(cpp, encoding="utf-8", errors="ignore").read(),
                _cyc(s.get("cosim")), model, arm, step=name)
    else:
        gen = glob.glob(os.path.join(cell, "*_generated.cpp"))
        if not gen or not _verified(d.get("csim"), d.get("cosim")):
            continue
        add(bench, open(gen[0], encoding="utf-8", errors="ignore").read(),
            _cyc(d.get("cosim")), model, arm)

# ---- build SFT (all distinct verified) + DPO + 90/10 split ----
sft, dpo = [], []
for bench in sorted(by_bench):
    prompt = make_user_prompt(bench)
    if not prompt:
        continue
    recs = sorted(by_bench[bench].values(), key=lambda r: r["cosim"])
    for r in recs:
        sft.append({
            "bench": bench, "model": r["model"], "arm": r["arm"], "step": r["step"],
            "cosim_cycles": r["cosim"], "hlshash": r["hlshash"],
            "sources": sorted(r["sources"]),
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": f"```cpp\n{r['hls']}\n```"},
            ],
        })
    chosen = recs[0]
    pairs = 0
    for rej in recs[1:]:
        if rej["cosim"] >= chosen["cosim"] * 1.10 and pairs < 12:
            dpo.append({
                "bench": bench, "prompt": prompt,
                "chosen": f"```cpp\n{chosen['hls']}\n```",
                "rejected": f"```cpp\n{rej['hls']}\n```",
                "chosen_cycles": chosen["cosim"], "rejected_cycles": rej["cosim"],
                "speedup": round(rej["cosim"] / chosen["cosim"], 2),
                "chosen_arm": chosen["arm"], "rejected_arm": rej["arm"],
            })
            pairs += 1

def _val(rec):  # deterministic 10% holdout by kernel hash
    return int(rec["hlshash"], 16) % 10 == 0

def dump(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

dump(f"{OUTDIR}/sft.jsonl", sft)
dump(f"{OUTDIR}/sft.train.jsonl", [r for r in sft if not _val(r)])
dump(f"{OUTDIR}/sft.val.jsonl", [r for r in sft if _val(r)])
dump(f"{OUTDIR}/dpo.jsonl", dpo)

by_model = defaultdict(int)
for r in sft:
    by_model[r["model"]] += 1
manifest = {
    "selection": "all cosim-verified kernels, deduped by (bench, content-hash)",
    "reward": "cosim latency_cycles (lower better), gated on csim+cosim PASS, vitis flow",
    "format": "chat messages (system/user/assistant); DPO {prompt,chosen,rejected}",
    "verified_kernels_mined": mined,
    "sft_examples": len(sft),
    "sft_train": sum(1 for r in sft if not _val(r)),
    "sft_val": sum(1 for r in sft if _val(r)),
    "dpo_pairs": len(dpo),
    "benches": len(by_bench),
    "sft_by_model": dict(by_model),
    "per_bench": {b: {"distinct_kernels": len(h),
                      "best_cosim": min(x["cosim"] for x in h.values())}
                  for b, h in sorted(by_bench.items())},
}
json.dump(manifest, open(f"{OUTDIR}/manifest.json", "w"), indent=2)
print(f"mined {mined} distinct cosim-verified kernels across {len(by_bench)} benches")
print(f"SFT: {len(sft)} (train {manifest['sft_train']} / val {manifest['sft_val']})  |  DPO: {len(dpo)}")
print("by model:", dict(by_model))
