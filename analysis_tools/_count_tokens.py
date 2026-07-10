"""Total LLM tokens per sweep experiment, from each cell's llm_usage block."""
import json, glob, os

dirs = sorted(d for d in glob.glob("results_matrix*") if os.path.isdir(d))
print(f'{"sweep dir":52}{"cells":>6}{"calls":>7}{"in_tok":>13}{"out_tok":>11}{"total_tok":>13}{"tot/cell":>10}')
print("-" * 112)
grand = 0
for d in dirs:
    files = set(glob.glob(f"{d}/hlsfactory_*/**/*_results.json", recursive=True))
    ti = to = tt = calls = cells = 0
    for f in files:
        try: lu = (json.load(open(f)).get("llm_usage") or {})
        except Exception: continue
        cells += 1
        ti += lu.get("input_tokens") or 0
        to += lu.get("output_tokens") or 0
        tt += lu.get("total_tokens") or 0
        calls += lu.get("calls") or 0
    if cells == 0:
        continue
    grand += tt
    print(f'{d[:52]:52}{cells:>6}{calls:>7}{ti:>13,}{to:>11,}{tt:>13,}{(tt//cells):>10,}')
print("-" * 112)
print(f'GRAND TOTAL tokens across all sweeps: {grand:,}')
