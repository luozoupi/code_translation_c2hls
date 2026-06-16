"""Quick 4-way technique comparison:
  OFF  = phase 8 flash, skills off
  OLD  = phase 8 flash, skills on (base)
  EXT  = phase 9 flash, skills on (base + extension)
  MULTI = phase 10 multistep, skills on (base only)

Aggregates inferred_skill_id and category presence per setup. Mirrors the
previous _off_vs_on_diff.py structure but adds MULTI as a 4th column.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # repo root (script lives in analysis/)
SETUPS = {
    "OFF": ROOT / "results_matrix_u280_fullcosim" / "{bench}" / "sonnet__flash__noskills",
    "OLD": ROOT / "results_matrix_u280_fullcosim" / "{bench}" / "sonnet__flash__skills",
    "EXT": ROOT / "results_matrix_u280_fullcosim_extended" / "{bench}" / "sonnet__flash__skills",
    "MULTI": ROOT / "results_matrix_u280_multistep_old_skills" / "{bench}" / "sonnet__multistep__skills",
}


def _load_all() -> dict[str, dict[str, dict]]:
    out: dict[str, dict[str, dict]] = {s: {} for s in SETUPS}
    phase8_root = ROOT / "results_matrix_u280_fullcosim"
    for bench_dir in sorted(phase8_root.iterdir()):
        if not bench_dir.is_dir():
            continue
        bench = bench_dir.name
        for setup_name, tmpl in SETUPS.items():
            sidecar = Path(str(tmpl).format(bench=bench)) / f"{bench}_techniques_detected.json"
            if sidecar.exists():
                out[setup_name][bench] = json.loads(sidecar.read_text())
    return out


def _aggregate(data: dict, key: str) -> Counter:
    c = Counter()
    for payload in data.values():
        if key == "_skill_ids":
            for sid in payload["inferred_skill_ids"]:
                c[sid] += 1
        elif key == "_categories":
            for cat, present in payload["categories"].items():
                if present:
                    c[cat] += 1
    return c


def _pct(n, d): return f"{100*n/d:5.1f}%" if d else "  n/a"


def main():
    data = _load_all()
    sizes = {s: len(b) for s, b in data.items()}
    print(f"Cells: OFF={sizes['OFF']}  OLD={sizes['OLD']}  EXT={sizes['EXT']}  MULTI={sizes['MULTI']}\n")

    # Skill-id freq
    all_ids = set()
    for d in data.values():
        for p in d.values():
            all_ids.update(p["inferred_skill_ids"])
    freq = {s: _aggregate(data[s], "_skill_ids") for s in data}

    print("=" * 105)
    print("  Skill-id frequency by setup")
    print(f"  {'skill_id':<55}{'OFF':>10}{'OLD':>10}{'EXT':>10}{'MULTI':>10}    M-O    M-N")
    print("-" * 105)

    def maxf(sid):
        return max((freq[s][sid] / sizes[s] if sizes[s] else 0) for s in data)
    for sid in sorted(all_ids, key=lambda x: -maxf(x)):
        off = freq["OFF"][sid]/sizes["OFF"] if sizes["OFF"] else 0
        old = freq["OLD"][sid]/sizes["OLD"] if sizes["OLD"] else 0
        ext = freq["EXT"][sid]/sizes["EXT"] if sizes["EXT"] else 0
        mlt = freq["MULTI"][sid]/sizes["MULTI"] if sizes["MULTI"] else 0
        d_mo = (mlt-old)*100  # MULTI - OLD (does multistep change vs flash OLD?)
        d_mn = (mlt-off)*100  # MULTI - OFF (does multistep beat no-skills?)
        flag = ""
        if abs(d_mo) >= 15: flag += " *MO"
        if abs(d_mn) >= 15: flag += " *MN"
        print(f"  {sid:<55}{_pct(freq['OFF'][sid], sizes['OFF']):>10}"
              f"{_pct(freq['OLD'][sid], sizes['OLD']):>10}"
              f"{_pct(freq['EXT'][sid], sizes['EXT']):>10}"
              f"{_pct(freq['MULTI'][sid], sizes['MULTI']):>10}    "
              f"{d_mo:+5.1f}pt {d_mn:+5.1f}pt{flag}")

    # Categories
    print()
    print("=" * 95)
    print("  Category presence by setup")
    sample = next(iter(data["OFF"].values()))
    cats = list(sample["categories"].keys())
    print(f"  {'category':<25}{'OFF':>10}{'OLD':>10}{'EXT':>10}{'MULTI':>10}    M-O    M-N")
    print("-" * 95)
    for cat in cats:
        cnt = {s: 0 for s in data}
        for setup, benches in data.items():
            for p in benches.values():
                if p["categories"].get(cat):
                    cnt[setup] += 1
        d_mo = (cnt["MULTI"]/sizes["MULTI"] - cnt["OLD"]/sizes["OLD"])*100 if sizes["MULTI"] and sizes["OLD"] else 0
        d_mn = (cnt["MULTI"]/sizes["MULTI"] - cnt["OFF"]/sizes["OFF"])*100 if sizes["MULTI"] and sizes["OFF"] else 0
        print(f"  {cat:<25}{_pct(cnt['OFF'], sizes['OFF']):>10}"
              f"{_pct(cnt['OLD'], sizes['OLD']):>10}"
              f"{_pct(cnt['EXT'], sizes['EXT']):>10}"
              f"{_pct(cnt['MULTI'], sizes['MULTI']):>10}    "
              f"{d_mo:+5.1f}pt {d_mn:+5.1f}pt")


if __name__ == "__main__":
    main()
