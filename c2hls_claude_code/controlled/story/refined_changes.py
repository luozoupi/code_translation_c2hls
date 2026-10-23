#!/usr/bin/env python3
"""Build refined_changes.html: every change from stock c2hls flash (Expertise) to neutral Refined-Expertise, verbatim.

Prompts and the resource block come from what the flow actually renders: refined_dump/stock.txt and
refined_dump/neutral.txt are the `run_flash_cc.py --dry-run --show-refined` outputs for
  stock:   --flavor skills
  neutral: --flavor skills --refined --refined-neutral
The skill changes are a field-by-field diff of the source pack against the neutral pack. The balance gate text is
taken from flash_refined.py.
"""
import difflib, html, json, re, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CC = HERE.parent.parent
sys.path.insert(0, str(CC / "flash_refined"))
import flash_refined  # noqa: E402
import make_neutral_skills  # noqa: E402

SRC_PACK = Path("/scratch/hpc-prf-llmfpga/asa582/projects/workspaces/code_translation_c2hls/hls_full_optimization_skills_schema_1_1_package/"
                "skills_ii_target_miss_solutions_added(90skills)_plus_gemm_flatten_v2.json")
NEU_PACK = flash_refined.NEUTRAL_PACK


def section(path, name):
    text = Path(path).read_text().split("\n")
    start = text.index(f"=== {name} ===") + 1
    end = next(i for i in range(start, len(text)) if text[i].startswith("=== "))
    return text[start:end]


def esc(lines):
    return html.escape("\n".join(lines)) if lines else "<span class='none'>(nothing)</span>"


def hunks(old, new):
    rows = []
    for op, a1, a2, b1, b2 in difflib.SequenceMatcher(None, old, new, autojunk=False).get_opcodes():
        if op == "equal":
            continue
        kind = {"replace": "Changed", "delete": "Removed", "insert": "Added"}[op]
        rows.append(f"<tr><td class='kind'>{kind}</td><td><pre class='verb old'>{esc(old[a1:a2])}</pre></td>"
                    f"<td><pre class='verb new'>{esc(new[b1:b2])}</pre></td></tr>")
    return rows


def table(rows, left="Stock c2hls flash (Expertise)", right="Refined-Expertise (neutral)"):
    return ("<div class='tablewrap'><table class='diff'><thead><tr><th scope='col'></th>"
            f"<th scope='col'>{left}</th><th scope='col'>{right}</th></tr></thead><tbody>" + "".join(rows)
            + "</tbody></table></div>")


def field_text(v):
    if isinstance(v, list):
        return "\n".join(f"- {x}" if isinstance(x, str) else f"- {json.dumps(x)}" for x in v)
    return v if isinstance(v, str) else json.dumps(v)


def skills_diff():
    src = {s["id"]: s for s in json.loads(SRC_PACK.read_text())["skills"]}
    neu = {s["id"]: s for s in json.loads(NEU_PACK.read_text())["skills"]}
    shown = ("pattern", "strategy", "required_steps", "guards", "template")
    edits, added = [], []
    for sid, s in neu.items():
        if sid not in src:
            body = "\n".join(f"{k}: {field_text(s.get(k))}" for k in ("id",) + shown if s.get(k))
            added.append(f"<tr><td class='kind'>Added</td><td><pre class='verb old'><span class='none'>(new skill)</span></pre></td>"
                         f"<td><pre class='verb new'>{html.escape(body)}</pre></td></tr>")
            continue
        for k in shown:
            a, b = src[sid].get(k), s.get(k)
            if a == b:
                continue
            if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
                for x, y in zip(a, b):
                    if x != y:
                        edits.append((sid, k, field_text([x]), field_text([y])))
            else:
                edits.append((sid, k, field_text(a), field_text(b)))
    rows = [f"<tr><td class='kind'>Changed<br><code>{html.escape(sid)}</code><br>{html.escape(k)}</td>"
            f"<td><pre class='verb old'>{html.escape(a)}</pre></td><td><pre class='verb new'>{html.escape(b)}</pre></td></tr>"
            for sid, k, a, b in edits]
    dropped = sorted(set(src) - set(neu))
    return rows, added, dropped, len(src), len(neu)


def main():
    d = HERE / "refined_dump"
    st, ne = d / "stock.txt", d / "neutral.txt"
    sys_rows = hunks(section(st, "flash system prompt"), section(ne, "flash system prompt"))
    usr_rows = hunks(section(st, "flash user prompt template"), section(ne, "flash user prompt template"))
    blk_rows = hunks(section(st, "flash resource block (sample report)"), section(ne, "flash resource block (sample report)"))
    sk_rows, sk_added, dropped, n_src, n_neu = skills_diff()
    gate_rule = (f"latency_cycles > {flash_refined.BALANCE_RATIO:g} × compute floor, where compute floor = "
                 f"(I·J·K) / (DSP / {flash_refined.DSP_PER_MAC:g})  →  reject")
    gate_msg = ("compute/transfer imbalance: latency {lat} cycles is {ratio}x the compute floor {floor} cycles "
                "(= {macs} MACs / ({dsp} DSP / 5 DSP per MAC)); the limit is 1.5x. The datapath is waiting on memory. "
                "Longest pipelined loops per call: {5 longest loops with latency, II and trip count}.")
    out = f"""
<p>Everything below is generated from the flow itself: the prompts and resource block are what c2hls renders for the
Expertise step and for this step, compared line by line. Lines not shown are identical.</p>
<details open><summary>System prompt ({len(sys_rows)} changes)</summary>{table(sys_rows)}</details>
<details open><summary>User prompt ({len(usr_rows)} changes)</summary>{table(usr_rows)}</details>
<details open><summary>Resource block shown to the model (sample report, {len(blk_rows)} changes)</summary>{table(blk_rows)}</details>
<details open><summary>Skill pack: {n_src} &rarr; {n_neu} skills ({len(sk_rows)} wording changes, {len(sk_added)} skills added{", " + str(len(dropped)) + " dropped" if dropped else ""})</summary>
<p class='meta'>Compared with the source pack <code>&hellip;plus_gemm_flatten_v2.json</code>, which the refined pack is built from.
The Expertise step itself uses the 90-skill + gemm-flatten v1 pack. Every wording change is listed, including each
place where &ldquo;resource explosion&rdquo; is redefined.</p>
{table(sk_rows + sk_added, left="Source pack", right="Neutral refined pack")}</details>
<details open><summary>Balance gate (new check after csynth)</summary>
<table class='diff'><tbody>
<tr><td class='kind'>Rule</td><td colspan='2'><pre class='verb new'>{html.escape(gate_rule)}</pre></td></tr>
<tr><td class='kind'>Rejection reason</td><td colspan='2'><pre class='verb new'>{html.escape(gate_msg)}</pre></td></tr>
<tr><td class='kind'>Repair prompt</td><td colspan='2'><pre class='verb new'>{html.escape(flash_refined.NEUTRAL_FIX)}</pre></td></tr>
</tbody></table></details>
<details open><summary>Flow settings</summary><table class='diff'><tbody>
<tr><td class='kind'>Changed</td><td><pre class='verb old'>STEP_REGRESSION_THRESHOLDS["flash"] = {{"latency": 1.50, "resources": {{"dsp": 12.0, "bram": 6.0, "ff": 8.0, "lut": 6.0, "default": 4.0}}}}
(a flash design may be up to 1.5&times; slower; DSP may grow at most 12&times;, BRAM/LUT 6&times;, FF 8&times;, others 4&times;)</pre></td>
<td><pre class='verb new'>C2HLS_STEP_REGRESSION_THRESHOLDS_JSON = {html.escape(flash_refined.FLOW_ENV["C2HLS_STEP_REGRESSION_THRESHOLDS_JSON"])}
(latency must not get worse; resource ratios removed)</pre></td></tr>
<tr><td class='kind'>Added</td><td><pre class='verb old'><span class='none'>(no device cap)</span></pre></td><td><pre class='verb new'>C2HLS_FLASH_MAX_DSP = 9024 (reject a design above the device)</pre></td></tr>
<tr><td class='kind'>Unchanged</td><td colspan='2'><pre class='verb'>No DSP floor, no fill gate (C2HLS_FLASH_DSP_REDO, C2HLS_FLASH_MIN_DSP, C2HLS_FLASH_DSP_FILL_PCT unset)</pre></td></tr>
</tbody></table></details>
"""
    (HERE / "refined_changes.html").write_text(out)
    print(len(sys_rows), len(usr_rows), len(blk_rows), len(sk_rows), len(sk_added), dropped)


if __name__ == "__main__":
    main()
