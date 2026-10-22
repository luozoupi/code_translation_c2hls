"""DSP window and seeded stages for flash_refined (applied after flash_refined.apply; c2hls stays read only).

apply_window(lo, hi) replaces flash_refined's ">= 90% of the device" fill target with a DSP window:
  * csynth DSP must be in [lo, hi]. hi is a hard roof (c2hls's ceiling gate, C2HLS_FLASH_MAX_DSP=hi).
    Below lo is rejected by the fill check (exact lo, not a rounded percent), unless another resource
    is >= 90% of the device and blocks more DSP (c2hls's own escape, unchanged).
  * Every place that told the model "fill the device" now states the window instead: the flash
    DSP-redo and ceiling guidance blocks, the ceiling reject text and repair template, the DEVICE
    BUDGET block, the refined system/user prompt sentences, and the fill/balance repair template.
  * The compute/transfer balance check (version 2) is unchanged and still runs after the window check.
An optional stage note (the ladder's "double it" / "increase it" instruction) is appended to the flash
guidance blocks.

apply_seed(path) makes Phase B start from an existing kernel instead of an LLM translation: c2hls
synthesizes and csims it as the step baseline, then the flash step rewrites it.
"""
from __future__ import annotations

import os
from pathlib import Path

DEVICE_DSP = 9024
BLOCK_PCT = 0.90  # c2hls flash_block_pct default: another resource this full blocks more DSP


def pct(n: int) -> str:
    return f"{100.0 * n / DEVICE_DSP:.0f}%"


def window_text(lo: int, hi: int) -> str:
    return f"{lo}–{hi} DSP ({pct(lo)}–{pct(hi)} of the U280's {DEVICE_DSP})"


def apply_window(lo: int, hi: int, stage_note: str = "", log=print) -> dict:
    import autosa_flow_gates as gates
    import c2hls
    import flash_refined
    import prompt_c2hls

    assert 0 < lo <= hi <= DEVICE_DSP
    win = window_text(lo, hi)
    os.environ.update({
        "C2HLS_FLASH_DSP_REDO": "1",
        "C2HLS_FLASH_MIN_DSP": str(min(300, lo)),
        "C2HLS_FLASH_MAX_DSP": str(hi),
        # c2hls's percent fill check is replaced by the exact lower bound below; 1% keeps its
        # resource-cap part (every resource < 100%) and never rejects on DSP by itself.
        "C2HLS_FLASH_DSP_FILL_PCT": "1",
        "FLASH_WINDOW_LO": str(lo), "FLASH_WINDOW_HI": str(hi),
    })

    # 1. Guidance blocks in the first flash prompt (c2hls imports these at call time).
    def redo_guidance(min_dsp: int, max_dsp: int) -> str:
        return (f"FLASH DSP WINDOW: csynth DSP must be between {lo} and {hi} ({pct(lo)}–{pct(hi)} of the "
                f"U280's {DEVICE_DSP} DSP). {hi} is a hard roof: a kernel above it is rejected. A kernel below "
                f"{lo} is rejected unless BRAM/FF/LUT/URAM is already >= {BLOCK_PCT:.0%} of the device and blocks "
                f"more DSP. Every resource stays strictly below 100%. Among legal kernels in the window, the lowest "
                f"latency_cycles wins. Keep the exact extern C top and kernel.h ABI. Do not clone AutoSA kernel0.")

    def ceiling_guidance(max_dsp: int) -> str:
        return (f"HARD DSP ROOF for this flash rewrite: csynth DSP must be <= {hi} ({pct(hi)} of the U280's "
                f"{DEVICE_DSP} DSP). A kernel above {hi} is rejected even if it compiles.")

    original_ceiling_error = gates.flash_dsp_ceiling_error

    def ceiling_error(report, *, max_dsp: int):
        err = original_ceiling_error(report, max_dsp=max_dsp)
        if err:
            err = err.replace("The Alveo U280 has 9024 DSP; a point above that cannot place.",
                              f"The DSP roof for this run is {hi} ({pct(hi)} of the U280's {DEVICE_DSP} DSP).")
        return err

    gates.flash_dsp_redo_initial_guidance = redo_guidance
    gates.flash_dsp_ceiling_initial_guidance = ceiling_guidance
    gates.flash_dsp_ceiling_error = ceiling_error
    c2hls.hls_flash_dsp_ceiling_fix = c2hls.hls_flash_dsp_ceiling_fix.replace(
        "U280 has 9024 DSP.", f"That is the DSP roof for this run ({pct(hi)} of the U280's {DEVICE_DSP} DSP).")

    # 2. Exact lower bound, ahead of flash_refined's balance check (which is already wrapped in).
    wrapped_fill = c2hls.flash_dsp_fill_reject_error

    def window_or_balance(step_name, report=None):
        if step_name == "flash" and report:
            utils = gates._resource_utils(report)
            dsp = (utils.get("dsp") or {}).get("used")
            blocked = [f"{k} {100 * r['utilization']:.1f}%" for k, r in utils.items()
                       if k != "dsp" and r["utilization"] >= BLOCK_PCT]
            if dsp is not None and dsp < lo and not blocked:
                others = ", ".join(f"{k} {100 * r['utilization']:.1f}%" for k, r in utils.items() if k != "dsp")
                return (f"DSP below the window: csynth DSP={int(dsp)} ({pct(int(dsp))} of {DEVICE_DSP}); the "
                        f"window is {win}. Headroom remains ({others}). Latency was "
                        f"{report.get('latency_cycles')} cycles. Add parallel MACs until DSP is inside the window; "
                        f"stay at or below the {hi} roof.")
        return wrapped_fill(step_name, report)

    c2hls.flash_dsp_fill_reject_error = window_or_balance
    fix = c2hls.hls_flash_dsp_fill_fix
    for old, new in (
        ("Device DSP budget {max_dsp} (fill target about {fill_pct}%).", f"DSP window for this run: {win}; {hi} is a hard roof."),
        ("until DSP is near the fill\n  target or another resource would exceed 100%.",
         f"until DSP is inside the window\n  ({lo}–{hi}) or another resource would exceed 100%."),
    ):
        if old in fix:
            fix = fix.replace(old, new)
    c2hls.hls_flash_dsp_fill_fix = fix

    # 3. DEVICE BUDGET block: DSP line states the window.
    original_budget = flash_refined._device_budget_block

    def budget_block(step_name, current_report, part, capacity):
        text = original_budget(step_name, current_report, part, capacity)
        lines = []
        for line in text.splitlines():
            if line.strip().startswith("dsp"):
                cur = current_report.get("dsp")
                cur_s = f"current={int(cur)}  " if cur is not None else ""
                line = f"  dsp  : {cur_s}target window {lo}–{hi} ({pct(lo)}–{pct(hi)} of device {DEVICE_DSP}); {hi} is a hard roof"
            elif line.startswith("Accept rule:"):
                line = ("Accept rule: latency improves AND DSP is inside the window AND every other resource stays "
                        "strictly below 100% of the device.")
            lines.append(line)
        return "\n".join(lines)

    flash_refined._device_budget_block = budget_block

    # 4. Prompt sentences that said "most of the device's DSP" / "toward the device budget".
    edits = [
        ("prompt_c2hls.q_optimize_flash",
         "the parallel datapath should use most of the device's DSP\nbudget while every resource stays below 100%; leaving most of the device idle\nis not the best endpoint.",
         f"the parallel datapath should use the DSP window given\nbelow ({win}) while every resource stays below 100%."),
        ("prompt_c2hls.Instruction_c2hls_flash",
         "scale the parallel datapath toward the device budget: a small datapath on a\n  large device is a performance bug, not a safe choice.",
         f"size the parallel datapath to the DSP window ({win}):\n  a design below the window or above its roof is rejected."),
    ]
    applied = []
    for name, old, new in edits:
        attr = name.split(".")[1]
        text = getattr(prompt_c2hls, attr)
        if text.count(old) == 1:
            setattr(prompt_c2hls, attr, text.replace(old, new))
            applied.append(attr)
    assert len(applied) == 2, f"window prompt edits did not match: {applied}"

    # 5. Stage note (ladder) appended to the flash guidance blocks.
    if stage_note:
        original_blocks = c2hls.flash_step_guidance_extra_blocks

        def blocks():
            return list(original_blocks()) + [stage_note]

        c2hls.flash_step_guidance_extra_blocks = blocks

    info = {"window": [lo, hi], "window_pct": [pct(lo), pct(hi)], "stage_note": stage_note}
    log(f"dsp_window applied: {info}")
    return info


def apply_seed(path: str, log=print) -> dict:
    """Phase B returns this kernel instead of asking the LLM to translate plain.cpp."""
    import c2hls

    code = Path(path).read_text()

    def seeded_translate(self):
        self.orch._append_history("system", f"[Phase B] Seeded from {path} (ladder stage input; no LLM call).")
        return code

    c2hls.TranslatorAgent.translate_initial = seeded_translate
    info = {"seed": str(path), "seed_bytes": len(code)}
    log(f"seed applied: {info}")
    return info
