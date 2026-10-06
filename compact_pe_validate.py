"""HLS-validate one compact PE candidate (csynth + csim; cosim off)."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from c2hls import _run_synth_csim_cosim, compile_check_cpp
from compact_pe_instantiate import instantiate_mm
from compact_pe_pack_instantiate import emit_pack_header, emit_pack_testbench
from compact_pe_search import search_candidate_id
from post_flash_pe_recipe import PeRecipe
from post_flash_stream import architecture_ok_for_recipe


def _csim_passed(csim: Any) -> bool:
    if not isinstance(csim, dict):
        return False
    return bool(csim.get("success") or csim.get("passed"))


def _terminal_result(
    rec: PeRecipe,
    *,
    hls_csim_pass: bool,
    hls_csynth_pass: bool,
    architecture_ok: bool,
    csynth_latency: Any = None,
    csynth_interval: Any = None,
    csynth_dsp: Any = None,
    reason: str = "",
) -> dict[str, Any]:
    return {
        "cand_id": search_candidate_id(rec),
        "pe": rec.pe,
        "simd": rec.simd,
        "pack_bits": rec.pack_bits,
        "expected_dsp": rec.expected_dsp,
        "layout": rec.layout,
        "pe_i": rec.pe_i if rec.pe_i else rec.pe,
        "pe_j": rec.pe_j,
        "hls_csim_pass": hls_csim_pass,
        "hls_csynth_pass": hls_csynth_pass,
        "architecture_ok": architecture_ok,
        "csynth_latency": csynth_latency,
        "csynth_interval": csynth_interval,
        "csynth_dsp": csynth_dsp,
        "reason": reason,
    }


def validate_candidate(
    rec: PeRecipe,
    out_root: Path,
    header_code: str,
    testbench_code: str,
    *,
    header_name: str = "kernel.h",
    top_function: str = "autosa_mm",
    part: str = "xcu280-fsvh2892-2L-e",
    clock_ns: float = 3.33,
) -> dict:
    cand_dir = out_root / search_candidate_id(rec)
    result_path = cand_dir / "result.json"
    if result_path.is_file():
        return json.loads(result_path.read_text(encoding="utf-8"))

    cand_dir.mkdir(parents=True, exist_ok=True)
    if rec.layout in ("pack", "io4", "io5"):
        header_code = emit_pack_header()
        testbench_code = emit_pack_testbench()
        top_function = "autosa_mm_pack"
    code = instantiate_mm(rec)
    (cand_dir / "kernel.cpp").write_text(code, encoding="utf-8")
    (cand_dir / "candidate.json").write_text(
        json.dumps(asdict(rec), indent=2) + "\n", encoding="utf-8"
    )

    ok, err = compile_check_cpp(
        code,
        header_code,
        header_name,
        work_dir=str(cand_dir),
    )
    if not ok:
        result = _terminal_result(
            rec,
            hls_csim_pass=False,
            hls_csynth_pass=False,
            architecture_ok=False,
            reason=err or "compile failed",
        )
        result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        return result

    outcome = _run_synth_csim_cosim(
        hls_code=code,
        header_code=header_code,
        header_name=header_name,
        top_function=top_function,
        part=part,
        clock_ns=clock_ns,
        extra_files=[],
        testbench_code=testbench_code,
        run_csim_check=True,
        run_cosim_check=False,
    )
    synth = outcome.get("synth") if isinstance(outcome, dict) else None
    if not isinstance(synth, dict):
        synth = {}
    csim = outcome.get("csim") if isinstance(outcome, dict) else None
    report = synth.get("report") if isinstance(synth.get("report"), dict) else {}
    hls_csynth_pass = bool(synth.get("success"))
    hls_csim_pass = _csim_passed(csim)
    arch_ok = bool(
        hls_csynth_pass and architecture_ok_for_recipe(code, report, rec)
    )
    reason = ""
    if not hls_csynth_pass:
        reason = "csynth failed"
    elif not hls_csim_pass:
        reason = "csim failed"
    elif not arch_ok:
        reason = "architecture miss"
    result = _terminal_result(
        rec,
        hls_csim_pass=hls_csim_pass,
        hls_csynth_pass=hls_csynth_pass,
        architecture_ok=arch_ok,
        csynth_latency=report.get("latency_cycles"),
        csynth_interval=report.get("interval"),
        csynth_dsp=report.get("dsp"),
        reason=reason,
    )
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result
