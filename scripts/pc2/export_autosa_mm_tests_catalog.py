#!/usr/bin/env python3
"""Export autosa_mm experiment + pytest catalog (knobs, results, LLM-call index).

Does not copy full transcripts (those stay in campaign *_history.json).
Writes a grep-able index: stage, role, message index, char count, preview, path.

Usage:
  python scripts/pc2/export_autosa_mm_tests_catalog.py
"""
from __future__ import annotations

import ast
import csv
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "artifacts" / "pc2"
DOCS = ROOT / "docs" / "pc2"
STAMP = "2026-09-10"
OUT_JSON = DOCS / f"{STAMP}-autosa-mm-tests-catalog.json"
OUT_CSV = DOCS / f"{STAMP}-autosa-mm-tests-catalog.csv"
OUT_MD = DOCS / f"{STAMP}-autosa-mm-tests-catalog.md"
OUT_LLM = DOCS / f"{STAMP}-autosa-mm-llm-calls.jsonl"
OUT_PYTEST = DOCS / f"{STAMP}-autosa-mm-pytest-knobs.json"

FROZEN = {
    "20260830_mmflow",
    "20260830_mm32x8",
    "20260829_123043",
    "20260909_085740",
    "20260909_101836",
    "20260909_114519",
    "20260909_121934",
    "20260909_175131",
    "20260904_131622",  # pe16 champion
    "20260831_io2",
}

KNOB_KEYS = [
    "enforcement",
    "enforcement_rounds",
    "keep_flash",
    "overlap_judge",
    "skip_flash",
    "skip_phase_b",
    "flash_seed_dir",
    "autosa_flow",
    "dse",
    "stream",
    "post_flash_dse",
    "post_flash_stream",
    "mm_flow_flavor",
    "skill_prompt_mode",
    "skills_pack",
    "packaged_skills_json",
    "flash_min_dsp",
    "flash_max_dsp",
    "flash_row_uf",
    "flash_pe_blk",
    "flash_tile_pp",
    "flash_onchip",
    "flash_skill_bin",
    "pe_recipe",
    "turns",
    "synth_timeout",
    "model",
    "latency_opt",
    "note",
]


def _load(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return None


def _lat(rep: Any) -> dict[str, Any] | None:
    if not isinstance(rep, dict):
        return None
    mn = rep.get("latency_cycles")
    mx = rep.get("latency_cycles_worst", mn)
    dsp = rep.get("dsp")
    if mn is None and dsp is None:
        return None
    return {
        "latency_min": mn,
        "latency_max": mx,
        "dsp": dsp,
        "bram": rep.get("bram"),
        "ff": rep.get("ff"),
        "lut": rep.get("lut"),
        "interval": rep.get("interval"),
        "work_dir": rep.get("work_dir"),
    }


def classify_family(dirname: str) -> str:
    n = dirname
    if "enf_aav_n_gf" in n or n.startswith("batch_parallel_autosa_mm_enf_"):
        return "enforcement_pingpong"
    if "flow_zero_shot" in n:
        return "ablation_zero_shot"
    if "flow_noskills" in n:
        return "ablation_noskills"
    if "32x8" in n:
        return "mmflow_32x8"
    if "mmflow" in n or "flow_aav_n_gf" in n:
        return "mmflow_3stage"
    if "tilepp" in n or "tile_pp" in n:
        return "flash_tile_pp"
    if "pe16" in n or "pe32" in n or "pe64" in n:
        return "flash_pe_blk"
    if "wideio" in n:
        return "flash_wide_io"
    if "computeii" in n:
        return "flash_compute_ii"
    if "flash_dsp_redo" in n:
        return "flash_dsp_redo"
    if "dsp1000" in n or "dsp500" in n or "dsp300" in n:
        return "flash_dsp_floor"
    if "rowuf" in n:
        return "flash_row_uf"
    if "seed_synth" in n:
        return "other_kernel_seed_synth"
    if "flash_onchip" in n:
        return "flash_onchip"
    if "flash_dataflow" in n:
        return "flash_dataflow_prompt"
    if "gap" in n:
        return "gap_flash"
    if "wave1" in n:
        return "wave1_other_kernels"
    if n.startswith("compact_pe_io"):
        return "family_c_io_mesh"
    if n.startswith("compact_pe_pack"):
        return "family_b_pack"
    if n.startswith("compact_pe"):
        return "family_a_pe_search"
    if n.startswith("manual_"):
        return "manual_handwritten"
    return "other"


def classify_aim(family: str, knobs: dict[str, Any]) -> str:
    aims = {
        "enforcement_pingpong": "Force explicit ping-pong + DATAFLOW on a flash kernel; judge code+csynth; keep-flash must not jack latency.",
        "ablation_zero_shot": "No skills, no PE recipe: what flash does with a generic prompt only.",
        "ablation_noskills": "3-stage flow without the 90-skill dump (JSON PE recipe still on for stream).",
        "mmflow_32x8": "Same 3-stage flow with locked 32x8 PE recipe.",
        "mmflow_3stage": "Flash then compute-rewrite then hide-load-store stream. Frozen slide column.",
        "flash_tile_pp": "Flash-only: PE_BLK=16 plus in-GEMM tile ping-pong prompt.",
        "flash_pe_blk": "Flash-only: pin PE_BLK 16/32/64 and spend U280 DSP.",
        "flash_wide_io": "Flash-only: wide AXI / LANES prompt (opt-in).",
        "flash_compute_ii": "Flash-only: compute-II pressure on DSP-floor kernels.",
        "flash_dsp_floor": "Flash-only: reject csynth if DSP below cutoff; repair toward more MACs.",
        "flash_dsp_redo": "Flash-only: fill U280 DSP under 100% of all resources; pick lowest latency among filled candidates.",
        "flash_row_uf": "Flash-only: unroll I in one tile (ROW_UF).",
        "flash_onchip": "Flash-only: distilled on-chip GEMM pack (940-class), not 90-skills.",
        "flash_dataflow_prompt": "Flash-only: DATAFLOW allowed when simple; no systolic stream stage.",
        "gap_flash": "Early gap-vs-rank-1 flash (Aug 18–23). Some ABI/model variants.",
        "wave1_other_kernels": "Other AutoSA kernels toward 2% of their rank-1.",
        "other_kernel_seed_synth": "Seed csynth of other AutoSA mm-family kernels (no LLM).",
        "family_a_pe_search": "Packed PE-count search (family A).",
        "family_b_pack": "Packing search (family B).",
        "family_c_io_mesh": "IO-mesh search (family C). First campaign frozen.",
        "manual_handwritten": "Hand-written kernels for ping-pong / PE16 reference (no LLM).",
    }
    extra = []
    if knobs.get("skip_flash"):
        extra.append("skip-flash: reuse a frozen flash_opt, no flash LLM.")
    if knobs.get("keep_flash"):
        extra.append("keep-flash gate: wrap max latency <= flash*1.10; do not drop flash LANES.")
    if knobs.get("flash_seed_dir"):
        extra.append("seed=" + str(knobs["flash_seed_dir"]).rsplit("/", 1)[-1][:80])
    base = aims.get(family, "See knobs.")
    return base + ((" " + " ".join(extra)) if extra else "")


def preview(text: str, n: int = 240) -> str:
    s = re.sub(r"\s+", " ", (text or "")).strip()
    return s[:n]


def dig_overlap(obj: Any) -> dict[str, Any] | None:
    """First nested dict that looks like a kernel csynth overlap block."""
    if isinstance(obj, dict):
        if obj.get("latency_cycles") is not None and obj.get("dsp") is not None:
            if "ok" in obj or "interval" in obj or "modules" in obj:
                return obj
        for v in obj.values():
            found = dig_overlap(v)
            if found:
                return found
    elif isinstance(obj, list):
        for v in obj:
            found = dig_overlap(v)
            if found:
                return found
    return None


def overlap_to_lat(ov: dict[str, Any] | None) -> dict[str, Any] | None:
    if not ov:
        return None
    mn = ov.get("latency_cycles")
    return {
        "latency_min": mn,
        "latency_max": ov.get("latency_cycles_worst", mn),
        "dsp": ov.get("dsp"),
        "bram": ov.get("bram"),
        "ff": ov.get("ff"),
        "lut": ov.get("lut"),
        "interval": ov.get("interval"),
        "from": "enforcement_initial_csynth",
    }


def tag_stage(content: str, hist_kind: str, current: str) -> str:
    if hist_kind in {"dse", "stream"}:
        return hist_kind
    c = content.lower()
    if any(
        x in c
        for x in (
            "overlap judge",
            "ping-pong/dataflow",
            "ping-pong + dataflow",
            "keep-flash",
            "keep_flash",
            "repair the kernel so load",
        )
    ):
        return "enforcement_judge_or_repair"
    if "flash mode" in c or "q_optimize_flash" in c or "instruction_c2hls_flash" in c:
        return "flash"
    if "[phase a]" in c or "phase b" in c or "phase-b" in c:
        return "phase_b"
    return current or hist_kind or "unknown"


def index_messages(
    path: Path,
    hist_kind: str,
    campaign: str,
    cell: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    data = _load(path)
    rows: list[dict[str, Any]] = []
    meta: dict[str, Any] = {
        "path": str(path),
        "kind": hist_kind,
        "exists": data is not None,
        "n_messages": 0,
        "llm_calls": None,
        "total_tokens": None,
        "model": None,
    }
    if not isinstance(data, dict):
        return rows, meta
    meta["model"] = data.get("model")
    usage = data.get("llm_usage") or {}
    if isinstance(usage, dict):
        meta["llm_calls"] = usage.get("calls")
        meta["total_tokens"] = usage.get("total_tokens")
        meta["by_agent"] = usage.get("by_agent")
    msgs = data.get("messages") or []
    meta["n_messages"] = len(msgs)
    current = "flash" if hist_kind == "flash_combined" else hist_kind
    for i, m in enumerate(msgs):
        if not isinstance(m, dict):
            continue
        content = m.get("content") or ""
        if not isinstance(content, str):
            content = json.dumps(content)[:500]
        current = tag_stage(content, hist_kind, current)
        rows.append(
            {
                "campaign": campaign,
                "cell": cell,
                "file": str(path),
                "hist_kind": hist_kind,
                "msg_index": i,
                "role": m.get("role"),
                "stage_guess": current,
                "chars": len(content),
                "preview": preview(content),
            }
        )
    return rows, meta


def report_from(cell: Path, stem: str) -> dict[str, Any] | None:
    p = cell / f"{stem}_report.json"
    if not p.exists():
        # some files are autosa_mm_<stage>_report.json
        alt = list(cell.glob(f"*_{stem}_report.json")) + list(cell.glob(f"{stem}.json"))
        p = alt[0] if alt else p
    if not p.exists():
        return None
    return _lat(_load(p))


def find_cells(camp: Path) -> list[Path]:
    cells = []
    var = camp / "variants"
    if var.is_dir():
        for p in var.rglob("*_flash_opt_report.json"):
            cells.append(p.parent)
        for p in var.rglob("*_selected_report.json"):
            if p.parent not in cells:
                cells.append(p.parent)
        for p in var.rglob("autosa_mm_enforcement.json"):
            if p.parent not in cells:
                cells.append(p.parent)
    # compact_pe / manual: look for json reports at top
    if not cells:
        for p in camp.glob("*.json"):
            if p.name.endswith("_report.json") or "cosim" in p.name or "summary" in p.name:
                cells.append(camp)
                break
        # also cpp next to reports
        if (camp / "autosa_mm_pe_pp_plus.cpp").exists() or list(camp.glob("*.cpp")):
            if camp not in cells:
                cells.append(camp)
    return sorted(set(cells))


def knobs_from_campaign(doc: dict[str, Any], camp: Path) -> dict[str, Any]:
    knobs: dict[str, Any] = {}
    for k in KNOB_KEYS:
        if k in doc and doc[k] not in (None, "", []):
            knobs[k] = doc[k]
    # sidecar txt
    for name in ("dse", "stream", "enforcement", "enforcement_rounds", "flavor", "skill_prompt", "model"):
        p = camp / f"{name}.txt"
        if p.exists() and name not in knobs:
            knobs[name] = p.read_text(encoding="utf-8", errors="replace").strip()
    cfg = doc.get("config") or {}
    pilot = cfg.get("pilot") or {}
    if isinstance(pilot, dict):
        knobs.setdefault("model", pilot.get("model") or doc.get("model"))
        knobs.setdefault("variant", pilot.get("variant"))
        knobs.setdefault("workflow", pilot.get("workflow"))
        knobs.setdefault("corpus", pilot.get("corpus"))
        knobs.setdefault("turns", pilot.get("turns") or doc.get("turns"))
        knobs.setdefault("benches", ",".join(pilot.get("benches") or []))
    knobs.setdefault("clock_ns", 3.33)
    knobs.setdefault("part", "xcu280-fsvh2892-2L-e")
    return knobs


def collect_campaign(camp: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    name = camp.name
    family = classify_family(name)
    doc = _load(camp / "campaign.json") if (camp / "campaign.json").exists() else {}
    if not isinstance(doc, dict):
        doc = {}
    knobs = knobs_from_campaign(doc, camp)
    stamp = doc.get("stamp") or ""
    frozen = any(s in name or s == stamp for s in FROZEN)
    rec: dict[str, Any] = {
        "dir": str(camp),
        "name": name,
        "stamp": stamp or name.rsplit("_", 1)[-1],
        "family": family,
        "aim": classify_aim(family, knobs),
        "status": doc.get("campaign_status") or ("manual" if name.startswith("manual_") else "unknown"),
        "created_at": doc.get("created_at"),
        "completed_at": doc.get("completed_at"),
        "frozen": frozen,
        "knobs": knobs,
        "cells": [],
        "llm_files": [],
        "notes": doc.get("note") or "",
    }
    llm_rows: list[dict[str, Any]] = []
    cells = find_cells(camp)
    if not cells:
        return rec, llm_rows

    for cell in cells:
        bench = cell.name
        # prefer autosa_mm reports
        stages: dict[str, Any] = {}
        for key, globpat in [
            ("reference", "reference_validation.json"),
            ("phase_b", "*_phase_b_report.json"),
            ("flash", "*_flash_opt_report.json"),
            ("flash_seed", "*_flash_seed_report.json"),
            ("dse", "*_dse_report.json"),
            ("stream", "*_stream_report.json"),
            ("selected", "*_selected_report.json"),
            ("enforcement_report", "*_enforcement.json"),
            ("manual_summary", "*summary.json"),
        ]:
            hits = list(cell.glob(globpat))
            if not hits:
                continue
            data = _load(hits[0])
            if key == "enforcement_report" and isinstance(data, dict):
                rpt = _lat(data.get("report")) if isinstance(data.get("report"), dict) else None
                stages["enforcement"] = {
                    "applied": data.get("applied"),
                    "passed": data.get("passed"),
                    "code_intended": data.get("code_intended"),
                    "overlap": data.get("overlap"),
                    "rounds_used": data.get("rounds_used"),
                    "rounds_limit": data.get("rounds_limit"),
                    "reason": preview(str(data.get("reason") or ""), 400),
                    "result": rpt,
                    "n_attempts": len(data.get("attempts") or []),
                    "path": str(hits[0]),
                }
                attempts_sum = []
                for a in data.get("attempts") or []:
                    if not isinstance(a, dict):
                        continue
                    jb = a.get("judge_before") or {}
                    ja = a.get("judge_after") or {}
                    attempts_sum.append(
                        {
                            "round": a.get("round"),
                            "status": a.get("status"),
                            "latency_cycles": a.get("latency_cycles"),
                            "judge_before_pass": jb.get("passed") if isinstance(jb, dict) else None,
                            "judge_after_pass": ja.get("passed") if isinstance(ja, dict) else None,
                            "judge_after_source": ja.get("source") if isinstance(ja, dict) else None,
                            "judge_after_reason": preview(str(ja.get("reason") or ""), 220)
                            if isinstance(ja, dict)
                            else "",
                        }
                    )
                stages["enforcement"]["attempts"] = attempts_sum
                before = overlap_to_lat(dig_overlap(data.get("initial")))
                if before:
                    stages["flash_before"] = before
            elif key == "reference":
                stages["reference"] = _lat((data or {}).get("report") if isinstance(data, dict) else None) or _lat(data)
            else:
                stages[key] = _lat(data)

        if stages.get("flash_seed") and not stages.get("flash_before"):
            stages["flash_before"] = dict(stages["flash_seed"])
            stages["flash_before"]["from"] = "flash_seed_report"
        # Enforcement often overwrites flash_opt.cpp/report with the wrap.
        flash_now = stages.get("flash") or {}
        enf_res = ((stages.get("enforcement") or {}).get("result") or {}) if isinstance(stages.get("enforcement"), dict) else {}
        if (
            stages.get("flash_before")
            and flash_now.get("latency_min") == enf_res.get("latency_min")
            and enf_res.get("latency_min") is not None
        ):
            stages["flash_opt_is_wrap"] = True

        manifest = _load(cell / "autosa_mm_flow_manifest.json") or _load(next(iter(cell.glob("*_flow_manifest.json")), Path("/dev/null")))
        if isinstance(manifest, dict) and manifest.get("latency_cycles"):
            stages["manifest_latency"] = manifest.get("latency_cycles")
            stages["selected_from"] = manifest.get("selected_from")

        skills = None
        sp = next(iter(cell.glob("*_flash_skills.json")), None)
        if sp and sp.exists():
            sd = _load(sp)
            if isinstance(sd, dict):
                fo = sd.get("flash_opt") or {}
                skills = {
                    "path": str(sp),
                    "injected_skill_count": fo.get("injected_skill_count"),
                    "routed_skill_id": fo.get("routed_skill_id"),
                    "skill_prompt_mode": fo.get("skill_prompt_mode"),
                    "pack_sha": ((sd.get("skills_source") or {}).get("sha256") or "")[:16],
                    "pack_path": (sd.get("skills_source") or {}).get("path"),
                }

        llm_meta = []
        for fname, kind in [
            ("autosa_mm_history.json", "flash_combined"),
            ("autosa_mm_dse_history.json", "dse"),
            ("autosa_mm_stream_history.json", "stream"),
        ]:
            p = cell / fname
            if not p.exists():
                # generic
                alts = list(cell.glob(f"*_{kind}_history.json")) if kind != "flash_combined" else list(cell.glob("*_history.json"))
                p = alts[0] if alts else p
            if p.exists():
                rows, meta = index_messages(p, kind, name, cell.name)
                llm_rows.extend(rows)
                llm_meta.append(meta)

        # enforcement judge traces sometimes only in history
        rec["llm_files"].extend(llm_meta)
        rec["cells"].append(
            {
                "path": str(cell),
                "name": bench,
                "stages": stages,
                "flash_skills": skills,
                "code": {
                    "flash_opt": str(next(iter(cell.glob("*_flash_opt.cpp")), "")) or None,
                    "selected": str(next(iter(cell.glob("*_selected.cpp")), "")) or None,
                    "dse": str(next(iter(cell.glob("*_dse.cpp")), "")) or None,
                    "stream": str(next(iter(cell.glob("*_stream.cpp")), "")) or None,
                    "enforcement_wrap": str(next(iter(cell.glob("*_enforcement_wrap.cpp")), "")) or None,
                },
            }
        )
    return rec, llm_rows


def collect_manual(camp: Path) -> dict[str, Any]:
    rec, _ = collect_campaign(camp)
    # handwritten reports
    stages = {}
    for p in camp.glob("*.json"):
        data = _load(p)
        lat = _lat(data)
        if lat:
            stages[p.stem] = lat
    if rec["cells"]:
        rec["cells"][0]["stages"].update(stages)
    elif stages:
        rec["cells"] = [{"path": str(camp), "name": camp.name, "stages": stages, "flash_skills": None, "code": {}}]
    rec["aim"] = classify_aim("manual_handwritten", {})
    rec["family"] = "manual_handwritten"
    rec["status"] = "manual"
    return rec


def pytest_inventory() -> list[dict[str, Any]]:
    files = sorted((ROOT / "tests").glob("test_flash*.py"))
    files += sorted((ROOT / "tests").glob("test_autosa_flow_gates.py"))
    files += sorted((ROOT / "tests").glob("test_autosa_mm_flow_flavors.py"))
    files += sorted((ROOT / "tests").glob("test_flash_skip_seed.py"))
    files += sorted((ROOT / "tests").glob("test_post_flash_*.py"))
    files += sorted((ROOT / "tests").glob("test_autosa_seed_synth.py"))
    seen = set()
    out = []
    for f in files:
        if f in seen:
            continue
        seen.add(f)
        text = f.read_text(encoding="utf-8", errors="replace")
        mod_doc = ast.get_docstring(ast.parse(text)) or ""
        tests = []
        for m in re.finditer(r"^def (test_[a-zA-Z0-9_]+)\(", text, re.M):
            tests.append(m.group(1))
        knobs = sorted(set(re.findall(r"C2HLS_[A-Z0-9_]+", text)))
        out.append(
            {
                "file": str(f.relative_to(ROOT)),
                "aim": preview(mod_doc, 400),
                "n_tests": len(tests),
                "tests": tests,
                "env_knobs": knobs,
            }
        )
    return out


def flatten_result_row(rec: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    cells = rec.get("cells") or [{}]
    if not cells:
        cells = [{}]
    for cell in cells:
        st = (cell.get("stages") or {}) if cell else {}
        flash = st.get("flash_before") or st.get("flash") or {}
        flash_now = st.get("flash") or {}
        enf = st.get("enforcement") or {}
        enf_res = (enf.get("result") or {}) if isinstance(enf, dict) else {}
        dse = st.get("dse") or {}
        stream = st.get("stream") or {}
        sel = st.get("selected") or st.get("manual_summary") or {}
        rows.append(
            {
                "name": rec["name"],
                "stamp": rec.get("stamp"),
                "family": rec["family"],
                "status": rec["status"],
                "frozen": rec["frozen"],
                "aim": rec["aim"],
                "model": (rec.get("knobs") or {}).get("model"),
                "knobs": json.dumps(rec.get("knobs") or {}, sort_keys=True),
                "skip_flash": (rec.get("knobs") or {}).get("skip_flash"),
                "keep_flash": (rec.get("knobs") or {}).get("keep_flash"),
                "enforcement": (rec.get("knobs") or {}).get("enforcement"),
                "dse": (rec.get("knobs") or {}).get("dse"),
                "stream": (rec.get("knobs") or {}).get("stream"),
                "flash_pe_blk": (rec.get("knobs") or {}).get("flash_pe_blk"),
                "flash_min_dsp": (rec.get("knobs") or {}).get("flash_min_dsp"),
                "flash_tile_pp": (rec.get("knobs") or {}).get("flash_tile_pp"),
                "flash_onchip": (rec.get("knobs") or {}).get("flash_onchip"),
                "flash_lat_min": flash.get("latency_min"),
                "flash_lat_max": flash.get("latency_max"),
                "flash_dsp": flash.get("dsp"),
                "flash_source": flash.get("from") or ("flash_opt_report" if flash else ""),
                "flash_opt_is_wrap": bool(st.get("flash_opt_is_wrap")),
                "wrap_lat_min": flash_now.get("latency_min") if st.get("flash_opt_is_wrap") else None,
                "enf_applied": enf.get("applied") if isinstance(enf, dict) else None,
                "enf_passed": enf.get("passed") if isinstance(enf, dict) else None,
                "enf_rounds": enf.get("rounds_used") if isinstance(enf, dict) else None,
                "enf_lat_min": enf_res.get("latency_min"),
                "enf_lat_max": enf_res.get("latency_max"),
                "enf_dsp": enf_res.get("dsp"),
                "dse_lat": dse.get("latency_min"),
                "dse_dsp": dse.get("dsp"),
                "stream_lat": stream.get("latency_min"),
                "stream_dsp": stream.get("dsp"),
                "selected_lat_min": sel.get("latency_min"),
                "selected_lat_max": sel.get("latency_max"),
                "selected_dsp": sel.get("dsp"),
                "llm_messages": sum(x.get("n_messages") or 0 for x in rec.get("llm_files") or []),
                "llm_calls": sum((x.get("llm_calls") or 0) for x in rec.get("llm_files") or []),
                "path": rec["dir"],
            }
        )
    return rows


def md_escape(s: Any) -> str:
    t = "" if s is None else str(s)
    return t.replace("|", "\\|").replace("\n", " ")


def write_md(catalog: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    lines = []
    a = lines.append
    a("# autosa_mm tests catalog (knobs, results, LLM history index)")
    a("")
    a(f"Generated: {catalog['generated_at']}")
    a("")
    a("This file is the **grep-able map**. Full LLM transcripts are **not** copied here (too large).")
    a("Each call is indexed in [`2026-09-10-autosa-mm-llm-calls.jsonl`](2026-09-10-autosa-mm-llm-calls.jsonl)")
    a("with `file` + `msg_index` pointing at the campaign `*_history.json`.")
    a("")
    a("Machine tables: [`2026-09-10-autosa-mm-tests-catalog.json`](2026-09-10-autosa-mm-tests-catalog.json),")
    a("[`.csv`](2026-09-10-autosa-mm-tests-catalog.csv).")
    a("Pytest knobs: [`2026-09-10-autosa-mm-pytest-knobs.json`](2026-09-10-autosa-mm-pytest-knobs.json).")
    a("Regenerate: `python scripts/pc2/export_autosa_mm_tests_catalog.py`.")
    a("")
    a("**Metrics:** latency min / max + DSP. Do not quote interval as the result.")
    a("**Two leaderboards (never mix):** iso-compute ~320 DSP vs AutoSA rank-1 **4228 / 320**; spend-DSP champion **940 / 5344** (`pe16_20260904_131622`).")
    a("")
    a("## 1. Knob legend")
    a("")
    a("| Knob | Env / campaign field | Aim | Default |")
    a("|---|---|---|---|")
    a("| 3-stage flow | `C2HLS_AUTOSA_FLOW=1` + DSE + stream | flash then compute-rewrite then hide-load-store | off except `start_autosa_mm_flow.sh` |")
    a("| Enforcement | `C2HLS_ENFORCEMENT=1` | ping-pong + DATAFLOW judged on code+csynth | off; flow sets 0 |")
    a("| Keep-flash | `keep_flash` + overlay skills | wrap must not drop flash LANES / jack latency > flash×1.10 | enforcement launcher |")
    a("| Skip-flash | `C2HLS_SKIP_FLASH=1` `--seed-flash DIR` | start enforcement from a frozen flash_opt (no flash LLM) | off |")
    a("| Explicit ping-pong | flow-gates | require `buf[2]` + `t&1`, B loaded once outside; DATAFLOW+arrays-inside is fail | after 101836 |")
    a("| Skills pack | `C2HLS_PACKAGED_SKILLS_JSON` | 90-skill GEMM flatten vs distilled on-chip vs keep-flash overlay | gemm_flatten_v1 |")
    a("| Prompt mode | `C2HLS_SKILL_PROMPT_MODE` | how skills are injected | `all_skills_avoids_global` |")
    a("| DSP floor | `C2HLS_FLASH_MIN_DSP` | reject flash if DSP below cutoff | off |")
    a("| PE_BLK | `C2HLS_FLASH_PE_BLK` 16/32/64 | pin PE width; spend-chip | off |")
    a("| Tile ping-pong | `C2HLS_FLASH_TILE_PP=1` | in-GEMM tile PP in **flash** (not enforcement) | off |")
    a("| ROW_UF | `C2HLS_FLASH_ROW_UF` | unroll I in one tile | off |")
    a("| On-chip pack | `C2HLS_FLASH_ONCHIP=1` | distilled 940-class skills, not 90 dump | off |")
    a("| Zero-shot | `C2HLS_MM_FLOW_FLAVOR=zero_shot` | no skills | off |")
    a("| No-skills | `C2HLS_MM_FLOW_FLAVOR=noskills` | 3-stage without 90-skill dump | off |")
    a("| Skip phase B | `C2HLS_SKIP_PHASE_B=1` | start from gold/plain, skip translator | enforcement seed path |")
    a("| DSE / stream | `C2HLS_POST_FLASH_DSE` / `_STREAM` | post-flash systolic stages (say compute rewrite / hide load-store on slides) | flow on; flash-* launchers off |")
    a("")
    a("## 2. Where each LLM call lives")
    a("")
    a("| Stage | File under the variant cell | What it contains |")
    a("|---|---|---|")
    a("| Phase B + flash (+ enforcement repairs) | `autosa_mm_history.json` → `messages[]` | Full chat: system, user, assistant. `llm_usage.calls` / tokens. |")
    a("| Flash skill injection | `autosa_mm_flash_skills.json` | Which skill IDs were stuffed into the flash prompt (`injected_prompt_text`). |")
    a("| Compute rewrite | `autosa_mm_dse_history.json` | Separate 3-message DSE call. |")
    a("| Hide load/store | `autosa_mm_stream_history.json` | Separate stream call. |")
    a("| Enforcement rounds | `autosa_mm_enforcement.json` → `attempts[]` | Per-round judge_before/after, latency, pass/fail. Repair *code* is in history.json, not duplicated here. |")
    a("| Orchestrator checkpoint | `pipelined/orchestrator_state.json` | Resume state; often duplicates history. Huge. Prefer `*_history.json`. |")
    a("| Drain log | `flow/gpu_drain.log` | Skip-flash accepted, 502s, round progress. |")
    a("")
    a("Index row example: `jq 'select(.campaign|test(\"101836\"))' docs/pc2/2026-09-10-autosa-mm-llm-calls.jsonl`")
    a("")
    a("## 3. Result table (one row per campaign cell)")
    a("")
    a("| Family | Stamp / name | Status | Flash min/max / DSP (before wrap) | After (enf or selected) min/max / DSP | Enf applied | LLM msgs | Frozen |")
    a("|---|---|---|---|---|---|---:|:---:|")
    for r in rows:
        flash = "—"
        if r.get("flash_lat_min") not in (None, ""):
            flash = f"{r['flash_lat_min']}/{r.get('flash_lat_max')} / {r.get('flash_dsp')}"
            if r.get("flash_opt_is_wrap") in (True, "True", "true"):
                flash += " (opt file overwritten)"
        after = "—"
        if r.get("enf_lat_min") not in (None, ""):
            after = f"{r['enf_lat_min']}/{r.get('enf_lat_max')} / {r.get('enf_dsp')}"
        elif r.get("selected_lat_min") not in (None, ""):
            after = f"{r['selected_lat_min']}/{r.get('selected_lat_max')} / {r.get('selected_dsp')}"
        elif r.get("stream_lat") not in (None, ""):
            after = f"{r['stream_lat']} / {r.get('stream_dsp')} (stream)"
        a(
            "| {fam} | `{name}` | {st} | {flash} | {after} | {ap} | {lm} | {fr} |".format(
                fam=md_escape(r.get("family")),
                name=md_escape(r.get("name")),
                st=md_escape(r.get("status")),
                flash=flash,
                after=after,
                ap=md_escape(r.get("enf_applied")),
                lm=r.get("llm_messages") or 0,
                fr="yes" if r.get("frozen") else "",
            )
        )
    a("")
    a("## 4. Campaign details")
    a("")
    for rec in catalog["campaigns"]:
        a(f"### `{rec['name']}`")
        a("")
        a(f"- **Family / aim:** {rec['family']} — {rec['aim']}")
        a(f"- **Status:** {rec['status']}  **Frozen:** {rec['frozen']}")
        a(f"- **Created / done:** {rec.get('created_at')} / {rec.get('completed_at')}")
        a(f"- **Path:** `{rec['dir']}`")
        kn = rec.get("knobs") or {}
        interesting = {k: kn[k] for k in kn if k in KNOB_KEYS or k in {"variant", "workflow", "benches", "turns"}}
        a(f"- **Knobs:** `{json.dumps(interesting, sort_keys=True)}`")
        if rec.get("notes"):
            a(f"- **Note:** {rec['notes']}")
        for cell in rec.get("cells") or []:
            st = cell.get("stages") or {}
            a(f"- **Cell:** `{cell.get('name')}`")
            for sk in ("flash", "dse", "stream", "selected"):
                v = st.get(sk)
                if isinstance(v, dict) and v.get("latency_min") is not None:
                    a(f"  - {sk}: **{v.get('latency_min')}–{v.get('latency_max')}** cycles, DSP **{v.get('dsp')}**")
            enf = st.get("enforcement")
            if isinstance(enf, dict) and enf:
                a(
                    f"  - enforcement: applied={enf.get('applied')} passed={enf.get('passed')} "
                    f"rounds={enf.get('rounds_used')}/{enf.get('rounds_limit')} "
                    f"overlap={enf.get('overlap')} reason={enf.get('reason')}"
                )
            skl = cell.get("flash_skills")
            if skl:
                a(f"  - flash skills injected={skl.get('injected_skill_count')} routed={skl.get('routed_skill_id')} pack_sha={skl.get('pack_sha')} `{skl.get('path')}`")
            code = cell.get("code") or {}
            for ck, cv in code.items():
                if cv:
                    a(f"  - code {ck}: `{cv}`")
        for lf in rec.get("llm_files") or []:
            a(
                f"- **LLM {lf.get('kind')}:** {lf.get('n_messages')} msgs, calls={lf.get('llm_calls')}, "
                f"tokens={lf.get('total_tokens')} `{lf.get('path')}`"
            )
        a("")
    a("## 5. Pytest files that lock these knobs")
    a("")
    a("These are **unit tests** (no Vitis, no LLM except mocked). They exist so a campaign knob cannot silently re-contaminate flash.")
    a("")
    a("| File | #tests | Aim | Env knobs |")
    a("|---|---:|---|---|")
    for t in catalog.get("pytest") or []:
        a(
            "| `{f}` | {n} | {aim} | {kn} |".format(
                f=md_escape(t["file"]),
                n=t["n_tests"],
                aim=md_escape(t["aim"]),
                kn=md_escape(", ".join(t.get("env_knobs") or [])[:200]),
            )
        )
    a("")
    a("## 6. Launchers (how a campaign is supposed to be started)")
    a("")
    a("All under `scripts/pc2/`. Unset `BATCH_PARALLEL_STAMP` and flash knobs you do not want before launch.")
    a("")
    a("| Script | Family |")
    a("|---|---|")
    for s, fam in [
        ("start_autosa_mm_flow.sh", "mmflow 3-stage (skills / noskills / zero_shot via flavor)"),
        ("start_autosa_mm_enforcement.sh", "enforcement; `--seed-flash DIR` skip-flash"),
        ("start_autosa_flash_enforcement.sh", "same, generic prefix"),
        ("start_autosa_mm_flash_dsp_floor.sh", "C2HLS_FLASH_MIN_DSP"),
        ("start_autosa_mm_flash_dsp_redo.sh", "fill DSP under 100%; pick lowest latency"),
        ("start_autosa_mm_flash_pe_blk.sh", "C2HLS_FLASH_PE_BLK"),
        ("start_autosa_mm_flash_tile_pp.sh", "PE16 + TILE_PP"),
        ("start_autosa_mm_flash_row_uf.sh", "ROW_UF"),
        ("start_autosa_mm_flash_onchip.sh", "distilled on-chip pack"),
        ("start_autosa_mm_flash_dataflow.sh", "flash-only DATAFLOW prompt"),
        ("start_autosa_mm_gap_flash_deepseek_aav_n_gf.sh", "early gap flash"),
        ("start_autosa_mm_pe_search.sh / pack / io", "family A/B/C"),
    ]:
        a(f"| `{s}` | {fam} |")
    a("")
    a("Do not overwrite frozen stamps listed in `FROZEN` in the exporter.")
    a("")
    return "\n".join(lines) + "\n"


def main() -> int:
    from datetime import datetime, timezone

    camps: list[Path] = []
    for pat in (
        "batch_parallel_autosa_mm_*",
        "batch_parallel_autosa_wave1_*",
        "compact_pe_*",
        "manual_mmflow_pe_pp",
        "manual_pe16_tile_pp",
        "manual_mm_lcst_tile_pp2",
        "manual_rank1_shaped",
    ):
        camps.extend(sorted(ART.glob(pat)))
    # unique dirs only
    seen = set()
    uniq = []
    for c in camps:
        if not c.is_dir():
            continue
        if c in seen:
            continue
        seen.add(c)
        uniq.append(c)

    campaigns = []
    llm_rows: list[dict[str, Any]] = []
    for c in uniq:
        if c.name.startswith("manual_"):
            rec = collect_manual(c)
            campaigns.append(rec)
            continue
        rec, rows = collect_campaign(c)
        campaigns.append(rec)
        llm_rows.extend(rows)

    pytest = pytest_inventory()
    catalog = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "root": str(ROOT),
        "n_campaigns": len(campaigns),
        "n_llm_index_rows": len(llm_rows),
        "rank1_reference": {"latency": 4228, "dsp": 320, "note": "AutoSA rank-1 iso-compute. Do not mix with spend-DSP."},
        "spend_dsp_champion": {"latency": 940, "dsp": 5344, "stamp": "20260904_131622"},
        "campaigns": campaigns,
        "pytest": pytest,
    }
    rows = []
    for rec in campaigns:
        rows.extend(flatten_result_row(rec))

    DOCS.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(catalog, indent=2, default=str) + "\n", encoding="utf-8")
    OUT_PYTEST.write_text(json.dumps(pytest, indent=2) + "\n", encoding="utf-8")

    if rows:
        keys = list(rows[0].keys())
        with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in rows:
                w.writerow(r)

    with OUT_LLM.open("w", encoding="utf-8") as f:
        for r in llm_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    OUT_MD.write_text(write_md(catalog, rows), encoding="utf-8")
    print(f"campaigns={len(campaigns)} llm_index_rows={len(llm_rows)} pytest_files={len(pytest)}")
    print(OUT_MD)
    print(OUT_JSON)
    print(OUT_CSV)
    print(OUT_LLM)
    return 0


if __name__ == "__main__":
    sys.exit(main())
