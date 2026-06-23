#!/usr/bin/env python3
"""Direct Nova/Rodinia sw_emu + hw_emu validation with canonical JSONL output.

No LLM: runs the upstream cpp through Vitis emulation and emits one schema-1.0
record per attempted sw_emu or hw_emu run, compatible with
results/references_philip.

Outputs:
  artifacts/nova_direct_emu.jsonl     - schema-1.0 records
  artifacts/nova_direct_emu_vs_ref.md - pass/fail + cycle delta vs references
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

from c2hls_paths import active_site, apply_runtime_defaults, configure_site, rodinia_nova_benchmarks_dir

configure_site()
apply_runtime_defaults()

import hls_eval  # noqa: E402
from export_schema_jsonl import SCHEMA_VERSION, validate_jsonl  # noqa: E402

_nova = rodinia_nova_benchmarks_dir()
if _nova is None or not _nova.is_dir():
    if active_site() == "pc2":
        raise SystemExit("Set C2HLS_RODINIA_NOVA_DIR in local.env (see local.env.example).")
    raise SystemExit(f"Nova benchmarks not found: {_nova}")
NOVA_ROOT = _nova
REF_DIR = REPO / "results" / "references_philip"
SW_EMU_REF = REF_DIR / "sw_emu_vitis_2023.2__device_xilinx_u280_gen3x16_xdma_1_202211_1.jsonl"
HW_EMU_REF = REF_DIR / "hw_emu_vitis_2023.2__device_xilinx_u280_gen3x16_xdma_1_202211_1.jsonl"
OUT_JSONL = REPO / "artifacts" / "nova_direct_emu.jsonl"
DELTA_MD  = REPO / "artifacts" / "nova_direct_emu_vs_ref.md"

NOVA_BENCHES = [
    (("cfd", "cfd_flux"), NOVA_ROOT / "cfd" / "cfd_flux", "cfd_flux"),
    (("pathfinder",), NOVA_ROOT / "pathfinder", "pathfinder"),
    (("knn",), NOVA_ROOT / "knn", "knn"),
    (("nw",), NOVA_ROOT / "nw", "nw"),
]

# Variants to hw_emu — limited subset due to ~30 min/variant cost. Comma-sep
# step names. Override with C2HLS_HW_EMU_STEPS=baseline,coalescing.
HW_EMU_STEPS = {
    s.strip() for s in os.getenv("C2HLS_HW_EMU_STEPS", "baseline,coalescing").split(",")
    if s.strip()
}
DEVICE = os.getenv("C2HLS_DEVICE_PLATFORM", "xilinx_u280_gen3x16_xdma_1_202211_1")
VITIS_VERSION = os.getenv("C2HLS_VITIS_VERSION", "2023.2")


def _list_variants(parent_dir: Path) -> list[str]:
    return sorted(p.name for p in parent_dir.iterdir() if p.is_dir())


def _short_step(variant_name: str) -> str:
    """Extract the step name from a variant_<index>_<step> tail. Handles
    multi-underscore bench names (cfd_flux_0_baseline → 'baseline')."""
    m = re.match(r"^.+_(\d+)_(.+)$", variant_name)
    if m:
        s = m.group(2)
        return s.replace("unrolling", "unroll").replace("double_buffer", "doublebuffer")
    return variant_name


def _variant_identity(variant_name: str) -> tuple[int, str]:
    m = re.match(r"^.+_(\d+)_(.+)$", variant_name)
    if not m:
        return 0, variant_name or "implementation"
    return int(m.group(1)), _short_step(variant_name)


def _parse_ref_lat_ns(s):
    if not s or s == "undef":
        return None
    try:
        n, u = s.split()
        return int(float(n) * {"ns": 1, "us": 1e3, "ms": 1e6, "s": 1e9}[u])
    except Exception:
        return None


def _normalize_ref_step(name: str) -> str:
    """Match the orientation of `_short_step` so driver/ref keys collide
    (unrolling↔unroll, double_buffer↔doublebuffer)."""
    return name.replace("unrolling", "unroll").replace("double_buffer", "doublebuffer")


def _emu_status(result: dict) -> str:
    if result.get("success"):
        return "pass"
    if "timed out" in (result.get("error") or "").lower():
        return "timeout"
    return "fail"


def _run_section(target: str, elapsed: float) -> dict:
    return {
        "target": target,
        "device": DEVICE,
        "vitis_version": VITIS_VERSION,
        "runtime_seconds": round(elapsed, 6),
    }


def _problem(group_path: tuple[str, ...]) -> dict:
    return {"suite": "rodinia_hls", "group_path": list(group_path)}


def _implementation(index: int, short_name: str,
                    variant_name: str, kernel_basename: str,
                    extra_meta: dict | None = None) -> dict:
    origin_meta = {
        "source_variant": variant_name,
        "kernel_basename": kernel_basename,
    }
    if extra_meta:
        origin_meta.update(extra_meta)
    return {
        "origin": "rodinia_hls_benchmark",
        "origin_version": "2023_port",
        "origin_meta": origin_meta,
        "variant": {"index": int(index), "name": short_name},
    }


def _sw_record(group_path: tuple[str, ...], variant_name: str, kernel_basename: str,
               sw: dict, elapsed: float) -> dict:
    index, short = _variant_identity(variant_name)
    status = _emu_status(sw)
    payload = {"status": status}
    if status != "pass":
        payload["error"] = (sw.get("error") or "sw_emu failed")[:300]
    return {
        "schema_version": SCHEMA_VERSION,
        "report_type": "sw_run",
        "run": _run_section("vitis.sw_emu", elapsed),
        "problem": _problem(group_path),
        "implementation": _implementation(index, short, variant_name, kernel_basename),
        "sw_run": payload,
    }


def _hw_record(group_path: tuple[str, ...], variant_name: str, kernel_basename: str,
               hw: dict, elapsed: float) -> dict:
    index, short = _variant_identity(variant_name)
    status = _emu_status(hw)
    payload = {
        "status": status,
        "kernel_runtime_cycles": hw.get("kernel_runtime_cycles"),
        "kernel_runtime_us": hw.get("kernel_runtime_us"),
        "kernel_clock_freq_mhz": hw.get("kernel_clock_freq_mhz"),
    }
    if status != "pass":
        payload["error"] = (hw.get("error") or "hw_emu failed")[:300]
    return {
        "schema_version": SCHEMA_VERSION,
        "report_type": "rtl_sim",
        "run": _run_section("vitis.hw_emu", elapsed),
        "problem": _problem(group_path),
        "implementation": _implementation(
            index,
            short,
            variant_name,
            kernel_basename,
            {
                "profile_csv": hw.get("profile_csv") or None,
                "profile_compute_unit_rows": hw.get("profile_compute_unit_rows"),
                "system_diagram_model": hw.get("system_diagram_model") or None,
                "clock_source": hw.get("clock_source") or None,
                "clock_fallback": hw.get("clock_fallback"),
            },
        ),
        "rtl_sim": payload,
    }


def load_sw_emu_reference():
    out = {}
    if not SW_EMU_REF.exists():
        print(f"  WARNING: no sw_emu reference at {SW_EMU_REF}")
        return out
    for line in SW_EMU_REF.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        gp = tuple(r["problem"]["group_path"])
        variant = r["implementation"]["variant"]
        v = _normalize_ref_step(variant["name"])
        out[(gp, int(variant["index"]), v)] = r.get("sw_run", {}).get("status")
    return out


def load_hw_emu_reference():
    """Load the canonical hw_emu reference JSONL. Each record is rtl_sim with
    `kernel_runtime_cycles`, `kernel_runtime_us`, and a status field
    (pass/timeout/fail). For pass records both cycles and us are populated
    and direct-comparable to our run."""
    out = {}
    if not HW_EMU_REF.exists():
        print(f"  WARNING: no hw_emu reference at {HW_EMU_REF}")
        return out
    for line in HW_EMU_REF.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        gp = tuple(r["problem"]["group_path"])
        variant = r["implementation"]["variant"]
        v = _normalize_ref_step(variant["name"])
        rs = r.get("rtl_sim") or {}
        out[(gp, int(variant["index"]), v)] = {
            "status": rs.get("status"),
            "kernel_runtime_cycles": rs.get("kernel_runtime_cycles"),
            "kernel_runtime_us": rs.get("kernel_runtime_us"),
            "kernel_clock_freq_mhz": rs.get("kernel_clock_freq_mhz"),
        }
    print(f"  loaded hw_emu reference: {len(out)} records")
    return out


def main() -> int:
    sw_ref = load_sw_emu_reference()
    hw_ref = load_hw_emu_reference()
    print(f"sw_emu ref: {len(sw_ref)} keys; hw cycle ref: {len(hw_ref)} keys", flush=True)

    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSONL.write_text("")
    emitted = []
    rows = []
    for ref_gp, parent_dir, kernel_basename in NOVA_BENCHES:
        if not parent_dir.is_dir():
            print(f"  SKIP {parent_dir.name}: missing")
            continue
        variants = _list_variants(parent_dir)
        print(f"\n=== {ref_gp[-1]} ({len(variants)} variants) ===", flush=True)
        for vname in variants:
            v_dir = parent_dir / vname
            if not (v_dir / "Makefile").exists():
                continue
            variant_index, short = _variant_identity(vname)
            ref_key = (ref_gp, variant_index, short)

            # sw_emu — run on every variant
            t0 = time.time()
            sw = hls_eval.run_sw_emu_via_nova(
                str(v_dir),
                kernel_basename=kernel_basename,
                timeout=int(os.getenv("C2HLS_SW_EMU_TIMEOUT", "1800")),
            )
            sw_elapsed = time.time() - t0
            sw_record = _sw_record(ref_gp, vname, kernel_basename, sw, sw_elapsed)
            emitted.append(sw_record)
            with OUT_JSONL.open("a") as f:
                f.write(json.dumps(sw_record) + "\n")
            sw_ref_status = sw_ref.get(ref_key)
            print(f"  {short:<14} sw_emu status={sw_record['sw_run']['status']} (ref={sw_ref_status})",
                  flush=True)

            # hw_emu — only for baselines + select steps, much slower
            hw_record = None
            hw_ref_entry = hw_ref.get(ref_key) or {}
            hw_ref_us     = hw_ref_entry.get("kernel_runtime_us")
            hw_ref_cycles = hw_ref_entry.get("kernel_runtime_cycles")
            hw_ref_status = hw_ref_entry.get("status")
            if short in HW_EMU_STEPS:
                t1 = time.time()
                hw = hls_eval.run_hw_emu_via_nova(
                    str(v_dir),
                    kernel_basename=kernel_basename,
                    timeout=int(os.getenv("C2HLS_HW_EMU_TIMEOUT", "21600")),
                )
                hw_elapsed = time.time() - t1
                hw_record = _hw_record(ref_gp, vname, kernel_basename, hw, hw_elapsed)
                emitted.append(hw_record)
                with OUT_JSONL.open("a") as f:
                    f.write(json.dumps(hw_record) + "\n")
                hw_us = hw.get("kernel_runtime_us")
                hw_lat_cycles = hw.get("kernel_runtime_cycles")
                ratio_us = (f"{hw_us/hw_ref_us:.3f}x"
                            if hw_us and hw_ref_us else "—")
                ratio_cy = (f"{hw_lat_cycles/hw_ref_cycles:.3f}x"
                            if hw_lat_cycles and hw_ref_cycles else "—")
                print(f"  {short:<14} hw_emu status={hw_record['rtl_sim']['status']} "
                      f"us={hw_us} cycles={hw_lat_cycles} "
                      f"clock={hw.get('kernel_clock_freq_mhz')} "
                      f"ref_us={hw_ref_us} ref_cycles={hw_ref_cycles} "
                      f"ratio_us={ratio_us} ratio_cy={ratio_cy} (ref_status={hw_ref_status}) "
                      f"({hw_elapsed}s)", flush=True)

            row = {
                "group_path": list(ref_gp), "variant_index": variant_index,
                "variant_name": vname, "variant_short": short,
                "kernel_basename": kernel_basename,
                "sw_emu": {
                    "status": sw_record["sw_run"]["status"],
                    "ref_status": sw_ref_status,
                    "matches_ref": sw_ref_status == sw_record["sw_run"]["status"],
                    "elapsed_sec": sw_elapsed,
                    "error": sw.get("error", ""),
                },
                "hw_record": hw_record,
                "hw_ref": hw_ref_entry,
            }
            rows.append(row)

    validation = validate_jsonl(OUT_JSONL)
    if validation["invalid"]:
        print(f"ERROR: {validation['invalid']} malformed JSONL records in {OUT_JSONL}", file=sys.stderr)
        return 1

    # Markdown delta
    lines = [
        "# Nova benchmarks: direct sw_emu + hw_emu vs reference",
        "",
        f"Vitis {VITIS_VERSION} / {DEVICE} / no LLM",
        "",
        f"hw_emu measured on steps: {sorted(HW_EMU_STEPS)}",
        "",
        "## Per-variant sw_emu correctness",
        "",
        "| bench | variant | sw_emu ours | sw_emu ref | match |",
        "|---|---:|:---:|:---:|:---:|",
    ]
    for r in rows:
        bench = "/".join(r["group_path"])
        sw = r["sw_emu"]
        ref_ok = sw.get("ref_status") or "-"
        match = "yes" if sw.get("matches_ref") else "no"
        lines.append(f"| {bench} | {r['variant_index']} {r['variant_short']} | {sw['status']} | {ref_ok} | {match} |")

    lines.append("")
    lines.append("## hw_emu kernel runtime (subset)")
    lines.append("")
    lines.append("| bench | variant | ours_status | ref_status | ours_us | ours_cycles | ref_us | ref_cycles | delta_cycles |")
    lines.append("|---|---:|:---:|:---:|---:|---:|---:|---:|---:|")
    for r in rows:
        hw_record = r["hw_record"]
        if hw_record is None:
            continue
        bench = "/".join(r["group_path"])
        hw = hw_record["rtl_sim"]
        ref = r["hw_ref"]
        ours_us = hw.get("kernel_runtime_us")
        ours_cy = hw.get("kernel_runtime_cycles")
        ref_us = ref.get("kernel_runtime_us")
        ref_cy = ref.get("kernel_runtime_cycles")
        ref_status = ref.get("status") or "-"
        delta_cy = ours_cy - ref_cy if isinstance(ours_cy, int) and isinstance(ref_cy, int) else None
        lines.append(f"| {bench} | {r['variant_index']} {r['variant_short']} | "
                     f"{hw.get('status')} | {ref_status} | "
                     f"{ours_us if ours_us is not None else '-'} | "
                     f"{ours_cy if ours_cy is not None else '-'} | "
                     f"{ref_us if ref_us is not None else '-'} | "
                     f"{ref_cy if ref_cy is not None else '-'} | "
                     f"{delta_cy if delta_cy is not None else '-'} |")

    DELTA_MD.write_text("\n".join(lines) + "\n")
    print(f"\nwrote {len(emitted)} schema records to {OUT_JSONL}")
    print(f"wrote delta table to {DELTA_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
