"""Helpers for HLSFactory gold/flash timeout fill (csynth_full + cosim_small on copies)."""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

REPO = Path(__file__).resolve().parents[2]
REPORT_DIR = (
    REPO
    / "artifacts"
    / "pc2"
    / "reports"
    / "hlsfactory_flash_cosim_vs_gold_latrag_off"
)
FILL_ROOT = REPORT_DIR / "fill_ab_20260730"
BENCHMARKS_COSIM = REPO / "benchmarks_cosim"

PART = "xcu280-fsvh2892-2L-e"
CLOCK_NS = 3.33

DEEPSEEK_CAMP = (
    REPO
    / "artifacts"
    / "pc2"
    / "batch_parallel_hlsfactory_ds_v4f_skills_20260725_084654_v4f_skills_ab"
)
DEVSTRAL_P28 = REPO / "artifacts" / "pc2" / "batch_parallel_20260701_bp_full_aav_n_p28"
DEVSTRAL_SISTER = (
    REPO
    / "artifacts"
    / "pc2"
    / "flash_fixed_cosim_aav_n_20260628_fixed_cosim_flash_r2_pipelined"
)

SMALL_MACROS: dict[str, dict[str, int]] = {
    "hlsfactory_fdtd-2d": {"TMAX": 10, "NX": 20, "NY": 24},
    "hlsfactory_heat-3d": {"N": 8, "TSTEPS": 10},
    "hlsfactory_seidel-2d": {"N": 40, "TSTEPS": 10},
    "hlsfactory_syr2k": {"N": 24, "M": 20},
    "hlsfactory_jacobi-1d": {"N": 40, "TSTEPS": 10},
    "hlsfactory_ludcmp": {"N": 40},
}

FULL_MACRO_NOTES: dict[str, str] = {
    "hlsfactory_fdtd-2d": "full: TMAX=40 NX=60 NY=80",
    "hlsfactory_heat-3d": "full: N=20 TSTEPS=40",
    "hlsfactory_seidel-2d": "full: N=120 TSTEPS=40",
    "hlsfactory_syr2k": "full: N=80 M=60",
    "hlsfactory_jacobi-1d": "full: N=120 TSTEPS=40",
    "hlsfactory_ludcmp": "full: N=120",
}

TIMEOUT_REASON: dict[tuple[str, str], str] = {
    ("deepseek_v4_flash_aav_n", "hlsfactory_fdtd-2d"): "gold_cosim_timeout",
    ("deepseek_v4_flash_aav_n", "hlsfactory_heat-3d"): "gold_cosim_timeout",
    ("deepseek_v4_flash_aav_n", "hlsfactory_seidel-2d"): "gold_cosim_timeout",
    ("deepseek_v4_flash_aav_n", "hlsfactory_syr2k"): "gold_cosim_timeout",
    ("deepseek_v4_flash_aav_n", "hlsfactory_jacobi-1d"): "flash_cosim_missing",
    ("deepseek_v4_flash_aav_n", "hlsfactory_ludcmp"): "flash_cosim_missing",
    ("devstral2_aav_n", "hlsfactory_fdtd-2d"): "gold_and_flash_missing",
    ("devstral2_aav_n", "hlsfactory_heat-3d"): "gold_and_flash_missing",
    ("devstral2_aav_n", "hlsfactory_seidel-2d"): "gold_and_flash_missing",
    ("devstral2_aav_n", "hlsfactory_syr2k"): "gold_cosim_timeout",
}

SCOPE: dict[str, list[str]] = {
    "deepseek_v4_flash_aav_n": [
        "hlsfactory_fdtd-2d",
        "hlsfactory_heat-3d",
        "hlsfactory_seidel-2d",
        "hlsfactory_syr2k",
        "hlsfactory_jacobi-1d",
        "hlsfactory_ludcmp",
    ],
    "devstral2_aav_n": [
        "hlsfactory_fdtd-2d",
        "hlsfactory_heat-3d",
        "hlsfactory_seidel-2d",
        "hlsfactory_syr2k",
    ],
}


@dataclass(frozen=True)
class FlashSource:
    path: Path
    note: str


def small_size_note(bench: str) -> str:
    m = SMALL_MACROS[bench]
    return "small: " + " ".join(f"{k}={v}" for k, v in m.items())


def resolve_flash_selected(model_tag: str, bench: str) -> FlashSource:
    if model_tag == "deepseek_v4_flash_aav_n":
        cell = DEEPSEEK_CAMP / "variants" / "aav_n" / bench
        hits = sorted(cell.rglob(f"{bench}_selected.cpp"))
        if not hits:
            raise FileNotFoundError(f"no selected.cpp for {model_tag} {bench} under {cell}")
        return FlashSource(hits[0], f"deepseek selected:{hits[0]}")
    if model_tag == "devstral2_aav_n":
        for camp, label in (
            (DEVSTRAL_P28 / "variants" / "aav_n" / bench, "p28"),
            (DEVSTRAL_SISTER / bench, "sister_r2"),
        ):
            if not camp.exists():
                continue
            for name in (f"{bench}_selected.cpp", f"{bench}_flash_opt.cpp"):
                hits = sorted(camp.rglob(name))
                # Prefer files that look complete (non-tiny)
                for h in hits:
                    if h.stat().st_size > 200:
                        return FlashSource(h, f"devstral {label}:{h}")
        raise FileNotFoundError(f"no flash kernel for {model_tag} {bench}")
    raise ValueError(model_tag)


def _patch_header_macros(header_text: str, macros: dict[str, int]) -> str:
    out = header_text
    for name, val in macros.items():
        # Replace #define NAME <number> whether guarded or not.
        out, n = re.subn(
            rf"(#define\s+{re.escape(name)}\s+)\d+",
            rf"\g<1>{val}",
            out,
        )
        if n == 0:
            # Insert after auto-macro guards start if present
            marker = "// >>> c2hls auto-macro guards"
            if marker in out:
                out = out.replace(
                    marker,
                    f"{marker}\n#ifndef {name}\n#define {name} {val}\n#endif",
                    1,
                )
            else:
                out = f"#ifndef {name}\n#define {name} {val}\n#endif\n" + out
    return out


def _guess_max_depth(macros: dict[str, int]) -> int:
    vals = list(macros.values())
    if not vals:
        return 1024
    # conservative product of two largest dims (arrays are often 2D/3D)
    vals_sorted = sorted(vals, reverse=True)
    if len(vals_sorted) >= 3:
        return max(vals_sorted[0] * vals_sorted[1] * vals_sorted[2], 64)
    if len(vals_sorted) == 2:
        return max(vals_sorted[0] * vals_sorted[1], 64)
    return max(vals_sorted[0] * vals_sorted[0], 64)


def _patch_m_axi_depths(src: str, depth: int) -> str:
    return re.sub(r"(depth\s*=\s*)\d+", rf"\g<1>{depth}", src)


def _load_meta(bench_dir: Path) -> dict[str, Any]:
    return json.loads((bench_dir / "metadata.json").read_text(encoding="utf-8"))


def infer_top(kernel_text: str, meta: dict[str, Any]) -> str:
    if re.search(r"\bvoid\s+workload\s*\(", kernel_text):
        return "workload"
    m = re.search(r"#pragma\s+HLS\s+top\s+name\s*=\s*(\w+)", kernel_text)
    if m:
        return m.group(1)
    return meta.get("translated_hls_top") or meta.get("hls_top") or meta.get("kernel_top") or "workload"


def prepare_bench_copy(
    *,
    model_tag: str,
    bench: str,
    size_kind: str,  # full | small
) -> Path:
    src = BENCHMARKS_COSIM / bench
    if not src.is_dir():
        raise FileNotFoundError(src)
    dest_root = FILL_ROOT / ("benches_full" if size_kind == "full" else "benches_small")
    dest = dest_root / model_tag / bench
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)
    for p in src.iterdir():
        if p.is_file():
            shutil.copy2(p, dest / p.name)

    flash = resolve_flash_selected(model_tag, bench)
    flash_text = flash.path.read_text(encoding="utf-8", errors="replace")
    (dest / "flash_kernel.cpp").write_text(flash_text, encoding="utf-8")
    (dest / "flash_kernel_source.json").write_text(
        json.dumps({"path": str(flash.path), "note": flash.note}, indent=2) + "\n",
        encoding="utf-8",
    )

    meta = _load_meta(dest)
    header_name = meta.get("header_file") or next(dest.glob("*.h")).name
    header_path = dest / header_name
    header = header_path.read_text(encoding="utf-8", errors="replace")

    if size_kind == "small":
        macros = SMALL_MACROS[bench]
        header = _patch_header_macros(header, macros)
        header_path.write_text(header, encoding="utf-8")
        depth = _guess_max_depth(macros)
        for fname in ("hls_baseline_cosim.cpp", "flash_kernel.cpp", "gold_kernel_for_cosim.cpp"):
            fp = dest / fname
            if fp.exists():
                fp.write_text(_patch_m_axi_depths(fp.read_text(encoding="utf-8", errors="replace"), depth), encoding="utf-8")

    note = {
        "model_tag": model_tag,
        "bench": bench,
        "size_kind": size_kind,
        "problem_size_note": FULL_MACRO_NOTES[bench] if size_kind == "full" else small_size_note(bench),
        "timeout_reason": TIMEOUT_REASON[(model_tag, bench)],
        "flash_source": flash.note,
        "part": PART,
        "clock_ns": CLOCK_NS,
    }
    (dest / "fill_ab_meta.json").write_text(json.dumps(note, indent=2) + "\n", encoding="utf-8")
    return dest


def build_jobs() -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    idx = 0
    for model_tag, benches in SCOPE.items():
        for bench in benches:
            for size_kind, metric_kind in (("full", "csynth_full"), ("small", "cosim_small")):
                bench_dir = FILL_ROOT / (
                    "benches_full" if size_kind == "full" else "benches_small"
                ) / model_tag / bench
                for side in ("gold", "flash"):
                    jobs.append(
                        {
                            "index": idx,
                            "model_tag": model_tag,
                            "bench": bench,
                            "size_kind": size_kind,
                            "metric_kind": metric_kind,
                            "side": side,
                            "bench_dir": str(bench_dir),
                            "timeout_reason": TIMEOUT_REASON[(model_tag, bench)],
                            "problem_size_note": (
                                FULL_MACRO_NOTES[bench]
                                if size_kind == "full"
                                else small_size_note(bench)
                            ),
                        }
                    )
                    idx += 1
    return jobs


def job_result_path(job: dict[str, Any]) -> Path:
    return (
        FILL_ROOT
        / "work"
        / job["metric_kind"]
        / job["model_tag"]
        / job["bench"]
        / f"{job['side']}.json"
    )


def _cosim_extra_files(bench_dir: Path, meta: dict[str, Any]) -> list[dict[str, Any]]:
    """Stage TB gold helper the same way flash_cosim_lib does (no c2hls import)."""
    extra_files: list[dict[str, Any]] = []
    seen: set[str] = set()
    support_rels: list[str] = []
    for key in ("cosim_support_files", "support_files"):
        for rel_path in meta.get(key) or []:
            if rel_path and rel_path not in support_rels:
                support_rels.append(rel_path)
    # Always include renamed gold helper used by testbench_cosim.cpp
    for rel in ("gold_kernel_for_cosim.cpp",):
        if rel not in support_rels and (bench_dir / rel).exists():
            support_rels.append(rel)
    for rel_path in support_rels:
        file_path = bench_dir / rel_path
        if not file_path.exists():
            raise FileNotFoundError(f"{bench_dir.name}: cosim support file missing ({rel_path})")
        is_header = Path(rel_path).suffix.lower() in {".h", ".hpp", ".hh"}
        extra_files.append(
            {
                "path": rel_path,
                "content": file_path.read_text(encoding="utf-8", errors="ignore"),
                "tb": not is_header,
            }
        )
        seen.add(rel_path)
    gold_src = meta.get("gold_hls_source_file") or "gold_hls_source.cpp"
    gold_src_path = bench_dir / gold_src
    if gold_src_path.exists() and gold_src not in seen:
        # Materialize for #include from gold_kernel_for_cosim.cpp only.
        # tb=False: do NOT compile as a separate TB TU (would redefine kernel_*).
        # .cpp is also skipped as a design source by hls_eval.
        extra_files.append(
            {
                "path": gold_src,
                "content": gold_src_path.read_text(encoding="utf-8", errors="ignore"),
                "tb": False,
            }
        )
        seen.add(gold_src)
    return extra_files


def run_one_job(job: dict[str, Any], *, force: bool = False) -> dict[str, Any]:
    import sys
    import time

    sys.path.insert(0, str(REPO))
    import hls_eval  # noqa: WPS410

    out_path = job_result_path(job)
    if out_path.exists() and not force:
        return json.loads(out_path.read_text(encoding="utf-8"))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    bench_dir = Path(job["bench_dir"])
    meta = _load_meta(bench_dir)
    header_name = meta.get("header_file") or next(bench_dir.glob("*.h")).name
    header_code = (bench_dir / header_name).read_text(encoding="utf-8", errors="replace")
    tb_name = meta.get("cosim_testbench_file") or "testbench_cosim.cpp"
    tb_code = (bench_dir / tb_name).read_text(encoding="utf-8", errors="replace")

    if job["side"] == "gold":
        kernel_name = meta.get("cosim_kernel_file") or "hls_baseline_cosim.cpp"
        kernel_path = bench_dir / kernel_name
    else:
        kernel_path = bench_dir / "flash_kernel.cpp"
    hls_code = kernel_path.read_text(encoding="utf-8", errors="replace")
    top = infer_top(hls_code, meta)

    work_dir = str(out_path.parent / f"workdir_{job['side']}")
    if force and Path(work_dir).exists():
        shutil.rmtree(work_dir)
    Path(work_dir).mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    if job["metric_kind"] == "csynth_full":
        result = hls_eval.run_hls_synthesis(
            hls_code,
            header_code=header_code,
            header_name=header_name,
            top_function=top,
            part=PART,
            clock_ns=CLOCK_NS,
            work_dir=work_dir,
        )
        report = result.get("report") or {}
        payload = {
            "status": "ok" if result.get("success") else "fail",
            "metric_kind": "csynth_full",
            "side": job["side"],
            "latency_cycles": report.get("latency_cycles"),
            "error": result.get("error") or "",
            "work_dir": result.get("work_dir") or work_dir,
            "top_function": top,
            "kernel_path": str(kernel_path),
            "runtime_seconds": round(time.time() - t0, 3),
            "job": job,
        }
    else:
        extra_files = _cosim_extra_files(bench_dir, meta)
        result = hls_eval.run_cosim(
            hls_code,
            tb_code,
            header_code=header_code,
            header_name=header_name,
            top_function=top,
            part=PART,
            clock_ns=CLOCK_NS,
            work_dir=work_dir,
            extra_files=extra_files or None,
            interface_depths=meta.get("cosim_depths") or None,
        )
        payload = {
            "status": "ok" if result.get("success") and result.get("passed") else (
                "timeout" if "timed out" in str(result.get("error") or "").lower() else "fail"
            ),
            "metric_kind": "cosim_small",
            "side": job["side"],
            "kernel_runtime_cycles": result.get("kernel_runtime_cycles"),
            "passed": bool(result.get("passed")),
            "error": result.get("error") or "",
            "work_dir": result.get("work_dir") or work_dir,
            "top_function": top,
            "kernel_path": str(kernel_path),
            "runtime_seconds": round(time.time() - t0, 3),
            "job": job,
        }
        (out_path.parent / f"{job['side']}.log").write_text(result.get("log") or "", encoding="utf-8")

    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload
