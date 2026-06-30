#!/usr/bin/env python3
"""Materialize HLSFactory Tier A (non-PolyBench) into tier_A_ready/ for c2hls flash."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from dataset_pipeline.external_adapter import infer_cosim_depths, infer_top_function  # noqa: E402
from dataset_pipeline.hls_normalize import (  # noqa: E402
    normalize_gold_header,
    normalize_gold_source,
    normalize_vitis_pragmas,
    parse_defines,
    remove_pragma_macro_definitions,
)
from prepare_benchmarks import _strip_hls_constructs  # noqa: E402

DEFAULT_TIER_A = _REPO / "related_work/benchmarks/HLSFactory_benchmarks/tier_A"
DEFAULT_OUTPUT = _REPO / "related_work/benchmarks/HLSFactory_benchmarks/tier_A_ready"

TARGET_PART = "xcu280-fsvh2892-2L-e"
TARGET_CLOCK_NS = 3.33

SET_TOP_RE = re.compile(r"^\s*set_top\s+(\S+)", re.MULTILINE | re.IGNORECASE)
ADD_FILES_RE = re.compile(
    r"^\s*add_files(?:\s+-tb)?\s+(\S+)",
    re.MULTILINE | re.IGNORECASE,
)

FORGEBENCH_SKIP_NO_TB = {
    "activation_module",
    "activation_op1",
    "activation_op2",
    "activation_op3",
    "attn_breakdown_module",
    "attn_breakdown_op1",
    "attn_breakdown_op2",
    "conv_block_module",
    "conv_block_op1",
    "conv_block_op2",
    "conv_block_op3",
    "conv_module",
    "diff_dims_module_large",
    "diff_dims_module_small",
    "diff_orders_module",
    "Llama_GPT_module",
    "mult_op_module_dot",
    "mult_op_module_mm",
    "mult_op_module_mmv",
    "vec_mtx_module",
}


@dataclass
class BenchSpec:
    dataset: str
    design: str
    source_dir: Path
    bench_name: str
    kernel_path: Path
    header_paths: List[Path] = field(default_factory=list)
    testbench_path: Optional[Path] = None
    support_paths: List[Path] = field(default_factory=list)
    top_function: Optional[str] = None
    supports_csim: bool = True
    skip_reason: Optional[str] = None
    extra_notes: List[str] = field(default_factory=list)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _extract_top_declaration(gold: str, top: str) -> Optional[str]:
    match = re.search(rf"void\s+{re.escape(top)}\s*\(([^)]*)\)", gold)
    if not match:
        return None
    return f"void {top}({match.group(1)});"


def _header_declares_top(header_text: str, top: str) -> bool:
    return bool(re.search(rf"\b{re.escape(top)}\s*\(", header_text))


def _ensure_kernel_prototype_header(
    out_dir: Path,
    *,
    design: str,
    top: str,
    gold: str,
    existing_headers: Dict[str, str],
) -> Optional[str]:
    for text in existing_headers.values():
        if _header_declares_top(text, top):
            return None
    decl = _extract_top_declaration(gold, top)
    if not decl:
        return None
    includes = [f'#include "{name}"' for name in sorted(existing_headers) if name.endswith(".h")]
    proto_name = f"{design}.h"
    guard = f"{design.upper()}_H_"
    body = "\n".join(includes + ["", decl, ""])
    content = f"#ifndef {guard}\n#define {guard}\n{body}\n#endif\n"
    (out_dir / proto_name).write_text(content, encoding="utf-8")
    return proto_name


def _patch_testbench_includes(tb: str, include_name: str) -> str:
    if f'"{include_name}"' in tb or f"<{include_name}>" in tb:
        return tb
    return f'#include "{include_name}"\n' + tb


def _parse_set_top(tcl_paths: List[Path]) -> Optional[str]:
    for path in tcl_paths:
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        match = SET_TOP_RE.search(text)
        if match:
            return match.group(1).strip("{}")
    return None


def _plain_has_pragma_leak(strip_report: dict) -> bool:
    return bool(
        strip_report.get("plain_contains_hls_pragmas")
        or strip_report.get("plain_contains_accel_pragmas")
    )


def _emit_tcl(
    *,
    project_name: str,
    top: str,
    kernel_file: str,
    header_files: List[str],
    support_files: List[str],
    tb_file: Optional[str],
    tb_data_files: List[str],
    csim: bool,
) -> str:
    lines = [
        f'open_project -reset {project_name}',
        f"set_top {top}",
        f"add_files {kernel_file}",
    ]
    for hf in header_files:
        lines.append(f"add_files {hf}")
    for sf in support_files:
        if sf not in header_files and sf != kernel_file:
            lines.append(f"add_files {sf}")
    if tb_file:
        lines.append(f"add_files -tb {tb_file}")
        for hf in header_files:
            lines.append(f"add_files -tb {hf}")
        for data in tb_data_files:
            lines.append(f"add_files -tb {data}")
    lines.extend(
        [
            'open_solution sol1 -flow_target vitis',
            f"set_part {{{TARGET_PART}}}",
            f"create_clock -period {TARGET_CLOCK_NS} -name default",
        ]
    )
    if csim:
        lines.append("csim_design")
    else:
        lines.append("csynth_design")
    lines.append("exit")
    return "\n".join(lines) + "\n"


def _fix_makefile(text: str) -> str:
    return text.replace(
        "VITIS_HLS_DIR := /tools/software/xilinx/Vitis_HLS/2023.1",
        "VITIS_HLS_DIR ?= $(XILINX_HLS)",
    )


def materialize_bench(spec: BenchSpec, output_root: Path) -> dict[str, Any]:
    if spec.skip_reason:
        return {
            "dataset": spec.dataset,
            "design": spec.design,
            "bench_name": spec.bench_name,
            "status": "skip",
            "reason": spec.skip_reason,
        }

    out_dir = output_root / spec.bench_name
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    kernel_raw = spec.kernel_path.read_text(encoding="utf-8", errors="ignore")
    header_texts = [
        p.read_text(encoding="utf-8", errors="ignore") for p in spec.header_paths if p.is_file()
    ]
    defines = parse_defines(kernel_raw, *header_texts)

    gold_kernel = normalize_gold_source(kernel_raw, header_texts)
    plain_kernel, strip_report = _strip_hls_constructs(kernel_raw, keep_ap_includes=True)

    if _plain_has_pragma_leak(strip_report):
        return {
            "dataset": spec.dataset,
            "design": spec.design,
            "bench_name": spec.bench_name,
            "status": "skip",
            "reason": "plain.cpp still contains HLS pragma leaks",
            "strip_report": strip_report,
        }

    (out_dir / "hls_baseline.cpp").write_text(gold_kernel, encoding="utf-8")
    (out_dir / "gold_hls_source.cpp").write_text(gold_kernel, encoding="utf-8")
    (out_dir / "plain.cpp").write_text(plain_kernel, encoding="utf-8")

    header_names: List[str] = []
    primary_header: Optional[str] = None
    header_contents: Dict[str, str] = {}
    for hp in spec.header_paths:
        if not hp.is_file():
            continue
        plain_header = remove_pragma_macro_definitions(hp.read_text(encoding="utf-8", errors="ignore"))
        dest = out_dir / hp.name
        dest.write_text(plain_header, encoding="utf-8")
        header_names.append(hp.name)
        header_contents[hp.name] = plain_header
        if primary_header is None and hp.name not in {"params.h"}:
            primary_header = hp.name
    if not primary_header and header_names:
        primary_header = header_names[0]

    top = spec.top_function or _parse_set_top(
        list(spec.source_dir.glob("*.tcl"))
    ) or infer_top_function(gold_kernel, fallback=spec.design)

    proto_name = _ensure_kernel_prototype_header(
        out_dir,
        design=spec.design,
        top=top,
        gold=gold_kernel,
        existing_headers=header_contents,
    )
    if proto_name:
        header_names.append(proto_name)
        header_contents[proto_name] = (out_dir / proto_name).read_text(encoding="utf-8")
        if primary_header is None or primary_header == "params.h":
            primary_header = proto_name

    support_rel: List[str] = []
    tb_data_rel: List[str] = []
    for sp in spec.support_paths:
        if not sp.is_file():
            continue
        if sp.name.startswith("DRAM_") or sp.name.startswith("BRAM_"):
            dest = out_dir / "support" / sp.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(sp, dest)
            rel = str(dest.relative_to(out_dir))
            support_rel.append(rel)
            tb_data_rel.append(rel)
        elif sp.name == "gnn_builder_lib.h":
            dest = out_dir / sp.name
            shutil.copy2(sp, dest)
            support_rel.append(sp.name)
        else:
            dest = out_dir / "support" / sp.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(sp, dest)
            support_rel.append(str(dest.relative_to(out_dir)))

    tb_dest_name: Optional[str] = None
    if spec.testbench_path and spec.testbench_path.is_file():
        tb_raw = spec.testbench_path.read_text(encoding="utf-8", errors="ignore")
        tb_gold = normalize_vitis_pragmas(tb_raw, defines)
        if proto_name:
            tb_gold = _patch_testbench_includes(tb_gold, proto_name)
        (out_dir / "testbench.cpp").write_text(tb_gold, encoding="utf-8")
        tb_dest_name = "testbench.cpp"

    kernel_out_name = "hls_baseline.cpp"
    project_name = f"{spec.bench_name}_prj"
    synth_tcl = _emit_tcl(
        project_name=project_name,
        top=top,
        kernel_file=kernel_out_name,
        header_files=header_names,
        support_files=support_rel,
        tb_file=tb_dest_name,
        tb_data_files=tb_data_rel,
        csim=False,
    )
    (out_dir / "dataset_hls.tcl").write_text(synth_tcl, encoding="utf-8")

    if tb_dest_name:
        csim_tcl = _emit_tcl(
            project_name=project_name,
            top=top,
            kernel_file=kernel_out_name,
            header_files=header_names,
            support_files=support_rel,
            tb_file=tb_dest_name,
            tb_data_files=tb_data_rel,
            csim=True,
        )
        (out_dir / "dataset_hls_csim.tcl").write_text(csim_tcl, encoding="utf-8")

    makefile_src = spec.source_dir / "makefile_testbench"
    if makefile_src.is_file():
        (out_dir / "makefile_testbench").write_text(
            _fix_makefile(makefile_src.read_text(encoding="utf-8", errors="ignore")),
            encoding="utf-8",
        )

    cosim_depths = infer_cosim_depths(gold_kernel, top)
    supports_csim = spec.supports_csim and bool(tb_dest_name)

    meta: dict[str, Any] = {
        "benchmark": spec.bench_name,
        "source_repo": "HLSFactory_tier_A",
        "dataset": spec.dataset,
        "design": spec.design,
        "algorithm_source_path": str(spec.kernel_path.resolve()),
        "gold_hls_source_path": str(spec.kernel_path.resolve()),
        "gold_hls_source_file": "gold_hls_source.cpp",
        "gold_hls_baseline_file": "hls_baseline.cpp",
        "kernel_file": spec.kernel_path.name,
        "header_file": primary_header,
        "baseline_variant": f"{spec.bench_name}_0_baseline",
        "translated_hls_top": top,
        "hls_top": top,
        "kernel_top": top,
        "testbench_file": tb_dest_name,
        "support_files": support_rel,
        "include_dirs": ["support"] if (out_dir / "support").exists() else [],
        "variants": [
            {
                "name": f"{spec.bench_name}_0_baseline",
                "file": "hls_baseline.cpp",
                "source_path": str(spec.kernel_path.resolve()),
            }
        ],
        "supports_csim": supports_csim,
        "supports_cosim": supports_csim,
        "cosim_depths": cosim_depths,
        "cosim_harness": "vitis_hls_c_rtl",
        "preferred_gt_file": "hls_baseline.cpp",
        "target_part": TARGET_PART,
        "target_clock_ns": TARGET_CLOCK_NS,
        "skip_phase_a": bool(
            strip_report.get("plain_contains_ap_uint")
            or "hls::" in plain_kernel
            or "ap_uint<" in plain_kernel
        ),
        "strip_report": strip_report,
        "provenance": {
            "gold_hls_source_sha256": _sha256(gold_kernel),
            "gold_hls_baseline_sha256": _sha256(gold_kernel),
            "plain_c_sha256": _sha256(plain_kernel),
            "plain_derived_from_gold_hls": True,
        },
    }
    if spec.extra_notes:
        meta["notes"] = spec.extra_notes
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    return {
        "dataset": spec.dataset,
        "design": spec.design,
        "bench_name": spec.bench_name,
        "status": "ok",
        "output_dir": str(out_dir),
        "top_function": top,
        "strip_report": strip_report,
        "supports_csim": supports_csim,
    }


def _discover_spector(design_dir: Path) -> BenchSpec:
    design = design_dir.name
    tcl = design_dir / "dataset_hls.tcl"
    tcl_text = tcl.read_text(encoding="utf-8", errors="ignore") if tcl.is_file() else ""
    kernel_name = None
    for match in ADD_FILES_RE.finditer(tcl_text):
        fname = match.group(1)
        if fname.endswith((".cpp", ".c")) and "-tb" not in match.group(0).lower():
            if "_tb" not in fname:
                kernel_name = fname
                break
    if kernel_name is None:
        cpp_files = [
            p for p in design_dir.glob("*.cpp")
            if "_tb" not in p.name and "testbench" not in p.name.lower()
        ]
        if not cpp_files:
            return BenchSpec("spector_hls", design, design_dir, f"spector_hls_{design}", design_dir / "missing.cpp", skip_reason="no kernel cpp")
        kernel_path = max(cpp_files, key=lambda p: p.stat().st_size)
    else:
        kernel_path = design_dir / kernel_name

    headers = sorted(
        p for p in design_dir.glob("*.h")
        if p.is_file()
    )
    tb_candidates = sorted(design_dir.glob("*_tb.cpp"))
    tb = tb_candidates[0] if tb_candidates else None
    top = _parse_set_top([tcl])

    return BenchSpec(
        dataset="spector_hls",
        design=design,
        source_dir=design_dir,
        bench_name=f"spector_hls_{design}",
        kernel_path=kernel_path,
        header_paths=headers,
        testbench_path=tb,
        top_function=top,
    )


def _discover_hp_fft(design_dir: Path) -> BenchSpec:
    design = design_dir.name
    kernel = design_dir / "FFT.cpp"
    header = design_dir / "FFT.h"
    tb = design_dir / "testbench.cpp"
    return BenchSpec(
        dataset="hp_fft_hls",
        design=design,
        source_dir=design_dir,
        bench_name=f"hp_fft_{design}",
        kernel_path=kernel,
        header_paths=[header] if header.is_file() else [],
        testbench_path=tb if tb.is_file() else None,
        top_function="FFT_TOP",
    )


def _discover_forgebench(design_dir: Path) -> BenchSpec:
    design = design_dir.name
    if design in FORGEBENCH_SKIP_NO_TB:
        return BenchSpec(
            "forgebench", design, design_dir, f"forgebench_{design}",
            design_dir / "top.cpp", skip_reason="no tb_top.cpp (module-only design)",
        )
    tb = design_dir / "tb_top.cpp"
    if not tb.is_file():
        reason = "tb_top.cpp referenced in TCL but missing on disk" if design == "tiled_attn_module" else "no tb_top.cpp"
        return BenchSpec(
            "forgebench", design, design_dir, f"forgebench_{design}",
            design_dir / "top.cpp", skip_reason=reason,
        )
    support = sorted(
        list(design_dir.glob("DRAM_*.txt")) + list(design_dir.glob("BRAM_*.txt"))
    )
    return BenchSpec(
        dataset="forgebench",
        design=design,
        source_dir=design_dir,
        bench_name=f"forgebench_{design}",
        kernel_path=design_dir / "top.cpp",
        header_paths=[design_dir / "top.h"],
        testbench_path=tb,
        support_paths=support,
        top_function="top",
    )


def _discover_gnnbuilder(design_dir: Path) -> BenchSpec:
    design = design_dir.name
    tb_data = design_dir / "tb_data"
    notes: List[str] = []
    supports_csim = True
    if not tb_data.is_dir():
        notes.append("tb_data/ missing — set supports_csim false until binary inputs are added")
        supports_csim = False
    support = [p for p in [design_dir / "gnn_builder_lib.h"] if p.is_file()]
    return BenchSpec(
        dataset="gnnbuilder",
        design=design,
        source_dir=design_dir,
        bench_name=f"gnnbuilder_{design}",
        kernel_path=design_dir / "model.cpp",
        header_paths=[design_dir / "model.h"],
        testbench_path=design_dir / "model_tb.cpp",
        support_paths=support,
        top_function=_parse_set_top(list(design_dir.glob("*.tcl"))) or f"{design}_top",
        supports_csim=supports_csim,
        extra_notes=notes,
    )


def discover_all(tier_a_root: Path) -> List[BenchSpec]:
    specs: List[BenchSpec] = []
    spector_root = tier_a_root / "spector_hls"
    if spector_root.is_dir():
        for d in sorted(p for p in spector_root.iterdir() if p.is_dir()):
            specs.append(_discover_spector(d))

    hp_root = tier_a_root / "hp_fft_hls"
    if hp_root.is_dir():
        for d in sorted(p for p in hp_root.iterdir() if p.is_dir()):
            specs.append(_discover_hp_fft(d))

    forge_root = tier_a_root / "forgebench"
    if forge_root.is_dir():
        for d in sorted(p for p in forge_root.iterdir() if p.is_dir()):
            specs.append(_discover_forgebench(d))

    gnn_root = tier_a_root / "gnnbuilder" / "designs"
    if gnn_root.is_dir():
        for d in sorted(p for p in gnn_root.iterdir() if p.is_dir()):
            specs.append(_discover_gnnbuilder(d))

    return specs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier-a-root", type=Path, default=DEFAULT_TIER_A)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=["spector_hls", "hp_fft_hls", "forgebench", "gnnbuilder"],
        help="Subset of datasets to materialize",
    )
    args = parser.parse_args()

    allowed = set(args.datasets)
    specs = [s for s in discover_all(args.tier_a_root) if s.dataset in allowed]
    args.output_root.mkdir(parents=True, exist_ok=True)

    results: List[dict[str, Any]] = []
    counts = {"ok": 0, "skip": 0}
    for spec in specs:
        result = materialize_bench(spec, args.output_root)
        results.append(result)
        counts[result["status"]] = counts.get(result["status"], 0) + 1
        print(f"[{result['status']}] {spec.bench_name}: {result.get('reason', result.get('output_dir', ''))}")

    manifest = {
        "tier": "A_ready",
        "target_part": TARGET_PART,
        "target_clock_ns": TARGET_CLOCK_NS,
        "output_root": str(args.output_root.resolve()),
        "datasets": sorted(allowed),
        "summary": counts,
        "designs": results,
    }
    manifest_path = args.output_root / "MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {manifest_path} ({counts.get('ok', 0)} ok, {counts.get('skip', 0)} skip)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
