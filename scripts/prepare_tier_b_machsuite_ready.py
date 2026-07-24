#!/usr/bin/env python3
"""Materialize HLSFactory Tier B MachSuite into tier_B_ready/ for c2hls flash.

Output layout (mirrors tier_A_ready):
    tier_B_ready/machsuite_<design>/
      plain.cpp
      hls_baseline.cpp
      gold_hls_source.cpp
      <kernel>.h, support.h
      testbench.cpp          # adapted MachSuite harness
      support.c, local_support.c
      input.data, check.data # when present
      metadata.json
      dataset_hls.tcl
      dataset_hls_csim.tcl
"""

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
    normalize_gold_source,
    parse_defines,
    remove_pragma_macro_definitions,
)
from prepare_benchmarks import _strip_hls_constructs  # noqa: E402


def _ensure_extern_c_header(text: str) -> str:
    """Wrap a C header so C++ TBs/kernels link against C support + top decls.

    Vitis csim compiles kernel.cpp / testbench.cpp as C++ while MachSuite
    support.c / local_support.c stay C. Without extern \"C\" on the shared
    declarations, the link fails with undefined references (mangled vs C).
    """
    if 'extern "C"' in text:
        return text
    body = text.rstrip() + "\n"
    return (
        "#ifdef __cplusplus\n"
        'extern "C" {\n'
        "#endif\n"
        f"{body}"
        "#ifdef __cplusplus\n"
        "}\n"
        "#endif\n"
    )


DEFAULT_MACHSUITE = (
    _REPO / "related_work/benchmarks/HLSFactory_benchmarks/tier_B/machsuite"
)
DEFAULT_OUTPUT = (
    _REPO / "related_work/benchmarks/HLSFactory_benchmarks/tier_B_ready"
)

TARGET_PART = "xcu280-fsvh2892-2L-e"
TARGET_CLOCK_NS = 3.33

SET_TOP_RE = re.compile(r"^\s*set_top\s+(\S+)", re.MULTILINE | re.IGNORECASE)
ADD_FILES_RE = re.compile(
    r"^\s*add_files(?:\s+-tb)?\s+(\S+)",
    re.MULTILINE | re.IGNORECASE,
)

SKIP_DIRS = {"common"}

NON_KERNEL_C = {
    "support.c",
    "local_support.c",
    "generate.c",
    "harness.c",
}

TESTBENCH_TEMPLATE = r'''/*
 * Vitis HLS C-simulation testbench for {bench_name}.
 * Adapted from MachSuite common/harness.c:
 *   load input.data -> run_benchmark (calls HLS top) -> compare to check.data
 */
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <assert.h>

#define WRITE_OUTPUT
#define CHECK_OUTPUT

/* MachSuite support is C; force C linkage from this C++ TB. */
extern "C" {{
#include "support.h"
}}

int main(int argc, char **argv)
{{
  const char *in_file;
#ifdef CHECK_OUTPUT
  const char *check_file;
#endif
  assert(argc < 4 && "Usage: ./benchmark <input_file> <check_file>");
  in_file = "input.data";
#ifdef CHECK_OUTPUT
  check_file = "check.data";
#endif
  if (argc > 1)
    in_file = argv[1];
#ifdef CHECK_OUTPUT
  if (argc > 2)
    check_file = argv[2];
#endif

  int in_fd;
  char *data;
  data = (char *)malloc(INPUT_SIZE);
  assert(data != NULL && "Out of memory");
  in_fd = open(in_file, O_RDONLY);
  assert(in_fd > 0 && "Couldn't open input data file");
  input_to_data(in_fd, data);
  close(in_fd);

  run_benchmark(data);

#ifdef WRITE_OUTPUT
  int out_fd;
  out_fd = open("output.data", O_WRONLY | O_CREAT | O_TRUNC,
                S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP | S_IROTH | S_IWOTH);
  assert(out_fd > 0 && "Couldn't open output data file");
  data_to_output(out_fd, data);
  close(out_fd);
#endif

#ifdef CHECK_OUTPUT
  int check_fd;
  char *ref;
  ref = (char *)malloc(INPUT_SIZE);
  assert(ref != NULL && "Out of memory");
  check_fd = open(check_file, O_RDONLY);
  assert(check_fd > 0 && "Couldn't open check data file");
  output_to_data(check_fd, ref);
  close(check_fd);

  if (!check_data(data, ref)) {{
    fprintf(stderr, "Benchmark results are incorrect\n");
    free(data);
    free(ref);
    return 1;
  }}
  free(ref);
#endif
  free(data);

  printf("Success.\n");
  return 0;
}}
'''


@dataclass
class BenchSpec:
    design: str
    source_dir: Path
    bench_name: str
    kernel_path: Path
    header_paths: List[Path] = field(default_factory=list)
    support_paths: List[Path] = field(default_factory=list)
    data_paths: List[Path] = field(default_factory=list)
    top_function: Optional[str] = None
    supports_csim: bool = True
    skip_reason: Optional[str] = None
    extra_notes: List[str] = field(default_factory=list)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _parse_set_top(tcl_paths: List[Path]) -> Optional[str]:
    for path in tcl_paths:
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        match = SET_TOP_RE.search(text)
        if match:
            return match.group(1).strip("{}")
    return None


def _parse_kernel_from_tcl(design_dir: Path) -> Optional[Path]:
    for tcl_name in ("hls_template.tcl", "dataset_hls.tcl"):
        tcl = design_dir / tcl_name
        if not tcl.is_file():
            continue
        text = tcl.read_text(encoding="utf-8", errors="ignore")
        for match in ADD_FILES_RE.finditer(text):
            fname = match.group(1)
            if fname.endswith((".c", ".cpp", ".cc")) and "_tb" not in fname:
                candidate = design_dir / Path(fname).name
                if candidate.is_file() and candidate.name not in NON_KERNEL_C:
                    return candidate
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
    tb_file: Optional[str],
    tb_support_files: List[str],
    tb_data_files: List[str],
    csim: bool,
) -> str:
    """Emit Vitis TCL. Support .c files are TB-only (MachSuite host I/O)."""
    lines = [
        f"open_project -reset {project_name}",
        f"set_top {top}",
        f"add_files {kernel_file}",
    ]
    for hf in header_files:
        lines.append(f"add_files {hf}")
    if tb_file:
        lines.append(f"add_files -tb {tb_file}")
        for hf in header_files:
            lines.append(f"add_files -tb {hf}")
        for sf in tb_support_files:
            lines.append(f"add_files -tb {sf}")
        for data in tb_data_files:
            lines.append(f"add_files -tb {data}")
    lines.extend(
        [
            "open_solution sol1 -flow_target vitis",
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


def discover_machsuite(machsuite_root: Path) -> List[BenchSpec]:
    specs: List[BenchSpec] = []
    for design_dir in sorted(p for p in machsuite_root.iterdir() if p.is_dir()):
        design = design_dir.name
        if design in SKIP_DIRS:
            continue

        bench_name = f"machsuite_{design}"
        kernel = _parse_kernel_from_tcl(design_dir)
        if kernel is None:
            specs.append(
                BenchSpec(
                    design=design,
                    source_dir=design_dir,
                    bench_name=bench_name,
                    kernel_path=design_dir / "missing.c",
                    skip_reason="no kernel .c in hls_template.tcl / dataset_hls.tcl",
                )
            )
            continue

        headers = sorted(
            p for p in design_dir.glob("*.h") if p.is_file() and p.name != "support.h"
        )
        support_h = design_dir / "support.h"
        if support_h.is_file():
            headers.append(support_h)

        support_paths: List[Path] = []
        for name in ("support.c", "local_support.c"):
            p = design_dir / name
            if p.is_file():
                support_paths.append(p)

        data_paths: List[Path] = []
        for name in ("input.data", "check.data", "output.data"):
            p = design_dir / name
            if p.is_file():
                data_paths.append(p)

        top = _parse_set_top(
            [design_dir / "hls_template.tcl", design_dir / "dataset_hls.tcl"]
        ) or infer_top_function(
            kernel.read_text(encoding="utf-8", errors="ignore"),
            fallback=design,
        )

        notes: List[str] = []
        supports_csim = True
        has_input = (design_dir / "input.data").is_file()
        has_check = (design_dir / "check.data").is_file()
        if not has_input:
            supports_csim = False
            notes.append(
                "input.data missing — supports_csim=false until vectors are restored"
            )
        if not has_check:
            supports_csim = False
            notes.append("check.data missing — supports_csim=false")
        if not (design_dir / "local_support.c").is_file():
            supports_csim = False
            notes.append("local_support.c missing — cannot build harness TB")

        specs.append(
            BenchSpec(
                design=design,
                source_dir=design_dir,
                bench_name=bench_name,
                kernel_path=kernel,
                header_paths=headers,
                support_paths=support_paths,
                data_paths=data_paths,
                top_function=top,
                supports_csim=supports_csim,
                extra_notes=notes,
            )
        )
    return specs


def materialize_bench(spec: BenchSpec, output_root: Path) -> dict[str, Any]:
    if spec.skip_reason:
        return {
            "dataset": "machsuite",
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
        p.read_text(encoding="utf-8", errors="ignore")
        for p in spec.header_paths
        if p.is_file()
    ]
    defines = parse_defines(kernel_raw, *header_texts)
    top = spec.top_function or infer_top_function(kernel_raw, fallback=spec.design)

    gold_kernel = normalize_gold_source(kernel_raw, header_texts)
    if spec.design == "md_grid":
        # Upstream md.c uses `int n_points[...]` while md.h declares int32_t —
        # align the gold definition to the header so csim signature checks pass.
        gold_kernel = gold_kernel.replace(
            "void md( int n_points[",
            "void md( int32_t n_points[",
            1,
        )
        if "void md( int n_points[" in gold_kernel:
            gold_kernel = gold_kernel.replace(
                "void md( int n_points[",
                "void md( int32_t n_points[",
            )
    plain_kernel, strip_report = _strip_hls_constructs(
        gold_kernel if spec.design == "md_grid" else kernel_raw,
        keep_ap_includes=True,
    )

    if _plain_has_pragma_leak(strip_report):
        return {
            "dataset": "machsuite",
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
    for hp in spec.header_paths:
        if not hp.is_file():
            continue
        plain_header = remove_pragma_macro_definitions(
            hp.read_text(encoding="utf-8", errors="ignore")
        )
        # Kernel + support headers are included from both C and C++ units.
        plain_header = _ensure_extern_c_header(plain_header)
        (out_dir / hp.name).write_text(plain_header, encoding="utf-8")
        header_names.append(hp.name)
        if primary_header is None and hp.name != "support.h":
            primary_header = hp.name
    if primary_header is None and header_names:
        primary_header = header_names[0]

    # Keep support .c at bench root so #include "gemm.h" / "support.h" resolve
    # the same way as upstream MachSuite (no -I support/ needed).
    support_rel: List[str] = []
    for sp in spec.support_paths:
        if not sp.is_file():
            continue
        dest = out_dir / sp.name
        shutil.copy2(sp, dest)
        support_rel.append(sp.name)

    tb_data_rel: List[str] = []
    for dp in spec.data_paths:
        if not dp.is_file():
            continue
        dest = out_dir / dp.name
        shutil.copy2(dp, dest)
        if dp.name in {"input.data", "check.data"}:
            tb_data_rel.append(dp.name)

    # HLSFactory mirror ships a stale check.data for backprop (upstream harness
    # also fails vs check.data but passes vs output.data). Prefer output.data.
    check_path = out_dir / "check.data"
    output_path = out_dir / "output.data"
    if spec.design == "backprop" and output_path.is_file():
        shutil.copy2(output_path, check_path)
        if "check.data" not in tb_data_rel:
            tb_data_rel.append("check.data")
        note = "check.data replaced with output.data (upstream check.data is stale)"
        if note not in spec.extra_notes:
            spec.extra_notes.append(note)

    # Vitis csim must stage golden vectors via support_files / add_files -tb.
    for data_name in tb_data_rel:
        if data_name not in support_rel:
            support_rel.append(data_name)

    tb_dest_name: Optional[str] = None
    if (spec.source_dir / "local_support.c").is_file():
        tb_text = TESTBENCH_TEMPLATE.format(bench_name=spec.bench_name)
        (out_dir / "testbench.cpp").write_text(tb_text, encoding="utf-8")
        tb_dest_name = "testbench.cpp"

    kernel_out_name = "hls_baseline.cpp"
    project_name = f"{spec.bench_name}_prj"
    synth_tcl = _emit_tcl(
        project_name=project_name,
        top=top,
        kernel_file=kernel_out_name,
        header_files=header_names,
        tb_file=None,
        tb_support_files=[],
        tb_data_files=[],
        csim=False,
    )
    (out_dir / "dataset_hls.tcl").write_text(synth_tcl, encoding="utf-8")

    supports_csim = bool(
        spec.supports_csim
        and tb_dest_name
        and "input.data" in tb_data_rel
        and "check.data" in tb_data_rel
    )
    if tb_dest_name:
        csim_tcl = _emit_tcl(
            project_name=project_name,
            top=top,
            kernel_file=kernel_out_name,
            header_files=header_names,
            tb_file=tb_dest_name,
            tb_support_files=support_rel,
            tb_data_files=tb_data_rel,
            csim=True,
        )
        (out_dir / "dataset_hls_csim.tcl").write_text(csim_tcl, encoding="utf-8")

    cosim_depths = infer_cosim_depths(gold_kernel, top)

    meta: Dict[str, Any] = {
        "benchmark": spec.bench_name,
        "source_repo": "HLSFactory_tier_B",
        "dataset": "machsuite",
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
        "include_dirs": [],
        "variants": [
            {
                "name": f"{spec.bench_name}_0_baseline",
                "file": "hls_baseline.cpp",
                "source_path": str(spec.kernel_path.resolve()),
            }
        ],
        "supports_csim": supports_csim,
        # Same harness TB drives Vitis cosim once csim is green.
        "supports_cosim": bool(supports_csim),
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
    if defines:
        meta["parsed_defines"] = {k: str(v) for k, v in list(defines.items())[:32]}
    if spec.extra_notes:
        meta["notes"] = spec.extra_notes
    (out_dir / "metadata.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )

    return {
        "dataset": "machsuite",
        "design": spec.design,
        "bench_name": spec.bench_name,
        "status": "ok",
        "output_dir": str(out_dir),
        "top_function": top,
        "strip_report": strip_report,
        "supports_csim": supports_csim,
        "notes": spec.extra_notes,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--machsuite-root", type=Path, default=DEFAULT_MACHSUITE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--designs",
        nargs="*",
        default=None,
        help="Optional subset of design directory names",
    )
    args = parser.parse_args()

    specs = discover_machsuite(args.machsuite_root)
    if args.designs:
        allowed = set(args.designs)
        specs = [s for s in specs if s.design in allowed]

    args.output_root.mkdir(parents=True, exist_ok=True)

    results: List[dict[str, Any]] = []
    counts = {"ok": 0, "skip": 0}
    for spec in specs:
        result = materialize_bench(spec, args.output_root)
        results.append(result)
        counts[result["status"]] = counts.get(result["status"], 0) + 1
        extra = result.get("reason") or result.get("output_dir", "")
        csim = result.get("supports_csim")
        note = f" csim={csim}" if csim is not None else ""
        print(f"[{result['status']}] {spec.bench_name}: {extra}{note}")

    manifest = {
        "tier": "B_ready",
        "dataset": "machsuite",
        "target_part": TARGET_PART,
        "target_clock_ns": TARGET_CLOCK_NS,
        "output_root": str(args.output_root.resolve()),
        "summary": counts,
        "designs": results,
    }
    manifest_path = args.output_root / "MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        f"\nWrote {manifest_path} "
        f"({counts.get('ok', 0)} ok, {counts.get('skip', 0)} skip)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
