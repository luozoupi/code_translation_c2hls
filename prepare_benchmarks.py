"""
Prepare the curated benchmark corpus from gold HLS sources.

Each benchmark records the provenance chain:
    gold_hls_source -> localized runnable hls_baseline.cpp -> stripped plain.cpp

The resulting corpus lives in code_translation-c2hls/benchmarks and is designed
to be runnable with the local Vitis HLS installation.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from pathlib import Path


ROOT = Path("/home/luo00466/code_translation-c2hls")
BENCHMARKS_DIR = ROOT / "benchmarks"
RODINIA_DIR = Path("/home/luo00466/rodinia-hls/Benchmarks")
ML4ACCEL_DIR = Path("/home/luo00466/ML4Accel-Dataset/fpga_ml_dataset/HLS_dataset")

RODINIA_COMMON_DIR = RODINIA_DIR / "common"


ML4ACCEL_ALGO_SOURCES = {
    "aes": ML4ACCEL_DIR / "hlsyn/sources/aes_kernel.c",
    "fft": ML4ACCEL_DIR / "machsuite/fft_transpose/fft.c",
    "gemm_ncubed": ML4ACCEL_DIR / "hlsyn/sources/gemm-ncubed_kernel.c",
    "md_knn": ML4ACCEL_DIR / "hlsyn/sources/md_kernel.c",
    "sort_merge": ML4ACCEL_DIR / "machsuite/sort_merge/sort.c",
    "spmv_crs": ML4ACCEL_DIR / "hlsyn/sources/spmv-crs_kernel.c",
    "stencil2D": ML4ACCEL_DIR / "hlsyn/sources/stencil_stencil2d_kernel.c",
    "viterbi": ML4ACCEL_DIR / "machsuite/viterbi/viterbi.c",
}

KERNEL_TOPS = {
    "StreamCluster": "workload",
    "aes": "aes256_encrypt_ecb",
    "fft": "fft1D_512",
    "gemm_ncubed": "gemm",
    "hotspot": "hotspot",
    "kmeans": "workload",
    "knn": "workload",
    "lavaMD": "lavaMD_baseline_padded",
    "lud": "lud",
    "md_knn": "md_kernel",
    "nw": "needwun",
    "pathfinder": "pathfinder_kernel",
    "sort_merge": "ms_mergesort",
    "spmv_crs": "spmv",
    "srad": "srad_kernel1",
    "stencil2D": "stencil",
    "viterbi": "viterbi",
}

HLS_TOPS = {bench: "workload" for bench in (
    "StreamCluster",
    "aes",
    "fft",
    "gemm_ncubed",
    "hotspot",
    "kmeans",
    "knn",
    "lavaMD",
    "lud",
    "md_knn",
    "nw",
    "pathfinder",
    "sort_merge",
    "spmv_crs",
    "srad",
    "stencil2D",
    "viterbi",
)}

COSIM_DEPTHS = {
    "spmv_crs": {
        "val": 1666,
        "cols": 1666,
        "rowDelimiters": 495,
        "vec": 494,
        "out": 494,
    },
    "nw": {
        "SEQA": 128,
        "SEQB": 128,
        "alignedA": 256,
        "alignedB": 256,
    },
}


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _read_text(path: Path) -> str:
    return path.read_text()


def _write_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _find_rodinia_variants(bench_dir: Path) -> list[str]:
    variants = []
    if not bench_dir.is_dir():
        return variants
    for child in sorted(bench_dir.iterdir()):
        if child.is_dir() and re.match(r"\w+_\d+_\w+", child.name):
            variants.append(child.name)
    return variants


def _load_existing_template(bench_name: str) -> dict[str, str]:
    bench_dir = BENCHMARKS_DIR / bench_name
    template = {}
    if not bench_dir.exists():
        return template
    for path in bench_dir.rglob("*"):
        if path.is_file():
            template[str(path.relative_to(bench_dir))] = path.read_text()
    return template


def _copy_common_support(support_dir: Path) -> list[str]:
    copied = []
    support_dir.mkdir(parents=True, exist_ok=True)
    for name in ("mc.h", "mars_wide_bus.h", "mars_wide_bus_2d.h", "mars_wide_bus_4d.h"):
        src = RODINIA_COMMON_DIR / name
        dst = support_dir / name
        shutil.copy2(src, dst)
        copied.append(str(dst.relative_to(support_dir.parent.parent)))
    return copied


def _needs_common_support(text: str) -> bool:
    return "common/mc.h" in text


def _localize_hls_support(text: str) -> str:
    return (
        text.replace('"../../../common/', '"support/common/')
            .replace('"../../common/', '"support/common/')
            .replace('"../common/', '"support/common/')
    )


def _strip_hls_constructs(text: str) -> tuple[str, dict]:
    report = {
        "removed_hls_pragmas": 0,
        "removed_accel_pragmas": 0,
        "removed_support_includes": 0,
        "removed_ap_int_includes": 0,
        "removed_extern_c_blocks": 0,
    }

    lines = []
    for line in text.splitlines():
        if re.match(r"\s*#pragma\s+HLS\b", line, re.IGNORECASE):
            report["removed_hls_pragmas"] += 1
            continue
        if re.match(r"\s*#pragma\s+ACCEL\b", line, re.IGNORECASE):
            report["removed_accel_pragmas"] += 1
            continue
        if re.match(r'\s*#include\s+"(?:\.\./)*common/mc\.h"', line):
            report["removed_support_includes"] += 1
            continue
        if re.match(r'\s*#include\s+"support/common/mc\.h"', line):
            report["removed_support_includes"] += 1
            continue
        if re.match(r'\s*#include\s+[<"]ap_int\.h[>"]', line):
            report["removed_ap_int_includes"] += 1
            continue
        lines.append(line)

    stripped = "\n".join(lines)
    new_text, count = re.subn(r'extern\s*"C"\s*\{', "", stripped)
    if count:
        report["removed_extern_c_blocks"] += count
        stripped = new_text
        all_lines = stripped.rstrip().splitlines()
        for idx in range(len(all_lines) - 1, -1, -1):
            if all_lines[idx].strip() == "}":
                all_lines.pop(idx)
                break
        stripped = "\n".join(all_lines)

    report["plain_contains_hls_pragmas"] = bool(re.search(r"#pragma\s+HLS\b", stripped))
    report["plain_contains_accel_pragmas"] = bool(re.search(r"#pragma\s+ACCEL\b", stripped))
    report["plain_contains_ap_uint"] = "ap_uint<" in stripped or "MARS_WIDE_BUS_TYPE" in stripped
    return stripped, report


def _rodinia_spec(bench_name: str, template_files: dict[str, str]) -> dict:
    bench_root = RODINIA_DIR / bench_name
    variants = _find_rodinia_variants(bench_root)
    baseline_variant = next((v for v in variants if "_0_baseline" in v), variants[0])
    baseline_src_dir = bench_root / baseline_variant / "src"
    cpp_files = [
        p.name for p in sorted(baseline_src_dir.iterdir())
        if p.suffix == ".cpp" and "local_support" not in p.name
    ]
    kernel_file = cpp_files[0]
    header_file = kernel_file.replace(".cpp", ".h")
    return {
        "benchmark": bench_name,
        "source_repo": "rodinia-hls",
        "algorithm_source_path": None,
        "gold_hls_source_path": str((baseline_src_dir / kernel_file).resolve()),
        "header_source_path": str((baseline_src_dir / header_file).resolve()) if (baseline_src_dir / header_file).exists() else None,
        "kernel_file": kernel_file,
        "header_file": header_file if (baseline_src_dir / header_file).exists() else None,
        "baseline_variant": baseline_variant,
        "variant_names": variants,
        "variant_source_paths": {
            v: str((bench_root / v / "src" / kernel_file).resolve())
            for v in variants
            if (bench_root / v / "src" / kernel_file).exists()
        },
        "template_files": template_files,
    }


def _ml4accel_spec(bench_name: str, template_files: dict[str, str]) -> dict:
    if "hls_baseline.cpp" not in template_files:
        raise FileNotFoundError(f"Missing curated HLS template for {bench_name}")
    meta_text = template_files.get("metadata.json")
    current_meta = json.loads(meta_text) if meta_text else {}
    header_file = current_meta.get("header_file", "")
    kernel_file = current_meta.get("kernel_file", bench_name + ".cpp")
    gold_source_path = ML4ACCEL_ALGO_SOURCES[bench_name].resolve()
    source_header_path = None
    if header_file:
        candidate_header = gold_source_path.parent / header_file
        if candidate_header.exists():
            source_header_path = str(candidate_header)
    return {
        "benchmark": bench_name,
        "source_repo": "ML4Accel-Dataset",
        "algorithm_source_path": str(gold_source_path),
        "gold_hls_source_path": str(gold_source_path),
        "header_source_path": source_header_path,
        "kernel_file": kernel_file,
        "header_file": header_file or None,
        "baseline_variant": f"{bench_name}_0_baseline",
        "variant_names": [f"{bench_name}_0_baseline"],
        "variant_source_paths": {
            f"{bench_name}_0_baseline": str(gold_source_path),
        },
        "template_files": template_files,
    }


def _build_spec(bench_name: str) -> dict:
    template_files = _load_existing_template(bench_name)
    if bench_name in ML4ACCEL_ALGO_SOURCES:
        return _ml4accel_spec(bench_name, template_files)
    return _rodinia_spec(bench_name, template_files)


def _benchmark_names() -> list[str]:
    index_path = BENCHMARKS_DIR / "index.json"
    with open(index_path) as f:
        metas = json.load(f)
    return [meta["benchmark"] for meta in metas]


def _variant_output_name(variant_name: str) -> str:
    if variant_name.endswith("_0_baseline"):
        return "hls_baseline.cpp"
    return f"hls_{variant_name}.cpp"


def _preferred_rodinia_variant_file(variant_names: list[str]) -> str | None:
    if not variant_names:
        return None
    coalescing = [name for name in variant_names if "coalescing" in name]
    if coalescing:
        return _variant_output_name(coalescing[-1])
    optimized = [name for name in variant_names if "baseline" not in name]
    if optimized:
        return _variant_output_name(optimized[-1])
    return _variant_output_name(variant_names[-1])


def _prepare_single_benchmark(bench_name: str) -> dict:
    spec = _build_spec(bench_name)
    out_dir = BENCHMARKS_DIR / bench_name
    support_files: list[str] = []

    if spec["source_repo"] == "rodinia-hls":
        gold_hls_source = _read_text(Path(spec["gold_hls_source_path"]))
        header_code = _read_text(Path(spec["header_source_path"])) if spec["header_source_path"] else ""
    else:
        gold_hls_source = spec["template_files"]["hls_baseline.cpp"]
        header_code = spec["template_files"].get(spec["header_file"] or "", "")

    localized_gold = _localize_hls_support(gold_hls_source)
    if _needs_common_support(gold_hls_source):
        support_files.extend(_copy_common_support(out_dir / "support/common"))

    plain_code, strip_report = _strip_hls_constructs(localized_gold)

    template_testbench = spec["template_files"].get("testbench.cpp", "")
    variant_entries = []
    variant_outputs = {}
    for variant_name in spec["variant_names"]:
        if spec["source_repo"] == "rodinia-hls":
            variant_source = _read_text(Path(spec["variant_source_paths"][variant_name]))
        else:
            variant_source = spec["template_files"]["hls_baseline.cpp"]
        localized_variant = _localize_hls_support(variant_source)
        variant_file = _variant_output_name(variant_name)
        variant_outputs[variant_file] = localized_variant
        variant_entries.append({
            "name": variant_name,
            "file": variant_file,
            "source_path": spec["variant_source_paths"][variant_name],
        })

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if _needs_common_support(gold_hls_source):
        support_files = _copy_common_support(out_dir / "support/common")

    _write_text(out_dir / "gold_hls_source.cpp", gold_hls_source)
    _write_text(out_dir / "hls_baseline.cpp", localized_gold)
    _write_text(out_dir / "plain.cpp", plain_code)

    if header_code and spec["header_file"]:
        _write_text(out_dir / spec["header_file"], _localize_hls_support(header_code))

    if template_testbench:
        _write_text(out_dir / "testbench.cpp", template_testbench)

    for variant_file, variant_text in variant_outputs.items():
        _write_text(out_dir / variant_file, variant_text)

    preferred_gt_file = None
    if spec["source_repo"] == "rodinia-hls":
        preferred_gt_file = _preferred_rodinia_variant_file(spec["variant_names"])

    metadata = {
        "benchmark": bench_name,
        "source_repo": spec["source_repo"],
        "algorithm_source_path": spec["algorithm_source_path"],
        "gold_hls_source_path": spec["gold_hls_source_path"],
        "gold_hls_source_file": "gold_hls_source.cpp",
        "gold_hls_baseline_file": "hls_baseline.cpp",
        "kernel_file": spec["kernel_file"],
        "header_file": spec["header_file"],
        "baseline_variant": spec["baseline_variant"],
        "variants": variant_entries,
        "variant_source_paths": spec["variant_source_paths"],
        "plain_c_file": "plain.cpp",
        "testbench_file": "testbench.cpp" if template_testbench else None,
        "kernel_top": KERNEL_TOPS.get(bench_name, "workload"),
        "hls_top": HLS_TOPS.get(bench_name, "workload"),
        "translated_hls_top": "workload",
        "support_files": sorted(set(support_files)),
        "include_dirs": [],
        "supports_csim": bool(template_testbench),
        "supports_cosim": bench_name in COSIM_DEPTHS,
        "cosim_depths": COSIM_DEPTHS.get(bench_name, {}),
        "strip_report": strip_report,
        "provenance": {
            "gold_hls_source_sha256": _sha256(gold_hls_source),
            "gold_hls_baseline_sha256": _sha256(localized_gold),
            "plain_c_sha256": _sha256(plain_code),
            "plain_derived_from_gold_hls": True,
        },
        "preferred_gt_file": preferred_gt_file,
    }

    _write_text(out_dir / "metadata.json", json.dumps(metadata, indent=2) + "\n")
    return metadata


def main():
    benchmark_names = _benchmark_names()
    results = []

    print(f"Regenerating {len(benchmark_names)} benchmarks into {BENCHMARKS_DIR}")
    for bench_name in benchmark_names:
        print(f"\nProcessing {bench_name}")
        metadata = _prepare_single_benchmark(bench_name)
        results.append(metadata)
        print(
            f"  source_repo={metadata['source_repo']} "
            f"supports_csim={metadata['supports_csim']} "
            f"supports_cosim={metadata['supports_cosim']}"
        )

    _write_text(BENCHMARKS_DIR / "index.json", json.dumps(results, indent=2) + "\n")
    print(f"\nDone. Regenerated {len(results)} benchmarks.")


if __name__ == "__main__":
    main()
