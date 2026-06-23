#!/usr/bin/env python3
"""Add cfd_flux, cfd_step_factor, lc_gicov, lc_mgvf, and lc_dilate from
rodinia-hls-nova to the c2hls corpus.

These are nested under cfd/ and leukocyte/ in the upstream repo (the parent
groups things-that-share-common-mk-and-data, not a single benchmark). We
treat each leaf sub-kernel as its own benchmark in our corpus, mirroring the
existing rodinia/<bench> structure.

Reuses helpers from prepare_benchmarks.py (pragma stripping, include
localisation, hashing) so the new entries match the existing corpus on
provenance fields (`source_repo`, `gold_hls_source_path`, etc.).

Testbenches are NOT generated — these benches will surface in the corpus
with `supports_csim: false` so the orchestrator's GT-validation skips csim
and uses csynth-only. Once the U280 dev platform lands, sw_emu/hw_emu via
`make check` will be the authoritative correctness + cycle path.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

from prepare_benchmarks import (  # noqa: E402
    _localize_hls_support,
    _strip_hls_constructs,
    _sha256,
)

from c2hls_paths import BENCHMARKS_DIR, active_site, configure_site, rodinia_nova_benchmarks_dir

configure_site()

NOVA_DIR = rodinia_nova_benchmarks_dir()


def _nova_benches() -> list[tuple[str, Path, str]]:
    if NOVA_DIR is None or not NOVA_DIR.is_dir():
        if active_site() == "pc2":
            raise RuntimeError(
                "Set C2HLS_RODINIA_NOVA_DIR in local.env (see local.env.example)."
            )
        raise RuntimeError(f"Nova benchmarks not found: {NOVA_DIR}")
    return [
        ("cfd_flux",         NOVA_DIR / "cfd"        / "cfd_flux",         "cfd_flux"),
        ("cfd_step_factor",  NOVA_DIR / "cfd"        / "cfd_step_factor",  "cfd_step_factor"),
        ("lc_gicov",         NOVA_DIR / "leukocyte"  / "lc_gicov",         "lc_gicov"),
        ("lc_mgvf",          NOVA_DIR / "leukocyte"  / "lc_mgvf",          "lc_mgvf"),
        ("lc_dilate",        NOVA_DIR / "leukocyte"  / "lc_dilate",        "dilate"),
    ]

# (benchmark_name, parent_dir, sub_kernel_dir, kernel_file_basename)
# kernel_file_basename is the cpp/.h stem expected under <variant>/src/.
def _read_variant(variant_dir: Path, kernel_basename: str) -> dict | None:
    src_dir = variant_dir / "src"
    cpp = src_dir / f"{kernel_basename}.cpp"
    hdr = src_dir / f"{kernel_basename}.h"
    if not cpp.exists():
        return None
    return {
        "cpp_path": cpp,
        "header_path": hdr if hdr.exists() else None,
        "cpp_text": cpp.read_text(),
        "header_text": hdr.read_text() if hdr.exists() else "",
    }


def _list_variants(parent_dir: Path, kernel_basename: str) -> list[str]:
    return sorted(
        p.name for p in parent_dir.iterdir()
        if p.is_dir() and (
            p.name.startswith(parent_dir.name + "_")
            or p.name.startswith(kernel_basename + "_")
        )
    )


def _output_variant_filename(variant_name: str) -> str:
    if variant_name.endswith("_0_baseline"):
        return "hls_baseline.cpp"
    return f"hls_{variant_name}.cpp"


def _copy_leukocyte_common(support_dir: Path) -> list[str]:
    """Leukocyte parent has common/ with mars_wide_bus*.h, mc.h, support.h.
    Mirror the same path layout the existing corpus uses (support/common/...).
    """
    src = NOVA_DIR / "leukocyte" / "common"
    if not src.exists():
        return []
    target = support_dir / "common"
    target.mkdir(parents=True, exist_ok=True)
    copied = []
    for name in ("mc.h", "mars_wide_bus.h", "mars_wide_bus_2d.h",
                 "mars_wide_bus_3d.h", "mars_wide_bus_4d.h"):
        s = src / name
        if s.exists():
            shutil.copy2(s, target / name)
            copied.append(f"support/common/{name}")
    return copied


def _prepare_one(bench_name: str, parent: Path, sub_kernel_dir: Path,
                 kernel_basename: str) -> dict:
    variants = _list_variants(sub_kernel_dir, kernel_basename)
    if not variants:
        raise RuntimeError(f"no variants found under {sub_kernel_dir}")
    baseline_name = next((v for v in variants if v.endswith("_0_baseline")), variants[0])

    out_dir = BENCHMARKS_DIR / bench_name
    out_dir.mkdir(parents=True, exist_ok=True)
    support_dir = out_dir / "support"

    # Copy the leukocyte common headers if this bench is from leukocyte.
    support_files = []
    if "leukocyte" in str(sub_kernel_dir):
        support_files = _copy_leukocyte_common(support_dir)

    # Process the baseline first to get the canonical header + plain C source.
    baseline_variant_dir = sub_kernel_dir / baseline_name
    baseline = _read_variant(baseline_variant_dir, kernel_basename)
    if not baseline:
        raise RuntimeError(f"baseline variant missing src/ for {bench_name}")

    # Localise common/ includes from upstream-style ../../common/foo to
    # support/common/foo, matching the existing corpus.
    baseline_cpp_local = _localize_hls_support(baseline["cpp_text"])
    header_local = _localize_hls_support(baseline["header_text"])

    # Strip pragmas → plain C version (LLM input).
    plain_text, strip_report = _strip_hls_constructs(baseline_cpp_local)

    # Write the cleaned baseline cpp + plain.cpp + header.
    (out_dir / "hls_baseline.cpp").write_text(baseline_cpp_local)
    (out_dir / "plain.cpp").write_text(plain_text)
    (out_dir / "gold_hls_source.cpp").write_text(baseline_cpp_local)
    (out_dir / f"{kernel_basename}.h").write_text(header_local)

    # Write each variant's cpp into the corpus.
    variant_records = []
    for variant_name in variants:
        v = _read_variant(sub_kernel_dir / variant_name, kernel_basename)
        if not v:
            continue
        out_name = _output_variant_filename(variant_name)
        (out_dir / out_name).write_text(_localize_hls_support(v["cpp_text"]))
        variant_records.append({
            "name": variant_name,
            "file": out_name,
            "source_path": str(v["cpp_path"].resolve()),
        })

    # NOTE: testbench.cpp is not auto-generated. The orchestrator will fall
    # back to csynth-only when supports_csim is false. The make-check
    # sw_emu/hw_emu flow (post U280-platform install) will use the upstream
    # local_support.cpp + data/ for correctness and cycle measurement.
    metadata = {
        "benchmark": bench_name,
        "source_repo": "rodinia-hls-nova",
        "algorithm_source_path": None,
        "gold_hls_source_path": str((baseline_variant_dir / "src" / f"{kernel_basename}.cpp").resolve()),
        "gold_hls_source_file": "gold_hls_source.cpp",
        "gold_hls_baseline_file": "hls_baseline.cpp",
        "kernel_file": f"{kernel_basename}.cpp",
        "header_file": f"{kernel_basename}.h",
        "baseline_variant": baseline_name,
        "variants": variant_records,
        "preferred_gt_file": _output_variant_filename(variants[-1]),
        "hls_top": "workload",
        "translated_hls_top": "workload",
        "supports_csim": False,
        "supports_cosim": False,
        "testbench_file": None,
        "support_files": support_files,
        "strip_report": strip_report,
        "gold_hls_source_sha256": _sha256(baseline["cpp_text"]),
        "_provenance": {
            "upstream_root": str(parent.resolve()),
            "added_by": "prepare_nova_benchmarks.py",
            "note": "csim/cosim disabled until testbench is generated; "
                    "use make check sw_emu/hw_emu after U280 platform install.",
        },
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def main() -> int:
    summaries = []
    metadata_by_bench = {}
    for bench_name, sub_kernel_dir, kernel_basename in _nova_benches():
        if not sub_kernel_dir.exists():
            print(f"  SKIP {bench_name}: missing {sub_kernel_dir}")
            continue
        parent = sub_kernel_dir.parent
        meta = _prepare_one(bench_name, parent, sub_kernel_dir, kernel_basename)
        metadata_by_bench[bench_name] = meta
        summaries.append({
            "benchmark": bench_name,
            "variants": [v["name"] for v in meta["variants"]],
            "support_files": meta["support_files"],
        })
        print(f"  {bench_name}: {len(meta['variants'])} variants prepared")

    # Update benchmarks/index.json to include the new entries.
    index_path = BENCHMARKS_DIR / "index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text())
    else:
        index = []
    existing = {entry.get("benchmark") for entry in index}
    index = [
        metadata_by_bench.get(entry.get("benchmark"), entry)
        for entry in index
    ]
    for s in summaries:
        if s["benchmark"] not in existing:
            index.append(metadata_by_bench[s["benchmark"]])
    index_path.write_text(json.dumps(index, indent=2) + "\n")
    print(f"\nupdated {index_path} with {len(summaries)} new entries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
