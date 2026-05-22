"""Adapt external HLS dataset kernels to the c2hls benchmark input shape.

Our pipeline expects each benchmark directory to contain:
    plain.cpp            stripped source (no HLS pragmas, no extern "C",
                         no ap_uint, no MARS_WIDE_BUS — pure C/C++)
    hls_baseline.cpp     the original gold HLS source
    metadata.json        benchmark descriptor (kernel_file, header_file,
                         variants, etc. — see prepare_benchmarks.py)
    [<bench>.h]          header (optional)
    [testbench.cpp]      csim testbench (optional but required for
                         validation_status='validated' under Pillar 9)

External datasets (HLSyn, HLSFactory, CollectiveHLS, HLSPilot) ship in
their own shapes. This module:

1. Surveys the cloned external_datasets/ tree, classifying each source
   file as `clean_c` (already plain), `hls_pragmas` (pragma-bearing),
   `encoded` (e.g. tar.gz), and reports counts → markdown.
2. Adapts a single external kernel into a c2hls-compatible directory by
   reusing the proven _strip_hls_constructs() stripper from
   prepare_benchmarks.py.

The point is to demonstrate that the Phase 1 pipeline (Pillar 1 feedback
parser + Pillar 9 robustness + Pillar 8 trajectory pipeline) is
*content-agnostic* — once a kernel is in our input shape, the same flow
applies regardless of source.
"""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# Reuse the production stripper without copy/pasting it.
import sys
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from prepare_benchmarks import _strip_hls_constructs  # noqa: E402


# Datasets with `.git` dirs we don't want to walk. Skipping them speeds up
# the survey ~10x.
_SKIP_DIRS = {".git", "__pycache__", "node_modules", ".github"}
_C_LIKE_SUFFIXES = {".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp"}

_PRAGMA_RE = re.compile(r"^\s*(?://+\s*)?#pragma\s+(HLS|ACCEL)\b", re.IGNORECASE)
_AP_INT_RE = re.compile(r"\bap_(?:u)?int\s*<")
_MARS_RE = re.compile(r"\bMARS_WIDE_BUS_TYPE\b|\bmemcpy_wide_bus_")
_EXTERN_C_RE = re.compile(r'extern\s*"C"\s*\{')
_HLS_TOP_RE = re.compile(r"#pragma\s+HLS\s+top\s+name\s*=\s*([A-Za-z_]\w*)")
_FUNC_DEF_RE = re.compile(
    r"^\s*(?:extern\s+\"C\"\s+)?(?:[\w:<>,~*&\s]+?)\s+([A-Za-z_]\w*)\s*\([^;]*\)\s*\{",
    re.MULTILINE,
)


@dataclass
class FileClassification:
    path: str
    suffix: str
    is_test: bool                  # name contains tb / test / testbench
    pragma_lines: int              # count of #pragma HLS / ACCEL lines
    extern_c_blocks: int
    ap_uint_uses: int
    mars_uses: int
    line_count: int
    classification: str            # clean_c | hls_pragmas | hls_apuint | hls_mars | testbench

    @property
    def is_clean(self) -> bool:
        return self.classification == "clean_c"


@dataclass
class DatasetReport:
    name: str
    root: str
    total_files: int = 0
    files_by_class: Dict[str, int] = field(default_factory=dict)
    encoded_archives: List[str] = field(default_factory=list)
    sample_clean: List[str] = field(default_factory=list)
    sample_hls: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


def classify_source_file(path: Path) -> FileClassification:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return FileClassification(str(path), path.suffix, False, 0, 0, 0, 0, 0, "unreadable")

    pragma_lines = sum(1 for ln in text.splitlines() if _PRAGMA_RE.match(ln))
    extern_c = len(_EXTERN_C_RE.findall(text))
    ap_uint = len(_AP_INT_RE.findall(text))
    mars = len(_MARS_RE.findall(text))
    name_low = path.name.lower()
    is_test = (
        "_tb." in name_low
        or "testbench" in name_low
        or name_low.startswith("test_")
        or name_low.startswith("tb_")
    )

    if is_test:
        cls = "testbench"
    elif pragma_lines > 0:
        cls = "hls_pragmas"
    elif ap_uint > 0:
        cls = "hls_apuint"
    elif mars > 0:
        cls = "hls_mars"
    else:
        cls = "clean_c"

    return FileClassification(
        path=str(path), suffix=path.suffix, is_test=is_test,
        pragma_lines=pragma_lines, extern_c_blocks=extern_c,
        ap_uint_uses=ap_uint, mars_uses=mars,
        line_count=text.count("\n") + 1,
        classification=cls,
    )


def infer_top_function(source_text: str, fallback: str = "workload") -> str:
    """Infer an HLS top function from pragma metadata or the first definition."""
    match = _HLS_TOP_RE.search(source_text)
    if match:
        return match.group(1)
    for match in _FUNC_DEF_RE.finditer(source_text):
        name = match.group(1)
        if name not in {"if", "for", "while", "switch"}:
            return name
    return fallback


def _walk(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        # Prune skip dirs.
        if any(part in _SKIP_DIRS for part in path.parts):
            continue
        yield path


def survey_dataset(root: Path, name: str, *,
                   sample_per_class: int = 3) -> DatasetReport:
    rep = DatasetReport(name=name, root=str(root))
    if not root.is_dir():
        rep.notes.append(f"directory missing: {root}")
        return rep

    for path in _walk(root):
        if path.suffix in {".gz", ".tar", ".tgz", ".zip"}:
            rep.encoded_archives.append(str(path.relative_to(root)))
            continue
        if path.suffix not in _C_LIKE_SUFFIXES:
            continue
        if not path.is_file():
            continue

        rep.total_files += 1
        fc = classify_source_file(path)
        rep.files_by_class[fc.classification] = rep.files_by_class.get(fc.classification, 0) + 1

        if fc.classification == "clean_c" and len(rep.sample_clean) < sample_per_class:
            rep.sample_clean.append(str(path.relative_to(root)))
        elif fc.classification == "hls_pragmas" and len(rep.sample_hls) < sample_per_class:
            rep.sample_hls.append(str(path.relative_to(root)))

    return rep


def adapt_external_kernel(
    *,
    kernel_path: Path,
    bench_name: str,
    output_dir: Path,
    header_path: Optional[Path] = None,
    testbench_path: Optional[Path] = None,
    support_paths: Optional[List[Path]] = None,
    root_support_paths: Optional[List[Path]] = None,
    source_repo: str = "external",
    top_function: Optional[str] = None,
) -> Dict[str, Any]:
    """Materialize one external HLS kernel into the c2hls benchmark dir
    layout. Returns a dict describing what was emitted plus the strip
    statistics from the proven _strip_hls_constructs() function.

    Output:
        <output_dir>/
            plain.cpp           — stripped, plain C/C++
            hls_baseline.cpp    — original kernel verbatim
            <bench>.h           — header (if provided)
            metadata.json       — minimal c2hls-compatible descriptor
    """
    if not kernel_path.is_file():
        raise FileNotFoundError(f"kernel source missing: {kernel_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = kernel_path.read_text(encoding="utf-8", errors="ignore")
    top_function = top_function or infer_top_function(raw)
    plain, strip_report = _strip_hls_constructs(raw)

    (output_dir / "plain.cpp").write_text(plain, encoding="utf-8")
    (output_dir / "hls_baseline.cpp").write_text(raw, encoding="utf-8")

    header_dest = None
    if header_path and header_path.is_file():
        header_dest = output_dir / header_path.name
        shutil.copy2(header_path, header_dest)

    testbench_dest = None
    if testbench_path and testbench_path.is_file():
        testbench_dest = output_dir / "testbench.cpp"
        shutil.copy2(testbench_path, testbench_dest)

    support_files: List[str] = []
    for support_path in support_paths or []:
        if not support_path.is_file():
            continue
        dst = output_dir / "support" / support_path.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(support_path, dst)
        support_files.append(str(dst.relative_to(output_dir)))

    for support_path in root_support_paths or []:
        if not support_path.is_file():
            continue
        dst = output_dir / support_path.name
        if dst.name in {"plain.cpp", "hls_baseline.cpp", "metadata.json", "testbench.cpp"}:
            continue
        shutil.copy2(support_path, dst)
        rel = str(dst.relative_to(output_dir))
        if rel not in support_files:
            support_files.append(rel)

    meta = {
        "benchmark": bench_name,
        "source_repo": source_repo,
        "algorithm_source_path": str(kernel_path.resolve()),
        "gold_hls_source_path": str(kernel_path.resolve()),
        "gold_hls_source_file": "hls_baseline.cpp",
        "gold_hls_baseline_file": "hls_baseline.cpp",
        "kernel_file": "hls_baseline.cpp",
        "header_file": header_dest.name if header_dest else None,
        "baseline_variant": f"{bench_name}_baseline",
        "translated_hls_top": top_function,
        "hls_top": top_function,
        "testbench_file": testbench_dest.name if testbench_dest else None,
        "support_files": support_files,
        "variants": [
            {
                "name": f"{bench_name}_0_baseline",
                "file": "hls_baseline.cpp",
                "source_path": str(kernel_path.resolve()),
            },
        ],
        "supports_csim": bool(testbench_dest),
        "supports_cosim": False,
        "preferred_gt_file": "hls_baseline.cpp",
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8",
    )

    return {
        "output_dir": str(output_dir),
        "strip_report": strip_report,
        "plain_lines": plain.count("\n") + 1,
        "raw_lines": raw.count("\n") + 1,
        "header_copied": header_dest.name if header_dest else None,
        "testbench_copied": testbench_dest.name if testbench_dest else None,
        "support_files": support_files,
        "top_function": top_function,
    }


def render_survey_markdown(reports: List[DatasetReport]) -> str:
    """Format a list of DatasetReports as a single markdown document."""
    lines: List[str] = ["# External Dataset Compatibility Survey", ""]
    lines.append("Generated by `dataset_pipeline.external_adapter.survey_dataset`. "
                 "Each row counts C/C++ source files in the cloned repo, by their "
                 "compatibility class with the c2hls plain.cpp input shape.")
    lines.append("")
    lines.append("Class definitions:")
    lines.append("- `clean_c` — file is already pragma-free and ap_uint-free; "
                 "drop-in compatible after wrapping into a c2hls bench dir.")
    lines.append("- `hls_pragmas` — file contains `#pragma HLS` / `#pragma ACCEL` "
                 "lines; needs `_strip_hls_constructs()` before use.")
    lines.append("- `hls_apuint` — uses `ap_uint<…>` types; needs the same "
                 "stripper plus type rewriting before plain-C use.")
    lines.append("- `hls_mars` — uses MARS_WIDE_BUS / memcpy_wide_bus helpers; "
                 "stripper handles those too.")
    lines.append("- `testbench` — name flagged as testbench (tb / test_ prefix); "
                 "kept as-is for csim, not used as plain.cpp.")
    lines.append("")
    lines.append("| Dataset | Source files | clean_c | hls_pragmas | hls_apuint | hls_mars | testbench | encoded |")
    lines.append("|---------|--------------|---------|-------------|------------|----------|-----------|---------|")
    for rep in reports:
        c = rep.files_by_class
        enc = len(rep.encoded_archives)
        lines.append(
            f"| **{rep.name}** | {rep.total_files} | "
            f"{c.get('clean_c', 0)} | {c.get('hls_pragmas', 0)} | "
            f"{c.get('hls_apuint', 0)} | {c.get('hls_mars', 0)} | "
            f"{c.get('testbench', 0)} | {enc} |"
        )
    lines.append("")
    for rep in reports:
        lines.append(f"### {rep.name}")
        lines.append("")
        lines.append(f"- root: `{rep.root}`")
        if rep.notes:
            for n in rep.notes:
                lines.append(f"- note: {n}")
        if rep.encoded_archives:
            lines.append(f"- encoded archives ({len(rep.encoded_archives)}): "
                         + ", ".join(f"`{p}`" for p in rep.encoded_archives[:5]))
        if rep.sample_clean:
            lines.append("- sample `clean_c` files (already plain C, drop-in):")
            for p in rep.sample_clean:
                lines.append(f"  - `{p}`")
        if rep.sample_hls:
            lines.append("- sample `hls_pragmas` files (need stripping):")
            for p in rep.sample_hls:
                lines.append(f"  - `{p}`")
        lines.append("")
    return "\n".join(lines) + "\n"
