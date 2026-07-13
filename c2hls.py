"""
C-to-HLS Translation Pipeline.

Adapts the Fortran-to-C++ pipeline for translating plain C kernels
into Xilinx Vitis HLS optimized code.

Pipeline:
  Reference Gate: Validate or load trusted reference evidence
  Phase A: Validate input C code compiles with g++
  Phase B: LLM translates C -> HLS-C, validate with Vitis HLS synthesis
  Phase C: Compare synthesis reports against the validated reference offline
"""

import copy
import hashlib
import json
import logging
import os
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional, Tuple

from dotenv import load_dotenv
from openai import OpenAI
from c2hls_temp import make_tempdir

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

from prompt_c2hls import *
from prompt_c2hls import (
    DEFAULT_OPT_STEPS,
    Instruction_c2hls_multistep,
    OPTIMIZATION_PROMPTS,
    hls_synthesis_timeout_fix,
)
from hls_eval import (
    DEFAULT_CLOCK_NS,
    DEFAULT_PART,
    compare_reports,
    format_report_summary,
    run_cosim,
    run_csim,
    run_hls_synthesis,
    run_native_testbench,
)

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(filename)s - %(lineno)d - %(levelname)s - %(message)s'
)

# === Configuration ===========================================================
# All constants below can be overridden via environment variables. See README
# for the full list and how to set them (.env, shell export, or direct edit).
REPO_ROOT = Path(__file__).resolve().parent

TRUSTED_EXTERNAL_REFERENCE_REPOS = {"rodinia-hls", "rodinia-hls-nova"}
_DIRECT_REFERENCE_CACHE: dict | None = None
REFERENCE_CACHE_DIR_ENV = "C2HLS_REFERENCE_CACHE_DIR"
REFERENCE_CACHE_REQUIRE_COSIM_ENV = "C2HLS_REFERENCE_CACHE_REQUIRE_COSIM"
REFERENCE_COSIM_SELECTED_ONLY_ENV = "C2HLS_REFERENCE_COSIM_SELECTED_ONLY"
REFERENCE_COSIM_BASELINE_ENV = "C2HLS_REFERENCE_COSIM_BASELINE"
REFERENCE_CACHE_SCHEMA_VERSION = "4.0"

# Paths to API key files (used only when ANTHROPIC_API_KEY / OPENAI_API_KEY
# environment variables are unset). The defaults point at the developer's
# local keys; set C2HLS_CLAUDE_KEY_FILE / C2HLS_OPENAI_KEY_FILE to override.
CLAUDE_API_KEY_FILE = Path(
    os.getenv("C2HLS_CLAUDE_KEY_FILE", "/home/luo00466/claude-api-key.txt")
)
OPENAI_API_KEY_FILE = Path(
    os.getenv("C2HLS_OPENAI_KEY_FILE", "/home/luo00466/gpt-key.txt")
)

# Hosted OpenAI API endpoint. Override with C2HLS_OPENAI_BASE_URL when using
# a compatible gateway (e.g. Together, OpenRouter).
OPENAI_HOSTED_BASE_URL = os.getenv("C2HLS_OPENAI_HOSTED_URL", "https://api.openai.com/v1")

# Default LLM model id. Override with --model or C2HLS_MODEL.
DEFAULT_MODEL_ID = os.getenv("C2HLS_MODEL", "nvidia/OpenCodeReasoning-Nemotron-1.1-32B")

# Quality-repair loop: how many candidate attempts per benchmark and the
# minimum score improvement (lower = better) required to accept a candidate.
DEFAULT_QUALITY_REPAIR_TURNS = int(os.getenv("C2HLS_QUALITY_REPAIR_TURNS", "2"))
QUALITY_SCORE_EPSILON = float(os.getenv("C2HLS_QUALITY_SCORE_EPSILON", "0.25"))
PHASEB_MODE_ENV = "C2HLS_PHASEB_MODE"
DEFAULT_PHASEB_MODE_SINGLE = "optimized"
DEFAULT_PHASEB_MODE_MULTISTEP = "functional"
STEP_CANDIDATES_ENV = "C2HLS_CANDIDATES_PER_STEP"
CANDIDATE_ATTEMPTS_ENV = "C2HLS_ATTEMPTS_PER_CANDIDATE"
EXHAUSTIVE_CANDIDATE_ATTEMPTS_ENV = "C2HLS_EXHAUSTIVE_CANDIDATE_ATTEMPTS"
COSIM_SKIP_SLOWER_THAN_GOLD_ENV = "C2HLS_COSIM_SKIP_SLOWER_THAN_GOLD"
COSIM_SKIP_GOLD_RATIO_ENV = "C2HLS_COSIM_SKIP_GOLD_RATIO"
DEFAULT_COSIM_SKIP_GOLD_RATIO = 10.0
REFERENCE_BLIND_ENV = "C2HLS_REFERENCE_BLIND"
GT_COMPARISON_IN_CONTROL_ENV = "C2HLS_GT_COMPARISON_IN_CONTROL"
SKILL_LIBRARY_FROZEN_ENV = "C2HLS_SKILL_LIBRARY_FROZEN"
SKILL_UPDATE_STATS_ENV = "C2HLS_SKILL_UPDATE_STATS"
SYNTHESIS_EVAL_BUDGET_ENV = "C2HLS_SYNTHESIS_EVAL_BUDGET"
LLM_CANDIDATE_BUDGET_ENV = "C2HLS_LLM_CANDIDATE_BUDGET"
COSIM_SELECTED_ONLY_ENV = "C2HLS_COSIM_SELECTED_ONLY"
FORCE_SELECTED_COSIM_ENV = "C2HLS_FORCE_SELECTED_COSIM"
CORRECTNESS_BEFORE_SYNTH_ENV = "C2HLS_CORRECTNESS_BEFORE_SYNTH"
FEASIBILITY_SELECTION_ENV = "C2HLS_FEASIBILITY_SELECTION"
LLM_TEMPERATURE_ENV = "C2HLS_LLM_TEMPERATURE"
LLM_TOP_P_ENV = "C2HLS_LLM_TOP_P"
LLM_SEED_ENV = "C2HLS_LLM_SEED"

# Multistep regression guard: a step is rejected if its latency_ns or its
# resource usage grew by this ratio vs the previous step. Defaults to 1.10
# (10% regression triggers a one-shot retry with regression-aware guidance,
# then revert if still regressing). Set to 0 to disable the guard entirely.
STEP_REGRESSION_THRESHOLD = float(os.getenv("C2HLS_STEP_REGRESSION_THRESHOLD", "1.10"))

# Max seconds for the g++ compile-check used in Phase A and before each
# Vitis synthesis attempt. Kept small since compile-check is quick.
TIMEOUT_LIMIT = int(os.getenv("C2HLS_COMPILE_CHECK_TIMEOUT", "60"))
# =============================================================================

# Benchmark-specific guidance. One dict keyed by benchmark name; sub-fields
# scope each hint class:
#   translation  — hints added to Phase B's benchmark_context (LLM translation)
#   quality      — hints added to Phase B's quality-repair prompt
#   priority     — one-liner that leads the quality-repair guidance
BENCHMARK_POLICIES = {
    "nw": {
        "priority": "Primary objective: improve timing/slack/Fmax while keeping the simple workload wrapper and dynamic-programming structure intact.",
        "translation": [
            "The header owns `bench_args_t`; do not redeclare it in the source.",
            "Preserve the existing `needwun` helper structure from the plain input instead of inventing a new algorithm decomposition.",
            "Keep the workload wrapper very close to the plain input: one pair of local dynamic-programming arrays named `M` and `ptr`, then a simple loop over jobs that calls `needwun`.",
            "Do not remove or rename dynamic-programming buffers like `M` and `ptr` if the existing helper logic still requires them.",
            "Avoid aggressive optimization on this benchmark: do not completely partition `M` or `ptr`, do not fully unroll the DP loops, and prefer only light inner-loop pipelining if any.",
        ],
        "quality": [
            "Treat the large dynamic-programming arrays `M` and `ptr` as simple memories; reduce or remove large partition factors on them if timing is poor.",
            "Avoid over-pipelining loops that repeatedly read and write `M` and `ptr` if it hurts timing closure.",
            "Keep the workload wrapper simple; do not add extra buffering layers or duplicated helper logic just to chase throughput.",
        ],
    },
    "spmv_crs": {
        "priority": "Primary objective: reduce resource usage, especially memory-heavy buffering, without making latency much worse.",
        "translation": [
            "Use the existing kernel interface `spmv(val, cols, rowDelimiters, vec, out)` from the header.",
            "Keep the workload wrapper very close to the plain input: preserve the existing local arrays `l_val`, `l_cols`, `l_rowDelimiters`, `l_vec`, and `l_out` plus their copy-in/copy-out loops.",
            "Do not invent new helper buffers beyond the existing plain-input locals unless they are clearly necessary and fully declared.",
            "Keep the wrapper ports aligned with the reference AXI-visible arrays: `val`, `cols`, `rowDelimiters`, `vec`, and `out`.",
            "Do not collapse the wrapper into a direct pointer call to `spmv`; the plain input already gives the intended wrapper structure.",
        ],
        "quality": [
            "Minimize BRAM-heavy local buffering and avoid complete partitioning of `out`/`l_out` or other large arrays unless it clearly pays off.",
            "Prefer modest cyclic factors or no partitioning on large arrays over aggressive partitioning that inflates memory resources.",
            "Keep the copy-in/copy-out loops simple and do not introduce extra array copies unless they materially help timing.",
            "When timing is already healthy, it is acceptable to spend a little area to shrink the remaining latency gap, especially in the compute loop.",
            "When timing is poor, keep the interface pragmas, but remove compute-side PIPELINE/ARRAY_PARTITION/INLINE directives unless they clearly help.",
            "For this benchmark, a simple, low-pressure pragma set is preferable to an over-pragmatized kernel.",
        ],
    },
    "StreamCluster": {
        "priority": "Primary objective: reduce FF/LUT blow-up from over-aggressive parallelism or duplicated logic.",
        "translation": [
            "Preserve the existing helper-call structure from the plain input instead of rewriting the whole benchmark around a new buffer scheme.",
        ],
        "quality": [
            "Reduce FF/LUT blow-up by avoiding aggressive unrolling, inlining, or duplicated helper pipelines.",
            "Prefer shared buffers and sequential helper calls over dataflow-like rewrites that replicate large logic blocks.",
            "Remove unnecessary complete partitioning on large state arrays and keep the design closer to the original helper structure.",
            "This benchmark has large latency headroom, so it is acceptable to relax throughput-oriented pipelining if that improves slack/Fmax or reduces DSP pressure.",
            "Prefer one reusable arithmetic pipeline over DSP-heavy parallel scheduling when timing is poor.",
        ],
    },
    "srad": {
        "priority": "Primary objective: preserve SRAD's tiled halo-row contract before optimizing bandwidth or parallelism.",
        "translation": [
            "SRAD uses one halo row above the active tile. Preserve the copy-back offsets exactly: write tile outputs to `Jout + (t*TILE_ROWS+1)*COLS` and `J + (t*TILE_ROWS+1)*COLS`, not to `t*TILE_ROWS*COLS`.",
            "Do not change the `workload(float J[(ROWS+3)*COLS], float Jout[(ROWS+3)*COLS])` array layout: rows 1..ROWS are the compared interior rows, while row 0 is halo/boundary context.",
            "Keep the local `J_buf` copy-in starting at `J + t*TILE_ROWS*COLS` / `Jout + t*TILE_ROWS*COLS`; only the copy-back to the global output arrays uses the `+1` row offset.",
        ],
        "quality": [
            "Any tiling, pipeline, unroll, doublebuffer, or coalescing edit must preserve SRAD's halo-row copy-back offsets: `(t*TILE_ROWS+1)*COLS` for both Jout and J updates.",
        ],
    },
}


def _policy(benchmark_name: str, field: str, default=None):
    """Look up legacy benchmark guidance without contaminating paper runs.

    ``BENCHMARK_POLICIES`` contains hand-written Rodinia advice learned while
    developing the reference-guided controller.  It is useful for the legacy
    mode, but it is benchmark-specific knowledge and therefore must not enter a
    reference-blind prompt or controller decision.
    """
    if _reference_blind_enabled():
        return default
    return (BENCHMARK_POLICIES.get(benchmark_name or "") or {}).get(field, default)


def _normalize_srad_halo_copy_offsets(code: str) -> tuple[str, list[str]]:
    """Repair the exact SRAD halo-copy offset mistake seen in generated code.

    The SRAD testbench compares interior rows 1..ROWS. Copy-in starts at the
    tile boundary, but copy-back must skip the top halo row. The LLM sometimes
    changes `(t*TILE_ROWS+1)*COLS` to `t*TILE_ROWS*COLS`, which shifts every
    output tile and fails csim/hw_emu. This preflight is intentionally narrow
    and records every edit so it is not a silent fallback.
    """
    if not code:
        return code, []

    replacements: list[tuple[str, str, str]] = [
        (
            r"memcpy\(\s*Jout\s*\+\s*t\s*\*\s*TILE_ROWS\s*\*\s*COLS\s*,",
            "memcpy(Jout+(t*TILE_ROWS+1)*COLS,",
            "Jout copy-back halo offset",
        ),
        (
            r"memcpy\(\s*J\s*\+\s*t\s*\*\s*TILE_ROWS\s*\*\s*COLS\s*,",
            "memcpy(J+(t*TILE_ROWS+1)*COLS,",
            "J copy-back halo offset",
        ),
    ]
    patched = code
    notes: list[str] = []
    for pattern, replacement, note in replacements:
        updated, count = re.subn(pattern, replacement, patched)
        if count:
            patched = updated
            notes.append(f"{note}: restored `(t*TILE_ROWS+1)*COLS` ({count} occurrence(s))")
    return patched, notes


def extract_cpp_code(text: str) -> Optional[str]:
    """Extract C/C++ code from the last fenced block in an LLM response."""
    if not text:
        return None
    fence_pattern = re.compile(r"```(?:cpp|c\+\+|c|hls)?\s*(.*?)```", re.DOTALL)
    matches = fence_pattern.findall(text)
    if matches:
        return matches[-1].strip()
    return None


def _normalize_extra_files(extra_files=None) -> List[Tuple[str, str]]:
    if not extra_files:
        return []
    normalized = []
    for item in extra_files:
        if isinstance(item, dict):
            rel_path = item.get("path")
            content = item.get("content", "")
        else:
            rel_path, content = item
        if rel_path:
            normalized.append((rel_path, content))
    return normalized


def compile_check_cpp(
    code: str,
    header_code: str = "",
    header_name: str = "kernel.h",
    work_dir: str = None,
    extra_files=None,
) -> Tuple[bool, str]:
    """Check if code compiles with g++ -c."""
    if work_dir is None:
        work_dir = make_tempdir(prefix="c2hls_compile_")
    os.makedirs(work_dir, exist_ok=True)

    src_file = os.path.join(work_dir, "kernel.cpp")
    with open(src_file, "w") as f:
        f.write(code)

    if header_code:
        hdr_file = os.path.join(work_dir, header_name)
        os.makedirs(os.path.dirname(hdr_file), exist_ok=True)
        with open(hdr_file, "w") as f:
            f.write(header_code)

    for rel_path, content in _normalize_extra_files(extra_files):
        file_path = os.path.join(work_dir, rel_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as f:
            f.write(content)

    cmd = ["g++", "-c", f"-I{work_dir}", "-o", "/dev/null", src_file]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=TIMEOUT_LIMIT, text=True)
        if result.returncode == 0:
            return True, ""
        return False, result.stderr
    except subprocess.TimeoutExpired:
        return False, "Compilation timed out"


def _binary_status(passed: bool) -> str:
    return "passed" if passed else "failed"


def _env_flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _reference_blind_enabled() -> bool:
    return _env_flag(REFERENCE_BLIND_ENV)


def _ground_truth_control_enabled() -> bool:
    """Whether expert code/metrics may influence controller decisions."""
    return (
        not _reference_blind_enabled()
        and _env_flag(GT_COMPARISON_IN_CONTROL_ENV, "1")
    )


def _skill_updates_enabled() -> bool:
    return (
        not _env_flag(SKILL_LIBRARY_FROZEN_ENV)
        and _env_flag(SKILL_UPDATE_STATS_ENV, "1")
    )


def _skill_library_control_enabled() -> bool:
    mode = os.getenv("C2HLS_SKILL_MODE", "").strip().lower()
    if mode in {"off", "skill_off", "skills_off", "none", "0", "false", "no"}:
        return False
    if mode in {
        "on", "skill_on", "skills_on", "frozen", "frozen_skill",
        "1", "true", "yes",
    }:
        return True
    return True


def _cosim_selected_only() -> bool:
    return _env_flag(COSIM_SELECTED_ONLY_ENV)


def _selected_cosim_measurement_ok(summary: Optional[dict]) -> bool:
    return bool(
        isinstance(summary, dict)
        and summary.get("ran")
        and summary.get("passed")
        and _positive_int(summary.get("kernel_runtime_cycles")) is not None
    )


def _generated_cosim_supported(meta: dict, testbench_code: str) -> bool:
    return bool(
        testbench_code
        and (
            meta.get("supports_cosim")
            or (
                _cosim_selected_only()
                and _env_flag(FORCE_SELECTED_COSIM_ENV)
            )
        )
    )


def _feasibility_selection_enabled() -> bool:
    """Whether candidate choice must use the paper's feasibility predicate."""
    return _reference_blind_enabled() or _env_flag(FEASIBILITY_SELECTION_ENV)


def _test_status(supported: bool, ran: bool, passed: bool) -> str:
    if not supported:
        return "not_supported"
    if not ran:
        return "not_run"
    return _binary_status(passed)


def _extract_failure_excerpt(log: str, fallback: str = "") -> str:
    if not log:
        return fallback
    interesting = []
    for line in log.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        lowered = stripped.lower()
        if (
            ("error" in lowered and "0 errors" not in lowered)
            or "simulation failed" in lowered
            or "segmentation violation" in lowered
            or "child killed" in lowered
            or "undefined symbol" in lowered
            or "did you mean to declare" in lowered
            or "ld.lld" in lowered
            or lowered.startswith("@e ")
        ):
            if stripped not in interesting:
                interesting.append(stripped)
        if len(interesting) >= 4:
            break
    if interesting:
        return "\n".join(interesting)
    return fallback


def _normalize_signature_text(text: str) -> str:
    # Strip C/C++ comments that appear inline in parameter lists
    # (e.g. `float *feature /*[N][F]*/`). The comparison should be on types
    # and names only, not on documentation.
    stripped = re.sub(r"/\*.*?\*/", " ", text or "", flags=re.DOTALL)
    stripped = re.sub(r"//[^\n]*", " ", stripped)
    normalized = " ".join(stripped.strip().split())
    normalized = re.sub(r"\s*([(),\[\]])\s*", r"\1", normalized)
    normalized = normalized.replace(",", ", ")
    normalized = re.sub(r"\s*\*\s*", "*", normalized)
    normalized = re.sub(r"\s*&\s*", "&", normalized)
    return normalized.strip()


def _extract_function_signature(code: str, function_name: str, definitions_only: bool = False) -> Optional[dict]:
    trailer_pattern = r"\{" if definitions_only else r"[;{]"
    pattern = re.compile(
        rf'(^|\n)(?P<indent>\s*)(?P<extern>extern\s*"C"\s+)?'
        rf'(?P<ret>[A-Za-z_][\w:\s\*&<>\[\]]*?)\b(?P<name>{re.escape(function_name)})\s*'
        rf'\((?P<params>.*?)\)\s*(?P<trailer>{trailer_pattern})',
        re.MULTILINE | re.DOTALL,
    )
    match = pattern.search(code or "")
    if not match:
        return None
    signature_start = match.start("extern") if match.group("extern") else match.start("ret")
    signature_end = match.start("trailer")
    return {
        "name": match.group("name"),
        "return_type": match.group("ret").strip(),
        "params": match.group("params").strip(),
        "extern_c": bool(match.group("extern")),
        "signature_start": signature_start,
        "signature_end": signature_end,
    }


def _canonical_function_signature(signature: Optional[dict]) -> str:
    if not signature:
        return ""
    return_type = _normalize_signature_text(signature.get("return_type", ""))
    params = _normalize_signature_text(signature.get("params", ""))
    return f"{return_type} {signature.get('name', '')}({params})".strip()


def _render_function_signature(signature: dict, include_extern: bool = False) -> str:
    prefix = 'extern "C" ' if include_extern else ""
    return (
        f"{prefix}{_normalize_signature_text(signature.get('return_type', ''))} "
        f"{signature.get('name', '')}({_normalize_signature_text(signature.get('params', ''))})"
    ).strip()


def _expected_top_signature(header_code: str, testbench_code: str, function_name: str) -> Optional[dict]:
    for source, label in ((testbench_code, "testbench"), (header_code, "header")):
        signature = _extract_function_signature(source, function_name, definitions_only=False)
        if signature:
            signature["source"] = label
            return signature
    return None


def _top_signature_mismatch_reason(code: str, header_code: str, testbench_code: str,
                                   function_name: str) -> str:
    expected = _expected_top_signature(header_code, testbench_code, function_name)
    current = _extract_function_signature(code, function_name, definitions_only=True)
    if not expected or not current:
        return ""

    expected_linkage = bool(expected.get("extern_c"))
    # `extern "C"` may appear with arbitrary whitespace (`extern"C"`,
    # `extern  "C"`) and the surrounding `extern "C" { ... }` block doesn't
    # land directly on the function — match either form via regex rather than
    # a literal substring.
    current_linkage = bool(
        current.get("extern_c") or
        re.search(r'extern\s*"C"', code or "")
    )
    same_signature = _canonical_function_signature(current) == _canonical_function_signature(expected)
    same_linkage = (not expected_linkage) or current_linkage
    if same_signature and same_linkage:
        return ""

    expected_text = _render_function_signature(expected, include_extern=expected_linkage)
    current_text = _render_function_signature(current, include_extern=current_linkage)
    return (
        f"`{function_name}` uses `{current_text}`, but the benchmark testbench expects "
        f"`{expected_text}`"
    )


def _align_generated_top_signature(code: str, header_code: str, testbench_code: str,
                                   function_name: str) -> tuple[str, str]:
    expected = _expected_top_signature(header_code, testbench_code, function_name)
    current = _extract_function_signature(code, function_name, definitions_only=True)
    if not expected or not current:
        return code, ""

    notes = []
    updated = code
    has_extern_linkage = current.get("extern_c") or ('extern "C"' in (code or ""))
    needs_extern = bool(expected.get("extern_c") and not has_extern_linkage)
    if _canonical_function_signature(current) != _canonical_function_signature(expected) or needs_extern:
        replacement = _render_function_signature(expected, include_extern=needs_extern)
        updated = (
            code[:current["signature_start"]]
            + replacement
            + " "
            + code[current["signature_end"]:]
        )
        if _canonical_function_signature(current) != _canonical_function_signature(expected):
            notes.append(f"normalized `{function_name}` signature to match the {expected.get('source', 'expected')} declaration")
        if needs_extern:
            notes.append(f'added `extern "C"` linkage to `{function_name}`')

    return updated, "; ".join(notes)


def _summarize_synth_result(result: Optional[dict]) -> dict:
    if result is None:
        return {
            "status": "failed",
            "ran": False,
            "success": False,
            "error": "",
            "report": {},
        }
    report = dict(result.get("report", {}) or {})
    passed = bool(result.get("success", False))
    return {
        "status": _binary_status(passed),
        "ran": True,
        "success": passed,
        "error": result.get("error", ""),
        "report": report,
        "work_dir": report.get("work_dir", ""),
    }


def _summarize_test_result(result: Optional[dict], supported: bool) -> dict:
    if not supported:
        return {
            "status": "not_supported",
            "supported": False,
            "ran": False,
            "success": False,
            "passed": False,
            "error": "",
        }
    if result is None:
        return {
            "status": "not_run",
            "supported": True,
            "ran": False,
            "success": False,
            "passed": False,
            "error": "",
        }
    passed = bool(result.get("passed", False))
    error = result.get("error", "")
    if not passed and not error:
        error = _extract_failure_excerpt(result.get("log", ""), "Testbench did not pass")
    timed_out = bool(result.get("timed_out")) or "timed out" in str(error).lower()
    summary = {
        "status": "timeout" if timed_out else _test_status(True, True, passed),
        "supported": True,
        "ran": True,
        "success": bool(result.get("success", False)),
        "passed": passed,
        "error": error,
    }
    work_dir = result.get("work_dir", "")
    if work_dir:
        summary["work_dir"] = work_dir
    # Cosim runs return kernel_runtime_cycles parsed from lat.rpt; preserve
    # it in the summary so downstream tooling (rubric, JSONL exporter) sees
    # the actual RTL cycle count instead of just pass/fail.
    for key in ("kernel_runtime_cycles", "kernel_runtime_us", "kernel_clock_freq_mhz"):
        if key in result:
            summary[key] = result.get(key)
    for key in ("correctness", "golden_output_sha256"):
        if key in result:
            summary[key] = result.get(key)
    log_excerpt = _extract_failure_excerpt(result.get("log", ""))
    if log_excerpt and not passed:
        summary["log_excerpt"] = log_excerpt
    return summary


def _summary_status(summary: Optional[dict], available: bool) -> str:
    if isinstance(summary, dict):
        status = summary.get("status")
        if status in {"passed", "failed", "timeout", "not_run", "not_supported"}:
            return status
        supported = bool(summary.get("supported", available))
        ran = bool(summary.get("ran", False))
        passed = bool(summary.get("passed", False))
        return _test_status(supported, ran, passed)
    return _test_status(available, False, False)


def _cosim_required_for_correctness() -> bool:
    raw = os.getenv("C2HLS_COSIM_REQUIRED", "1").strip().lower()
    return raw in ("1", "true", "yes", "on")


def _positive_int(value: Any) -> Optional[int]:
    try:
        parsed = int(round(float(value)))
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _cosim_skip_gold_ratio() -> float:
    raw = os.getenv(
        COSIM_SKIP_GOLD_RATIO_ENV,
        str(DEFAULT_COSIM_SKIP_GOLD_RATIO),
    )
    try:
        ratio = float(raw)
    except (TypeError, ValueError):
        ratio = DEFAULT_COSIM_SKIP_GOLD_RATIO
    return max(1.0, ratio)


def _cosim_gold_precheck_policy(
    synth_report: Optional[dict],
    reference_cycle_info: Optional[dict],
) -> Optional[dict]:
    """Return structured provenance when cosim should be preempted.

    The policy compares the generated csynth latency estimate with an
    authoritative gold runtime cycle count when available. It is opt-in so
    historical experiments keep their original behavior.
    """
    if _reference_blind_enabled():
        return None
    enabled = os.getenv(COSIM_SKIP_SLOWER_THAN_GOLD_ENV, "0").strip().lower()
    if enabled not in {"1", "true", "yes", "on"}:
        return None

    report = synth_report or {}
    reference = reference_cycle_info or {}
    generated_cycles = _positive_int(
        report.get("latency_cycles_worst") or report.get("latency_cycles")
    )
    gold_cycles = _positive_int(reference.get("cycles"))
    if generated_cycles is None or gold_cycles is None:
        return None

    threshold_ratio = _cosim_skip_gold_ratio()
    observed_ratio = generated_cycles / gold_cycles
    if observed_ratio < threshold_ratio:
        return None

    return {
        "schema_version": "1.0",
        "policy": "gold_relative_csynth_precheck",
        "decision": "skip",
        "classification": "predicted_timeout",
        "ran": False,
        "reason": "generated_csynth_latency_exceeds_gold_ratio_threshold",
        "generated_csynth_latency_cycles": generated_cycles,
        "gold_reference_cycles": gold_cycles,
        "gold_reference_source": reference.get("source") or "unknown",
        "gold_reference_metric": reference.get("metric") or "runtime_cycles",
        "ratio_generated_over_gold": round(observed_ratio, 6),
        "threshold_ratio": threshold_ratio,
    }


def _predicted_cosim_timeout_summary(policy: dict) -> dict:
    generated_cycles = policy["generated_csynth_latency_cycles"]
    gold_cycles = policy["gold_reference_cycles"]
    ratio = policy["ratio_generated_over_gold"]
    threshold = policy["threshold_ratio"]
    return {
        "status": "timeout",
        "supported": True,
        "ran": False,
        "success": False,
        "passed": False,
        "error": (
            "Cosim classified as predicted timeout and not run: generated "
            f"csynth latency {generated_cycles} cycles is {ratio:.3f}x the "
            f"gold reference {gold_cycles} cycles (skip threshold {threshold:.3f}x)."
        ),
        "skip_reason": "predicted_longer_than_gold",
        "classification": "predicted_timeout",
        "profile_required": True,
        "cosim_policy": policy,
    }


def _test_summary_is_timeout(summary: Optional[dict]) -> bool:
    if not isinstance(summary, dict):
        return False
    status = str(summary.get("status") or "").strip().lower()
    error = str(summary.get("error") or "").strip().lower()
    return status == "timeout" or "timed out" in error


def _skill_prompt_injection_enabled() -> bool:
    """Honor an explicit skill-off request while preserving legacy default."""
    raw = os.getenv("C2HLS_FORCE_SKILL_PROMPTS")
    if raw is None or not raw.strip():
        return True
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _run_synth_csim_cosim(
    hls_code: str,
    header_code: str,
    header_name: str,
    top_function: str,
    part: str,
    clock_ns: float,
    extra_files: list,
    testbench_code: str = "",
    run_csim_check: bool = True,
    run_cosim_check: bool = False,
    cosim_depths: Optional[dict] = None,
    cosim_requires_csim_pass: bool = False,
    cosim_reference_cycle_info: Optional[dict] = None,
    golden_output_text: str = "",
    golden_output_specs: Optional[dict] = None,
    log_prefix: str = "",
) -> dict:
    """Synthesize HLS code, then optionally run csim and cosim.

    csim runs when the synthesis succeeded, a testbench is provided, and
    run_csim_check is True. cosim runs when synthesis succeeded, a testbench
    is provided, run_cosim_check is True, and (when cosim_requires_csim_pass
    is True) the csim run either passed or was not attempted.

    Returns {synth, csim, cosim} where synth is the raw synthesis result and
    csim/cosim are _summarize_test_result dicts (or None when skipped).
    """
    csim_summary = None
    correctness_first = _env_flag(CORRECTNESS_BEFORE_SYNTH_ENV)
    if correctness_first and testbench_code and run_csim_check:
        if log_prefix:
            logging.info("%s Running correctness gate before synthesis (csim)...", log_prefix)
        csim_result = run_csim(
            hls_code,
            testbench_code,
            header_code,
            header_name=header_name,
            top_function=top_function,
            part=part,
            clock_ns=clock_ns,
            extra_files=extra_files,
            golden_output_text=golden_output_text,
            golden_output_specs=golden_output_specs or {},
        )
        csim_summary = _summarize_test_result(csim_result, True)

    correctness_gate_failed = bool(
        correctness_first
        and isinstance(csim_summary, dict)
        and csim_summary.get("ran")
        and not csim_summary.get("passed")
    )
    if correctness_gate_failed:
        synth_result = {
            "success": False,
            "ran": False,
            "skipped": True,
            "skip_reason": "csim_correctness_gate_failed",
            "error": (
                "Synthesis not run because the candidate failed the "
                "pre-synthesis CSim/golden correctness gate"
            ),
            "report": {},
        }
    else:
        synth_result = run_hls_synthesis(
            hls_code,
            header_code,
            header_name=header_name,
            top_function=top_function,
            part=part,
            clock_ns=clock_ns,
            extra_files=extra_files,
        )
        synth_result.setdefault("ran", True)

    if (
        not correctness_first
        and synth_result.get("success")
        and testbench_code
        and run_csim_check
    ):
        if log_prefix:
            logging.info("%s Running C-simulation (csim)...", log_prefix)
        csim_result = run_csim(
            hls_code,
            testbench_code,
            header_code,
            header_name=header_name,
            top_function=top_function,
            part=part,
            clock_ns=clock_ns,
            extra_files=extra_files,
            golden_output_text=golden_output_text,
            golden_output_specs=golden_output_specs or {},
        )
        csim_summary = _summarize_test_result(csim_result, True)

    cosim_gate_open = (
        not cosim_requires_csim_pass
        or csim_summary is None
        or csim_summary.get("passed")
    )
    cosim_summary = None
    if (
        synth_result.get("success")
        and testbench_code
        and run_cosim_check
        and cosim_gate_open
    ):
        policy = _cosim_gold_precheck_policy(
            synth_result.get("report") or {},
            cosim_reference_cycle_info,
        )
        if policy:
            cosim_summary = _predicted_cosim_timeout_summary(policy)
            logging.warning(
                "%s Skipping cosim: csynth latency %s cycles is %.3fx gold "
                "(%s cycles; threshold %.3fx)",
                log_prefix or "[cosim precheck]",
                policy["generated_csynth_latency_cycles"],
                policy["ratio_generated_over_gold"],
                policy["gold_reference_cycles"],
                policy["threshold_ratio"],
            )
        else:
            if log_prefix:
                logging.info("%s Running co-simulation (cosim)...", log_prefix)
            cosim_result = run_cosim(
                hls_code,
                testbench_code,
                header_code,
                header_name=header_name,
                top_function=top_function,
                part=part,
                clock_ns=clock_ns,
                extra_files=extra_files,
                interface_depths=cosim_depths or {},
                golden_output_text=golden_output_text,
                golden_output_specs=golden_output_specs or {},
            )
            cosim_summary = _summarize_test_result(cosim_result, True)

    return {"synth": synth_result, "csim": csim_summary, "cosim": cosim_summary}


def _repo_root_for_benchmark(bench_dir: Path) -> Path:
    bench_dir = bench_dir.resolve()
    for candidate in [bench_dir] + list(bench_dir.parents):
        if (candidate / "c2hls.py").exists() and (candidate / "benchmarks").exists():
            return candidate
    return REPO_ROOT


def _default_output_dir(bench_dir: str, bench_name: str, multistep: bool = False) -> Path:
    root = _repo_root_for_benchmark(Path(bench_dir))
    results_dir = root / ("results_multistep" if multistep else "results")
    return results_dir / bench_name


def _build_coverage(meta: dict, reference_validation: dict, generated_csim: Optional[dict], generated_cosim: Optional[dict]) -> dict:
    gt_csim = reference_validation.get("csim", {})
    gt_cosim = reference_validation.get("cosim", {})
    gen_csim = generated_csim or {"status": "failed", "ran": False}
    gen_cosim = generated_cosim or {"status": "failed", "ran": False}
    cosim_available = bool(
        meta.get("testbench_file")
        and (
            meta.get("supports_cosim")
            or (_cosim_selected_only() and _env_flag(FORCE_SELECTED_COSIM_ENV))
        )
    )
    return {
        "ground_truth_csim_available": bool(meta.get("supports_csim") and meta.get("testbench_file")),
        "ground_truth_csim_ran": bool(gt_csim.get("ran", False)),
        "ground_truth_cosim_available": cosim_available,
        "ground_truth_cosim_ran": bool(gt_cosim.get("ran", False)),
        "generated_csim_available": bool(meta.get("supports_csim") and meta.get("testbench_file")),
        "generated_csim_ran": bool(gen_csim.get("ran", False)),
        "generated_cosim_available": cosim_available,
        "generated_cosim_ran": bool(gen_cosim.get("ran", False)),
    }


def _load_anthropic_api_key() -> str:
    key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if key:
        return key
    if CLAUDE_API_KEY_FILE.exists():
        return CLAUDE_API_KEY_FILE.read_text().strip()
    return ""


def _load_openai_api_key() -> str:
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if key:
        return key
    if OPENAI_API_KEY_FILE.exists():
        return OPENAI_API_KEY_FILE.read_text().strip()
    return ""


def _llm_timeout_seconds(default: float = 600.0) -> float:
    value = os.getenv("C2HLS_LLM_TIMEOUT", str(default)).strip()
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        logging.warning(
            "Invalid C2HLS_LLM_TIMEOUT=%r; using %.1fs",
            value,
            default,
        )
        return default
    return max(1.0, parsed)


def _is_hosted_openai_model(model_name: str) -> bool:
    model = (model_name or "").lower()
    return model.startswith(("gpt-", "o1", "o3", "o4", "codex-"))


def _extract_struct_names(header_code: str) -> List[str]:
    return sorted(set(re.findall(r"\bstruct\s+([A-Za-z_][A-Za-z0-9_]*)", header_code or "")))


def _extract_prototype_names(header_code: str) -> List[str]:
    if not header_code:
        return []
    pattern = re.compile(r"^\s*(?:[A-Za-z_][\w:\s\*<>]*?)\b([A-Za-z_][A-Za-z0-9_]*)\s*\([^;{}]*\)\s*;", re.MULTILINE)
    names = []
    for name in pattern.findall(header_code):
        if name not in {"if", "for", "while", "switch", "return"}:
            names.append(name)
    return sorted(set(names))


def _extract_defined_function_names(code: str) -> List[str]:
    if not code:
        return []
    pattern = re.compile(r"^\s*(?:[A-Za-z_][\w:\s\*<>\[\]]*?)\b([A-Za-z_][A-Za-z0-9_]*)\s*\([^;]*\)\s*\{", re.MULTILINE)
    names = []
    for name in pattern.findall(code):
        if name not in {"if", "for", "while", "switch"}:
            names.append(name)
    return sorted(set(names))


def _extract_interface_ports(hls_code: str) -> List[str]:
    if not hls_code:
        return []
    ports = re.findall(r"#pragma\s+HLS\s+INTERFACE\s+[^\n]*?\bport\s*=\s*([A-Za-z_][A-Za-z0-9_]*)", hls_code)
    return sorted(dict.fromkeys(ports))


def _normalize_vitis_s_axilite_bundles(hls_code: str) -> tuple[str, str]:
    """Ensure Vitis kernel-control pragmas are explicit and single-bundle.

    Vitis 2023.2 treats `m_axi ... offset=slave` ports as requiring a matching
    AXI-lite control entry. If the generated code only adds `s_axilite` for
    scalar arguments and return, synthesis can fail with HLS 214-219 because
    the pointer/array offsets are assigned a different bundle. Normalize this
    before synthesis so external datasets with native array signatures do not
    depend on the model remembering every control pragma.
    """
    if not hls_code:
        return hls_code, ""

    lines = hls_code.splitlines()
    m_axi_ports: list[str] = []
    s_axilite_ports: set[str] = set()
    updated_lines: list[str] = []
    changed_bundle = False

    for line in lines:
        m_axi = re.search(r"#pragma\s+HLS\s+INTERFACE\s+m_axi\b[^\n]*?\bport\s*=\s*([A-Za-z_][A-Za-z0-9_]*)", line)
        if m_axi:
            port = m_axi.group(1)
            if port not in m_axi_ports:
                m_axi_ports.append(port)

        s_axilite = re.search(r"#pragma\s+HLS\s+INTERFACE\s+s_axilite\b[^\n]*?\bport\s*=\s*([A-Za-z_][A-Za-z0-9_]*)", line)
        if s_axilite:
            s_axilite_ports.add(s_axilite.group(1))
            if re.search(r"\bbundle\s*=", line):
                normalized = re.sub(r"\bbundle\s*=\s*[A-Za-z_][A-Za-z0-9_]*", "bundle=control", line)
            else:
                normalized = line.rstrip() + " bundle=control"
            changed_bundle = changed_bundle or normalized != line
            line = normalized
        updated_lines.append(line)

    missing_ports = [port for port in m_axi_ports if port not in s_axilite_ports]
    if not missing_ports and not changed_bundle:
        return hls_code, ""

    final_lines: list[str] = []
    inserted: set[str] = set()
    for line in updated_lines:
        final_lines.append(line)
        m_axi = re.search(r"#pragma\s+HLS\s+INTERFACE\s+m_axi\b[^\n]*?\bport\s*=\s*([A-Za-z_][A-Za-z0-9_]*)", line)
        if m_axi:
            port = m_axi.group(1)
            if port in missing_ports and port not in inserted:
                indent = re.match(r"^(\s*)", line).group(1)
                final_lines.append(f"{indent}#pragma HLS INTERFACE s_axilite port={port} bundle=control")
                inserted.add(port)

    notes = []
    if inserted:
        notes.append("added missing s_axilite control pragmas for m_axi ports: " + ", ".join(sorted(inserted)))
    if changed_bundle:
        notes.append("normalized all s_axilite pragmas to bundle=control")
    return "\n".join(final_lines) + ("\n" if hls_code.endswith("\n") else ""), "; ".join(notes)


def _build_benchmark_context(meta: dict, header_name: str, header_code: str,
                             c_code: str,
                             testbench_code: str = "") -> str:
    """Build the benchmark-specific prompt context.

    IMPORTANT: This function must NEVER read ground-truth HLS code. The context
    it produces is seen by the LLM at generation time; leaking GT interface
    ports, pragmas, or structure into the prompt contaminates any downstream
    RL dataset and defeats the purpose of measuring model skill. Only inspect
    plain C source, the header, the testbench-visible top-function signature,
    and static per-benchmark hints stored in BENCHMARK_POLICIES.
    """
    hints = []
    bench = meta.get("benchmark", "unknown")
    wrapper_top = meta.get("translated_hls_top", "workload")
    kernel_top = meta.get("kernel_top")

    hints.append(f"Benchmark name: `{bench}`.")
    hints.append(f"Required HLS wrapper top function: `{wrapper_top}`.")
    if kernel_top and kernel_top != wrapper_top:
        hints.append(f"Preserve or call the existing kernel/helper function `{kernel_top}` inside `{wrapper_top}`.")
    if header_name:
        hints.append(f"Include `{header_name}` exactly once and reuse its declarations.")

    expected_signature = _expected_top_signature(header_code, testbench_code, wrapper_top)
    if expected_signature:
        sig_text = _render_function_signature(
            expected_signature,
            include_extern=bool(expected_signature.get("extern_c")),
        )
        hints.append(f"Exact testbench-visible `{wrapper_top}` signature to preserve: `{sig_text}`.")

    struct_names = _extract_struct_names(header_code)
    if struct_names:
        joined = ", ".join(f"`{name}`" for name in struct_names)
        hints.append(f"Header-owned structs/types that must not be redeclared in the source: {joined}.")

    prototype_names = _extract_prototype_names(header_code)
    if prototype_names:
        joined = ", ".join(f"`{name}`" for name in prototype_names[:6])
        hints.append(f"Header-declared functions available for reuse: {joined}.")

    defined_names = _extract_defined_function_names(c_code)
    if defined_names:
        joined = ", ".join(f"`{name}`" for name in defined_names[:8])
        hints.append(f"Functions already defined in the plain input whose names/signatures should be preserved unless wrapping is required: {joined}.")

    # Deliberately no GT-derived hints here. The reference wrapper's interface
    # ports, pragmas, and structure must remain invisible to the LLM.

    for manual_hint in _policy(bench, "translation", []):
        hints.append(manual_hint)

    return "\n".join(f"- {hint}" for hint in hints)


def _format_attempt_history(turn_records: list, current_phase: str = "B",
                            max_recent: int = 4) -> str:
    """One-line-per-attempt summary of past failed turns, intended to land
    above the per-error repair guidance in fix prompts. The goal is to break
    the LLM's tendency to oscillate between two failure modes by surfacing
    the chain of mistakes it has already made in this Phase B / step.

    Returns "" when there's nothing to say (first attempt, or no history),
    otherwise a markdown block with a trailing newline so it can be dropped
    into a {attempt_history} field.
    """
    if not turn_records:
        return ""
    relevant = [r for r in turn_records if r.get("phase") == current_phase][-max_recent:]
    if len(relevant) < 1:
        return ""

    lines = ["## Previous attempts in this phase"]
    for rec in relevant:
        turn = rec.get("turn", "?")
        if rec.get("success"):
            lines.append(f"- Attempt {turn}: SUCCESS (then a later step regressed)")
            continue
        err = (rec.get("error") or "").strip()
        # First non-blank line of the error is usually the most actionable.
        first_line = err.split("\n", 1)[0][:200] if err else "(no error message)"
        klass = _classify_synth_error(err) if err else "unknown"
        lines.append(f"- Attempt {turn}: {klass.upper()} — {first_line}")
    lines.append(
        f"You are now on attempt {len(relevant)}. Avoid repeating the same fix "
        f"category if you've already tried it; pick a different angle."
    )
    return "\n".join(lines) + "\n\n"


def _build_repair_guidance(error: str) -> str:
    if not error:
        return "- Keep the wrapper minimal, syntactically valid, and consistent with the header and plain input."

    error_lower = error.lower()
    hints = []
    if "redefinition" in error_lower:
        hints.append("- Remove duplicate structs, typedefs, constants, or prototypes that already come from the header.")
    if "undeclared identifier" in error_lower or "was not declared" in error_lower:
        hints.append("- Do not reference invented helper arrays/buffers unless you declare and initialize them first.")
    if "pragma hls" in error_lower and "function scope" in error_lower:
        hints.append("- Move every `#pragma HLS` inside a function body or loop body; none may appear at global scope.")
    if "no matching function" in error_lower or "too many arguments" in error_lower or "too few arguments" in error_lower:
        hints.append("- Match the exact function signatures from the header and the plain input.")
    if "timed out" in error_lower:
        hints.append("- Prefer a simpler wrapper and modest loop pragmas over aggressive buffering or full unrolling.")
    if "214-219" in error or "must be bundled into one bundle" in error_lower:
        hints.append(
            "- All `s_axilite` ports must share the SAME bundle. Use `bundle=control` "
            "on every s_axilite line including `port=return`. Do not let any port pick "
            "an auto-generated bundle name like `control_r`. Vitis kernel mode rejects "
            "split bundles with HLS 214-219."
        )
        hints.append(
            "- For every `m_axi port=<name> offset=slave ...` pragma, also add a separate "
            "`#pragma HLS INTERFACE s_axilite port=<name> bundle=control` line."
        )
    if not hints:
        hints.append("- Preserve the existing helper/kernel structure and make the smallest change that fixes the reported error.")
    return "\n".join(hints)


# Device-resource limits used by the profile-signal extractor. Keep this in
# sync with rubric._DEVICE_TABLE; we duplicate the small lookup here to avoid
# a circular import.
def _resource_capacity_for(part: str) -> dict:
    try:
        from rubric import _device_limits_for_part  # type: ignore
    except Exception:
        return {}
    return dict(_device_limits_for_part(part) or {})


def _target_context_for_prompt(part: str = "", clock_ns: float | None = None) -> str:
    active_part = part or os.getenv("C2HLS_PART", DEFAULT_PART)
    active_clock = clock_ns if clock_ns is not None else os.getenv("C2HLS_CLOCK_NS", str(DEFAULT_CLOCK_NS))
    flow = os.getenv("C2HLS_FLOW_TARGET", "vitis")
    return f"{flow} flow, part {active_part}, target clock {active_clock} ns"


def _resource_utilization(report: dict, part: str = "") -> dict:
    capacity = _resource_capacity_for(part or os.getenv("C2HLS_PART", DEFAULT_PART))
    out = {}
    for key in ("bram", "dsp", "ff", "lut", "uram"):
        value = _as_float((report or {}).get(key))
        cap = _as_float((capacity or {}).get(key))
        if value is None or not cap or cap <= 0:
            continue
        out[key] = {
            "used": value,
            "capacity": cap,
            "utilization": value / cap,
        }
    return out


def _resource_over_device(report: dict, part: str = "") -> list[str]:
    over = []
    for key, data in _resource_utilization(report, part).items():
        if data["utilization"] > 1.0:
            over.append(
                f"{key} {int(data['used'])} > device cap {int(data['capacity'])}"
            )
    return over


def _fits_device(report: dict, part: str = "") -> bool:
    return not _resource_over_device(report, part)


def _paper_candidate_feasibility(
    report: Optional[dict],
    *,
    csim: Optional[dict] = None,
    correctness_required: bool = True,
    part: str = "",
    clock_ns: Optional[float] = None,
) -> dict:
    """Apply the paper's single correctness/fit/timing feasibility rule."""
    report = report or {}
    active_part = part or os.getenv("C2HLS_PART", DEFAULT_PART)
    target_clock = _as_float(
        report.get("requested_clock_period_ns")
        or clock_ns
        or os.getenv("C2HLS_CLOCK_NS", DEFAULT_CLOCK_NS)
    )

    correctness_ok = True
    if correctness_required:
        correctness_ok = bool(isinstance(csim, dict) and csim.get("passed"))

    resource_capacity = _resource_capacity_for(active_part)
    # Require evidence for every resource class physically present on the
    # selected device.  U280 URAM is a real placement constraint; silently
    # treating a missing URAM count as zero could admit an unverified design.
    resource_keys = tuple(
        key
        for key in ("bram", "dsp", "ff", "lut", "uram")
        if (_as_float(resource_capacity.get(key)) or 0) > 0
    )
    resource_evidence_complete = bool(resource_capacity) and all(
        _as_float(report.get(key)) is not None for key in resource_keys
    )
    resource_ok = resource_evidence_complete and _fits_device(report, active_part)

    slack = _as_float(report.get("slack_ns"))
    estimated = _as_float(report.get("estimated_clock_period_ns"))
    fmax = _as_float(report.get("fmax_mhz"))
    timing_evidence = ""
    timing_ok = False
    if slack is not None:
        timing_evidence = "slack_ns"
        timing_ok = slack >= -1e-9
    elif estimated is not None and target_clock is not None:
        timing_evidence = "estimated_clock_period_ns"
        timing_ok = estimated <= target_clock + 1e-9
    elif fmax is not None and fmax > 0 and target_clock is not None:
        timing_evidence = "fmax_mhz"
        timing_ok = fmax + 1e-9 >= 1000.0 / target_clock

    latency = _as_float(
        report.get("latency_cycles_worst")
        or report.get("latency_cycles")
        or report.get("latency_ns_worst")
        or report.get("latency_ns")
    )
    reasons = []
    if not correctness_ok:
        reasons.append("golden_correctness_not_passed")
    if not resource_evidence_complete:
        reasons.append("resource_evidence_incomplete")
    elif not resource_ok:
        reasons.append("device_resource_fit_failed")
    if not timing_evidence:
        reasons.append("target_timing_evidence_missing")
    elif not timing_ok:
        reasons.append("target_timing_failed")
    if latency is None or latency <= 0:
        reasons.append("estimated_latency_missing")
    return {
        "schema_version": "c2hls.candidate-feasibility.v1",
        "feasible": not reasons,
        "correctness_ok": correctness_ok,
        "resource_fit": resource_ok,
        "resource_evidence_complete": resource_evidence_complete,
        "timing_met": timing_ok,
        "timing_evidence": timing_evidence or None,
        "target_clock_ns": target_clock,
        "estimated_latency": latency,
        "reasons": reasons,
    }


def _actionable_timing_bottlenecks(report: dict, limit: int = 3) -> list[str]:
    feedback = (report or {}).get("feedback") or {}
    bottlenecks = feedback.get("bottlenecks") or []
    out = []
    for bn in bottlenecks:
        if not isinstance(bn, dict):
            continue
        kind = str(bn.get("kind") or "")
        if kind not in {"ii_target_miss", "pipeline_blocked", "loop_carried_dep", "port_conflict"}:
            continue
        scope = bn.get("scope_id") or bn.get("scope") or "unknown_scope"
        evidence = bn.get("evidence") or bn.get("message") or kind
        out.append(f"{scope}: {evidence}")
        if len(out) >= limit:
            break
    return out


def _classify_synth_error(error: str) -> str:
    """Coarse error-class buckets for streak detection. Distinct classes
    should drive distinct repair strategies; the same class repeating is
    a strong signal the LLM is stuck."""
    if not error:
        return "unknown"
    e = error.lower()
    if "timed out" in e or "timeout" in e:
        return "timeout"
    if e.startswith("csim_failed") or "csim failed" in e:
        return "csim_failed"
    if e.startswith("cosim_failed") or "cosim failed" in e:
        return "cosim_failed"
    if "214-219" in error or "must be bundled into one bundle" in e:
        return "axilite_bundle_split"
    if "synthesis report not found" in e:
        return "missing_report"
    if "redefinition" in e or "undeclared" in e or "no matching function" in e \
            or "too many arguments" in e or "too few arguments" in e:
        return "compile_error"
    if "pragma hls" in e and "function scope" in e:
        return "pragma_scope"
    if "memory" in e and ("exceeded" in e or "overflow" in e):
        return "resource_overflow"
    if "scheduler" in e or "could not schedule" in e:
        return "scheduling_error"
    return "synth_other"


def _build_profile_signal(report: dict, part: str = "",
                          requested_clock_ns: float = None) -> str:
    """Turn a synthesis report into a structured bottleneck summary.

    Returns a multi-line bullet list intended to be embedded in a repair
    prompt next to the raw error log. Concrete signals beat free-text
    error dumps for the LLM's next-attempt focus. Empty string when the
    report has no actionable signal.
    """
    if not report:
        return ""

    signals: list[str] = []
    capacity = _resource_capacity_for(part) if part else {}

    # Timing
    slack = _as_float(report.get("slack_ns"))
    if slack is not None and slack < 0:
        signals.append(
            f"- TIMING_VIOLATION: slack={slack:+.3f} ns. "
            f"The combinational path is longer than the target clock; "
            f"reduce loop body complexity or break critical paths with "
            f"an extra pipeline stage."
        )
    fmax = _as_float(report.get("fmax_mhz"))
    if requested_clock_ns and fmax is not None and fmax > 0:
        actual_period = 1000.0 / fmax
        target = float(requested_clock_ns)
        if actual_period > target * 1.10:
            signals.append(
                f"- FMAX_BELOW_TARGET: actual ~{fmax:.1f} MHz "
                f"(period {actual_period:.3f} ns vs target {target:.3f} ns). "
                f"Simplify the critical path."
            )

    # Latency
    lat_ns = _as_float(report.get("latency_ns"))
    if lat_ns is not None and lat_ns > 1e8:  # > 100 ms
        signals.append(
            f"- LATENCY_HIGH: {lat_ns/1e6:.1f} ms. Loop trip counts may be "
            f"variable or unrolled inefficiently; check for unnecessary "
            f"sequential dependencies."
        )

    # Resources — flag any > 80% of device capacity (warning) or > 100%
    # (infeasible). Bigger signal means tighter constraint.
    res_keys = (("lut", "LUT"), ("ff", "FF"), ("dsp", "DSP"), ("bram", "BRAM"))
    for key, label in res_keys:
        used = _as_float(report.get(key))
        cap = capacity.get(key)
        if used is None or not cap:
            continue
        pct = 100.0 * used / cap
        if pct > 100.0:
            signals.append(
                f"- {label}_OVERFLOW: {int(used)} / {cap} ({pct:.1f}% of device). "
                f"This will not place-and-route. Reduce parallelism, "
                f"partition factors, or unroll factors on this resource type."
            )
        elif pct > 80.0:
            signals.append(
                f"- {label}_PRESSURE: {int(used)} / {cap} ({pct:.1f}% of device). "
                f"Approaching the ceiling; further optimisation should not add "
                f"to {label} usage."
            )

    if not signals:
        return ""
    return ("Profile signals from the latest synthesis "
            "(treat these as the primary repair targets):\n"
            + "\n".join(signals))


# Legacy reference-guided per-step regression thresholds (Phase 5 tuning).
#
# The original single-threshold design (1.10x for everything) was too tight
# for steps that *legitimately* trade resources for throughput — unroll
# typically grows DSP/FF, doublebuffer doubles BRAM by definition, and
# coalescing widens the AXI port (often 8x DSP on knn-style kernels).
#
# These ceilings were calibrated against the upstream Rodinia-HLS reference.
# They are consequently disabled in reference-blind evaluation.  Paper runs
# rely on the benchmark-independent correctness/device-fit/timing gates plus
# best-state recovery instead of these expert-derived relative ratios.
#
# Schema:
#   {
#     "<step_name>": {
#       "latency": <float>,             # max latency growth ratio (lat_ns)
#       "resources": {
#         "<key>": <float>,              # per-resource max ratio
#         "default": <float>,            # fallback for unlisted resources
#       },
#     },
#     ...
#   }
#
# A step is regressed when *either* latency exceeds its threshold,
# OR 3+ resources each exceed their per-resource thresholds.
#
# Override at runtime via:
#   C2HLS_STEP_REGRESSION_THRESHOLD       — single global number (legacy;
#                                            overrides everything when set)
#   C2HLS_STEP_REGRESSION_THRESHOLDS_JSON — JSON of the per-step dict
STEP_REGRESSION_THRESHOLDS = {
    "_default": {
        "latency": 1.10,
        "resources": {"default": 1.10},
    },
    "tiling": {
        # Tiling adds outer-loop control + load/compute/store split. Latency
        # may regress significantly (philip ref: knn 4.08x; pathfinder 1.50x)
        # — it's a structural prerequisite for the dataflow/buffer wins.
        # Resources mostly hold steady (just BRAM for the tile buffer).
        "latency": 5.0,
        "resources": {"bram": 4.0, "default": 1.30},
    },
    "pipeline": {
        # Pipeline shouldn't grow latency. Some FF growth from added
        # pipeline registers; LUT mostly stable.
        "latency": 1.10,
        "resources": {"ff": 1.50, "default": 1.20},
    },
    "unroll": {
        # Unroll grows compute resources. philip ref knn: DSP 2.0x, FF 5.55x,
        # LUT 2.16x; latency 0.95x. Allow generous compute growth, latency
        # must not regress.
        "latency": 1.10,
        "resources": {"dsp": 8.0, "ff": 6.0, "lut": 2.5, "default": 1.30},
    },
    "doublebuffer": {
        # Doublebuffer literally doubles the load buffer's BRAM by design.
        # FF/LUT grow for dataflow control. philip ref: BRAM 1.68x, FF 1.09x,
        # LUT 1.39x; latency 0.50x.
        "latency": 1.10,
        "resources": {"bram": 2.50, "ff": 2.50, "lut": 2.50, "default": 1.30},
    },
    "coalescing": {
        # Coalescing widens AXI to 512-bit. philip ref knn: DSP 8.0x,
        # BRAM 0.94x; latency 0.15x. Allow huge DSP/BRAM growth (the whole
        # point). Slight latency slack (1.20x) since the LLM may add control
        # overhead before the coalescing kicks in.
        "latency": 1.20,
        "resources": {"dsp": 10.0, "bram": 5.0, "ff": 2.5, "lut": 2.5, "default": 1.50},
    },
    # Phase 3 combo strategies — bundle several techniques in one rewrite.
    "combo_full": {
        # All-in-one: tolerant on every axis.
        "latency": 4.0,
        "resources": {"default": 4.0},
    },
    "combo_structural": {
        # Tiling + doublebuffer + dataflow combo: tolerates latency growth
        # (it's a structural setup) and modest resource growth.
        "latency": 5.0,
        "resources": {"bram": 4.0, "default": 2.0},
    },
    "combo_parallel": {
        # Pipeline + unroll + coalescing combo: latency must shrink, but
        # compute resources can grow significantly.
        "latency": 1.20,
        "resources": {"dsp": 10.0, "ff": 6.0, "lut": 2.5, "default": 2.0},
    },
    "flash": {
        # Flash is a single all-in endpoint. It should normally improve
        # latency, but allow broad resource motion so one-shot candidate
        # search can discover compact HLSFactory/PolyBench rewrites.
        "latency": 1.50,
        "resources": {"dsp": 12.0, "bram": 6.0, "ff": 8.0, "lut": 6.0, "default": 4.0},
    },
}


ONE_SHOT_STRATEGIES = {"combo", "combo_full", "flash"}


def _resolve_step_thresholds(step_name: str,
                              global_override: float = None) -> dict:
    """Return the threshold dict for ``step_name``. Order of precedence:

    In reference-blind mode all relative ratio limits are disabled, including
    environment overrides, because the built-in defaults were calibrated from
    reference-isolated expert implementations. Hard correctness, device-fit, and target
    timing checks remain active elsewhere.

    Outside reference-blind mode, precedence is:

    1. ``C2HLS_STEP_REGRESSION_THRESHOLDS_JSON`` (per-step JSON) — full
       per-step override.
    2. ``C2HLS_STEP_REGRESSION_THRESHOLD`` env var **explicitly set** — a
       single number applies to everything (legacy behaviour preserved).
    3. ``STEP_REGRESSION_THRESHOLDS[step_name]`` — the new per-step default.
    4. ``STEP_REGRESSION_THRESHOLDS["_default"]`` — fallback.
    """
    if _reference_blind_enabled():
        return {
            "latency": float("inf"),
            "resources": {"default": float("inf")},
            "source": "reference_blind_feasibility_and_best_state_only",
        }
    json_blob = os.getenv("C2HLS_STEP_REGRESSION_THRESHOLDS_JSON")
    if json_blob:
        try:
            override = json.loads(json_blob)
            if step_name in override:
                return override[step_name]
            if "_default" in override:
                return override["_default"]
        except (json.JSONDecodeError, TypeError):
            pass

    # Legacy single-number override only fires when the env var is
    # *explicitly set* by the caller. The constant
    # `STEP_REGRESSION_THRESHOLD` is default-fallback 1.10 even when the
    # env is unset, so we can't use it directly to detect intent.
    if "C2HLS_STEP_REGRESSION_THRESHOLD" in os.environ:
        try:
            v = float(os.environ["C2HLS_STEP_REGRESSION_THRESHOLD"])
            if v > 0:
                return {
                    "latency": v,
                    "resources": {"default": v},
                }
        except ValueError:
            pass

    if step_name in STEP_REGRESSION_THRESHOLDS:
        return STEP_REGRESSION_THRESHOLDS[step_name]
    return STEP_REGRESSION_THRESHOLDS["_default"]


def _step_regression_reasons(new_report: dict, prev_report: dict,
                             threshold: float = 1.10,
                             step_name: str = "",
                             part: str = "") -> list:
    """Return a list of human-readable regression reasons. Empty list = no
    regression detected.

    A step "regresses" when at least one of:
      - latency_ns grew by > the step's latency threshold
      - 3+ resource counts (lut/ff/bram/dsp) all grew by > their per-resource
        threshold (catches the "LLM added pragmas without measuring" failure
        mode where the design got bigger across the board with no benefit)
      UNLESS the two-tier override applies: if latency improved by ≥2× AND no
      resource exceeds absolute device capacity, the step is accepted even when
      per-step resource ratios are over ceiling. This allows aggressive Sonnet-
      style DSP parallelization (e.g., tiling attempt 0 at 67K cycles with 30×
      DSP) to pass when the design is genuinely faster and still fits on chip.

    In legacy/reference-guided mode, per-step thresholds come from
    STEP_REGRESSION_THRESHOLDS, which was calibrated against the upstream
    Rodinia-HLS implementation.  Reference-blind mode disables those relative
    thresholds and retains only benchmark-independent timing/device-fit gates.
    Pass ``threshold > 0`` for the legacy single-threshold path.  When
    ``step_name`` is empty, it falls through to the ``_default`` entry.
    """
    reasons: list[str] = []
    if not new_report or not prev_report:
        return reasons

    cfg = _resolve_step_thresholds(step_name, threshold if threshold and threshold > 0 else None)
    lat_threshold = float(cfg.get("latency", 1.10))
    resource_thresholds = cfg.get("resources") or {"default": 1.10}
    default_resource_t = float(resource_thresholds.get("default", 1.10))

    new_lat = _as_float(new_report.get("latency_ns"))
    prev_lat = _as_float(prev_report.get("latency_ns"))
    lat_ratio = (new_lat / prev_lat) if (new_lat and prev_lat and prev_lat > 0) else None
    if lat_ratio is not None and lat_ratio > lat_threshold:
        reasons.append(
            f"latency_ns regressed {lat_ratio:.2f}x (limit "
            f"{lat_threshold:.2f}x for step '{step_name or '_default'}'): "
            f"({prev_lat:.0f} -> {new_lat:.0f})"
        )

    new_slack = _as_float(new_report.get("slack_ns"))
    new_est_clock = _as_float(new_report.get("estimated_clock_period_ns"))
    new_req_clock = _as_float(new_report.get("requested_clock_period_ns"))
    timing_bad = False
    timing_detail = ""
    if new_slack is not None and new_slack < 0:
        timing_bad = True
        timing_detail = f"slack_ns={new_slack:.3f}"
    elif (
        new_est_clock is not None
        and new_req_clock is not None
        and new_est_clock > new_req_clock + 1e-9
    ):
        timing_bad = True
        timing_detail = (
            f"estimated_clock_period_ns {new_est_clock:.3f} > "
            f"requested_clock_period_ns {new_req_clock:.3f}"
        )
    timing_allowed_by_latency_fit = (
        lat_ratio is not None
        and lat_ratio <= 0.5
        and _fits_device(new_report, part or os.getenv("C2HLS_PART", DEFAULT_PART))
    )
    if timing_bad and not timing_allowed_by_latency_fit:
        reasons.append(
            f"timing_not_clean for step '{step_name or '_default'}': "
            f"{timing_detail}"
        )

    grown_resources: list[str] = []
    for key in ("lut", "ff", "bram", "dsp"):
        new_v = _as_float(new_report.get(key))
        prev_v = _as_float(prev_report.get(key))
        if new_v is not None and prev_v is not None and prev_v > 0:
            r = new_v / prev_v
            t = float(resource_thresholds.get(key, default_resource_t))
            if r > t:
                grown_resources.append(
                    f"{key} {int(prev_v)}->{int(new_v)} ({r:.2f}x; limit {t:.2f}x)"
                )
    if len(grown_resources) >= 3:
        # Two-tier override: if latency improved by ≥2× AND every resource
        # stays within absolute device capacity, accept the step even though
        # per-step ratios are over ceiling. This lets aggressive DSP
        # parallelization through when it genuinely speeds up the design.
        latency_improved_2x = (lat_ratio is not None and lat_ratio <= 0.5)
        if latency_improved_2x and part:
            capacity = _resource_capacity_for(part)
            over_device = []
            for key in ("lut", "ff", "bram", "dsp"):
                new_v = _as_float(new_report.get(key))
                cap = _as_float((capacity or {}).get(key))
                if new_v and cap and new_v > cap:
                    over_device.append(f"{key} {int(new_v)} > device cap {int(cap)}")
            if not over_device:
                # Two-tier pass: big latency win, fits on chip — accept
                pass
            else:
                reasons.append(
                    f"resource_growth (>=3 resources over per-resource limits for "
                    f"step '{step_name or '_default'}'): "
                    + ", ".join(grown_resources)
                    + f" [device overflow: {'; '.join(over_device)}]"
                )
        else:
            reasons.append(
                f"resource_growth (>=3 resources over per-resource limits for "
                f"step '{step_name or '_default'}'): "
                + ", ".join(grown_resources)
            )

    return reasons


_NO_OP_FIELDS = ("latency_cycles", "interval", "bram", "dsp", "ff", "lut")


def _step_no_op_reasons(new_report: dict, prev_report: dict) -> list:
    """Pillar 9 (MVP) — detect the 'no-op trap'. The agentic smoke test
    surfaced trajectories where pipeline / unroll / doublebuffer steps all
    produced byte-identical synthesis numbers (knn, lud), meaning the LLM's
    edit didn't reach the scheduler at all (wrong loop labelled, pragma
    placed on a function that gets inlined away, etc.). When that happens we
    want to re-prompt with 'your last variant changed nothing' rather than
    silently accept the no-op as a successful step.

    Two reasons returned (caller cares only that the list is non-empty):
        - 'identical_synth_tuple' when (latency, interval, resources) all match
        - one human-readable summary line for prompt feedback
    """
    if not new_report or not prev_report:
        return []
    new_tuple = tuple(new_report.get(k) for k in _NO_OP_FIELDS)
    prev_tuple = tuple(prev_report.get(k) for k in _NO_OP_FIELDS)
    # All fields populated and equal — that's the no-op signature. We require
    # at least latency_cycles + ANY two other fields to be equal so we don't
    # false-positive on a benchmark where Vitis genuinely emits identical
    # numbers (rare, but possible for tiny kernels).
    if new_tuple == prev_tuple and any(v is not None for v in new_tuple):
        populated = sum(1 for v in new_tuple if v is not None)
        if populated >= 3:
            return [
                "identical_synth_tuple",
                f"all of {_NO_OP_FIELDS} unchanged from previous step "
                f"(lat={new_tuple[0]}, ii={new_tuple[1]}, "
                f"bram={new_tuple[2]}, dsp={new_tuple[3]}, "
                f"ff={new_tuple[4]}, lut={new_tuple[5]})",
            ]
    return []


def _render_no_op_guidance(step_name: str, reasons: list) -> str:
    """Prompt fragment delivered to the LLM when a no-op is detected. Tells
    it bluntly that its last edit had zero observable effect on synthesis."""
    if not reasons:
        return ""
    summary = reasons[-1] if reasons else ""
    return (
        f"Your previous attempt at the `{step_name}` step did NOT change any "
        f"synthesized metric: {summary}. That means the edit either never "
        f"reached the scheduler (pragma on the wrong loop / function got "
        f"inlined away / loop label mismatched) or your change was a pure "
        f"comment / formatting tweak. For this retry: identify the specific "
        f"loop or function that needs the optimization, place the pragma on "
        f"its first line inside the loop body, and verify the pragma name "
        f"matches a `for` loop or `function` that actually appears in the "
        f"emitted RTL. If the requested optimization is genuinely not "
        f"applicable to this kernel, say so explicitly in a comment and "
        f"return the previous code unchanged."
    )


# === Phase 8: baseline alignment ============================================
#
# Phase B's translation is a single-shot rewrite of plain C -> HLS. If the
# LLM lands a baseline that's significantly worse than the offline reference
# envelope, every downstream optimization step compounds the bad starting
# point.
#
# Phase 8 adds an opt-in baseline-alignment loop that runs *between*
# Phase B (translate + synth) and Phase C (offline reference comparison). When
# our baseline is more than ``C2HLS_PHASE8_BASELINE_LATENCY_TOL`` (default
# 1.20×) over the reference's baseline cycles, or any single resource is
# more than ``C2HLS_PHASE8_BASELINE_RESOURCE_TOL`` (default 2.00×) over,
# we re-translate with metric-only, ratio-only feedback and re-synth. Up to
# 3 attempts.
#
# Critical constraint: the feedback must NOT include the reference HLS
# source. The translator agent should still be solving the
# C-to-HLS task, not reconstructing the reference implementation. We render
# only:
#   - latency_cycles ratio (ours / ref)
#   - per-resource ratios
#   - per-loop diagnostics from the agent's own report (already in
#     `feedback["scopes"]` from Pillar 1)


def _compute_baseline_gap(
    ours: dict,
    reference: dict,
    *,
    latency_tolerance: float = 1.20,
    resource_tolerance: float = 2.00,
    fmax_floor: float = 0.80,
) -> dict:
    """Pure function: compute the gap between our baseline synth report
    and the reference baseline. Returns a dict with ratios per axis +
    a ``within_tolerance`` flag.

    The Fmax floor (default 0.80) catches structurally inferior translations
    that happen to match the GT cycle count but run at much lower clock
    frequency. Example: a translation producing 149K cycles at 167 MHz is
    within 1.20× of GT's 142K cycles, but its Fmax is only 40% of GT's
    411 MHz — indicating a fundamentally different (slower) microarchitecture
    that Phase 8 should retranslate rather than accept.

    Disable the Fmax floor by setting fmax_floor=0.0 or via env var
    C2HLS_PHASE8_FMAX_FLOOR (float, default 0.80).
    """
    fmax_floor = float(os.getenv("C2HLS_PHASE8_FMAX_FLOOR", str(fmax_floor)))

    if not ours or not reference:
        return {"within_tolerance": False, "reason": "missing reports"}

    def _f(d, k):
        try:
            return float(d.get(k)) if d.get(k) is not None else None
        except (TypeError, ValueError):
            return None

    cyc_ours = _f(ours, "latency_cycles")
    cyc_ref = _f(reference, "latency_cycles")
    lat_ratio = (cyc_ours / cyc_ref) if (cyc_ours and cyc_ref and cyc_ref > 0) else None

    resource_ratios: dict = {}
    over_resources: list = []
    for k in ("bram", "dsp", "ff", "lut"):
        v_ours, v_ref = _f(ours, k), _f(reference, k)
        if v_ours is None or v_ref is None or v_ref <= 0:
            continue
        r = v_ours / v_ref
        resource_ratios[k] = r
        if r > resource_tolerance:
            over_resources.append((k, v_ours, v_ref, r))

    latency_over = lat_ratio is not None and lat_ratio > latency_tolerance

    # Fmax floor: reject baseline if our Fmax is below fmax_floor × ref Fmax.
    # A low Fmax with matching cycle count means the design runs at a slower
    # clock and has much longer real-time latency than the reference.
    fmax_ours = _f(ours, "fmax_mhz")
    fmax_ref = _f(reference, "fmax_mhz")
    fmax_ratio = (fmax_ours / fmax_ref) if (fmax_ours and fmax_ref and fmax_ref > 0) else None
    fmax_below_floor = (
        fmax_floor > 0
        and fmax_ratio is not None
        and fmax_ratio < fmax_floor
    )

    return {
        "within_tolerance": not latency_over and not over_resources and not fmax_below_floor,
        "latency_ratio": lat_ratio,
        "latency_threshold": latency_tolerance,
        "latency_over": latency_over,
        "resource_ratios": resource_ratios,
        "resource_threshold": resource_tolerance,
        "over_resources": over_resources,
        "fmax_ratio": fmax_ratio,
        "fmax_floor": fmax_floor,
        "fmax_below_floor": fmax_below_floor,
        "ours_summary": {
            "latency_cycles": cyc_ours,
            "latency_ns":     _f(ours, "latency_ns"),
            "interval":       _f(ours, "interval"),
            "bram": _f(ours, "bram"), "dsp": _f(ours, "dsp"),
            "ff": _f(ours, "ff"),     "lut": _f(ours, "lut"),
            "fmax_mhz":       fmax_ours,
        },
        "reference_summary": {
            "latency_cycles": cyc_ref,
            "latency_ns":     _f(reference, "latency_ns"),
            "interval":       _f(reference, "interval"),
            "bram": _f(reference, "bram"), "dsp": _f(reference, "dsp"),
            "ff": _f(reference, "ff"),     "lut": _f(reference, "lut"),
            "fmax_mhz":       fmax_ref,
        },
    }


def _render_baseline_alignment_guidance(gap: dict, attempt: int = 0) -> str:
    """Build a prompt fragment for re-translating with metric-only
    feedback. **Never** includes the reference HLS source or absolute
    reference metrics; only ratios plus diagnostics from our own report's
    ``feedback`` field are exposed to the LLM.
    """
    if gap.get("within_tolerance"):
        return ""

    o, r = gap.get("ours_summary") or {}, gap.get("reference_summary") or {}

    lines: list = [
        f"Your previous translation has a baseline that's significantly "
        f"worse than the offline reference envelope for this kernel.",
        "",
        "Per-axis ratios against the offline reference (ratio > 1 is worse for latency/resources; ratio < 1 is worse for Fmax):",
    ]
    if o.get("latency_cycles") and r.get("latency_cycles"):
        ratio = o["latency_cycles"] / r["latency_cycles"]
        lines.append(f"  - latency_cycles_ratio={ratio:.2f}")
    for k in ("bram", "dsp", "ff", "lut"):
        v_o, v_r = o.get(k), r.get(k)
        if v_o is not None and v_r is not None and v_r > 0:
            ratio = v_o / v_r
            lines.append(f"  - {k}_ratio={ratio:.2f}")
    if o.get("fmax_mhz") and r.get("fmax_mhz"):
        lines.append(f"  - fmax_ratio={o['fmax_mhz'] / r['fmax_mhz']:.2f}")

    if gap.get("over_resources"):
        lines.append("")
        lines.append("Resources over the alignment tolerance:")
        for k, vo, vr, rr in gap["over_resources"]:
            lines.append(f"  - {k}: ratio={rr:.2f}")

    if gap.get("fmax_below_floor"):
        fmax_ratio = gap.get("fmax_ratio", 0)
        fmax_floor = gap.get("fmax_floor", 0.80)
        lines.append("")
        lines.append(
            f"Fmax ratio is too low: {fmax_ratio:.2f} with floor {fmax_floor:.2f}. "
            "A cycle-count match at much lower Fmax indicates a long combinational "
            "critical path; fix structural timing before optimization."
        )
        lines.append(
            "  - Avoid deep nested function calls without `inline`"
        )
        lines.append(
            "  - Avoid wide combinational expressions; break multi-step "
            "float arithmetic across temporary variables so HLS can register"
        )
        lines.append(
            "  - Avoid memory-mapped accesses inside tight loops; buffer "
            "into local variables first"
        )

    lines.append("")
    lines.append(
        "Likely causes for an inflated baseline include:"
    )
    lines.append(
        "  - Helper functions left non-`static`/`inline`, preventing "
        "specialization"
    )
    lines.append(
        "  - Excess local arrays / temporaries adding BRAM and FF state"
    )
    lines.append(
        "  - Loop bounds the synthesizer can't prove static, forcing "
        "conservative scheduling — annotate with "
        "`#pragma HLS loop_tripcount` only when actually needed"
    )
    lines.append(
        "  - Missed `extern \"C\"` wrapper or AXI-control bundle mistakes "
        "that force fallback resource allocation"
    )
    lines.append("")
    lines.append(
        "Re-translate the kernel from scratch, targeting a baseline "
        "that fits the configured device budget and avoids unnecessary "
        "local state. Do NOT add optimization pragmas (PIPELINE / UNROLL / "
        "DATAFLOW / array_partition) — those belong to the multistep "
        "optimization phase that runs AFTER this alignment. Keep this "
        "translation conservative: just AXI INTERFACE pragmas + the "
        "minimal kernel structure."
    )
    if attempt > 0:
        lines.append("")
        lines.append(
            f"This is alignment retry {attempt + 1}; prior retries did "
            "not close the baseline gap."
        )
    return "\n".join(lines)


def _normalize_phaseb_mode(value: str, *, multistep: bool) -> str:
    """Resolve Phase B prompt mode.

    `functional` keeps multistep baselines deliberately conservative.
    `optimized` preserves the legacy/single-shot behavior.
    """
    mode = (value or "").strip().lower()
    if not mode:
        return DEFAULT_PHASEB_MODE_MULTISTEP if multistep else DEFAULT_PHASEB_MODE_SINGLE
    aliases = {
        "conservative": "functional",
        "baseline": "functional",
        "legacy": "optimized",
    }
    mode = aliases.get(mode, mode)
    if mode not in {"functional", "optimized"}:
        logging.warning(
            "Unknown %s=%r; using %s default",
            PHASEB_MODE_ENV, value,
            DEFAULT_PHASEB_MODE_MULTISTEP if multistep else DEFAULT_PHASEB_MODE_SINGLE,
        )
        return DEFAULT_PHASEB_MODE_MULTISTEP if multistep else DEFAULT_PHASEB_MODE_SINGLE
    return mode


def _step_candidate_count(step_name: str) -> int:
    """Return bounded candidate count for an optimization step.

    Env forms:
      C2HLS_CANDIDATES_PER_STEP=3
      C2HLS_CANDIDATES_PER_STEP='{"coalescing": 3, "default": 1}'
    """
    raw = os.getenv(STEP_CANDIDATES_ENV, "").strip()
    if not raw:
        return 1
    try:
        if raw.startswith("{"):
            payload = json.loads(raw)
            value = payload.get(step_name, payload.get("default", 1))
        else:
            value = raw
        count = int(value)
    except (TypeError, ValueError, json.JSONDecodeError):
        logging.warning("Invalid %s=%r; using 1", STEP_CANDIDATES_ENV, raw)
        return 1
    return max(1, min(count, 8))


def _candidate_attempt_count(default: int = 1) -> int:
    """Return bounded per-candidate attempt count.

    This is separate from candidate count. In exhaustive mode each candidate
    can produce multiple synth-tested attempts, and the best successful
    attempt becomes that candidate's representative.
    """
    raw = os.getenv(CANDIDATE_ATTEMPTS_ENV, "").strip()
    if not raw:
        return max(1, min(int(default or 1), 10))
    try:
        count = int(raw)
    except (TypeError, ValueError):
        logging.warning("Invalid %s=%r; using %s", CANDIDATE_ATTEMPTS_ENV, raw, default)
        return max(1, min(int(default or 1), 10))
    return max(1, min(count, 10))


def _exhaustive_candidate_attempts_enabled() -> bool:
    raw = os.getenv(EXHAUSTIVE_CANDIDATE_ATTEMPTS_ENV, "0")
    try:
        return bool(int(raw or "0"))
    except (TypeError, ValueError):
        logging.warning("Invalid %s=%r; using 0", EXHAUSTIVE_CANDIDATE_ATTEMPTS_ENV, raw)
        return False


def _metric_stats_from_reports(reports: list[dict]) -> dict:
    """Compute min/max/avg for metrics present in synth reports."""
    metrics = (
        "latency_cycles", "latency_ns", "interval",
        "bram", "dsp", "ff", "lut", "uram", "fmax_mhz",
    )
    stats = {}
    for metric in metrics:
        values = []
        for report in reports:
            if not isinstance(report, dict):
                continue
            value = _as_float(report.get(metric))
            if value is not None:
                values.append(value)
        if not values:
            continue
        stats[metric] = {
            "min": min(values),
            "max": max(values),
            "avg": sum(values) / len(values),
            "count": len(values),
        }
    return stats


def _compact_attempt_record(record: dict) -> dict:
    """Strip heavy code blobs from nested candidate/attempt telemetry."""
    if not isinstance(record, dict):
        return record
    item = {k: v for k, v in record.items() if k != "code"}
    if "attempt_results" in item:
        item["attempt_results"] = [
            _compact_attempt_record(entry)
            for entry in (item.get("attempt_results") or [])
        ]
    if "candidate_attempts" in item:
        item["candidate_attempts"] = [
            _compact_attempt_record(entry)
            for entry in (item.get("candidate_attempts") or [])
        ]
    return item


def _detect_in_place_neighbor_update(code: str | None) -> dict:
    """Detect generic in-place 2D neighbor updates.

    This intentionally avoids benchmark names. It looks for assignments to a
    2D array element where the RHS reads the same array at a neighboring index,
    e.g. ``A[i][j] = f(A[i-1][j], A[i][j-1], ...)``. Such patterns usually
    carry true loop dependencies, so generic false-dependence/coalescing advice
    is unsafe unless an algorithmic proof says otherwise.
    """
    if not code:
        return {"detected": False}
    text = re.sub(r"/\*.*?\*/", " ", code, flags=re.S)
    text = re.sub(r"//.*", " ", text)
    assign_re = re.compile(
        r"\b(?P<arr>[A-Za-z_]\w*)\s*\[[^\]]+\]\s*\[[^\]]+\]\s*="
        r"(?P<rhs>[^;]+);",
        re.S,
    )
    for match in assign_re.finditer(text):
        arr = match.group("arr")
        rhs = match.group("rhs")
        if not re.search(rf"\b{re.escape(arr)}\s*\[", rhs):
            continue
        neighbor_read = (
            re.search(rf"\b{re.escape(arr)}\s*\[[^\]]*[\+\-]\s*1[^\]]*\]\s*\[", rhs)
            or re.search(rf"\b{re.escape(arr)}\s*\[[^\]]+\]\s*\[[^\]]*[\+\-]\s*1[^\]]*\]", rhs)
        )
        if neighbor_read:
            return {
                "detected": True,
                "array": arr,
                "reason": "same 2D array is written and read at neighboring indices",
            }
    return {"detected": False}


def _lint_in_place_stencil_guardrails(source_code: str | None,
                                      generated_code: str | None) -> list[str]:
    """Reject unsafe generic transforms for in-place stencil-like kernels."""
    source_match = _detect_in_place_neighbor_update(source_code)
    if not source_match.get("detected"):
        return []
    if not generated_code:
        return []
    issues: list[str] = []
    dep_false_lines = [
        line.strip()
        for line in generated_code.splitlines()
        if re.search(r"#\s*pragma\s+HLS\s+DEPENDENCE\b", line, re.I)
        and re.search(r"\bfalse\b", line, re.I)
    ]
    if dep_false_lines:
        issues.append(
            "in-place neighbor-update guardrail violation: generated code uses "
            "`DEPENDENCE ... false` even though the source writes and reads a "
            "neighboring element of the same 2D array in the same loop nest. "
            "Treat this dependence as true unless an explicit legal transform "
            "proves otherwise. Offending pragma(s): "
            + " | ".join(dep_false_lines[:4])
        )
    return issues


def _render_candidate_improvement_prompt(step_name: str, candidate_index: int,
                                         candidate_count: int, attempt_index: int,
                                         attempt_count: int, report: dict,
                                         current_code: str) -> str:
    """Prompt fragment for exhaustive candidate attempts after a pass."""
    return (
        f"CANDIDATE SEARCH CONTINUATION: candidate {candidate_index + 1} "
        f"of {candidate_count}, attempt {attempt_index + 1} of {attempt_count} "
        f"for `{step_name}`.\n\n"
        "The previous attempt compiled, synthesized, and passed available "
        "correctness checks. Produce another complete HLS C++ implementation "
        "for the same optimization step that tries to improve latency while "
        "preserving correctness and staying within the same resource budget. "
        "Do not remove required interface pragmas or change the public "
        "function signature.\n\n"
        "Previous successful attempt report:\n"
        f"{format_report_summary(report)}\n\n"
        "Current code to improve:\n"
        "```cpp\n"
        f"{current_code[:6000]}\n"
        "```\n\n"
        "Return only the complete revised C++ code in a fenced cpp block."
    )


def _render_regression_guidance(step_name: str, reasons: list) -> str:
    """Format a regression-reason list as a prompt fragment for the LLM's
    next attempt. Tells it bluntly that the previous output was rejected
    and what specifically regressed."""
    if not reasons:
        return ""
    bullets = "\n".join(f"  - {r}" for r in reasons)
    return (
        f"Your previous attempt at the `{step_name}` step was REJECTED because "
        f"it regressed against the previous step's metrics:\n"
        f"{bullets}\n\n"
        f"Produce a more conservative version that PRESERVES or IMPROVES on the "
        f"previous step's latency and does not inflate resource usage. If the "
        f"requested optimization cannot help here, return the previous code "
        f"with only minor tweaks."
    )


def _render_baseline_scope_diff(baseline_report: dict, current_report: dict,
                                step_name: str) -> str:
    """Render a per-loop diff between the baseline synthesis report and the
    current step's report using Pillar 1 bottleneck records.

    Tells the LLM exactly which loops regressed (or improved) relative to the
    baseline, so it can understand WHY latency changed — not just that it did.
    Returns empty string when either report lacks scope data.
    """
    def _get_bns(report: dict) -> dict:
        """Return {scope_id: bottleneck_record} for the top bottlenecks."""
        feedback = (report or {}).get("feedback") or {}
        bns = feedback.get("bottlenecks") or []
        return {bn.get("scope_id", ""): bn for bn in bns if bn.get("scope_id")}

    base_bns = _get_bns(baseline_report)
    cur_bns = _get_bns(current_report)
    if not base_bns and not cur_bns:
        return ""

    base_cyc = (baseline_report or {}).get("latency_cycles")
    cur_cyc = (current_report or {}).get("latency_cycles")
    ratio = ""
    if base_cyc and cur_cyc and base_cyc > 0:
        ratio = f" ({cur_cyc/base_cyc:.2f}× baseline)"

    lines = [
        f"BASELINE vs `{step_name}` per-loop diff "
        f"(baseline={base_cyc} cyc → current={cur_cyc} cyc{ratio}):"
    ]

    # Scopes that were bottlenecks in baseline
    for sid, bn in base_bns.items():
        cur = cur_bns.get(sid)
        base_ev = bn.get("evidence", "")
        if cur:
            cur_ev = cur.get("evidence", "")
            changed = " ← CHANGED" if cur_ev != base_ev else ""
            lines.append(f"  {sid}: baseline=({base_ev}) → now=({cur_ev}){changed}")
        else:
            lines.append(f"  {sid}: baseline=({base_ev}) → now=RESOLVED ✓")

    # New bottlenecks that did not exist in baseline
    new_sids = set(cur_bns) - set(base_bns)
    for sid in sorted(new_sids):
        bn = cur_bns[sid]
        lines.append(
            f"  {sid}: NEW bottleneck → ({bn.get('evidence','')}) ← introduced by {step_name}"
        )

    if len(lines) == 1:
        return ""  # Only header, nothing to show
    lines.append(
        "Target: eliminate the NEW bottlenecks and reduce II on loops that "
        "worsened vs baseline. Do NOT introduce new non-pipelined outer loops "
        "unless they enable a larger II improvement inside."
    )
    return "\n".join(lines)


def _render_step_resource_constraints(step_name: str, current_report: dict,
                                      part: str = "") -> str:
    """Render the per-step resource ceilings as a compact constraint block
    so the LLM knows what budget it has BEFORE generating code.

    The agent is told:
      - Current resource usage on each axis (from the previous step's report)
      - The per-step ratio ceiling (from STEP_REGRESSION_THRESHOLDS)
      - The resulting absolute ceiling in raw units (so it can reason concretely)
      - The two-tier override rule (latency ≥2× improvement + fits on chip)
      - Device capacity so it knows the hard upper bound

    This prevents the pattern where Sonnet aggressively unrolls to fix II=144,
    blows through the DSP ceiling (29.67×), and gets reverted — wasting a
    synthesis run and triggering a cascade of regressed subsequent steps.
    """
    if not current_report:
        return ""

    if _reference_blind_enabled():
        capacity = _resource_capacity_for(part) if part else {}
        lines = [
            f"DEVICE-FIT CONSTRAINTS for the `{step_name}` step:",
            "  Preserve golden-output correctness and meet the target clock.",
            "  No benchmark-specific or expert-calibrated relative resource "
            "ceiling is used; every reported resource must fit the device.",
        ]
        for key in ("dsp", "bram", "uram", "ff", "lut"):
            cur_v = _as_float(current_report.get(key))
            cap = _as_float((capacity or {}).get(key))
            if cur_v is None and cap is None:
                continue
            current = "unknown" if cur_v is None else str(int(cur_v))
            device_cap = "unknown" if cap is None else str(int(cap))
            lines.append(f"  {key:<6}: current={current}  device_cap={device_cap}")
        return "\n".join(lines)

    cfg = _resolve_step_thresholds(step_name)
    lat_threshold = float(cfg.get("latency", 1.10))
    resource_thresholds = cfg.get("resources") or {"default": 1.10}
    default_t = float(resource_thresholds.get("default", 1.10))

    capacity = _resource_capacity_for(part) if part else {}

    cur_lat_ns = _as_float(current_report.get("latency_ns"))
    lines = [
        f"RESOURCE CONSTRAINTS for the `{step_name}` step "
        f"(enforced by the synthesis regression guard — exceeding these "
        f"causes automatic revert and retry):",
    ]
    if cur_lat_ns:
        max_lat_ns = cur_lat_ns * lat_threshold
        lines.append(
            f"  latency_ns: current={cur_lat_ns:.0f} → max={max_lat_ns:.0f} "
            f"(limit {lat_threshold:.2f}× current)"
        )

    for key in ("dsp", "bram", "ff", "lut"):
        cur_v = _as_float(current_report.get(key))
        if cur_v is None:
            continue
        t = float(resource_thresholds.get(key, default_t))
        max_v = cur_v * t
        cap = _as_float((capacity or {}).get(key))
        cap_str = f"  device_cap={int(cap)}" if cap else ""
        lines.append(
            f"  {key:<6}: current={int(cur_v):>6} → max={int(max_v):>7} "
            f"(limit {t:.2f}×){cap_str}"
        )

    lines.append(
        "Two-tier override: if your optimization reduces latency_ns by ≥2× "
        "AND every resource stays below device capacity, the step is accepted "
        "even if per-step ratio limits are exceeded. Use this when aggressive "
        "parallelization (e.g. DSP unrolling) genuinely halves latency."
    )
    lines.append(
        "Strategy guidance for II violations from AXI bus dependencies: "
        "prefer LOCAL BUFFER staging (load tile into local array, then compute) "
        "over aggressive unrolling — local buffers resolve memory port conflicts "
        "without DSP growth, and stay well within the tiling/pipeline ceilings. "
        "Reserve DSP-heavy unrolling for the `unroll` and later steps where "
        "the DSP ceiling is 8× or more."
    )
    return "\n".join(lines)


def _as_float(value) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _comparison_ratio(comparison: dict, key: str) -> Optional[float]:
    vals = (comparison or {}).get(key, {})
    return _as_float(vals.get("ratio"))


def _reference_ratio_summary(comparison: dict) -> dict:
    """Small prompt/history-safe comparison: ratios only, no GT absolutes."""
    if _reference_blind_enabled():
        return {}
    out = {}
    for key in ("latency_cycles", "latency_ns", "interval", "fmax_mhz"):
        ratio = _comparison_ratio(comparison, key)
        if ratio is not None:
            out[key] = {
                "ratio": round(ratio, 6),
                "semantics": "lower_is_better" if key != "fmax_mhz" else "higher_is_better",
            }
    return out


def _coalescing_diagnostics(hls_code: str, report: dict | None = None) -> dict:
    code = hls_code or ""
    uses_widening = bool(
        re.search(r"max_widen_bitwidth\s*=\s*(256|512|1024)", code)
        or re.search(r"\bap_uint\s*<\s*(256|512|1024)\s*>", code)
    )
    mentions_lanes = bool(re.search(r"\bLANES?\b|WIDTH_FACTOR|lane", code, re.IGNORECASE))
    has_compute_parallelism = bool(
        re.search(r"#\s*pragma\s+HLS\s+UNROLL", code)
        or re.search(r"#\s*pragma\s+HLS\s+ARRAY_PARTITION", code)
        or mentions_lanes
    )
    has_burst_or_staging = bool(
        re.search(r"max_(read|write)_burst_length|num_(read|write)_outstanding", code)
        or re.search(r"\b(load|store)_[A-Za-z0-9_]*:", code)
    )
    if uses_widening and not (has_compute_parallelism or has_burst_or_staging):
        status = "interface_only"
    elif uses_widening:
        status = "compound_or_partial"
    else:
        status = "not_applied"
    return {
        "status": status,
        "uses_widening": uses_widening,
        "has_compute_parallelism_marker": has_compute_parallelism,
        "has_burst_or_staging_marker": has_burst_or_staging,
        "latency_cycles": (report or {}).get("latency_cycles"),
    }


def _build_quality_guidance(benchmark_name: str, report: dict, ground_truth_report: dict, comparison: dict) -> str:
    # Defence in depth: paper runs normally make quality repair unreachable by
    # removing the reference report before control begins.  Also erase every
    # reference-derived and benchmark-specific input here so an accidental
    # direct call cannot reintroduce oracle information.
    reference_blind = _reference_blind_enabled()
    bench = "" if reference_blind else (benchmark_name or "")
    if reference_blind:
        ground_truth_report = {}
        comparison = {}
    issues = []

    slack = _as_float((report or {}).get("slack_ns"))
    timing_bottlenecks = _actionable_timing_bottlenecks(report)
    if slack is not None and slack < 0 and timing_bottlenecks:
        issues.append(
            "Timing is not clean and the report names actionable loop bottlenecks; "
            "repair the listed II/dependency or port-pressure causes rather than "
            "blindly removing parallelism."
        )

    fmax_ratio = _comparison_ratio(comparison, "fmax_mhz")
    if fmax_ratio is not None and fmax_ratio < 0.8:
        issues.append(
            f"Clock quality ratio to the hidden reference is {fmax_ratio:.3f} "
            "(higher is better); improve scheduling only if this does not give up "
            "a large latency win."
        )

    latency_ratio = _comparison_ratio(comparison, "latency_ns")
    if latency_ratio is not None and latency_ratio > 1.10:
        issues.append(
            f"Latency ratio to the hidden reference is {latency_ratio:.3f} "
            "(ratio > 1 is slower); reduce avoidable serialization or buffering."
        )

    part = os.getenv("C2HLS_PART", DEFAULT_PART)
    for item in _resource_over_device(report, part):
        issues.append(f"Device resource budget exceeded: {item}.")
    for key, data in _resource_utilization(report, part).items():
        if data["utilization"] >= 0.90:
            issues.append(
                f"{key.upper()} utilization is {100.0 * data['utilization']:.1f}% "
                f"of the configured device; reduce this only if it threatens fit "
                "or blocks timing."
            )

    if bench == "spmv_crs" and latency_ratio is not None and latency_ratio > 1.5 and (slack is None or slack >= 0) and (fmax_ratio is None or fmax_ratio >= 1.0):
        issues.insert(0, "Timing is already healthy, so focus this repair on reducing latency while keeping slack non-negative.")

    if bench == "spmv_crs" and ((slack is not None and slack < 0) or (fmax_ratio is not None and fmax_ratio < 0.8)):
        issues.insert(0, "Timing is still poor on this benchmark; prefer local-memory and dependency fixes over additional aggressive compute-side directives.")

    if bench == "StreamCluster" and latency_ratio is not None and latency_ratio < 1.0 and ((slack is not None and slack < 0) or (fmax_ratio is not None and fmax_ratio < 0.5)):
        issues.insert(0, "Latency headroom is ample, so it is acceptable to trade some extra cycles for better slack/Fmax and lower DSP pressure.")

    if not issues:
        return ""

    guidance = []
    priority = _policy(bench, "priority")
    if priority:
        guidance.append(priority)
    guidance.extend(issues)
    guidance.extend(_policy(bench, "quality", []))
    return "\n".join(f"- {line}" for line in guidance)


def _build_quality_context(report: dict, comparison: dict, *, part: str = "", clock_ns: float | None = None) -> str:
    """Agent-visible quality context with no absolute reference metrics."""
    reference_blind = _reference_blind_enabled()
    if reference_blind:
        comparison = {}
        lines = [
            "Use only the generated candidate's compiler measurements below; "
            "no expert comparison is available to the controller."
        ]
    else:
        lines = [
            "Reference data is held offline. Ratios below are directional only:",
            "  - latency/fmax ratios are generated/reference; latency ratio < 1 is faster, fmax ratio > 1 is faster.",
        ]
        for key, label in [
            ("latency_cycles", "latency_cycles"),
            ("latency_ns", "latency_ns"),
            ("fmax_mhz", "fmax_mhz"),
        ]:
            ratio = _comparison_ratio(comparison, key)
            if ratio is not None:
                lines.append(f"  - {label}_ratio={ratio:.3f}")

    active_part = part or os.getenv("C2HLS_PART", DEFAULT_PART)
    active_clock = clock_ns if clock_ns is not None else os.getenv("C2HLS_CLOCK_NS", str(DEFAULT_CLOCK_NS))
    lines.append(f"Configured target: {_target_context_for_prompt(active_part, active_clock)}")

    util = _resource_utilization(report, active_part)
    if util:
        lines.append("Device utilization:")
        for key in ("bram", "dsp", "ff", "lut", "uram"):
            data = util.get(key)
            if not data:
                continue
            lines.append(
                f"  - {key}: used={int(data['used'])} cap={int(data['capacity'])} "
                f"util={100.0 * data['utilization']:.1f}%"
            )

    slack = _as_float((report or {}).get("slack_ns"))
    est = _as_float((report or {}).get("estimated_clock_period_ns"))
    if slack is not None or est is not None:
        lines.append(
            f"Timing estimate: slack_ns={slack if slack is not None else 'unknown'}, "
            f"estimated_clock_period_ns={est if est is not None else 'unknown'}."
        )

    bottlenecks = _actionable_timing_bottlenecks(report)
    if bottlenecks:
        lines.append("Actionable report bottlenecks:")
        lines.extend(f"  - {item}" for item in bottlenecks)
    else:
        lines.append("No actionable timing path detail was parsed; avoid blind timing-only rewrites.")

    return "\n".join(lines)


def _quality_score(benchmark_name: str, report: dict, comparison: dict) -> float:
    if _reference_blind_enabled():
        # Match the paper's decision rule: infeasible reports rank behind every
        # feasible report; within a feasibility class, lower estimated latency
        # wins.  No benchmark identity or expert-relative quantity participates.
        feasibility = _paper_candidate_feasibility(
            report,
            correctness_required=False,
            part=os.getenv("C2HLS_PART", DEFAULT_PART),
            clock_ns=_as_float(os.getenv("C2HLS_CLOCK_NS", DEFAULT_CLOCK_NS)),
        )
        latency = _as_float(
            (report or {}).get("latency_cycles_worst")
            or (report or {}).get("latency_cycles")
            or (report or {}).get("latency_ns_worst")
            or (report or {}).get("latency_ns")
        )
        if latency is None or latency <= 0:
            latency = 1.0e14
        return round((0.0 if feasibility["feasible"] else 1.0e15) + latency, 3)

    bench = benchmark_name or ""
    score = 0.0

    slack = _as_float((report or {}).get("slack_ns"))
    if slack is not None and slack < 0 and _actionable_timing_bottlenecks(report):
        score += abs(slack) * 5.0

    fmax_ratio = _comparison_ratio(comparison, "fmax_mhz")
    if fmax_ratio is not None and fmax_ratio < 0.8:
        score += (0.8 - fmax_ratio) * 25.0

    latency_ratio = _comparison_ratio(comparison, "latency_ns")
    if latency_ratio is not None and latency_ratio > 1.0:
        score += (latency_ratio - 1.0) * 35.0
    cycles_ratio = _comparison_ratio(comparison, "latency_cycles")
    if cycles_ratio is not None and cycles_ratio > 1.0:
        score += (cycles_ratio - 1.0) * 20.0

    part = os.getenv("C2HLS_PART", DEFAULT_PART)
    for data in _resource_utilization(report, part).values():
        util = data["utilization"]
        if util > 1.0:
            score += (util - 1.0) * 1000.0
        elif util > 0.90:
            score += (util - 0.90) * 25.0

    if bench == "nw":
        if slack is not None and slack < 0:
            score += abs(slack) * 5.0
        if fmax_ratio is not None and fmax_ratio < 0.8:
            score += (0.8 - fmax_ratio) * 80.0
    elif bench == "spmv_crs":
        latency_focus = (slack is None or slack >= 0) and (fmax_ratio is None or fmax_ratio >= 1.0)
        if latency_focus:
            latency_ratio = _comparison_ratio(comparison, "latency_ns")
            if latency_ratio is not None and latency_ratio > 1.0:
                score += (latency_ratio - 1.0) * 35.0
    elif bench == "StreamCluster":
        if fmax_ratio is not None and fmax_ratio < 0.5:
            score += (0.5 - fmax_ratio) * 120.0

    return round(score, 3)


def _preserves_passed_test(current_summary: Optional[dict], candidate_summary: Optional[dict]) -> bool:
    if current_summary and current_summary.get("passed"):
        return bool(candidate_summary and candidate_summary.get("passed"))
    return True


def _quality_focus(benchmark_name: str, report: dict, comparison: dict) -> str:
    bench = benchmark_name or ""
    slack = _as_float((report or {}).get("slack_ns"))
    fmax_ratio = _comparison_ratio(comparison, "fmax_mhz")
    latency_ratio = _comparison_ratio(comparison, "latency_ns")
    part = os.getenv("C2HLS_PART", DEFAULT_PART)

    if _resource_over_device(report, part):
        return "device_budget"

    if _reference_blind_enabled():
        if slack is not None and slack < 0:
            return "timing"
        estimated = _as_float((report or {}).get("estimated_clock_period_ns"))
        target = _as_float(
            (report or {}).get("requested_clock_period_ns")
            or os.getenv("C2HLS_CLOCK_NS", DEFAULT_CLOCK_NS)
        )
        if estimated is not None and target is not None and estimated > target:
            return "timing"
        latency = _as_float(
            (report or {}).get("latency_cycles_worst")
            or (report or {}).get("latency_cycles")
            or (report or {}).get("latency_ns_worst")
            or (report or {}).get("latency_ns")
        )
        return "latency" if latency is not None and latency > 0 else "general"

    if bench == "spmv_crs":
        if ((slack is not None and slack < 0 and _actionable_timing_bottlenecks(report))
                or (fmax_ratio is not None and fmax_ratio < 0.8)):
            return "timing"
        if latency_ratio is not None and latency_ratio > 1.5:
            return "latency"
        return "general"

    if bench == "StreamCluster":
        if ((slack is not None and slack < 0 and _actionable_timing_bottlenecks(report))
                or (fmax_ratio is not None and fmax_ratio < 0.5)):
            return "timing"
        return "general"

    if bench == "nw":
        if ((slack is not None and slack < 0 and _actionable_timing_bottlenecks(report))
                or (fmax_ratio is not None and fmax_ratio < 0.8)):
            return "timing"
        return "general"

    if latency_ratio is not None and latency_ratio > 1.10:
        return "latency"

    return "general"


def _quality_focus_improved(benchmark_name: str, focus: str, current_report: dict, current_comparison: dict,
                            candidate_report: dict, candidate_comparison: dict) -> bool:
    if _reference_blind_enabled():
        part = os.getenv("C2HLS_PART", DEFAULT_PART)
        clock_ns = _as_float(os.getenv("C2HLS_CLOCK_NS", DEFAULT_CLOCK_NS))
        current_feasible = _paper_candidate_feasibility(
            current_report,
            correctness_required=False,
            part=part,
            clock_ns=clock_ns,
        )["feasible"]
        candidate_feasible = _paper_candidate_feasibility(
            candidate_report,
            correctness_required=False,
            part=part,
            clock_ns=clock_ns,
        )["feasible"]
        if candidate_feasible != current_feasible:
            return candidate_feasible

        if focus == "device_budget":
            cur_over = len(_resource_over_device(current_report, part))
            cand_over = len(_resource_over_device(candidate_report, part))
            if cand_over != cur_over:
                return cand_over < cur_over
            cur_max = max(
                (d["utilization"] for d in _resource_utilization(current_report, part).values()),
                default=0.0,
            )
            cand_max = max(
                (d["utilization"] for d in _resource_utilization(candidate_report, part).values()),
                default=0.0,
            )
            return cand_max < cur_max - 0.02

        if focus == "timing":
            current_slack = _as_float((current_report or {}).get("slack_ns"))
            candidate_slack = _as_float((candidate_report or {}).get("slack_ns"))
            if current_slack is not None and candidate_slack is not None:
                return candidate_slack > current_slack + 0.5
            current_period = _as_float((current_report or {}).get("estimated_clock_period_ns"))
            candidate_period = _as_float((candidate_report or {}).get("estimated_clock_period_ns"))
            if current_period is not None and candidate_period is not None:
                return candidate_period < current_period - 0.05
            current_fmax = _as_float((current_report or {}).get("fmax_mhz"))
            candidate_fmax = _as_float((candidate_report or {}).get("fmax_mhz"))
            if current_fmax is not None and candidate_fmax is not None:
                return candidate_fmax > current_fmax + 0.05
            return False

        current_latency = _as_float(
            (current_report or {}).get("latency_cycles_worst")
            or (current_report or {}).get("latency_cycles")
            or (current_report or {}).get("latency_ns_worst")
            or (current_report or {}).get("latency_ns")
        )
        candidate_latency = _as_float(
            (candidate_report or {}).get("latency_cycles_worst")
            or (candidate_report or {}).get("latency_cycles")
            or (candidate_report or {}).get("latency_ns_worst")
            or (candidate_report or {}).get("latency_ns")
        )
        return bool(
            current_latency is not None
            and candidate_latency is not None
            and candidate_latency < current_latency
        )

    current_slack = _as_float((current_report or {}).get("slack_ns"))
    candidate_slack = _as_float((candidate_report or {}).get("slack_ns"))
    current_fmax = _comparison_ratio(current_comparison, "fmax_mhz") or 0.0
    candidate_fmax = _comparison_ratio(candidate_comparison, "fmax_mhz") or 0.0
    current_latency = _comparison_ratio(current_comparison, "latency_ns") or float("inf")
    candidate_latency = _comparison_ratio(candidate_comparison, "latency_ns") or float("inf")
    part = os.getenv("C2HLS_PART", DEFAULT_PART)

    timing_better = False
    if current_slack is not None and candidate_slack is not None and candidate_slack > current_slack + 0.5:
        timing_better = True
    if candidate_fmax > current_fmax + 0.05:
        timing_better = True

    if focus == "device_budget":
        cur_over = len(_resource_over_device(current_report, part))
        cand_over = len(_resource_over_device(candidate_report, part))
        if cand_over < cur_over:
            return True
        cur_max = max((d["utilization"] for d in _resource_utilization(current_report, part).values()), default=0.0)
        cand_max = max((d["utilization"] for d in _resource_utilization(candidate_report, part).values()), default=0.0)
        return cand_max < cur_max - 0.02
    if focus == "timing":
        return timing_better
    if focus == "latency":
        return candidate_latency < current_latency - 0.05

    return True


# =============================================================================
# Agent split (P3): Phase logic lives on dedicated agent classes; the
# orchestrator below instantiates them and delegates. State + shared helpers
# (history, messages, _synth_and_test, _evaluate_candidate_with_repairs)
# remain on the orchestrator so this is a layering refactor, not a state
# migration. Each agent has its own model, resolvable via env var, with a
# fallback to the orchestrator's default. The lazy LLM client cache in
# C2HLSOrchestrator._call_llm_with_model handles cross-backend mixing
# (e.g. a Claude translator + a vLLM-served Qwen synthesizer) without
# forcing all callers to manage clients.

# Per-agent model env vars. Unset -> falls through to gpt_model.
TRANSLATOR_MODEL_ENV     = "C2HLS_TRANSLATOR_MODEL"
SYNTHESIS_MODEL_ENV      = "C2HLS_SYNTHESIS_MODEL"
QUALITY_REPAIR_MODEL_ENV = "C2HLS_QUALITY_REPAIR_MODEL"
FEEDBACK_MODEL_ENV       = "C2HLS_FEEDBACK_MODEL"


class _AgentBase:
    """Base class for phase-specific agents.

    Each agent owns its phase's prompts and control flow. State (messages,
    history, hls_code, synth_report, etc.) lives on the parent orchestrator;
    agents read and mutate via `self.orch.<field>`. Helper methods that are
    shared across phases (compile_check_cpp via the orchestrator's
    _synth_and_test, the candidate-with-repairs evaluator) stay on the
    orchestrator so multistep mode and quality repair can keep reusing them.

    The only thing each agent owns independently is its LLM model id. The
    orchestrator's _call_llm_with_model() routes to the right backend.
    """
    AGENT_NAME = "base"
    MODEL_ENV = ""

    def __init__(self, orch: "C2HLSOrchestrator"):
        self.orch = orch
        env_model = os.getenv(self.MODEL_ENV) if self.MODEL_ENV else ""
        self.model = env_model or orch.gpt_model

    def _call_llm(self, messages: list, max_tokens: int = None) -> str:
        return self.orch._call_llm_with_model(
            messages, model=self.model, max_tokens=max_tokens,
            agent_name=self.AGENT_NAME,
        )

    def _request_code_revision(self, prompt: str) -> Optional[str]:
        """Append a user prompt to orch.messages, call this agent's LLM,
        record both turns in history, and return the extracted code."""
        self.orch.messages.append({"role": "user", "content": prompt})
        reply = self._call_llm(self.orch.messages)
        self.orch.messages.append({"role": "assistant", "content": reply})
        self.orch._append_history("user", prompt)
        self.orch._append_history("assistant", reply)
        code = extract_cpp_code(reply)
        if not code:
            self.orch._finalize_candidate_evaluation(
                correctness_status="not_run",
                synthesis_status="not_run",
                failure_class="malformed_output",
            )
        return code


class TranslatorAgent(_AgentBase):
    """Phase A (validate plain C compiles) + initial Phase B translate.

    Owns the conversation initialization (system prompt + first user prompt)
    so the SynthesisAgent's repair turns layer on top of a coherent thread.
    """
    AGENT_NAME = "translator"
    MODEL_ENV = TRANSLATOR_MODEL_ENV

    def run_phase_a(self, c_code: str, header_code: str = "",
                    header_name: str = "kernel.h") -> bool:
        orch = self.orch
        orch.c_code = c_code
        orch.header_code = header_code
        orch.header_name = header_name

        logging.info("=== [Phase A] Validating C code compilation ===")
        orch._append_history("system", Instruction_c2hls)

        ok, err = compile_check_cpp(
            c_code, header_code, header_name, extra_files=orch.extra_files,
        )
        if ok:
            logging.info("[Phase A] C code compiles successfully")
            orch._append_history("system", "[Phase A] Input C code compiles OK.")
            return True

        logging.warning("[Phase A] C code fails to compile: %s", err)
        for turn in range(orch.turns_limitation):
            prompt = q_validate_c_code.format(
                c_code=orch.c_code,
                header_code=orch.header_code,
                benchmark_context=orch.benchmark_context,
            )
            orch.messages = [
                {"role": "system", "content": Instruction_c2hls},
                {"role": "user", "content": prompt},
            ]
            reply = self._call_llm(orch.messages)
            orch._append_history("assistant", reply)

            fixed = extract_cpp_code(reply)
            if fixed:
                orch.c_code = fixed
                ok, err = compile_check_cpp(
                    orch.c_code, orch.header_code, orch.header_name,
                    extra_files=orch.extra_files,
                )
                if ok:
                    orch._finalize_candidate_evaluation(
                        code=orch.c_code,
                        correctness_status="not_run",
                        synthesis_status="not_run",
                        failure_class="other",
                        failure_detail=(
                            "LLM response repaired the plain-C input preflight; "
                            "it is not an HLS candidate"
                        ),
                    )
                    logging.info("[Phase A] Fixed C code compiles (turn %d)", turn)
                    return True
                orch._finalize_candidate_evaluation(
                    code=orch.c_code,
                    correctness_status="not_run",
                    synthesis_status="not_run",
                    failure_class="compile_or_interface_failure",
                )
                logging.warning("[Phase A] Still fails (turn %d): %s", turn, err[:200])
            else:
                orch._finalize_candidate_evaluation(
                    correctness_status="not_run",
                    synthesis_status="not_run",
                    failure_class="malformed_output",
                )

        orch._append_history(
            "system",
            f"[Phase A] FAIL: C code does not compile after {orch.turns_limitation} turns",
        )
        return False

    def translate_initial(self) -> Optional[str]:
        """Run Phase B's initial translate prompt and return the extracted
        HLS code. Does not synthesize — the SynthesisAgent does that."""
        orch = self.orch
        mode = _normalize_phaseb_mode(
            getattr(orch, "phaseb_mode", ""),
            multistep=bool(getattr(orch, "_phaseb_multistep_context", False)),
        )
        orch.phaseb_mode = mode
        logging.info("=== [Phase B] Translating C to HLS (mode=%s) ===", mode)

        prompt_template = (
            q_translate_c_to_hls_functional
            if mode == "functional"
            else q_translate_c_to_hls
        )
        prompt = prompt_template.format(
            c_code=orch.c_code,
            header_code=orch.header_code,
            benchmark_context=orch.benchmark_context,
        )
        orch.messages = [
            {"role": "system", "content": Instruction_c2hls},
            {"role": "user", "content": prompt},
        ]

        reply = self._call_llm(orch.messages)
        orch._append_history("user", prompt)
        orch._append_history("assistant", reply)
        orch.messages.append({"role": "assistant", "content": reply})

        hls_code = extract_cpp_code(reply)
        if not hls_code:
            orch._finalize_candidate_evaluation(
                correctness_status="not_run",
                synthesis_status="not_run",
                failure_class="malformed_output",
            )
            logging.error("[Phase B] No code block in LLM response")
            orch._append_history("system", "[Phase B] FAIL: no code in response")
            return None
        orch._append_history("system", f"[Phase B] Translation mode: {mode}.")
        return hls_code

    def retranslate_with_guidance(self, guidance: str,
                                    *, attempt: int = 1) -> Optional[str]:
        """Phase 8: re-run the translator prompt with a metric-only
        feedback block appended. Used by the baseline-alignment loop
        when our Phase B baseline is significantly worse than the
        reference baseline. ``guidance`` is a metric-only string
        produced by `_render_baseline_alignment_guidance` — it MUST
        NOT include the reference HLS source.
        """
        orch = self.orch
        logging.info(
            "=== [Phase 8] Re-translating with baseline-alignment "
            "guidance (attempt %d) ===", attempt,
        )
        # Baseline alignment always uses the functional prompt in multistep:
        # the goal is a clean starting point, not accidental optimization.
        prompt_template = (
            q_translate_c_to_hls_functional
            if _normalize_phaseb_mode(getattr(orch, "phaseb_mode", ""), multistep=True) == "functional"
            else q_translate_c_to_hls
        )
        base_prompt = prompt_template.format(
            c_code=orch.c_code,
            header_code=orch.header_code,
            benchmark_context=orch.benchmark_context,
        )
        prompt = base_prompt + "\n\n=== BASELINE ALIGNMENT FEEDBACK ===\n" + guidance
        orch.messages = [
            {"role": "system", "content": Instruction_c2hls},
            {"role": "user", "content": prompt},
        ]
        reply = self._call_llm(orch.messages)
        orch._append_history("user", "[Phase 8 retranslate] " + prompt[:200] + "...")
        orch._append_history("assistant", reply)
        orch.messages.append({"role": "assistant", "content": reply})
        hls_code = extract_cpp_code(reply)
        if not hls_code:
            orch._finalize_candidate_evaluation(
                correctness_status="not_run",
                synthesis_status="not_run",
                failure_class="malformed_output",
            )
            logging.warning("[Phase 8] No code in LLM retranslation response")
            return None
        return hls_code


class SynthesisAgent(_AgentBase):
    """Phase B synth + test loop with LLM-driven repair on compile or synth
    failures. Mutates orch.hls_code, orch.synth_report, orch.generated_csim,
    orch.generated_cosim, orch.turn_results.

    P5 enhancements:
      - **Profile-feedback loop.** Each repair prompt now carries a
        structured bottleneck summary (TIMING_VIOLATION / *_OVERFLOW /
        LATENCY_HIGH / FMAX_BELOW_TARGET) derived from the synth report,
        not just the raw error log. Empty when nothing is actionable.
      - **Best-state tracking + revert-on-streak.** If `C2HLS_SYNTH_REVERT_THRESHOLD`
        consecutive same-class errors hit, we either revert to the last
        successful synth (if any) and exit early, or — when no good run
        exists yet — abort the loop early instead of burning more LLM
        budget on the same dead end. Default 0 = disabled (preserves
        current behavior).
    """
    AGENT_NAME = "synthesis"
    MODEL_ENV = SYNTHESIS_MODEL_ENV

    @property
    def revert_threshold(self) -> int:
        """Read at access time so tests / env-overrides don't have to
        re-instantiate the orchestrator."""
        try:
            return int(os.getenv("C2HLS_SYNTH_REVERT_THRESHOLD", "0"))
        except ValueError:
            return 0

    def _compose_repair_guidance(self, err: str, report: dict = None) -> str:
        """Combine error-text hints with structured profile signals."""
        parts = [_build_repair_guidance(err)]
        if report:
            profile = _build_profile_signal(
                report,
                part=self.orch.part,
                requested_clock_ns=self.orch.clock_ns,
            )
            if profile:
                parts.append("")  # blank line between sections
                parts.append(profile)
        return "\n".join(parts)

    def _record_best(self, hls_code: str, result: dict, outcome: dict) -> dict:
        """Capture a synth-success snapshot for revert-on-streak."""
        return {
            "code": hls_code,
            "report": result.get("report", {}),
            "csim": outcome.get("csim"),
            "cosim": outcome.get("cosim"),
        }

    def _correctness_gate_failure(self, outcome: dict) -> "tuple[str, str]":
        """Return (gate_name, error_text) when Phase B synth passes but a
        generated csim/cosim check that actually ran fails.

        Phase B is only a valid baseline when it is functionally correct under
        the available generated testbench. Reference csim can be externally
        trusted, but generated code must still pass its own correctness gate.
        """
        if os.getenv("C2HLS_DISABLE_CORRECTNESS_REPAIR", "0").lower() in ("1", "true", "yes"):
            return "", ""
        for gate_name in ("csim", "cosim"):
            if gate_name == "cosim" and not _cosim_required_for_correctness():
                continue
            summary = outcome.get(gate_name)
            if not isinstance(summary, dict):
                continue
            # A timeout, including a gold-relative predicted timeout, is an
            # inconclusive performance outcome rather than evidence that the
            # generated algorithm is functionally wrong. Do not spend repair
            # turns regenerating the same finite design.
            if _test_summary_is_timeout(summary):
                continue
            if not summary.get("ran") or summary.get("passed"):
                continue
            gate_error = (
                (summary.get("error") or "").strip() + "\n"
                + (summary.get("log_excerpt") or "").strip()
            ).strip() or "(testbench reported a mismatch but did not capture an error message)"
            return gate_name, gate_error
        return "", ""

    def _restore_from_best(self, best: dict) -> None:
        orch = self.orch
        orch.hls_code = best["code"]
        orch.synth_report = best["report"]
        orch.generated_csim = best.get("csim")
        orch.generated_cosim = best.get("cosim")

    def synthesize_with_repair(self) -> bool:
        orch = self.orch
        best_state: Optional[dict] = None
        error_class_history: list[str] = []
        threshold = self.revert_threshold

        for turn in range(orch.turns_limitation):
            logging.info("[Phase B] Synthesis attempt %d", turn)
            orch.hls_code = orch._preflight_generated_hls_code(
                orch.hls_code, f"Phase B attempt {turn}",
            )

            ok, err = compile_check_cpp(
                orch.hls_code, orch.header_code, orch.header_name,
                extra_files=orch.extra_files,
            )
            if not ok:
                logging.warning("[Phase B] HLS code doesn't compile: %s", err[:200])
                orch._finalize_candidate_evaluation(
                    code=orch.hls_code,
                    correctness_status="not_run",
                    synthesis_status="not_run",
                    failure_class="compile_or_interface_failure",
                )
                error_class_history.append(_classify_synth_error(err))
                if self._should_revert(error_class_history, best_state, threshold):
                    return self._revert_and_exit(error_class_history, best_state, threshold)
                fix_prompt = c_compilation_fix.format(
                    compile_error=err,
                    hls_code=orch.hls_code,
                    benchmark_context=orch.benchmark_context,
                    repair_guidance=self._compose_repair_guidance(err, report=None),
                    attempt_history=_format_attempt_history(orch.turn_results, "B"),
                )
                orch.messages.append({"role": "user", "content": fix_prompt})
                reply = self._call_llm(orch.messages)
                orch.messages.append({"role": "assistant", "content": reply})
                orch._append_history("user", fix_prompt)
                orch._append_history("assistant", reply)
                fixed = extract_cpp_code(reply)
                if fixed:
                    orch.hls_code = fixed
                else:
                    orch._finalize_candidate_evaluation(
                        correctness_status="not_run",
                        synthesis_status="not_run",
                        failure_class="malformed_output",
                    )
                continue

            outcome = orch._synth_and_test(orch.hls_code, log_prefix="[Phase B]")
            result = outcome["synth"]
            orch.turn_results.append({
                "turn": turn,
                "phase": "B",
                "success": result["success"],
                "report": result.get("report", {}),
                "error": result.get("error", ""),
            })

            if result.get("budget_exhausted"):
                logging.warning("[Phase B] %s", result.get("error"))
                return False

            if result.get("skip_reason") == "csim_correctness_gate_failed":
                gate_summary = outcome.get("csim") or {}
                gate_error = (
                    (gate_summary.get("error") or "").strip()
                    + "\n"
                    + (gate_summary.get("log_excerpt") or "").strip()
                ).strip() or "pre-synthesis CSim/golden comparison failed"
                failure = f"csim_failed: {gate_error[:300]}"
                orch.turn_results.append({
                    "turn": turn,
                    "phase": "B",
                    "success": False,
                    "stage": "csim",
                    "csim": gate_summary,
                    "error": failure,
                })
                if turn >= orch.turns_limitation - 1:
                    continue
                fix_prompt = hls_correctness_repair_fix.format(
                    step_name="initial translation",
                    gate_name="csim",
                    gate_error=gate_error[:2000],
                    hls_code=orch.hls_code,
                    header_code=orch.header_code,
                    benchmark_context=orch.benchmark_context,
                    attempt_history=_format_attempt_history(orch.turn_results, "B"),
                )
                orch.messages.append({"role": "user", "content": fix_prompt})
                reply = self._call_llm(orch.messages)
                orch.messages.append({"role": "assistant", "content": reply})
                orch._append_history("user", fix_prompt)
                orch._append_history("assistant", reply)
                fixed = extract_cpp_code(reply)
                if fixed:
                    orch.hls_code = fixed
                else:
                    orch._finalize_candidate_evaluation(
                        correctness_status="not_run",
                        synthesis_status="not_run",
                        failure_class="malformed_output",
                    )
                continue

            if result["success"]:
                orch.synth_report = result["report"]
                logging.info("[Phase B] Synthesis SUCCESS!\n%s",
                             format_report_summary(result["report"]))
                orch._append_history(
                    "system",
                    f"[Phase B] Synthesis SUCCESS. Report:\n"
                    f"{format_report_summary(result['report'])}",
                )

                orch.generated_csim = outcome["csim"]
                if orch.generated_csim is not None:
                    if orch.generated_csim.get("passed"):
                        logging.info("[Phase B] Csim PASSED")
                    else:
                        logging.warning(
                            "[Phase B] Csim FAILED: %s",
                            (orch.generated_csim.get("error") or "")[:200],
                        )

                orch.generated_cosim = outcome["cosim"]
                if orch.generated_cosim is not None:
                    if orch.generated_cosim.get("passed"):
                        logging.info("[Phase B] Cosim PASSED")
                    elif _test_summary_is_timeout(orch.generated_cosim):
                        qualifier = "predicted; not run" if not orch.generated_cosim.get("ran") else "wall-clock"
                        logging.warning(
                            "[Phase B] Cosim TIMEOUT (%s): %s",
                            qualifier,
                            (orch.generated_cosim.get("error") or "")[:240],
                        )
                    else:
                        logging.warning(
                            "[Phase B] Cosim FAILED: %s",
                            (orch.generated_cosim.get("error") or "")[:200],
                        )

                gate_name, gate_error = self._correctness_gate_failure(outcome)
                if gate_name:
                    logging.warning(
                        "[Phase B] %s FAILED on attempt %d — asking translator/synthesis "
                        "thread to repair functionality before optimization",
                        gate_name,
                        turn,
                    )
                    failure = f"{gate_name}_failed: {gate_error[:300]}"
                    orch.turn_results.append({
                        "turn": turn,
                        "phase": "B",
                        "success": False,
                        "stage": gate_name,
                        "report": result.get("report", {}),
                        "csim": outcome.get("csim"),
                        "cosim": outcome.get("cosim"),
                        "error": failure,
                    })
                    error_class_history.append(f"{gate_name}_failed")
                    if self._should_revert(error_class_history, best_state, threshold):
                        return self._revert_and_exit(error_class_history, best_state, threshold)
                    if turn >= orch.turns_limitation - 1:
                        continue
                    fix_prompt = hls_correctness_repair_fix.format(
                        step_name="initial translation",
                        gate_name=gate_name,
                        gate_error=gate_error[:2000],
                        hls_code=orch.hls_code,
                        header_code=orch.header_code,
                        benchmark_context=orch.benchmark_context,
                        attempt_history=_format_attempt_history(orch.turn_results, "B"),
                    )
                    orch.messages.append({"role": "user", "content": fix_prompt})
                    reply = self._call_llm(orch.messages)
                    orch.messages.append({"role": "assistant", "content": reply})
                    orch._append_history("user", fix_prompt)
                    orch._append_history("assistant", reply)
                    fixed = extract_cpp_code(reply)
                    if fixed:
                        orch.hls_code = fixed
                    else:
                        orch._finalize_candidate_evaluation(
                            correctness_status="not_run",
                            synthesis_status="not_run",
                            failure_class="malformed_output",
                        )
                    continue

                # Snapshot only after correctness passes (or no generated
                # testbench was available). A synth-only success with failing
                # csim is not a usable Phase B baseline.
                best_state = self._record_best(orch.hls_code, result, outcome)
                error_class_history.clear()
                return True

            logging.warning("[Phase B] Synthesis failed: %s", result["error"][:300])
            error_class_history.append(_classify_synth_error(result["error"]))
            if self._should_revert(error_class_history, best_state, threshold):
                return self._revert_and_exit(error_class_history, best_state, threshold)

            is_timeout = "timed out" in result["error"].lower()
            guidance = self._compose_repair_guidance(
                result["error"], report=result.get("report"),
            )
            history_block = _format_attempt_history(orch.turn_results, "B")
            if is_timeout:
                fix_prompt = hls_synthesis_timeout_fix.format(
                    timeout=600,
                    hls_code=orch.hls_code,
                    header_code=orch.header_code,
                    benchmark_context=orch.benchmark_context,
                    repair_guidance=guidance,
                    attempt_history=history_block,
                )
            else:
                fix_prompt = hls_synthesis_fix.format(
                    synth_error=result["error"],
                    hls_code=orch.hls_code,
                    header_code=orch.header_code,
                    target_context=_target_context_for_prompt(orch.part, orch.clock_ns),
                    benchmark_context=orch.benchmark_context,
                    repair_guidance=guidance,
                    attempt_history=history_block,
                )
            orch.messages.append({"role": "user", "content": fix_prompt})
            reply = self._call_llm(orch.messages)
            orch.messages.append({"role": "assistant", "content": reply})
            orch._append_history("user", fix_prompt)
            orch._append_history("assistant", reply)
            fixed = extract_cpp_code(reply)
            if fixed:
                orch.hls_code = fixed
            else:
                orch._finalize_candidate_evaluation(
                    correctness_status="not_run",
                    synthesis_status="not_run",
                    failure_class="malformed_output",
                )

        orch._append_history(
            "system",
            f"[Phase B] FAIL: HLS synthesis/correctness not achieved in {orch.turns_limitation} turns",
        )
        return False

    def _should_revert(self, history: list, best_state: Optional[dict],
                       threshold: int) -> bool:
        """True when we've seen `threshold` consecutive errors of the same
        class. threshold<=0 disables the check entirely."""
        if threshold <= 0 or len(history) < threshold:
            return False
        recent = history[-threshold:]
        return len(set(recent)) == 1

    def _revert_and_exit(self, history: list, best_state: Optional[dict],
                         threshold: int) -> bool:
        """Either restore the last successful state or abort the loop early.
        Returning True means "Phase B succeeded (via the restored state)";
        False means "Phase B is giving up early because the LLM is stuck"."""
        orch = self.orch
        stuck_class = history[-1] if history else "unknown"
        if best_state is not None:
            self._restore_from_best(best_state)
            logging.info(
                "[Phase B] Stuck on %s for %d turns; reverting to last good synth",
                stuck_class, threshold,
            )
            orch._append_history(
                "system",
                f"[Phase B] Reverted to last good synth after "
                f"{threshold} consecutive {stuck_class} errors",
            )
            return True
        logging.warning(
            "[Phase B] Stuck on %s for %d turns and no good state to revert to; "
            "exiting Phase B early",
            stuck_class, threshold,
        )
        orch._append_history(
            "system",
            f"[Phase B] Early exit: {threshold} consecutive {stuck_class} "
            f"errors with no successful synth to revert to",
        )
        return False


class QualityRepairAgent(_AgentBase):
    """Iterative LLM-driven candidate generation + accept/reject loop to
    close the gen-vs-GT gap on rubric-tracked metrics."""
    AGENT_NAME = "quality_repair"
    MODEL_ENV = QUALITY_REPAIR_MODEL_ENV

    def run(self, ground_truth_report: dict,
            initial_comparison: Optional[dict] = None) -> dict:
        orch = self.orch
        summary = {"attempted": False, "applied": False, "attempts": []}
        orch.quality_repair_result = summary

        if (orch.quality_repair_turns <= 0
                or not ground_truth_report
                or not orch.synth_report
                or not orch.hls_code):
            summary["reason"] = "Quality repair disabled or missing reports"
            return summary

        current_comparison = initial_comparison or compare_reports(
            orch.synth_report, ground_truth_report,
        )
        quality_guidance = orch.feedback.render(
            "quality_gap",
            bench_name=orch.benchmark_name,
            report=orch.synth_report,
            ground_truth_report=ground_truth_report,
            comparison=current_comparison,
        )
        if not quality_guidance:
            summary["reason"] = "No timing/resource issues triggered quality repair"
            return summary

        summary["attempted"] = True
        summary["initial_score"] = _quality_score(
            orch.benchmark_name, orch.synth_report, current_comparison,
        )
        summary["initial_comparison"] = current_comparison
        current_score = summary["initial_score"]

        for turn in range(orch.quality_repair_turns):
            current_focus = _quality_focus(
                orch.benchmark_name, orch.synth_report, current_comparison,
            )
            prompt = hls_quality_repair.format(
                hls_code=orch.hls_code,
                current_report=format_report_summary(orch.synth_report),
                quality_context=_build_quality_context(
                    orch.synth_report,
                    current_comparison,
                    part=orch.part,
                    clock_ns=orch.clock_ns,
                ),
                benchmark_context=orch.benchmark_context,
                quality_guidance=quality_guidance,
            )
            proposed_code = self._request_code_revision(prompt)
            attempt = {
                "turn": turn,
                "focus": current_focus,
                "score_before": current_score,
            }
            if not proposed_code:
                attempt["status"] = "no_code"
                summary["attempts"].append(attempt)
                continue

            candidate = orch._evaluate_candidate_with_repairs(
                proposed_code, "[Quality Repair]",
            )
            if not candidate.get("success"):
                attempt["status"] = "failed"
                attempt["error"] = candidate.get("error", "")
                summary["attempts"].append(attempt)
                continue

            candidate_comparison = compare_reports(
                candidate["report"], ground_truth_report,
            )
            candidate_score = _quality_score(
                orch.benchmark_name, candidate["report"], candidate_comparison,
            )
            tests_preserved = (
                _preserves_passed_test(orch.generated_csim, candidate.get("csim"))
                and _preserves_passed_test(orch.generated_cosim, candidate.get("cosim"))
            )
            attempt.update({
                "candidate_score": candidate_score,
                "comparison": candidate_comparison,
                "tests_preserved": tests_preserved,
            })

            focus_improved = _quality_focus_improved(
                orch.benchmark_name, current_focus,
                orch.synth_report, current_comparison,
                candidate["report"], candidate_comparison,
            )
            attempt["focus_improved"] = focus_improved

            if (tests_preserved and focus_improved
                    and candidate_score + QUALITY_SCORE_EPSILON < current_score):
                orch.hls_code = candidate["code"]
                orch.synth_report = candidate["report"]
                orch.generated_csim = candidate.get("csim")
                orch.generated_cosim = candidate.get("cosim")
                current_comparison = candidate_comparison
                current_score = candidate_score
                summary["applied"] = True
                attempt["status"] = "accepted"
                summary["attempts"].append(attempt)
                logging.info("[Quality Repair] Accepted improved candidate with score %.3f",
                             candidate_score)

                quality_guidance = orch.feedback.render(
                    "quality_gap",
                    bench_name=orch.benchmark_name,
                    report=orch.synth_report,
                    ground_truth_report=ground_truth_report,
                    comparison=current_comparison,
                )
                if not quality_guidance:
                    break
                continue

            attempt["status"] = "rejected"
            if not tests_preserved:
                attempt["rejection_reason"] = "Functional checks regressed"
            elif not focus_improved:
                attempt["rejection_reason"] = (
                    f"Candidate did not improve the active {current_focus} focus"
                )
            else:
                attempt["rejection_reason"] = (
                    f"Quality score did not improve enough "
                    f"({candidate_score:.3f} vs {current_score:.3f})"
                )
            summary["attempts"].append(attempt)

        summary["final_score"] = current_score
        summary["final_comparison"] = current_comparison
        orch.quality_repair_result = summary
        return summary


# =============================================================================


class FeedbackAgent(_AgentBase):
    """Phase 4: single owner of "given a typed failure record, produce an
    LLM prompt fragment."

    Consolidates the seven feedback renderers that used to live as free
    module-level functions and one SynthesisAgent method. Two consumers
    (SynthesisAgent's repair loop and the orchestrator's multistep loop)
    now delegate here so feedback shape stays consistent and a future
    LLM-aided variant can plug in at one call site.

    Default behavior is **deterministic templates** (zero LLM calls) —
    same on-disk output as Phase 1+2+3 today, just re-homed. The
    optional ``compose_with_llm()`` path (gated by
    ``C2HLS_FEEDBACK_LLM=1``) reads the LLM's actual edit + the typed
    failure record and routes through this agent's own model
    (``C2HLS_FEEDBACK_MODEL``, defaults to a cheap Haiku) to compose a
    more strategic prompt. Off by default; not exercised in production
    yet.
    """
    AGENT_NAME = "feedback"
    MODEL_ENV = FEEDBACK_MODEL_ENV

    # ---- compile / synth error feedback (ex-SynthesisAgent helpers) ----

    def render_for_compile_error(self, err: str) -> str:
        """Hint block for a g++ pre-flight or Vitis compile error."""
        return _build_repair_guidance(err)

    def render_for_synth_error(self, err: str,
                                report: Optional[dict] = None) -> str:
        """Hint block for a Vitis synth failure. Combines the error-text
        guidance with a structured profile-bottleneck summary when a
        partial report is available."""
        parts = [_build_repair_guidance(err)]
        if report:
            profile = _build_profile_signal(
                report,
                part=self.orch.part,
                requested_clock_ns=self.orch.clock_ns,
            )
            if profile:
                parts.append("")
                parts.append(profile)
        return "\n".join(parts)

    # ---- multistep regression / no-op / alignment feedback (Phase 1+2+3) ----

    def render_for_regression(self, step_name: str,
                                reasons: List[str]) -> str:
        """Phase 1 regression-revert prompt fragment."""
        return _render_regression_guidance(step_name, reasons)

    def render_for_no_op(self, step_name: str,
                          reasons: List[str]) -> str:
        """Phase 9 no-op-trap prompt fragment."""
        return _render_no_op_guidance(step_name, reasons)

    def render_for_alignment(self, step_name: str,
                              decision) -> str:
        """Phase 3 trajectory-alignment "this step is an enabler" block.
        ``decision`` is a TrajectoryAlignmentDecision; rendering returns
        empty string when no message is needed."""
        from trajectory_alignment import render_alignment_for_prompt
        return render_alignment_for_prompt(decision, step_name)

    def render_for_throughput_regression(
        self, step_name: str, check) -> str:
        """Phase 9 throughput-regression hint. ``check`` is a
        robustness.ThroughputCheck."""
        if not check or not getattr(check, "flagged", False):
            return ""
        bullets = "\n".join(f"  - {r}" for r in (check.reasons or []))
        return (
            f"NOTE on the `{step_name}` step: throughput regression "
            f"flagged on the just-synthesized variant:\n{bullets}\n"
            f"Common causes: top-level kernel cannot start a new "
            f"transaction until the previous one drains (no DATAFLOW); "
            f"or unroll factor exceeds memory-port partition factor; "
            f"or pipeline depth was extended without rebalancing. Aim "
            f"for `Interval ≤ Latency` on the workload top function."
        )

    # ---- quality-gap feedback (ex-_build_quality_guidance) ----

    def render_for_quality_gap(
        self,
        bench_name: str,
        report: dict,
        ground_truth_report: dict,
        comparison: dict,
    ) -> str:
        """QualityRepairAgent's gap-closing prompt fragment."""
        return _build_quality_guidance(
            bench_name, report, ground_truth_report, comparison,
        )

    # ---- top-level dispatch ----

    def render(
        self,
        kind: str,
        *,
        step_name: str = "",
        err: str = "",
        report: Optional[dict] = None,
        reasons: Optional[List[str]] = None,
        decision=None,
        check=None,
        bench_name: str = "",
        ground_truth_report: Optional[dict] = None,
        comparison: Optional[dict] = None,
    ) -> str:
        """Single dispatcher by failure ``kind``. Returns the prompt
        fragment (or empty string when nothing actionable)."""
        if kind == "compile_error":
            return self.render_for_compile_error(err)
        if kind == "synth_error":
            return self.render_for_synth_error(err, report)
        if kind == "regression":
            return self.render_for_regression(step_name, reasons or [])
        if kind == "no_op":
            return self.render_for_no_op(step_name, reasons or [])
        if kind == "alignment":
            return self.render_for_alignment(step_name, decision)
        if kind == "throughput_regression":
            return self.render_for_throughput_regression(step_name, check)
        if kind == "quality_gap":
            return self.render_for_quality_gap(
                bench_name, report or {}, ground_truth_report or {},
                comparison or {},
            )
        return ""

    # ---- Phase-4 stretch: optional LLM-aided composition ----

    def compose_with_llm(
        self,
        kind: str,
        *,
        kernel_diff: str = "",
        prior_template: str = "",
        bottleneck_record: Optional[dict] = None,
        **render_kwargs,
    ) -> str:
        """Call this agent's own model (cheap by default) to compose a
        more strategic retry prompt by reading the LLM's actual edit.
        Off by default — only fires when ``C2HLS_FEEDBACK_LLM=1`` AND
        the deterministic template (``prior_template``) has already
        been tried at least once.

        Returns the LLM-composed prompt fragment, or falls back to the
        deterministic template on any error."""
        if not int(os.getenv("C2HLS_FEEDBACK_LLM", "0") or "0"):
            return prior_template or self.render(kind, **render_kwargs)

        try:
            from prompt_c2hls import Instruction_c2hls  # noqa
        except ImportError:
            Instruction_c2hls = ""  # type: ignore

        prompt_lines = [
            "You are a senior FPGA engineer reviewing an LLM-generated "
            "HLS edit that just failed in synthesis. Read the failure "
            "record and the diff of the LLM's edit, then produce a "
            "concise (≤ 8 sentences) retry guidance that names the "
            "specific code construct (loop label, function, array) to "
            "fix and the pragma syntax to use.",
            "",
            f"Failure kind: {kind}",
        ]
        if bottleneck_record:
            prompt_lines.append(
                f"Bottleneck record: {json.dumps(bottleneck_record, indent=2)}"
            )
        if prior_template:
            prompt_lines.append("")
            prompt_lines.append("Prior deterministic feedback (already tried, didn't work):")
            prompt_lines.append(prior_template)
        if kernel_diff:
            prompt_lines.append("")
            prompt_lines.append("LLM's edit (unified diff or full new code):")
            prompt_lines.append("```")
            prompt_lines.append(kernel_diff[:6000])
            prompt_lines.append("```")
        prompt_lines.append("")
        prompt_lines.append("Output: just the retry guidance text. No code fences.")

        try:
            messages = [
                {"role": "user", "content": "\n".join(prompt_lines)},
            ]
            return self._call_llm(messages, max_tokens=800)
        except Exception as exc:  # pragma: no cover - LLM-aided is best-effort
            logging.warning("FeedbackAgent.compose_with_llm failed (%s); "
                             "falling back to deterministic template", exc)
            return prior_template or self.render(kind, **render_kwargs)


# =============================================================================


class C2HLSOrchestrator:
    """Pipeline orchestrator for C-to-HLS translation.

    Holds shared state (messages, history, hls_code, synth_report, ...) and
    coordinates phase-specific agents:

      - self.translator      (TranslatorAgent)      Phase A + initial translate
      - self.synthesis       (SynthesisAgent)        Phase B synth+repair loop
      - self.quality_repair  (QualityRepairAgent)    candidate-improvement loop

    Multistep mode (run_optimization_step) and reference helpers (_synth_and_test,
    _evaluate_candidate_with_repairs, _preflight_generated_hls_code) remain on
    the orchestrator so existing callers (multistep, run_benchmark, etc.) are
    unchanged.
    """

    def __init__(self, max_completion_tokens=8192, gpt_model=DEFAULT_MODEL_ID,
                 turns_limitation=3, idx=0, quality_repair_turns=DEFAULT_QUALITY_REPAIR_TURNS):
        self.max_completion_tokens = max_completion_tokens
        self.gpt_model = gpt_model
        self.turns_limitation = turns_limitation
        self.idx = idx
        self.quality_repair_turns = quality_repair_turns

        self.use_anthropic = gpt_model.lower().startswith("claude")
        self.use_hosted_openai = _is_hosted_openai_model(gpt_model)
        if self.use_anthropic:
            assert HAS_ANTHROPIC, "anthropic package required for Claude models: pip install anthropic"
            api_key = _load_anthropic_api_key()
            assert api_key, f"Missing Anthropic API key. Set ANTHROPIC_API_KEY or populate {CLAUDE_API_KEY_FILE}."
            self.anthropic_client = anthropic.Anthropic(
                api_key=api_key,
                timeout=_llm_timeout_seconds(),
            )
        else:
            if self.use_hosted_openai:
                self.key = _load_openai_api_key()
                assert self.key, f"Missing OpenAI API key. Set OPENAI_API_KEY or populate {OPENAI_API_KEY_FILE}."
                self.base_url = OPENAI_HOSTED_BASE_URL
            else:
                self.key = os.getenv("OPENAI_API_KEY", "EMPTY")
                self.base_url = os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1")
            self.client = OpenAI(
                base_url=self.base_url,
                api_key=self.key,
                timeout=_llm_timeout_seconds(),
            )

        # Per-agent LLM cache. Populated lazily when an agent's model differs
        # from the orchestrator's default. Keys: model id; values: tuple
        # (kind, client) where kind is "anthropic" or "openai".
        self._extra_clients: dict = {}

        self.messages = []
        self.history = []
        self.c_code = None
        self.header_code = ""
        self.header_name = "kernel.h"
        self.phaseb_mode = os.getenv(PHASEB_MODE_ENV, "").strip().lower()
        self.phase_b_fast_candidate = None
        self.hls_code = None
        self.synth_report = None
        self.testbench_code = ""
        self.extra_files = []
        self.translated_hls_top = "workload"
        self.reference_hls_top = "workload"
        self.part = DEFAULT_PART
        self.clock_ns = DEFAULT_CLOCK_NS
        self.supports_cosim = False
        self.cosim_depths = {}
        self.generated_csim = None
        self.generated_cosim = None
        self.cosim_reference_cycle_info: dict = {}
        self.independent_golden_output: str = ""
        self.independent_golden_specs: dict = {}
        self.independent_golden_provenance: dict = {}
        self.benchmark_name = ""
        self.benchmark_context = ""
        self.preflight_patches = []
        self.turn_results = []  # tracks each synthesis attempt: {turn, phase, success, report, error}
        self.quality_repair_result = {
            "attempted": False,
            "applied": False,
            "attempts": [],
        }

        # ---- Phase 2 wiring (Pillars 3 / 5 / 6 / 7 / 9) ----
        # All opt-in. dynamic_routing=False keeps the static
        # tiling→pipeline→… order unchanged. Toggle via
        # `C2HLS_DYNAMIC_ROUTING=1`, the `--dynamic-routing` CLI flag,
        # `C2HLS_STRATEGY=dynamic`, or by setting `orch.dynamic_routing = True`
        # directly. Historically the sweep runner set only C2HLS_STRATEGY,
        # which loaded the skill library but still used static ordering; keep
        # the two knobs coherent so "dynamic" always means routed.
        env_strategy = os.getenv("C2HLS_STRATEGY", "").strip().lower()
        dynamic_env = os.getenv("C2HLS_DYNAMIC_ROUTING", "0").strip().lower()
        self.dynamic_routing: bool = (
            dynamic_env in {"1", "true", "yes", "on"}
            or env_strategy == "dynamic"
        )
        self.skill_library = None  # SkillLibrary, lazy-loaded when dynamic_routing kicks in
        self.skill_library_provenance: dict = {}
        self.vitis_version: str = os.getenv("C2HLS_VITIS_VERSION", "")
        # Trajectory-collapse / throughput-regression telemetry, populated
        # by run_multistep so callers can inspect the new robustness
        # signals without enabling dynamic routing.
        self.robustness_log: list = []
        self.llm_usage_events: list = []
        self.llm_controller_transcript: list[dict] = []
        self.llm_candidate_request_count = 0
        self._candidate_stream_started_monotonic: Optional[float] = None
        self.selected_winner_cosim_count = 0
        self.selected_code_sha256: Optional[str] = None
        self.cosim_target_code_sha256: Optional[str] = None
        try:
            parsed_llm_budget = int(os.getenv(LLM_CANDIDATE_BUDGET_ENV, "0") or "0")
        except (TypeError, ValueError):
            parsed_llm_budget = 0
        self.llm_candidate_budget: Optional[int] = (
            parsed_llm_budget if parsed_llm_budget > 0 else None
        )
        self.synthesis_eval_count = 0
        self.synthesis_eval_events: list[dict] = []
        try:
            parsed_budget = int(os.getenv(SYNTHESIS_EVAL_BUDGET_ENV, "0") or "0")
        except (TypeError, ValueError):
            parsed_budget = 0
        self.synthesis_eval_budget: Optional[int] = parsed_budget if parsed_budget > 0 else None

        # ---- Phase 3 wiring: strategy + GT-aware revert tolerance ----
        # `strategy` is the source of truth. dynamic_routing flag
        # remains for backward compat (it implies strategy="dynamic").
        # Allowed values: "static" | "dynamic" | "combo" | "combo_full" |
        # "combo_progressive" | "forward_eval" | "flash".
        # `gt_aware_revert` toggles the trajectory-alignment check that
        # tolerates regressions on enabling steps when the GT trajectory
        # also regresses there.
        if env_strategy:
            self.strategy: str = env_strategy
        elif self.dynamic_routing:
            self.strategy = "dynamic"
        else:
            self.strategy = "static"
        self.reference_blind: bool = _reference_blind_enabled()
        self.gt_aware_revert: bool = (
            _ground_truth_control_enabled()
            and _env_flag("C2HLS_GT_AWARE_REVERT", "0")
        )
        # Cache of GT step reports keyed by step_name, populated by
        # run_multistep before the optimization loop begins. Used by
        # _step_alignment_decision() to consult GT shape per-step.
        self._gt_step_reports: dict = {}
        self._gt_baseline_report: dict = {}
        # Agent baseline report stored so per-step prompts can diff their
        # per-loop bottlenecks against it (Pillar 1 scope comparison).
        self._baseline_report: dict = {}

        # Phase-specific agents. They share state with this orchestrator and
        # call _call_llm_with_model() for routing. Each picks up its own
        # model from C2HLS_TRANSLATOR_MODEL / _SYNTHESIS_MODEL / _QUALITY_REPAIR_MODEL,
        # falling back to gpt_model when unset.
        # Phase 4 adds FeedbackAgent (model: C2HLS_FEEDBACK_MODEL) — single
        # owner of "given a typed failure record, produce an LLM prompt
        # fragment". Existing call sites get a stable interface; the
        # FeedbackAgent's optional LLM-aided composition (off by default)
        # is the natural Phase-4-and-beyond extension point.
        self.translator = TranslatorAgent(self)
        self.synthesis = SynthesisAgent(self)
        self.quality_repair = QualityRepairAgent(self)
        self.feedback = FeedbackAgent(self)

    def configure_benchmark(
        self,
        extra_files=None,
        translated_hls_top: str = "workload",
        reference_hls_top: str = "workload",
        part: str = DEFAULT_PART,
        clock_ns: float = DEFAULT_CLOCK_NS,
        supports_cosim: bool = False,
        cosim_depths: Optional[dict] = None,
        benchmark_name: str = "",
        benchmark_context: str = "",
        independent_golden_output: str = "",
        independent_golden_specs: Optional[dict] = None,
        independent_golden_provenance: Optional[dict] = None,
    ):
        self.extra_files = list(extra_files or [])
        self.translated_hls_top = translated_hls_top or "workload"
        self.reference_hls_top = reference_hls_top or "workload"
        self.part = part or DEFAULT_PART
        self.clock_ns = clock_ns or DEFAULT_CLOCK_NS
        self.supports_cosim = supports_cosim
        self.cosim_depths = dict(cosim_depths or {})
        self.benchmark_name = benchmark_name or ""
        self.benchmark_context = benchmark_context or ""
        self.independent_golden_output = independent_golden_output or ""
        self.independent_golden_specs = dict(independent_golden_specs or {})
        self.independent_golden_provenance = dict(independent_golden_provenance or {})

    def _call_llm(self, messages: list, max_tokens: int = None) -> str:
        """Default-model LLM call. Kept as the public interface so existing
        callers (multistep, run_optimization_step, anything outside the
        agent classes) continue to work unchanged."""
        return self._call_llm_with_model(messages, model=self.gpt_model,
                                         max_tokens=max_tokens,
                                         agent_name="orchestrator")

    @staticmethod
    def _usage_value(obj, name: str, default: int = 0) -> int:
        if obj is None:
            return default
        if isinstance(obj, dict):
            value = obj.get(name, default)
        else:
            value = getattr(obj, name, default)
        try:
            return int(value or 0)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _configured_decoding() -> dict:
        def parse_float(name: str) -> Optional[float]:
            raw = os.getenv(name, "").strip()
            if not raw:
                return None
            value = float(raw)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {raw!r}")
            return value

        raw_seed = os.getenv(LLM_SEED_ENV, "").strip()
        return {
            "temperature": parse_float(LLM_TEMPERATURE_ENV),
            "top_p": parse_float(LLM_TOP_P_ENV),
            "seed": int(raw_seed) if raw_seed else None,
        }

    def _record_llm_usage(self, *, provider: str, model: str, agent_name: str,
                          usage, messages: list, max_tokens: int,
                          decoding: Optional[dict] = None,
                          candidate_evaluation_index: Optional[int] = None) -> None:
        """Record provider-reported token usage for bench-level accounting."""
        if candidate_evaluation_index is None:
            candidate_evaluation_index = len(self.llm_usage_events)
        prompt_payload = json.dumps(
            messages or [], sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        ).encode("utf-8")
        common = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "provider": provider,
            "model": model,
            "model_revision": os.getenv("C2HLS_MODEL_REVISION") or None,
            "agent": agent_name or "unknown",
            "message_count": len(messages or []),
            "max_tokens": max_tokens,
            "prompt_sha256": hashlib.sha256(prompt_payload).hexdigest(),
            "decoding": dict(decoding or {}),
            "candidate_evaluation_index": candidate_evaluation_index,
        }
        if usage is None:
            event = {
                **common,
                "usage_available": False,
            }
            self.llm_usage_events.append(event)
            self._start_candidate_evaluation_event(
                candidate_evaluation_index,
                agent_name=agent_name,
                model=model,
            )
            return

        if provider == "anthropic":
            input_tokens = self._usage_value(usage, "input_tokens")
            output_tokens = self._usage_value(usage, "output_tokens")
            cache_creation = self._usage_value(usage, "cache_creation_input_tokens")
            cache_read = self._usage_value(usage, "cache_read_input_tokens")
            normalized = {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
                "cache_creation_input_tokens": cache_creation,
                "cache_read_input_tokens": cache_read,
            }
        else:
            prompt_tokens = self._usage_value(usage, "prompt_tokens")
            completion_tokens = self._usage_value(usage, "completion_tokens")
            total_tokens = self._usage_value(
                usage, "total_tokens", prompt_tokens + completion_tokens
            )
            prompt_details = getattr(usage, "prompt_tokens_details", None)
            completion_details = getattr(usage, "completion_tokens_details", None)
            normalized = {
                "input_tokens": prompt_tokens,
                "output_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "cached_tokens": self._usage_value(prompt_details, "cached_tokens"),
                "reasoning_tokens": self._usage_value(completion_details, "reasoning_tokens"),
            }

        event = {
            **common,
            "usage_available": True,
            **normalized,
        }
        self.llm_usage_events.append(event)
        self._start_candidate_evaluation_event(
            candidate_evaluation_index,
            agent_name=agent_name,
            model=model,
        )

    def _llm_usage_summary(self) -> dict:
        fields = [
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "cache_creation_input_tokens",
            "cache_read_input_tokens",
            "cached_tokens",
            "reasoning_tokens",
        ]
        totals = {field: 0 for field in fields}
        by_agent: dict[str, dict] = {}
        by_model: dict[str, dict] = {}

        def add(bucket: dict, key: str, event: dict) -> None:
            item = bucket.setdefault(key or "unknown", {"calls": 0, **{field: 0 for field in fields}})
            item["calls"] += 1
            for field in fields:
                item[field] += int(event.get(field) or 0)

        for event in self.llm_usage_events:
            for field in fields:
                totals[field] += int(event.get(field) or 0)
            add(by_agent, event.get("agent"), event)
            add(by_model, event.get("model"), event)

        return {
            "schema_version": "1.0",
            "calls": len(self.llm_usage_events),
            "candidate_requests": self.llm_candidate_request_count,
            "candidate_budget": self.llm_candidate_budget,
            **totals,
            "by_agent": by_agent,
            "by_model": by_model,
            "usage_missing_calls": sum(
                1 for event in self.llm_usage_events
                if not event.get("usage_available")
            ),
            "events": self.llm_usage_events,
        }

    def _candidate_elapsed_seconds(self) -> float:
        started = getattr(self, "_candidate_stream_started_monotonic", None)
        if started is None:
            started = time.monotonic()
            self._candidate_stream_started_monotonic = started
        return max(0.0, time.monotonic() - started)

    def _llm_cumulative_totals(self) -> tuple[int, int]:
        events = getattr(self, "llm_usage_events", []) or []
        return (
            sum(int(event.get("total_tokens") or 0) for event in events),
            len(events),
        )

    def _start_candidate_evaluation_event(
        self,
        candidate_index: int,
        *,
        agent_name: str = "",
        model: str = "",
    ) -> dict:
        """Open one unified candidate event for one budgeted LLM response."""
        if not hasattr(self, "synthesis_eval_events"):
            self.synthesis_eval_events = []
        if not hasattr(self, "synthesis_eval_count"):
            self.synthesis_eval_count = 0
        if getattr(self, "_candidate_stream_started_monotonic", None) is None:
            self._candidate_stream_started_monotonic = time.monotonic()
        if candidate_index != len(self.synthesis_eval_events):
            raise RuntimeError(
                "candidate_evaluation_index must be contiguous: "
                f"got {candidate_index}, expected {len(self.synthesis_eval_events)}"
            )
        cumulative_tokens, cumulative_calls = self._llm_cumulative_totals()
        event = {
            "candidate_evaluation_index": candidate_index,
            "synthesis_index": None,
            "label": agent_name or "generated_candidate",
            "model": model,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "cumulative_tokens": cumulative_tokens,
            "cumulative_llm_calls": cumulative_calls,
            "cumulative_synthesis_evaluations": self.synthesis_eval_count,
            "cumulative_elapsed_seconds": self._candidate_elapsed_seconds(),
            "correctness_status": "not_run",
            "synthesis_status": "not_run",
            "resource_fit": None,
            "timing_met": None,
            "synthesized_latency_cycles": None,
            "latency_source": "none",
            "report_sha256": None,
            "failure_class": "other",
            "failure_detail": "candidate response has not completed evaluation",
            "selected_for_executed_cosim": False,
            "_llm_joined": True,
            "_telemetry_finalized": False,
        }
        self.synthesis_eval_events.append(event)
        return event

    def _pending_candidate_evaluation_index(self) -> Optional[int]:
        for index, event in enumerate(getattr(self, "synthesis_eval_events", []) or []):
            if not event.get("_telemetry_finalized"):
                return index
        return None

    def _ensure_candidate_evaluation_event(self, label: str = "") -> int:
        pending = self._pending_candidate_evaluation_index()
        if pending is not None:
            return pending
        # Compatibility for direct controller calls that bypass the LLM
        # producer. Such an event is deliberately not declared complete.
        index = len(getattr(self, "synthesis_eval_events", []) or [])
        if not hasattr(self, "synthesis_eval_events"):
            self.synthesis_eval_events = []
        cumulative_tokens, cumulative_calls = self._llm_cumulative_totals()
        event = {
            "candidate_evaluation_index": index,
            "synthesis_index": None,
            "label": label or "generated_candidate",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "cumulative_tokens": cumulative_tokens,
            "cumulative_llm_calls": cumulative_calls,
            "cumulative_synthesis_evaluations": getattr(self, "synthesis_eval_count", 0),
            "cumulative_elapsed_seconds": self._candidate_elapsed_seconds(),
            "correctness_status": "not_run",
            "synthesis_status": "not_run",
            "resource_fit": None,
            "timing_met": None,
            "synthesized_latency_cycles": None,
            "latency_source": "none",
            "report_sha256": None,
            "failure_class": "other",
            "failure_detail": "candidate evaluation has no attributed LLM response",
            "selected_for_executed_cosim": False,
            "_llm_joined": False,
            "_telemetry_finalized": False,
        }
        self.synthesis_eval_events.append(event)
        return index

    @staticmethod
    def _candidate_test_status(summary: Any) -> str:
        if not isinstance(summary, dict) or summary.get("ran") is not True:
            return "not_run"
        status = str(summary.get("status") or "").strip().lower()
        error = str(summary.get("error") or "").lower()
        if status == "timeout" or "timed out" in error:
            return "timeout"
        if status in {"tool_error", "tool_failure", "error"}:
            return "tool_failure"
        return "passed" if summary.get("passed") is True else "failed"

    @staticmethod
    def _candidate_synthesis_status(synth: Any, *, synthesis_ran: bool) -> str:
        if not synthesis_ran:
            return "not_run"
        synth = synth if isinstance(synth, dict) else {}
        error = str(synth.get("error") or "").lower()
        status = str(synth.get("status") or "").strip().lower()
        if synth.get("timed_out") or status == "timeout" or "timed out" in error:
            return "timeout"
        if synth.get("tool_failure") or status in {
            "tool_error", "tool_failure", "error",
        }:
            return "tool_failure"
        return "passed" if synth.get("success") is True else "failed"

    @staticmethod
    def _candidate_report_latency_cycles(report: Any) -> Optional[int]:
        if not isinstance(report, dict):
            return None
        for key in ("latency_cycles_worst", "latency_cycles"):
            value = report.get(key)
            if isinstance(value, bool):
                continue
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            if numeric > 0 and numeric.is_integer():
                return int(numeric)
        return None

    @staticmethod
    def _candidate_report_sha256(report: Any) -> Optional[str]:
        if not isinstance(report, dict) or not report:
            return None
        try:
            payload = json.dumps(
                report,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
                default=str,
            ).encode("utf-8")
        except (TypeError, ValueError):
            return None
        return hashlib.sha256(payload).hexdigest()

    def _finalize_candidate_evaluation(
        self,
        candidate_index: Optional[int] = None,
        *,
        code: str = "",
        correctness_status: str = "not_run",
        synthesis_status: str = "not_run",
        resource_fit: Optional[bool] = None,
        timing_met: Optional[bool] = None,
        latency_cycles: Optional[int] = None,
        report: Optional[dict] = None,
        failure_class: Optional[str] = "other",
        failure_detail: str = "",
    ) -> dict:
        if candidate_index is None:
            candidate_index = self._ensure_candidate_evaluation_event()
        event = self.synthesis_eval_events[candidate_index]
        event.update({
            "code_sha256": hashlib.sha256((code or "").encode("utf-8")).hexdigest(),
            "cumulative_synthesis_evaluations": getattr(self, "synthesis_eval_count", 0),
            "cumulative_elapsed_seconds": self._candidate_elapsed_seconds(),
            "correctness_status": correctness_status,
            "synthesis_status": synthesis_status,
            "resource_fit": resource_fit,
            "timing_met": timing_met,
            "synthesized_latency_cycles": latency_cycles,
            "latency_source": "vitis_csynth_report" if latency_cycles is not None else "none",
            "report_sha256": self._candidate_report_sha256(report),
            "failure_class": failure_class,
            "selected_for_executed_cosim": bool(
                event.get("selected_for_executed_cosim")
            ),
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "_telemetry_finalized": True,
        })
        if failure_class == "other":
            event["failure_detail"] = failure_detail or "unclassified candidate failure"
        else:
            event.pop("failure_detail", None)
        return event

    def _finalize_candidate_from_outcome(
        self,
        candidate_index: int,
        code: str,
        outcome: dict,
        *,
        synthesis_ran: bool,
    ) -> dict:
        csim = outcome.get("csim")
        synth = outcome.get("synth") if isinstance(outcome.get("synth"), dict) else {}
        correctness_status = self._candidate_test_status(csim)
        synthesis_status = self._candidate_synthesis_status(
            synth, synthesis_ran=synthesis_ran,
        )
        report = synth.get("report") if synthesis_status == "passed" else {}
        latency_cycles = self._candidate_report_latency_cycles(report)
        resource_fit = None
        timing_met = None
        failure_class: Optional[str]
        failure_detail = ""
        if correctness_status != "passed":
            if correctness_status == "failed":
                failure_class = "wrong_output"
            elif correctness_status in {"timeout", "tool_failure"}:
                failure_class = "tool_failure"
            else:
                failure_class = "other"
                failure_detail = "candidate did not execute the required CSim gate"
        elif synthesis_status == "timeout":
            failure_class = "synthesis_timeout"
        elif synthesis_status == "tool_failure":
            failure_class = "tool_failure"
        elif synthesis_status == "failed":
            failure_class = "compile_or_interface_failure"
        elif synthesis_status == "passed":
            feasibility = _paper_candidate_feasibility(
                report or {},
                csim=csim,
                correctness_required=bool(getattr(self, "testbench_code", "")),
                part=getattr(self, "part", DEFAULT_PART),
                clock_ns=getattr(self, "clock_ns", DEFAULT_CLOCK_NS),
            )
            resource_fit = (
                feasibility.get("resource_fit")
                if isinstance(feasibility.get("resource_fit"), bool)
                else None
            )
            timing_met = (
                feasibility.get("timing_met")
                if isinstance(feasibility.get("timing_met"), bool)
                else None
            )
            if resource_fit is False or resource_fit is None:
                failure_class = "infeasible_resources"
            elif timing_met is False or timing_met is None:
                failure_class = "timing_failure"
            elif latency_cycles is None:
                failure_class = "other"
                failure_detail = "passing synthesis lacks exact Vitis CSynth latency cycles"
            else:
                failure_class = None
        else:
            failure_class = "other"
            failure_detail = "candidate synthesis status is unavailable"
        return self._finalize_candidate_evaluation(
            candidate_index,
            code=code,
            correctness_status=correctness_status,
            synthesis_status=synthesis_status,
            resource_fit=resource_fit,
            timing_met=timing_met,
            latency_cycles=latency_cycles if synthesis_status == "passed" else None,
            report=report if synthesis_status == "passed" else None,
            failure_class=failure_class,
            failure_detail=failure_detail,
        )

    def _seal_candidate_event_stream(self, reason: str = "controller terminated") -> None:
        for index, event in enumerate(getattr(self, "synthesis_eval_events", []) or []):
            if event.get("_telemetry_finalized"):
                continue
            self._finalize_candidate_evaluation(
                index,
                correctness_status="not_run",
                synthesis_status="not_run",
                failure_class="other",
                failure_detail=reason,
            )

    def _client_for_model(self, model: str):
        """Get or create the right backend client for a given model id.

        Returns a tuple (kind, client) where kind is "anthropic" or
        "openai". The orchestrator's default-model client is reused; non-
        default models populate self._extra_clients on first use.
        """
        if model == self.gpt_model:
            if self.use_anthropic:
                return ("anthropic", self.anthropic_client)
            return ("openai", self.client)
        cached = self._extra_clients.get(model)
        if cached is not None:
            return cached

        is_claude = model.lower().startswith("claude")
        if is_claude:
            assert HAS_ANTHROPIC, (
                "anthropic package required for Claude models: pip install anthropic"
            )
            api_key = _load_anthropic_api_key()
            assert api_key, (
                f"Missing Anthropic API key for agent model {model!r}; "
                f"set ANTHROPIC_API_KEY or populate {CLAUDE_API_KEY_FILE}."
            )
            client = anthropic.Anthropic(
                api_key=api_key,
                timeout=_llm_timeout_seconds(),
            )
            entry = ("anthropic", client)
        else:
            if _is_hosted_openai_model(model):
                api_key = _load_openai_api_key()
                assert api_key, (
                    f"Missing OpenAI API key for agent model {model!r}."
                )
                base_url = OPENAI_HOSTED_BASE_URL
            else:
                api_key = os.getenv("OPENAI_API_KEY", "EMPTY")
                base_url = os.getenv("OPENAI_BASE_URL",
                                     "http://127.0.0.1:8000/v1")
            entry = ("openai", OpenAI(
                base_url=base_url,
                api_key=api_key,
                timeout=_llm_timeout_seconds(),
            ))
        self._extra_clients[model] = entry
        return entry

    def _call_llm_with_model(self, messages: list, model: str = None,
                             max_tokens: int = None,
                             agent_name: str = "orchestrator") -> str:
        """Route an LLM call to the requested model's backend. Used by
        agents to support per-agent model overrides without forcing every
        caller to manage clients.
        """
        if max_tokens is None:
            max_tokens = self.max_completion_tokens
        if not model:
            model = self.gpt_model

        if (
            self.llm_candidate_budget is not None
            and self.llm_candidate_request_count >= self.llm_candidate_budget
        ):
            raise RuntimeError(
                "llm_candidate_budget_exhausted: used "
                f"{self.llm_candidate_request_count}/{self.llm_candidate_budget}"
            )
        self.llm_candidate_request_count += 1
        if not hasattr(self, "llm_controller_transcript"):
            self.llm_controller_transcript = []
        call_index = self.llm_candidate_request_count - 1
        for message_index, message in enumerate(messages or []):
            self.llm_controller_transcript.append({
                "call_index": call_index,
                "message_index": message_index,
                "agent": agent_name or "orchestrator",
                "model": model,
                "role": message.get("role", "unknown"),
                "content": str(message.get("content") or ""),
            })

        kind, client = self._client_for_model(model)
        configured_decoding = self._configured_decoding()
        if kind == "anthropic":
            system_text = ""
            conv_messages = []
            for message in messages:
                if message["role"] == "system":
                    system_text += message["content"] + "\n"
                else:
                    conv_messages.append({"role": message["role"],
                                          "content": message["content"]})
            anthropic_kwargs = {
                "model": model,
                "max_tokens": max_tokens,
                "system": system_text.strip() if system_text else "",
                "messages": conv_messages,
            }
            for key in ("temperature", "top_p"):
                if configured_decoding[key] is not None:
                    anthropic_kwargs[key] = configured_decoding[key]
            effective_decoding = {
                **configured_decoding,
                "seed": None,
                "seed_supported": False,
            }
            response = client.messages.create(
                **anthropic_kwargs,
            )
            self._record_llm_usage(
                provider="anthropic",
                model=model,
                agent_name=agent_name,
                usage=getattr(response, "usage", None),
                messages=messages,
                max_tokens=max_tokens,
                decoding=effective_decoding,
                candidate_evaluation_index=call_index,
            )
            return response.content[0].text

        kwargs = {"model": model, "messages": messages}
        if _is_hosted_openai_model(model):
            kwargs["max_completion_tokens"] = max_tokens
        else:
            kwargs["max_tokens"] = max_tokens
        for key in ("temperature", "top_p", "seed"):
            if configured_decoding[key] is not None:
                kwargs[key] = configured_decoding[key]
        if "qwen" in model.lower():
            kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
        response = client.chat.completions.create(**kwargs)
        self._record_llm_usage(
            provider="openai",
            model=model,
            agent_name=agent_name,
            usage=getattr(response, "usage", None),
            messages=messages,
            max_tokens=max_tokens,
            decoding={**configured_decoding, "seed_supported": True},
            candidate_evaluation_index=call_index,
        )
        return response.choices[0].message.content

    def _append_history(self, role: str, content: str):
        self.history.append({"role": role, "content": content})

    def _request_code_revision(self, prompt: str) -> Optional[str]:
        self.messages.append({"role": "user", "content": prompt})
        reply = self._call_llm(self.messages)
        self.messages.append({"role": "assistant", "content": reply})
        self._append_history("user", prompt)
        self._append_history("assistant", reply)
        code = extract_cpp_code(reply)
        if not code:
            self._finalize_candidate_evaluation(
                correctness_status="not_run",
                synthesis_status="not_run",
                failure_class="malformed_output",
            )
        return code

    def _preflight_generated_hls_code(self, code: str, context: str) -> str:
        normalized, note = _align_generated_top_signature(
            code,
            self.header_code,
            self.testbench_code,
            self.translated_hls_top,
        )
        normalized, interface_note = _normalize_vitis_s_axilite_bundles(normalized)
        if interface_note:
            note = "; ".join(part for part in (note, interface_note) if part)
        if self.benchmark_name == "srad" and not _reference_blind_enabled():
            normalized, srad_notes = _normalize_srad_halo_copy_offsets(normalized)
            if srad_notes:
                for srad_note in srad_notes:
                    self.preflight_patches.append({
                        "context": context,
                        "kind": "srad_halo_copy_offset",
                        "detail": srad_note,
                        "profile_required": True,
                    })
                note = "; ".join(part for part in (note, "SRAD preflight: " + "; ".join(srad_notes)) if part)
        if note:
            logging.info("[%s] ABI preflight: %s", context, note)
            self._append_history("system", f"[{context}] ABI preflight: {note}")
        return normalized

    def _synth_and_test(self, code: str, log_prefix: str = "") -> dict:
        """Synthesize `code` with the orchestrator's current config and run
        csim/cosim if a testbench is available. Returns the same shape as
        _run_synth_csim_cosim: {synth, csim, cosim}."""
        if (
            self.synthesis_eval_budget is not None
            and self.synthesis_eval_count >= self.synthesis_eval_budget
        ):
            candidate_index = self._pending_candidate_evaluation_index()
            if candidate_index is not None:
                self._finalize_candidate_evaluation(
                    candidate_index,
                    code=code,
                    correctness_status="not_run",
                    synthesis_status="not_run",
                    failure_class="candidate_budget_exhausted",
                )
            return {
                "synth": {
                    "success": False,
                    "report": {},
                    "error": (
                        "synthesis_evaluation_budget_exhausted: "
                        f"used {self.synthesis_eval_count}/{self.synthesis_eval_budget}"
                    ),
                    "budget_exhausted": True,
                },
                "csim": None,
                "cosim": None,
            }
        candidate_index = self._ensure_candidate_evaluation_event(log_prefix)
        event = self.synthesis_eval_events[candidate_index]
        event["label"] = log_prefix or event.get("label") or "generated_candidate"
        event["code_sha256"] = hashlib.sha256((code or "").encode("utf-8")).hexdigest()
        outcome = _run_synth_csim_cosim(
            code,
            header_code=self.header_code,
            header_name=self.header_name,
            top_function=self.translated_hls_top,
            part=self.part,
            clock_ns=self.clock_ns,
            extra_files=self.extra_files,
            testbench_code=self.testbench_code,
            run_csim_check=bool(self.testbench_code),
            run_cosim_check=bool(
                self.testbench_code
                and self.supports_cosim
                and _cosim_required_for_correctness()
                and not _cosim_selected_only()
            ),
            cosim_depths=self.cosim_depths,
            cosim_reference_cycle_info=(
                {} if self.reference_blind else self.cosim_reference_cycle_info
            ),
            golden_output_text=self.independent_golden_output,
            golden_output_specs=self.independent_golden_specs,
            log_prefix=log_prefix,
        )
        synth = outcome.get("synth", {})
        synthesis_ran = synth.get("ran") is not False and not synth.get("skipped")
        if synthesis_ran:
            event["synthesis_index"] = self.synthesis_eval_count
            self.synthesis_eval_count += 1
        event["synthesis_ran"] = synthesis_ran
        event["correctness_gate_passed"] = bool(
            not isinstance(outcome.get("csim"), dict)
            or outcome.get("csim", {}).get("passed")
        )
        event["success"] = bool(synth.get("success"))
        event["error"] = str(synth.get("error") or "")
        event["timed_out"] = bool(
            synth.get("timed_out")
            or "timed out" in event["error"].lower()
        )
        event["tool_failure"] = bool(
            synth.get("tool_failure")
            or str(synth.get("status") or "").lower()
            in {"tool_error", "tool_failure", "error"}
        )
        event["status"] = (
            "timeout"
            if event["timed_out"]
            else "tool_failure"
            if event["tool_failure"]
            else "passed"
            if event["success"]
            else "failed"
            if synthesis_ran
            else "not_run"
        )
        self._finalize_candidate_from_outcome(
            candidate_index,
            code,
            outcome,
            synthesis_ran=synthesis_ran,
        )
        return outcome

    def _synthesis_evaluation_summary(self) -> dict:
        llm_events = getattr(self, "llm_usage_events", []) or []
        candidate_events = getattr(self, "synthesis_eval_events", []) or []
        complete = bool(candidate_events) and (
            len(candidate_events)
            == getattr(self, "llm_candidate_request_count", len(llm_events))
            == len(llm_events)
            and all(
                event.get("_llm_joined") is True
                and event.get("_telemetry_finalized") is True
                and event.get("candidate_evaluation_index") == index
                for index, event in enumerate(candidate_events)
            )
            and all(
                event.get("candidate_evaluation_index") == index
                for index, event in enumerate(llm_events)
            )
        )
        return {
            "schema_version": "c2hls.synthesis-evaluations.v1",
            "count": self.synthesis_eval_count,
            "budget": self.synthesis_eval_budget,
            "budget_exhausted": bool(
                self.synthesis_eval_budget is not None
                and self.synthesis_eval_count >= self.synthesis_eval_budget
            ),
            "events": [
                {
                    key: value
                    for key, value in event.items()
                    if not key.startswith("_")
                }
                for event in candidate_events
            ],
            "candidate_evaluations": len(self.synthesis_eval_events),
            "complete_candidate_event_stream": complete,
        }

    def _run_selected_winner_cosim(self) -> Optional[dict]:
        """Execute RTL cosim exactly once, after the winner is selected."""
        if not (
            _cosim_selected_only()
            and self.hls_code
            and self.testbench_code
            and self.supports_cosim
        ):
            return self.generated_cosim
        selected_hash = hashlib.sha256(self.hls_code.encode("utf-8")).hexdigest()
        self.selected_code_sha256 = selected_hash
        for event in getattr(self, "synthesis_eval_events", []) or []:
            event["selected_for_executed_cosim"] = False
        matching = [
            event
            for event in (getattr(self, "synthesis_eval_events", []) or [])
            if event.get("code_sha256") == selected_hash
            and event.get("correctness_status") == "passed"
            and event.get("synthesis_status") == "passed"
            and event.get("resource_fit") is True
            and event.get("timing_met") is True
            and isinstance(event.get("synthesized_latency_cycles"), int)
            and not isinstance(event.get("synthesized_latency_cycles"), bool)
            and event.get("synthesized_latency_cycles") > 0
        ]
        if matching:
            min(
                matching,
                key=lambda event: (
                    event["synthesized_latency_cycles"],
                    event["candidate_evaluation_index"],
                ),
            )["selected_for_executed_cosim"] = True
        self.selected_winner_cosim_count = (
            getattr(self, "selected_winner_cosim_count", 0) + 1
        )
        self.cosim_target_code_sha256 = selected_hash
        result = run_cosim(
            self.hls_code,
            self.testbench_code,
            self.header_code,
            header_name=self.header_name,
            top_function=self.translated_hls_top,
            part=self.part,
            clock_ns=self.clock_ns,
            extra_files=self.extra_files,
            interface_depths=self.cosim_depths,
            golden_output_text=self.independent_golden_output,
            golden_output_specs=self.independent_golden_specs,
        )
        self.generated_cosim = _summarize_test_result(result, True)
        return self.generated_cosim

    def _total_synthesis_calls(self) -> int:
        return int(getattr(self, "synthesis_eval_count", 0)) + int(
            getattr(self, "selected_winner_cosim_count", 0)
        )

    def _evaluate_candidate_with_repairs(self, candidate_code: str, label: str) -> dict:
        current_code = candidate_code
        last_error = ""
        local_turn_records: list[dict] = []  # for attempt_history feedback

        for turn in range(self.turns_limitation):
            logging.info("%s Candidate attempt %d", label, turn)
            current_code = self._preflight_generated_hls_code(current_code, f"{label} attempt {turn}")

            ok, err = compile_check_cpp(
                current_code,
                self.header_code,
                self.header_name,
                extra_files=self.extra_files,
            )
            if not ok:
                last_error = err
                self._finalize_candidate_evaluation(
                    code=current_code,
                    correctness_status="not_run",
                    synthesis_status="not_run",
                    failure_class="compile_or_interface_failure",
                )
                local_turn_records.append({"turn": turn, "phase": "B",
                                           "success": False, "error": err})
                logging.warning("%s Compile error: %s", label, err[:200])
                fixed = self._request_code_revision(
                    c_compilation_fix.format(
                        compile_error=err,
                        hls_code=current_code,
                        benchmark_context=self.benchmark_context,
                        repair_guidance=_build_repair_guidance(err),
                        attempt_history=_format_attempt_history(local_turn_records, "B"),
                    )
                )
                if fixed:
                    current_code = fixed
                continue

            outcome = self._synth_and_test(current_code, log_prefix=label)
            result = outcome["synth"]
            if result["success"]:
                return {
                    "success": True,
                    "code": current_code,
                    "report": result["report"],
                    "csim": outcome["csim"],
                    "cosim": outcome["cosim"],
                }

            if result.get("budget_exhausted"):
                return {
                    "success": False,
                    "code": current_code,
                    "error": result["error"],
                    "budget_exhausted": True,
                }

            last_error = result["error"]
            local_turn_records.append({"turn": turn, "phase": "B",
                                       "success": False, "error": result["error"]})
            logging.warning("%s Synthesis failed: %s", label, result["error"][:300])
            is_timeout = "timed out" in result["error"].lower()
            history_block = _format_attempt_history(local_turn_records, "B")
            if is_timeout:
                fix_prompt = hls_synthesis_timeout_fix.format(
                    timeout=600,
                    hls_code=current_code,
                    header_code=self.header_code,
                    benchmark_context=self.benchmark_context,
                    repair_guidance=_build_repair_guidance(result["error"]),
                    attempt_history=history_block,
                )
            else:
                fix_prompt = hls_synthesis_fix.format(
                    synth_error=result["error"],
                    hls_code=current_code,
                    header_code=self.header_code,
                    target_context=_target_context_for_prompt(self.part, self.clock_ns),
                    benchmark_context=self.benchmark_context,
                    repair_guidance=_build_repair_guidance(result["error"]),
                    attempt_history=history_block,
                )
            fixed = self._request_code_revision(fix_prompt)
            if fixed:
                current_code = fixed

        return {
            "success": False,
            "code": current_code,
            "error": last_error or f"{label} failed after {self.turns_limitation} attempts",
        }

    def run_quality_repair(self, ground_truth_report: dict,
                           initial_comparison: Optional[dict] = None) -> dict:
        """Delegate to QualityRepairAgent. Logic lives there; this method
        keeps the existing public API for callers like run() and
        run_benchmark()."""
        return self.quality_repair.run(ground_truth_report, initial_comparison)

    def run_phase_a(self, c_code: str, header_code: str = "",
                    header_name: str = "kernel.h") -> bool:
        """Delegate to TranslatorAgent."""
        return self.translator.run_phase_a(c_code, header_code, header_name)

    def run_phase_b(self, *, multistep: bool = False) -> bool:
        """Delegate to TranslatorAgent for the initial translate, then to
        SynthesisAgent for the synth+repair loop. Splitting these two lets
        future RL work plug in different policies for "produce code" vs
        "fix a synth failure" without restructuring the full pipeline."""
        prev_context = getattr(self, "_phaseb_multistep_context", False)
        self._phaseb_multistep_context = bool(multistep)
        self.phaseb_mode = _normalize_phaseb_mode(
            os.getenv(PHASEB_MODE_ENV, ""),
            multistep=bool(multistep),
        )
        try:
            hls_code = self.translator.translate_initial()
            if not hls_code:
                return False
            self.hls_code = hls_code
            return self.synthesis.synthesize_with_repair()
        finally:
            self._phaseb_multistep_context = prev_context

    def _baseline_alignment_loop(
        self,
        reference_report: Optional[dict],
        *,
        max_attempts: Optional[int] = None,
    ) -> dict:
        """Phase 8: ensure our Phase B baseline is competitive with the
        reference baseline before optimization steps run. Re-translate
        with **metric-only** feedback (no gold-code leak) when the gap
        is significant. Returns a record with the loop outcome.

        Opt-in via ``C2HLS_PHASE8_BASELINE_ALIGN=1``. Tunable thresholds:
            ``C2HLS_PHASE8_BASELINE_LATENCY_TOL``  (default 1.20)
            ``C2HLS_PHASE8_BASELINE_RESOURCE_TOL`` (default 2.00)
            ``C2HLS_PHASE8_MAX_ATTEMPTS``          (default 3)
        """
        outcome: dict = {
            "enabled": False,
            "attempts": 0,
            "history": [],
            "aligned": False,
            "skipped": None,
        }
        if not bool(int(os.getenv("C2HLS_PHASE8_BASELINE_ALIGN", "0") or "0")):
            return outcome
        outcome["enabled"] = True
        if not reference_report:
            outcome["skipped"] = "no reference report"
            return outcome
        if not self.synth_report:
            outcome["skipped"] = "no Phase B synth report"
            return outcome

        if max_attempts is None:
            try:
                max_attempts = int(os.getenv("C2HLS_PHASE8_MAX_ATTEMPTS", "3"))
            except ValueError:
                max_attempts = 3
        try:
            lat_tol = float(os.getenv("C2HLS_PHASE8_BASELINE_LATENCY_TOL", "1.20"))
        except ValueError:
            lat_tol = 1.20
        try:
            res_tol = float(os.getenv("C2HLS_PHASE8_BASELINE_RESOURCE_TOL", "2.00"))
        except ValueError:
            res_tol = 2.00

        for attempt in range(max_attempts):
            gap = _compute_baseline_gap(
                self.synth_report, reference_report,
                latency_tolerance=lat_tol, resource_tolerance=res_tol,
            )
            outcome["history"].append({
                "attempt": attempt,
                "latency_ratio": gap.get("latency_ratio"),
                "resource_ratios": gap.get("resource_ratios"),
                "within_tolerance": gap.get("within_tolerance"),
            })
            if gap.get("within_tolerance"):
                outcome["aligned"] = True
                outcome["attempts"] = attempt
                logging.info(
                    "[Phase 8] Baseline aligned at attempt %d: lat_ratio=%s "
                    "(limit %.2f), resource ratios %s",
                    attempt, gap.get("latency_ratio"), lat_tol,
                    {k: round(v, 3) for k, v in (gap.get("resource_ratios") or {}).items()},
                )
                self._append_history(
                    "system",
                    f"[Phase 8] Baseline aligned within tolerance at attempt {attempt}.",
                )
                return outcome

            logging.info(
                "[Phase 8] Baseline misaligned (attempt %d): lat_ratio=%.3f "
                "(limit %.2f), resource over-thresholds: %s",
                attempt, gap.get("latency_ratio") or 0.0, lat_tol,
                [k for k, *_ in (gap.get("over_resources") or [])],
            )

            # Build metric-only retranslation feedback.
            guidance = _render_baseline_alignment_guidance(gap, attempt=attempt)
            if not guidance:
                # within_tolerance was false but no guidance — likely
                # missing reference fields. Bail.
                outcome["skipped"] = "no actionable guidance"
                outcome["attempts"] = attempt
                return outcome

            self._append_history(
                "system",
                f"[Phase 8] Re-translating (attempt {attempt + 1}/{max_attempts}) "
                f"with baseline-alignment guidance.",
            )
            new_code = self.translator.retranslate_with_guidance(
                guidance, attempt=attempt + 1,
            )
            if not new_code:
                outcome["skipped"] = "translator returned no code"
                outcome["attempts"] = attempt + 1
                return outcome
            self.hls_code = new_code

            if not self.synthesis.synthesize_with_repair():
                outcome["skipped"] = "synthesis failed after retranslation"
                outcome["attempts"] = attempt + 1
                return outcome

        # Exhausted attempts without alignment.
        gap = _compute_baseline_gap(
            self.synth_report, reference_report,
            latency_tolerance=lat_tol, resource_tolerance=res_tol,
        )
        outcome["attempts"] = max_attempts
        outcome["aligned"] = bool(gap.get("within_tolerance"))
        outcome["final_gap"] = {
            "latency_ratio": gap.get("latency_ratio"),
            "resource_ratios": gap.get("resource_ratios"),
        }
        return outcome

    def run_phase_c(self, ground_truth_report: dict) -> dict:
        logging.info("=== [Phase C] Offline reference comparison ===")

        if not self.synth_report:
            logging.error("[Phase C] No synthesis report from Phase B")
            return {"success": False, "error": "No synthesis report", "invalid_reference": False}

        if not ground_truth_report:
            logging.error("[Phase C] Missing validated ground-truth report")
            return {
                "success": False,
                "error": "Missing validated ground-truth report",
                "invalid_reference": True,
            }

        comparison = compare_reports(self.synth_report, ground_truth_report)
        logging.info("[Phase C] Reference report is kept offline from LLM prompts.")
        logging.info("[Phase C] Ratio comparison:")
        for metric, vals in comparison.items():
            if isinstance(vals, dict) and vals.get("ratio") is not None:
                logging.info(
                    "  %s: ratio=%.3f",
                    metric,
                    vals["ratio"],
                )

        self._append_history(
            "system",
            "[Phase C] Offline reference ratios recorded for controller scoring only: "
            f"{json.dumps(_reference_ratio_summary(comparison), sort_keys=True)}",
        )

        return {
            "success": True,
            "valid_reference": True,
            "invalid_reference": False,
            "generated_report": self.synth_report,
            "ground_truth_report": ground_truth_report,
            "comparison": comparison,
        }

    def _previous_gt_report_for_step(self, step_name: str) -> Optional[dict]:
        """Walk the canonical optimization order backwards from
        ``step_name`` to find the most recent populated GT report. Used
        by the trajectory-alignment check to compare gen vs GT step deltas
        on the SAME step boundary."""
        order = list(DEFAULT_OPT_STEPS) + ["combo_full",
                                            "combo_structural",
                                            "combo_parallel"]
        if step_name not in order:
            return self._gt_baseline_report or None
        idx = order.index(step_name)
        for prev in reversed(order[:idx]):
            r = self._gt_step_reports.get(prev)
            if r:
                return r
        return self._gt_baseline_report or None

    def run_optimization_step(self, step_name: str, gt_code: str = None,
                               gt_header_code: str = None,
                               skill_id: Optional[str] = None) -> dict:
        """Run one optimization step with a regression guard.

        Outer logic: try the step (LLM + synth + repair). If it succeeds but
        regresses against the previous step's metrics (`_step_regression_reasons`
        non-empty at threshold STEP_REGRESSION_THRESHOLD), retry once with an
        explicit "you regressed, here's how" prompt. If the retry still
        regresses, revert: keep the previous step's code and report and mark
        this step as failed-by-regression. This stops a bad LLM step from
        poisoning the rest of the multistep chain.
        """
        logging.info("=== [Step: %s] Applying optimization ===", step_name)

        if not self.hls_code:
            return {"success": False, "step_name": step_name,
                    "error": "No HLS code to optimize"}

        # Snapshot for revert-on-regression. self.hls_code / self.synth_report
        # represent the previous step's accepted output before this step runs.
        prev_code = self.hls_code
        prev_report = self.synth_report
        prev_csim = self.generated_csim
        prev_cosim = self.generated_cosim
        threshold = STEP_REGRESSION_THRESHOLD

        # Phase-5a: when C2HLS_PHASE5_LLM_RETRY=1, after the deterministic
        # regression template fails once, fire one more attempt with
        # FeedbackAgent.compose_with_llm() — which reads the LLM's actual
        # last edit + the bottleneck record and returns a strategic,
        # kernel-aware retry prompt. This converts what would otherwise be
        # a reverted step into one more chance with surgical guidance.
        # Off by default to preserve cost; turn on for high-value runs.
        phase5_llm_retry = bool(int(os.getenv("C2HLS_PHASE5_LLM_RETRY", "0") or "0"))

        # Up to 2 outer attempts (or 3 with phase5 LLM-aided retry on):
        #   turn 0  clean attempt
        #   turn 1  deterministic-template re-prompt
        #   turn 2  LLM-aided composition (only if phase5_llm_retry=1)
        # then revert. LLMs that miss thrice rarely converge.
        max_outer_turns = 3 if phase5_llm_retry else 2
        regression_guidance = ""
        last_llm_edit_code = ""  # populated for compose_with_llm to read
        for outer_turn in range(max_outer_turns):
            attempt = self._optimization_step_attempt(
                step_name, gt_code,
                additional_guidance=regression_guidance,
                gt_header_code=gt_header_code,
                skill_id=skill_id,
            )
            if not attempt.get("success"):
                # Synth itself failed (compile/synth budget exhausted). Don't
                # retry-with-regression-guidance; just return the failure.
                return attempt

            # Synth succeeded — but may have regressed OR be a no-op.
            new_report = attempt["report"]
            new_code = attempt["code"]

            # Pillar 9 (MVP): no-op-trap check first. If the new variant
            # produced byte-identical synthesis numbers, re-prompt with that
            # specific feedback before any regression check (a no-op is by
            # definition not a regression). On the second consecutive no-op,
            # mark the step as failed-by-no-op so the trajectory is honest.
            no_op_reasons = _step_no_op_reasons(new_report, prev_report)
            if no_op_reasons:
                logging.warning(
                    "[Step: %s] No-op detected on attempt %d: %s",
                    step_name, outer_turn, no_op_reasons[-1],
                )
                if outer_turn == 0:
                    regression_guidance = self.feedback.render(
                        "no_op", step_name=step_name, reasons=no_op_reasons,
                    )
                    self._append_history(
                        "system",
                        f"[Step: {step_name}] No-op on attempt 0; retrying "
                        f"with no-op-aware guidance.",
                    )
                    continue
                # outer_turn == 1: still a no-op. Don't pretend the step
                # succeeded — keep prev code/report and surface the failure
                # so downstream (dataset_pipeline) can record `no_op` cleanly.
                self.hls_code = prev_code
                self.synth_report = prev_report
                self.generated_csim = prev_csim
                self.generated_cosim = prev_cosim
                self._append_history(
                    "system",
                    f"[Step: {step_name}] Reverted: no-op persisted after retry.",
                )
                return {
                    "success": False,
                    "step_name": step_name,
                    "error": "no_op_persisted",
                    "no_op_reasons": no_op_reasons,
                    "rejected_report": new_report,
                    "reverted_to_prev": True,
                }

            # Phase-5 follow-up: pass step_name so the per-step threshold
            # lookup (STEP_REGRESSION_THRESHOLDS) can apply step-aware
            # tolerance (e.g. unroll allowed 8x DSP, coalescing 5x BRAM).
            # When C2HLS_STEP_REGRESSION_THRESHOLD env is set as a single
            # number it still overrides everything (legacy behavior).
            reasons = _step_regression_reasons(
                new_report, prev_report, threshold,
                step_name=step_name,
                part=self.part,
            )

            if not reasons:
                # Accept: commit and return.
                self.hls_code = new_code
                self.synth_report = new_report
                self.generated_csim = attempt.get("csim")
                self.generated_cosim = attempt.get("cosim")
                if outer_turn > 0:
                    attempt["regression_retry_succeeded"] = True
                return attempt

            # Regression detected.
            logging.warning(
                "[Step: %s] Regression detected on attempt %d: %s",
                step_name, outer_turn, "; ".join(reasons),
            )
            # Capture the failing edit so a subsequent compose_with_llm()
            # call can read it for strategic, code-specific feedback.
            last_llm_edit_code = new_code

            if outer_turn == 0:
                # Re-prompt with the specific regression info; loop.
                regression_guidance = self.feedback.render(
                    "regression", step_name=step_name, reasons=reasons,
                )
                self._append_history(
                    "system",
                    f"[Step: {step_name}] Regression on attempt 0; retrying with "
                    f"regression-aware guidance.",
                )
                continue

            # outer_turn == 1 AND phase5_llm_retry → one more attempt with
            # LLM-aided composition. compose_with_llm reads the failing
            # edit + bottleneck record and emits surgical guidance.
            if outer_turn == 1 and phase5_llm_retry and outer_turn < max_outer_turns - 1:
                bottleneck_record = {
                    "kind": "regression",
                    "step": step_name,
                    "regression_reasons": reasons,
                    "metrics_delta": {
                        k: {"prev": (prev_report or {}).get(k),
                            "new":  new_report.get(k)}
                        for k in ("latency_cycles", "latency_ns", "interval",
                                  "bram", "dsp", "ff", "lut", "fmax_mhz")
                    },
                }
                static_extras = ((new_report.get("feedback") or {}).get("static_extras") or {})
                if static_extras:
                    try:
                        from hls_feedback import render_static_extras_for_prompt
                        bottleneck_record["static_extras_summary"] = static_extras.get("summary")
                        bottleneck_record["static_extras_prompt"] = render_static_extras_for_prompt(static_extras)
                    except Exception as exc:  # pragma: no cover
                        logging.warning("static_extras render failed: %s", exc)
                deterministic = self.feedback.render(
                    "regression", step_name=step_name, reasons=reasons,
                )
                regression_guidance = self.feedback.compose_with_llm(
                    "regression",
                    kernel_diff=last_llm_edit_code[:6000],
                    prior_template=deterministic,
                    bottleneck_record=bottleneck_record,
                    step_name=step_name,
                    reasons=reasons,
                )
                self._append_history(
                    "system",
                    f"[Step: {step_name}] Regression persisted on attempt 1; "
                    f"retrying with LLM-aided composition (Phase 5a).",
                )
                attempt["llm_aided_retry_used"] = True
                continue

            # Final outer_turn (1 in legacy 2-turn path, 2 with Phase 5a's
            # LLM-aided 3-turn path): still regressed after all retries.
            # Before reverting, consult the GT trajectory if we have it:
            # this step might be a structural enabler (the canonical
            # reference also regresses here, e.g. tiling alone is +4x
            # latency on knn — required prerequisite for
            # doublebuffer/coalescing wins). When GT shape and gen shape
            # match within tolerance, KEEP the step instead of reverting
            # (Pillar 5b).
            if self.gt_aware_revert:
                from trajectory_alignment import (
                    is_consistent_with_gt_trajectory,
                    render_alignment_for_history,
                )
                gt_step_report = self._gt_step_reports.get(step_name)
                # Parent GT report comes from whichever step preceded this
                # one in the canonical trajectory. We walk the cache in
                # order to find the most recent populated parent.
                gt_parent_report = self._previous_gt_report_for_step(step_name)
                alignment = is_consistent_with_gt_trajectory(
                    gen_report=new_report,
                    parent_gen_report=prev_report,
                    gt_report=gt_step_report,
                    parent_gt_report=gt_parent_report,
                )
                self._append_history(
                    "system",
                    render_alignment_for_history(alignment, step_name),
                )
                if alignment.consistent_with_gt:
                    # KEEP — accept the regression as a structural enabler.
                    logging.info(
                        "[Step: %s] Keeping enabling regression (consistent with GT shape: %s)",
                        step_name, alignment.reason,
                    )
                    self.hls_code = new_code
                    self.synth_report = new_report
                    self.generated_csim = attempt.get("csim")
                    self.generated_cosim = attempt.get("cosim")
                    attempt["alignment_decision"] = {
                        "consistent_with_gt": True,
                        "reason": alignment.reason,
                        "gen_latency_ratio": alignment.gen_latency_ratio,
                        "gt_latency_ratio": alignment.gt_latency_ratio,
                    }
                    attempt["enabling_regress_kept"] = True
                    return attempt

            logging.warning(
                "[Step: %s] Reverting to previous step's code after regression retry",
                step_name,
            )
            self.hls_code = prev_code
            self.synth_report = prev_report
            self.generated_csim = prev_csim
            self.generated_cosim = prev_cosim
            self._append_history(
                "system",
                f"[Step: {step_name}] Reverted: regression persisted after retry.",
            )
            return {
                "success": False,
                "step_name": step_name,
                "error": "Reverted: regression after retry",
                "regression_reasons": reasons,
                "rejected_report": new_report,
                "reverted_to_prev": True,
            }

        # Unreachable; satisfies static analysis.
        return {"success": False, "step_name": step_name,
                "error": "Optimization step exited unexpectedly"}

    def _optimization_step_attempt(self, step_name: str, gt_code: str = None,
                                   additional_guidance: str = "",
                                   gt_header_code: str = None,
                                   skill_id: Optional[str] = None) -> dict:
        """Run one or more independent candidates for an optimization step.

        Candidate count is controlled by C2HLS_CANDIDATES_PER_STEP. This is
        bounded search, not RL: every candidate goes through the existing
        synth/correctness-repair path, and the best successful candidate by
        latency/resource score is returned to the existing regression guard.
        """
        count = _step_candidate_count(step_name)
        exhaustive = _exhaustive_candidate_attempts_enabled()
        attempt_count = _candidate_attempt_count(self.turns_limitation) if exhaustive else 1
        if count <= 1:
            return self._optimization_step_attempt_single(
                step_name,
                gt_code,
                additional_guidance=additional_guidance,
                gt_header_code=gt_header_code,
                skill_id=skill_id,
                candidate_index=0,
                candidate_count=1,
            )

        attempts: list = []
        for candidate_index in range(count):
            attempt = self._optimization_step_attempt_single(
                step_name,
                gt_code,
                additional_guidance=additional_guidance,
                gt_header_code=gt_header_code,
                skill_id=skill_id,
                candidate_index=candidate_index,
                candidate_count=count,
            )
            attempt["candidate_index"] = candidate_index
            attempt["candidate_count"] = count
            attempts.append(attempt)
            if attempt.get("budget_exhausted"):
                break

        synthesized_successes = [
            a for a in attempts
            if (a.get("success") or a.get("synthesis_success")) and a.get("report")
        ]
        if _feasibility_selection_enabled():
            successes = [
                a for a in synthesized_successes
                if (a.get("feasibility") or {}).get("feasible")
            ]
        else:
            successes = synthesized_successes
        candidate_search = {
            "candidate_count": count,
            "attempts_per_candidate": attempt_count,
            "exhaustive_attempts": exhaustive,
            "synthesis_successful_candidates": len(synthesized_successes),
            "feasible_candidates": len(successes),
            "successful_candidates": len(successes),
            "candidate_stats": _metric_stats_from_reports(
                [a.get("report") for a in successes if a.get("report")]
            ),
            "all_attempt_stats": _metric_stats_from_reports([
                entry.get("report")
                for attempt in attempts
                for entry in (attempt.get("attempt_results") or [])
                if entry.get("success") and entry.get("report")
            ]),
        }
        if not successes:
            chosen = attempts[-1] if attempts else {
                "success": False,
                "step_name": step_name,
                "error": "no candidates attempted",
            }
            if synthesized_successes and _feasibility_selection_enabled():
                best_infeasible = min(
                    synthesized_successes,
                    key=lambda a: self._best_so_far_score(a.get("report") or {}),
                )
                chosen = dict(best_infeasible)
                chosen.update({
                    "success": False,
                    "synthesis_success": True,
                    "feasibility_rejected": True,
                    "error": "no_feasible_candidates",
                })
            chosen["candidate_attempts"] = [
                _compact_attempt_record(a) for a in attempts
            ]
            chosen["candidate_search"] = candidate_search
            return chosen

        chosen = min(successes, key=lambda a: self._best_so_far_score(a.get("report") or {}))
        chosen["candidate_selected"] = True
        chosen["selected_candidate_index"] = chosen.get("candidate_index")
        chosen["candidate_attempts"] = [
            _compact_attempt_record(a) for a in attempts
        ]
        candidate_search["selected_candidate_index"] = chosen.get("candidate_index")
        candidate_search["selected_attempt_index"] = chosen.get("selected_attempt_index")
        chosen["candidate_search"] = candidate_search
        self._append_history(
            "system",
            f"[Step: {step_name}] selected candidate "
            f"{chosen.get('candidate_index')} of {count}.",
        )
        return chosen

    def _optimization_step_attempt_single(self, step_name: str, gt_code: str = None,
                                          additional_guidance: str = "",
                                          gt_header_code: str = None,
                                          skill_id: Optional[str] = None,
                                          candidate_index: int = 0,
                                          candidate_count: int = 1) -> dict:
        """One optimization-step pass: LLM → synth + repair loop. Returns a
        step_result dict with success / code / report / vs_previous / vs_ground_truth.

        Does NOT commit self.hls_code / self.synth_report — the outer
        run_optimization_step decides whether to accept based on the
        regression check. This deferred-commit design is what makes the
        regression guard possible.
        """
        exhaustive = _exhaustive_candidate_attempts_enabled()
        attempt_limit = (
            _candidate_attempt_count(self.turns_limitation)
            if exhaustive else self.turns_limitation
        )
        report_str = (
            format_report_summary(self.synth_report)
            if self.synth_report else "(no prior report)"
        )
        prompt_template = OPTIMIZATION_PROMPTS.get(step_name)
        if prompt_template is None:
            prompt_template = q_optimize_generic
            prompt = prompt_template.format(
                optimization_name=step_name,
                optimization_description=f"Apply {step_name} optimization.",
                synth_report=report_str,
                header_code=self.header_code,
                current_code=self.hls_code,
            )
        else:
            prompt = prompt_template.format(
                synth_report=report_str,
                header_code=self.header_code,
                current_code=self.hls_code,
            )

        # Inject profile-signal hints + (if retrying) regression guidance into
        # the prompt as an additional section. Keeps prompt templates
        # untouched while letting the LLM see the actual bottlenecks.
        signal = _build_profile_signal(
            self.synth_report or {}, part=self.part,
            requested_clock_ns=self.clock_ns,
        )
        extra_blocks = []
        skill_prompt_mode = os.getenv("C2HLS_SKILL_PROMPT_MODE", "").strip().lower()
        skill_prompt_scope = os.getenv("C2HLS_SKILL_PROMPT_SCOPE", "").strip().lower()
        all_positive_skill_prompt = skill_prompt_scope in {
            "all_positive",
            "all-positive",
            "positive_all",
            "positive-all",
        }
        render_skill_prompt_mode = "action_only" if all_positive_skill_prompt else (skill_prompt_mode or None)
        action_only_skill_prompt = all_positive_skill_prompt or skill_prompt_mode in {
            "action_only",
            "action-only",
            "positive",
            "positive_only",
            "positive-only",
        }
        skill_prompt_injection_enabled = _skill_prompt_injection_enabled()
        skill_prompt_meta = {
            "enabled": self.skill_library is not None and skill_prompt_injection_enabled,
            "requested_skill_id": skill_id,
            "prompt_mode": render_skill_prompt_mode or "default",
            "prompt_scope": skill_prompt_scope or "matched",
            "avoid_skills_suppressed": False,
            "query_bottleneck_kind": None,
            "matched_skill_ids": [],
            "matched_skill_count": 0,
            "avoid_skill_ids": [],
            "injected_skill_ids": [],
            "injected": False,
            "reason": (
                "skill_library_not_loaded"
                if self.skill_library is None
                else "disabled_by_env"
                if not skill_prompt_injection_enabled
                else "no_matching_skills"
            ),
        }
        store_meta = (self.skill_library_provenance or {}).get("store") or {}
        if store_meta:
            skill_prompt_meta["library_schema"] = store_meta.get("schema")
            skill_prompt_meta["library_sha256"] = store_meta.get("sha256")
            skill_prompt_meta["library_skill_count"] = store_meta.get("skill_count")
        if candidate_count > 1:
            extra_blocks.append(
                f"CANDIDATE SEARCH: this is candidate {candidate_index + 1} "
                f"of {candidate_count} for `{step_name}`. Produce a distinct, "
                "correct implementation of the requested step. Do not copy a "
                "previous candidate unless it is clearly the only safe option."
            )
        if exhaustive:
            extra_blocks.append(
                f"EXHAUSTIVE ATTEMPT MODE: this candidate will be evaluated for "
                f"{attempt_limit} attempts. Each attempt must be a complete, "
                "synthesizable design variant. Later attempts should use the "
                "reported feedback from earlier attempts to improve latency, "
                "Fmax, or resource balance while preserving correctness."
            )
        if signal:
            extra_blocks.append(signal)

        # Pillar 1: inject baseline-vs-current per-loop scope diff so the LLM
        # understands which loops regressed and which new bottlenecks were
        # introduced by the intermediate steps. Only injected when the baseline
        # report has scope data and we are past the first optimization step.
        if (self._baseline_report
                and step_name != "baseline"
                and self.synth_report
                and self.synth_report is not self._baseline_report):
            scope_diff = _render_baseline_scope_diff(
                self._baseline_report, self.synth_report, step_name
            )
            if scope_diff:
                extra_blocks.append(scope_diff)

        # Inject per-step resource constraints so the LLM knows its budget
        # before generating code. Prevents aggressive over-parallelization
        # (e.g. 30× DSP growth to fix II=144) that the regression guard would
        # catch and revert, wasting a synthesis run and degrading subsequent
        # steps. Always injected when we have a prior report to measure from.
        if self.synth_report and step_name:
            constraints = _render_step_resource_constraints(
                step_name, self.synth_report, part=self.part
            )
            if constraints:
                extra_blocks.append(constraints)

        # Phase 5b: when a skill library is loaded (dynamic-routing path) and
        # the bottleneck-router has matched a skill for this step's bottleneck
        # kind, inject the skill's pattern / strategy / template into the
        # prompt so the LLM sees the proven recipe — not just the bare step
        # name. Dynamic routing may still load the library while an explicit
        # C2HLS_FORCE_SKILL_PROMPTS=0 keeps prompt injection disabled.
        if (
            skill_prompt_injection_enabled
            and self.skill_library is not None
            and self.synth_report is not None
        ):
            try:
                from skill_library import TIER_AVOID, render_skill_set_for_prompt
                matching = []
                avoid_matching = []
                context_matching = []
                top_bottleneck_kind = None
                feedback = (self.synth_report or {}).get("feedback") or {}
                bns = feedback.get("bottlenecks") or []
                if bns:
                    top_bottleneck_kind = bns[0].get("kind")
                if all_positive_skill_prompt:
                    matching = [
                        sk for sk in self.skill_library.all()
                        if getattr(sk, "confidence", "") != TIER_AVOID
                    ]
                    skill_prompt_meta["reason"] = "all_positive_requested"
                    skill_prompt_meta["avoid_skills_suppressed"] = True
                else:
                    selected_skill = self.skill_library.get(skill_id) if skill_id else None
                    if selected_skill:
                        matching = [selected_skill]
                        skill_prompt_meta["reason"] = "requested_skill_id_matched"
                    else:
                        if top_bottleneck_kind:
                            matching = self.skill_library.query(
                                bottleneck_kind=top_bottleneck_kind,
                                vitis_version=self.vitis_version,
                                fpga=self.part,
                            )
                        else:
                            skill_prompt_meta["reason"] = "no_top_bottleneck_kind"
                    in_place_neighbor = _detect_in_place_neighbor_update(self.hls_code)
                    if in_place_neighbor.get("detected"):
                        stencil_skill = self.skill_library.get("hls-inplace-stencil-true-dependence")
                        if stencil_skill is not None:
                            context_matching = [stencil_skill]
                        skill_prompt_meta["context_match"] = {
                            "kind": "in_place_neighbor_update",
                            **in_place_neighbor,
                        }
                        skill_prompt_meta["context_guardrail_active"] = True
                skill_prompt_meta["query_bottleneck_kind"] = top_bottleneck_kind
                if context_matching:
                    seen_skill_ids = {getattr(sk, "id", "") for sk in context_matching}
                    support_ids = {"hls-pipeline-realistic-ii-selection"}
                    matching = context_matching + [
                        sk for sk in matching
                        if getattr(sk, "id", "") in support_ids
                        and getattr(sk, "id", "") not in seen_skill_ids
                    ]
                    skill_prompt_meta["context_guardrail_filtered_generic_skills"] = True
                skill_prompt_meta["matched_skill_ids"] = [getattr(sk, "id", "") for sk in matching]
                skill_prompt_meta["matched_skill_count"] = len(matching)
                if matching:
                    if top_bottleneck_kind and not action_only_skill_prompt:
                        avoid_matching = [
                            sk for sk in self.skill_library.query(
                                bottleneck_kind=top_bottleneck_kind,
                                vitis_version=self.vitis_version,
                                fpga=self.part,
                                include_avoid=True,
                            )
                            if getattr(sk, "confidence", "") == "avoid"
                        ][:2]
                    elif top_bottleneck_kind and action_only_skill_prompt:
                        skill_prompt_meta["avoid_skills_suppressed"] = True
                    skill_prompt_meta["avoid_skill_ids"] = [getattr(sk, "id", "") for sk in avoid_matching]
                    prompt_skills = list(matching) + avoid_matching if all_positive_skill_prompt else list(matching)[:3] + avoid_matching
                    skill_block = render_skill_set_for_prompt(
                        prompt_skills,
                        max_skills=len(prompt_skills) if all_positive_skill_prompt else 4,
                        prompt_mode=render_skill_prompt_mode,
                    )
                    if skill_block and "No matching skills" not in skill_block:
                        skill_prompt_meta["injected"] = True
                        skill_prompt_meta["injected_skill_ids"] = [
                            getattr(sk, "id", "") for sk in prompt_skills
                        ]
                        skill_prompt_meta["reason"] = "injected"
                        prompt_shape = (
                            "pattern → strategy → required steps → template/example"
                            if action_only_skill_prompt else
                            "pattern → strategy → required steps → guardrails → template/example"
                        )
                        if all_positive_skill_prompt:
                            extra_blocks.append(
                                f"ALL POSITIVE SKILLS from library ({prompt_shape}). "
                                "Use the applicable positive recipes as a menu for this "
                                f"`{step_name}` rewrite; no avoid-tier skills or guard "
                                "sections are included:\n\n" + skill_block
                            )
                        else:
                            extra_blocks.append(
                                f"RELEVANT SKILLS from library ({prompt_shape}). "
                                "Apply the highest-confidence one that "
                                f"addresses the bottleneck/route '{top_bottleneck_kind or skill_id}' "
                                f"on the `{step_name}` step:\n\n" + skill_block
                            )
            except Exception as exc:  # pragma: no cover - skill injection best-effort
                skill_prompt_meta["reason"] = "skill_injection_error"
                skill_prompt_meta["error"] = str(exc)
                logging.warning("Phase 5b skill-template injection failed: %s", exc)

        if additional_guidance:
            extra_blocks.append(additional_guidance)
        if extra_blocks:
            prompt = prompt + "\n\n" + "\n\n".join(extra_blocks)

        self.messages = [
            {"role": "system", "content": Instruction_c2hls_multistep},
            {"role": "user", "content": prompt},
        ]

        reply = self._call_llm(self.messages)
        self._append_history("user", f"[Step: {step_name}]\n{prompt}")
        self._append_history("assistant", reply)
        self.messages.append({"role": "assistant", "content": reply})

        new_code = extract_cpp_code(reply)
        if not new_code:
            self._finalize_candidate_evaluation(
                correctness_status="not_run",
                synthesis_status="not_run",
                failure_class="malformed_output",
            )
            logging.error("[Step: %s] No code in LLM response", step_name)
            return {"success": False, "step_name": step_name,
                    "error": "No code in response",
                    "skill_prompt": skill_prompt_meta,
                    "candidate_index": candidate_index,
                    "candidate_count": candidate_count,
                    "attempt_count": attempt_limit,
                    "attempt_results": []}

        step_turn_records: list[dict] = []  # per-step attempt history
        attempt_results: list[dict] = []
        successful_attempts: list[dict] = []

        for turn in range(attempt_limit):
            logging.info("[Step: %s] Synthesis attempt %d", step_name, turn)
            new_code = self._preflight_generated_hls_code(
                new_code, f"Step {step_name} attempt {turn}",
            )

            guardrail_issues = _lint_in_place_stencil_guardrails(self.hls_code, new_code)
            if guardrail_issues:
                err = "\n".join(guardrail_issues)
                self._finalize_candidate_evaluation(
                    code=new_code,
                    correctness_status="not_run",
                    synthesis_status="not_run",
                    failure_class="compile_or_interface_failure",
                )
                attempt_results.append({
                    "attempt_index": turn,
                    "candidate_index": candidate_index,
                    "candidate_count": candidate_count,
                    "success": False,
                    "stage": "guardrail_lint",
                    "error": err,
                    "skill_prompt": skill_prompt_meta,
                })
                step_turn_records.append({
                    "turn": turn,
                    "phase": "B",
                    "success": False,
                    "error": err,
                })
                logging.warning("[Step: %s] Guardrail lint failed: %s", step_name, err[:300])
                fix_prompt = (
                    "GUARDRAIL LINT FAILED before synthesis.\n\n"
                    f"{err}\n\n"
                    "Revise the complete HLS C++ code. Preserve the public "
                    "function signature and correctness. For an in-place "
                    "neighbor-update/stencil pattern, do not suppress true "
                    "dependencies with `DEPENDENCE ... false`, and do not "
                    "convert the algorithm to a separate-input/output update "
                    "unless you can preserve the original update order. If no "
                    "safe optimization is available, return the previous code "
                    "with only legal interface/metadata pragmas and a short "
                    "comment explaining the true-dependence limit.\n\n"
                    "Current rejected code:\n"
                    "```cpp\n"
                    f"{new_code[:9000]}\n"
                    "```\n\n"
                    "Return only the complete revised C++ code in a fenced cpp block."
                )
                self.messages.append({"role": "user", "content": fix_prompt})
                self._append_history("user", f"[Step: {step_name} guardrail repair]\n{fix_prompt}")
                reply = self._call_llm(self.messages)
                self.messages.append({"role": "assistant", "content": reply})
                self._append_history("assistant", reply)
                fixed = extract_cpp_code(reply)
                if fixed:
                    new_code = fixed
                else:
                    self._finalize_candidate_evaluation(
                        correctness_status="not_run",
                        synthesis_status="not_run",
                        failure_class="malformed_output",
                    )
                continue

            ok, err = compile_check_cpp(
                new_code, self.header_code, self.header_name,
                extra_files=self.extra_files,
            )
            if not ok:
                self._finalize_candidate_evaluation(
                    code=new_code,
                    correctness_status="not_run",
                    synthesis_status="not_run",
                    failure_class="compile_or_interface_failure",
                )
                attempt_results.append({
                    "attempt_index": turn,
                    "candidate_index": candidate_index,
                    "candidate_count": candidate_count,
                    "success": False,
                    "stage": "compile_check",
                    "error": err,
                })
                step_turn_records.append({"turn": turn, "phase": "B",
                                          "success": False, "error": err})
                logging.warning("[Step: %s] Compile error: %s", step_name, err[:200])
                fix_prompt = c_compilation_fix.format(
                    compile_error=err,
                    hls_code=new_code,
                    benchmark_context=self.benchmark_context,
                    repair_guidance=self.synthesis._compose_repair_guidance(err, report=None),
                    attempt_history=_format_attempt_history(step_turn_records, "B"),
                )
                self.messages.append({"role": "user", "content": fix_prompt})
                reply = self._call_llm(self.messages)
                self.messages.append({"role": "assistant", "content": reply})
                self._append_history("assistant", reply)
                fixed = extract_cpp_code(reply)
                if fixed:
                    new_code = fixed
                else:
                    self._finalize_candidate_evaluation(
                        correctness_status="not_run",
                        synthesis_status="not_run",
                        failure_class="malformed_output",
                    )
                continue

            outcome = self._synth_and_test(new_code, log_prefix=f"[Step: {step_name}]")
            result = outcome["synth"]

            if result.get("budget_exhausted"):
                return {
                    "success": False,
                    "step_name": step_name,
                    "code": new_code,
                    "error": result.get("error"),
                    "budget_exhausted": True,
                    "attempt_results": [
                        _compact_attempt_record(entry) for entry in attempt_results
                    ],
                }

            if result.get("skip_reason") == "csim_correctness_gate_failed":
                gate_summary = outcome.get("csim") or {}
                gate_error = (
                    (gate_summary.get("error") or "").strip()
                    + "\n"
                    + (gate_summary.get("log_excerpt") or "").strip()
                ).strip() or "pre-synthesis CSim/golden comparison failed"
                attempt_results.append({
                    "attempt_index": turn,
                    "candidate_index": candidate_index,
                    "candidate_count": candidate_count,
                    "success": False,
                    "stage": "csim",
                    "csim": gate_summary,
                    "error": f"csim_failed: {gate_error[:300]}",
                })
                step_turn_records.append({
                    "turn": turn,
                    "phase": "B",
                    "success": False,
                    "error": f"csim_failed: {gate_error[:200]}",
                })
                if turn >= attempt_limit - 1:
                    continue
                fix_prompt = hls_correctness_repair_fix.format(
                    step_name=step_name,
                    gate_name="csim",
                    gate_error=gate_error[:2000],
                    hls_code=new_code,
                    header_code=self.header_code,
                    benchmark_context=self.benchmark_context,
                    attempt_history=_format_attempt_history(step_turn_records, "B"),
                )
                self.messages.append({"role": "user", "content": fix_prompt})
                self._append_history("user", fix_prompt)
                reply = self._call_llm(self.messages)
                self.messages.append({"role": "assistant", "content": reply})
                self._append_history("assistant", reply)
                fixed = extract_cpp_code(reply)
                if fixed:
                    new_code = fixed
                else:
                    self._finalize_candidate_evaluation(
                        correctness_status="not_run",
                        synthesis_status="not_run",
                        failure_class="malformed_output",
                    )
                continue

            if result["success"]:
                logging.info("[Step: %s] Synthesis SUCCESS!\n%s",
                             step_name, format_report_summary(result["report"]))

                step_result = {
                    "success": True,
                    "step_name": step_name,
                    "report": result["report"],
                    "code": new_code,
                    "skill_prompt": skill_prompt_meta,
                }
                if step_name == "coalescing" or "max_widen_bitwidth" in (new_code or ""):
                    step_result["coalescing_diagnostics"] = _coalescing_diagnostics(
                        new_code, result.get("report")
                    )
                if self.synth_report:
                    step_result["vs_previous"] = compare_reports(
                        result["report"], self.synth_report,
                    )

                if gt_code:
                    cached_gt_report = self._gt_step_reports.get(step_name)
                    if cached_gt_report:
                        step_result["vs_ground_truth"] = compare_reports(
                            result["report"], cached_gt_report,
                        )
                        step_result["gt_report"] = cached_gt_report
                        step_result["gt_report_status"] = {
                            "status": "passed",
                            "source": "trusted_external_direct_jsonl",
                        }
                    else:
                        # Per-variant headers carry step-specific `#define`s
                        # (TILE_SIZE, COALESCING_5_512bit, …). Falling back to the
                        # local header here makes GT synth fail with "undeclared
                        # identifier", which silently zeroes the per-step gen-vs-gt
                        # comparison.
                        gt_hdr = gt_header_code if gt_header_code else self.header_code
                        gt_result = run_hls_synthesis(
                            gt_code,
                            gt_hdr,
                            header_name=self.header_name,
                            top_function=self.reference_hls_top,
                            part=self.part,
                            clock_ns=self.clock_ns,
                            extra_files=self.extra_files,
                        )
                        if gt_result["success"]:
                            step_result["vs_ground_truth"] = compare_reports(
                                result["report"], gt_result["report"],
                            )
                            step_result["gt_report"] = gt_result["report"]
                            # Phase 3: cache the GT step report so
                            # _previous_gt_report_for_step can find it later
                            # when the alignment check kicks in.
                            if step_name and gt_result.get("report"):
                                self._gt_step_reports[step_name] = gt_result["report"]
                        else:
                            step_result["gt_report_status"] = _summarize_synth_result(gt_result)

                if outcome["csim"] is not None:
                    step_result["csim"] = outcome["csim"]
                if outcome["cosim"] is not None:
                    step_result["cosim"] = outcome["cosim"]

                # Pillar 9 / Phase 9: correctness gate. Synth passing
                # is necessary but not sufficient — if csim or cosim
                # actually ran and failed, the LLM's optimization
                # broke the algorithm. Re-prompt with the failure log
                # and retry under the same turn budget as csynth-fail
                # repair. Default-on; disable via
                # ``C2HLS_DISABLE_CORRECTNESS_REPAIR=1`` for legacy
                # comparison runs.
                csim_summary = outcome["csim"]
                cosim_summary = outcome["cosim"]
                csim_failed = (
                    isinstance(csim_summary, dict)
                    and csim_summary.get("ran")
                    and not csim_summary.get("passed")
                )
                cosim_failed = (
                    _cosim_required_for_correctness()
                    and isinstance(cosim_summary, dict)
                    and cosim_summary.get("ran")
                    and not cosim_summary.get("passed")
                    and not _test_summary_is_timeout(cosim_summary)
                )
                correctness_disabled = bool(int(
                    os.getenv("C2HLS_DISABLE_CORRECTNESS_REPAIR", "0") or "0"
                ))
                if (csim_failed or cosim_failed) and not correctness_disabled:
                    gate_name = "csim" if csim_failed else "cosim"
                    gate_summary = csim_summary if csim_failed else cosim_summary
                    gate_error = (
                        (gate_summary.get("error") or "").strip() + "\n"
                        + (gate_summary.get("log_excerpt") or "").strip()
                    ).strip() or "(testbench reported a mismatch but did not capture an error message)"
                    logging.warning(
                        "[Step: %s] %s FAILED on attempt %d — entering "
                        "correctness-repair loop",
                        step_name, gate_name, turn,
                    )
                    attempt_results.append({
                        "attempt_index": turn,
                        "candidate_index": candidate_index,
                        "candidate_count": candidate_count,
                        "success": False,
                        "stage": gate_name,
                        "report": result.get("report"),
                        "csim": csim_summary,
                        "cosim": cosim_summary,
                        "error": f"{gate_name}_failed: {gate_error[:300]}",
                    })
                    step_turn_records.append({
                        "turn": turn, "phase": "B",
                        "success": False,
                        "error": f"{gate_name}_failed: {gate_error[:200]}",
                    })
                    fix_prompt = hls_correctness_repair_fix.format(
                        step_name=step_name,
                        gate_name=gate_name,
                        gate_error=gate_error[:2000],
                        hls_code=new_code,
                        header_code=self.header_code,
                        benchmark_context=self.benchmark_context,
                        attempt_history=_format_attempt_history(
                            step_turn_records, "B",
                        ),
                    )
                    self.messages.append({"role": "user", "content": fix_prompt})
                    reply = self._call_llm(self.messages)
                    self.messages.append({"role": "assistant", "content": reply})
                    self._append_history("assistant", reply)
                    fixed = extract_cpp_code(reply)
                    if fixed:
                        new_code = fixed
                    else:
                        self._finalize_candidate_evaluation(
                            correctness_status="not_run",
                            synthesis_status="not_run",
                            failure_class="malformed_output",
                        )
                    continue

                step_result.update({
                    "attempt_index": turn,
                    "attempt_count": attempt_limit,
                    "candidate_index": candidate_index,
                    "candidate_count": candidate_count,
                })
                feasibility = _paper_candidate_feasibility(
                    result.get("report") or {},
                    csim=outcome.get("csim"),
                    correctness_required=bool(self.testbench_code),
                    part=self.part,
                    clock_ns=self.clock_ns,
                )
                step_result["feasibility"] = feasibility
                if _feasibility_selection_enabled() and not feasibility.get("feasible"):
                    rejected = dict(step_result)
                    rejected.update({
                        "success": False,
                        "synthesis_success": True,
                        "stage": "feasibility",
                        "error": "candidate_infeasible: " + ", ".join(
                            feasibility.get("reasons") or ["unknown"]
                        ),
                    })
                    attempt_results.append(_compact_attempt_record(rejected))
                    step_turn_records.append({
                        "turn": turn,
                        "phase": "feasibility",
                        "success": False,
                        "error": rejected["error"],
                        "report": result.get("report"),
                    })
                    if turn >= attempt_limit - 1:
                        break
                    feasibility_prompt = _render_candidate_improvement_prompt(
                        step_name,
                        candidate_index,
                        candidate_count,
                        turn + 1,
                        attempt_limit,
                        result["report"],
                        new_code,
                    ) + (
                        "\n\nThe prior design is ineligible under the evaluation "
                        "feasibility rule. Fix all of these conditions without "
                        "changing functional behavior: "
                        + ", ".join(feasibility.get("reasons") or ["unknown"])
                        + "."
                    )
                    self.messages.append({"role": "user", "content": feasibility_prompt})
                    self._append_history("user", feasibility_prompt)
                    reply = self._call_llm(self.messages)
                    self.messages.append({"role": "assistant", "content": reply})
                    self._append_history("assistant", reply)
                    improved = extract_cpp_code(reply)
                    if not improved:
                        self._finalize_candidate_evaluation(
                            correctness_status="not_run",
                            synthesis_status="not_run",
                            failure_class="malformed_output",
                        )
                        break
                    new_code = improved
                    continue
                attempt_results.append(_compact_attempt_record(step_result))
                successful_attempts.append(step_result)

                if not exhaustive:
                    return step_result

                if turn >= attempt_limit - 1:
                    break

                improve_prompt = _render_candidate_improvement_prompt(
                    step_name,
                    candidate_index,
                    candidate_count,
                    turn + 1,
                    attempt_limit,
                    result["report"],
                    new_code,
                )
                self.messages.append({"role": "user", "content": improve_prompt})
                self._append_history("user", improve_prompt)
                reply = self._call_llm(self.messages)
                self.messages.append({"role": "assistant", "content": reply})
                self._append_history("assistant", reply)
                improved = extract_cpp_code(reply)
                if not improved:
                    self._finalize_candidate_evaluation(
                        correctness_status="not_run",
                        synthesis_status="not_run",
                        failure_class="malformed_output",
                    )
                    attempt_results.append({
                        "attempt_index": turn + 1,
                        "candidate_index": candidate_index,
                        "candidate_count": candidate_count,
                        "success": False,
                        "stage": "llm_improvement",
                        "error": "No code in improvement response",
                    })
                    break
                step_turn_records.append({
                    "turn": turn, "phase": "B", "success": True,
                    "report": result.get("report"),
                })
                new_code = improved
                continue

            logging.warning("[Step: %s] Synthesis failed: %s",
                            step_name, result["error"][:300])
            attempt_results.append({
                "attempt_index": turn,
                "candidate_index": candidate_index,
                "candidate_count": candidate_count,
                "success": False,
                "stage": "synthesis",
                "report": result.get("report"),
                "error": result["error"],
            })
            step_turn_records.append({"turn": turn, "phase": "B",
                                      "success": False, "error": result["error"]})
            is_timeout = "timed out" in result["error"].lower()
            guidance = self.synthesis._compose_repair_guidance(
                result["error"], report=result.get("report"),
            )
            history_block = _format_attempt_history(step_turn_records, "B")
            if is_timeout:
                fix_prompt = hls_synthesis_timeout_fix.format(
                    timeout=600,
                    hls_code=new_code,
                    header_code=self.header_code,
                    benchmark_context=self.benchmark_context,
                    repair_guidance=guidance,
                    attempt_history=history_block,
                )
            else:
                fix_prompt = hls_synthesis_fix.format(
                    synth_error=result["error"],
                    hls_code=new_code,
                    header_code=self.header_code,
                    target_context=_target_context_for_prompt(self.part, self.clock_ns),
                    benchmark_context=self.benchmark_context,
                    repair_guidance=guidance,
                    attempt_history=history_block,
                )
            self.messages.append({"role": "user", "content": fix_prompt})
            reply = self._call_llm(self.messages)
            self.messages.append({"role": "assistant", "content": reply})
            self._append_history("assistant", reply)
            fixed = extract_cpp_code(reply)
            if fixed:
                new_code = fixed
            else:
                self._finalize_candidate_evaluation(
                    correctness_status="not_run",
                    synthesis_status="not_run",
                    failure_class="malformed_output",
                )

        if exhaustive and successful_attempts:
            chosen = min(
                successful_attempts,
                key=lambda a: self._best_so_far_score(a.get("report") or {}),
            )
            chosen["attempt_selected"] = True
            chosen["selected_attempt_index"] = chosen.get("attempt_index")
            chosen["attempt_results"] = [
                _compact_attempt_record(entry) for entry in attempt_results
            ]
            chosen["attempt_stats"] = _metric_stats_from_reports([
                entry.get("report")
                for entry in attempt_results
                if entry.get("success") and entry.get("report")
            ])
            chosen["successful_attempt_count"] = len(successful_attempts)
            chosen["attempt_count"] = attempt_limit
            self._append_history(
                "system",
                f"[Step: {step_name}] candidate {candidate_index + 1}/"
                f"{candidate_count} selected attempt "
                f"{chosen.get('selected_attempt_index')} of {attempt_limit}.",
            )
            return chosen

        # Loop exhausted. Preserve the best synthesized-but-infeasible design
        # as typed evidence; it is never eligible for selection.
        infeasible_attempts = [
            entry for entry in attempt_results
            if entry.get("synthesis_success") and entry.get("report")
        ]
        if _feasibility_selection_enabled() and infeasible_attempts:
            best_infeasible = min(
                infeasible_attempts,
                key=lambda entry: self._best_so_far_score(entry.get("report") or {}),
            )
            return {
                **best_infeasible,
                "success": False,
                "synthesis_success": True,
                "step_name": step_name,
                "code": new_code,
                "error": "no_feasible_attempts",
                "feasibility_rejected": True,
                "attempt_results": [
                    _compact_attempt_record(entry) for entry in attempt_results
                ],
                "attempt_stats": _metric_stats_from_reports([
                    entry.get("report")
                    for entry in attempt_results
                    if entry.get("report")
                ]),
                "successful_attempt_count": 0,
                "attempt_count": attempt_limit,
            }

        # Loop exhausted. Tail the last attempt record so the caller can
        # tell whether we ran out of synth budget or correctness budget
        # — Pillar 9's no-op trap and the new correctness-repair loop
        # both consume turns from the same pool.
        last_err = ""
        if step_turn_records:
            last_err = step_turn_records[-1].get("error", "") or ""
        if last_err.startswith("csim_failed") or last_err.startswith("cosim_failed"):
            error_msg = (
                f"Correctness repair exhausted after "
                f"{attempt_limit} attempts ({last_err[:160]})"
            )
        else:
            error_msg = f"Synthesis failed after {attempt_limit} attempts"
        return {
            "success": False,
            "step_name": step_name,
            "error": error_msg,
            "skill_prompt": skill_prompt_meta,
            "attempt_results": [
                _compact_attempt_record(entry) for entry in attempt_results
            ],
            "attempt_stats": _metric_stats_from_reports([
                entry.get("report")
                for entry in attempt_results
                if entry.get("success") and entry.get("report")
            ]),
            "successful_attempt_count": len(successful_attempts),
            "attempt_count": attempt_limit,
        }

    # ---- Phase 6a: best-so-far tracking helpers ----

    @staticmethod
    def _best_so_far_score(report: dict) -> float:
        """Lower-is-better score after feasibility has been established.

        The paper selects minimum estimated worst-case latency in cycles.
        Latency in nanoseconds is only a fallback for legacy reports; a tiny
        resource term deterministically breaks exact latency ties.
        """
        if not report:
            return float("inf")
        lat = _as_float(
            report.get("latency_cycles_worst")
            or report.get("latency_cycles")
            or report.get("latency_ns_worst")
            or report.get("latency_ns")
        )
        if lat is None or lat <= 0:
            lat = float("inf")
        rsum = 0.0
        for k in ("bram", "dsp", "ff", "lut"):
            try:
                rsum += float(report.get(k) or 0)
            except (TypeError, ValueError):
                pass
        return lat + rsum * 1e-12

    def _record_best_so_far(self, history: list, *, step_index: int,
                             step_name: str, source: str) -> None:
        """Append a snapshot of the current orchestrator state to the
        best-so-far history. ``source`` is one of {"baseline", "step",
        "step_forward", "alignment_kept"} for downstream attribution."""
        if not self.synth_report:
            return
        generated_csim = getattr(self, "generated_csim", None)
        generated_cosim = getattr(self, "generated_cosim", None)
        feasibility = _paper_candidate_feasibility(
            self.synth_report,
            csim=generated_csim,
            correctness_required=bool(getattr(self, "testbench_code", None)),
            part=getattr(self, "part", DEFAULT_PART),
            clock_ns=getattr(self, "clock_ns", DEFAULT_CLOCK_NS),
        )
        if _feasibility_selection_enabled() and not feasibility.get("feasible"):
            self._append_history(
                "system",
                f"[Best-state] Excluded infeasible snapshot '{step_name}': "
                + ", ".join(feasibility.get("reasons") or ["unknown"]),
            )
            return
        history.append({
            "step_index": step_index,
            "step_name": step_name,
            "source": source,
            "score": self._best_so_far_score(self.synth_report),
            "code": self.hls_code,
            "report": dict(self.synth_report),
            "csim": _sanitize_test_summary(generated_csim),
            "cosim": _sanitize_test_summary(generated_cosim),
            "feasibility": feasibility,
        })

    def _record_phase_b_fast_candidate(self, reference_report: Optional[dict]) -> None:
        """Record an unusually fast Phase B result as provenance.

        The record is metadata only. It preserves accidental fast baselines
        for analysis while letting the multistep chain start from the chosen
        Phase B mode without hidden fallback behavior.
        """
        self.phase_b_fast_candidate = None
        if not reference_report or not self.synth_report:
            return
        try:
            threshold = float(os.getenv("C2HLS_PHASEB_FAST_CANDIDATE_RATIO", "0.80"))
        except ValueError:
            threshold = 0.80
        gap = _compute_baseline_gap(
            self.synth_report,
            reference_report,
            latency_tolerance=max(threshold, 1e-9),
            resource_tolerance=1e9,
            fmax_floor=0.0,
        )
        ratio = gap.get("latency_ratio")
        if ratio is None or ratio >= threshold:
            return
        self.phase_b_fast_candidate = {
            "recorded": True,
            "reason": (
                f"Phase B baseline is faster than reference baseline "
                f"({ratio:.3f}x < {threshold:.3f}x)"
            ),
            "phaseb_mode": self.phaseb_mode,
            "latency_ratio": ratio,
            "report": dict(self.synth_report),
            "code": self.hls_code,
        }
        self._append_history(
            "system",
            f"[Phase B] Recorded fast candidate: latency_ratio={ratio:.3f}, "
            f"mode={self.phaseb_mode}.",
        )

    def _promote_best_so_far(self, history: list) -> Optional[dict]:
        """If the orchestrator's current state isn't the best one
        observed in the trajectory, snap it back. Returns the snapshot
        promoted (or None if no promotion was needed)."""
        if not history:
            return None
        best = min(history, key=lambda h: h.get("score") or float("inf"))
        cur_score = self._best_so_far_score(self.synth_report)
        if best.get("score", float("inf")) < cur_score:
            logging.info(
                "[Phase 6a] best-so-far promotes step '%s' (idx %d, score %.6f) "
                "over current state (score %.6f)",
                best.get("step_name"), best.get("step_index"),
                best.get("score"), cur_score,
            )
            self.hls_code = best.get("code")
            self.synth_report = dict(best.get("report") or {})
            self.generated_csim = best.get("csim")
            self.generated_cosim = best.get("cosim")
            self._append_history(
                "system",
                f"[Phase 6a] Promoted best-so-far snapshot from step "
                f"'{best.get('step_name')}' (idx {best.get('step_index')}, "
                f"source={best.get('source')}). Final state replaced.",
            )
            return best
        return None

    # ---- Phase 6b: forward_eval mode ----

    def run_optimization_step_forward(self, step_name: str,
                                       gt_code: str = None,
                                       gt_header_code: str = None,
                                       skill_id: Optional[str] = None) -> dict:
        """Forward-only step: csynth + csim + cosim are gates (correctness
        guards) but a regression in PPA does NOT revert. Lets the
        trajectory explore freely; the outer loop's best-so-far tracker
        commits whichever state is best at the end.

        Returns the same step_result shape as run_optimization_step so
        downstream consumers (dataset_pipeline, history) don't have to
        special-case forward_eval."""
        logging.info("=== [Step: %s] Applying optimization (forward_eval) ===",
                     step_name)

        if not self.hls_code:
            return {"success": False, "step_name": step_name,
                    "error": "No HLS code to optimize"}

        prev_code = self.hls_code
        prev_report = self.synth_report

        attempt = self._optimization_step_attempt(
            step_name, gt_code,
            additional_guidance="",
            gt_header_code=gt_header_code,
            skill_id=skill_id,
        )
        if not attempt.get("success"):
            return attempt

        # Regression check is informational only in forward_eval — log it
        # but commit anyway. Correctness (csynth/csim/cosim) was already
        # gated by _optimization_step_attempt.
        new_report = attempt.get("report") or {}
        new_code = attempt.get("code") or self.hls_code
        from_per_step = _step_regression_reasons(
            new_report, prev_report, threshold=1e9, step_name=step_name, part=self.part,
        )
        if from_per_step:
            attempt["regression_warnings"] = from_per_step
            logging.info(
                "[Step: %s] forward_eval committing despite regression: %s",
                step_name, "; ".join(from_per_step)[:240],
            )

        self.hls_code = new_code
        self.synth_report = new_report
        self.generated_csim = attempt.get("csim")
        self.generated_cosim = attempt.get("cosim")
        attempt["forward_eval_committed"] = True
        return attempt

    def _prepare_skill_library_for_run(self) -> None:
        """Resolve skill control before any model or compiler call."""
        force_skill_prompts = os.getenv(
            "C2HLS_FORCE_SKILL_PROMPTS", ""
        ).strip().lower() in {"1", "true", "yes", "on"}
        skills_enabled = _skill_library_control_enabled()
        if not skills_enabled:
            self.skill_library = None
            self.skill_library_provenance = {
                "loaded": False,
                "control_enabled": False,
                "reason": "C2HLS_SKILL_MODE explicitly disables all skill-library use",
                "frozen": _env_flag(SKILL_LIBRARY_FROZEN_ENV),
                "persistence_enabled": False,
                "online_updates_enabled": False,
            }
            return
        should_load = (
            self.dynamic_routing
            or self.strategy == "dynamic"
            or force_skill_prompts
        )
        if should_load and self.skill_library is None:
            from skill_library import load_frozen_library, make_default_library

            persist_skills = (
                not _env_flag(SKILL_LIBRARY_FROZEN_ENV)
                and not self.reference_blind
                and _env_flag("C2HLS_SKILL_LIBRARY_PERSIST", "1")
            )
            if _env_flag(SKILL_LIBRARY_FROZEN_ENV):
                frozen_path = os.getenv("C2HLS_SKILL_LIBRARY_PATH", "").strip()
                if not frozen_path:
                    raise RuntimeError(
                        "frozen skill mode requires C2HLS_SKILL_LIBRARY_PATH"
                    )
                self.skill_library = load_frozen_library(Path(frozen_path))
            else:
                self.skill_library = make_default_library(persist=persist_skills)
            self.skill_library_provenance = _skill_library_provenance(
                self.skill_library
            )
            self.skill_library_provenance.update(
                {
                    "frozen": _env_flag(SKILL_LIBRARY_FROZEN_ENV),
                    "control_enabled": True,
                    "persistence_enabled": persist_skills,
                    "online_updates_enabled": _skill_updates_enabled(),
                }
            )
        elif self.skill_library is not None and not self.skill_library_provenance:
            self.skill_library_provenance = _skill_library_provenance(
                self.skill_library
            )

    def run_multistep(self, c_code: str, header_code: str = "",
                      header_name: str = "kernel.h",
                      steps: list = None,
                      gt_variants: dict = None,
                      gt_variant_headers: dict = None,
                      reference_report: dict = None):
        if getattr(self, "_candidate_stream_started_monotonic", None) is None:
            self._candidate_stream_started_monotonic = time.monotonic()
        if steps is None:
            steps = list(DEFAULT_OPT_STEPS)
        if gt_variants is None:
            gt_variants = {}
        if gt_variant_headers is None:
            gt_variant_headers = {}

        # The evaluator may validate expert implementations in the same
        # process, but the reference-isolated controller receives no expert code,
        # metrics, headers, or trajectory state.  Oracle runs must opt in
        # explicitly through C2HLS_GT_COMPARISON_IN_CONTROL=1 with reference
        # blindness disabled.
        if not _ground_truth_control_enabled():
            gt_variants = {}
            gt_variant_headers = {}
            reference_report = {}
            self._gt_step_reports.clear()
            self._gt_baseline_report = {}
            self.cosim_reference_cycle_info = {}

        # A missing, malformed, or contaminated frozen snapshot must fail
        # before Phase A/B can spend an LLM or synthesis call.
        self._prepare_skill_library_for_run()

        if not self.run_phase_a(c_code, header_code, header_name):
            return False, {"phase": "A", "error": "C code validation failed"}

        if not self.run_phase_b(multistep=True):
            return False, {
                "phase": "B",
                "error": "Baseline HLS synthesis/correctness failed",
                "turn_results": self.turn_results,
                "csim": self.generated_csim,
                "cosim": self.generated_cosim,
                "baseline_csim": self.generated_csim,
                "baseline_cosim": self.generated_cosim,
                "preflight_patches": self.preflight_patches,
            }

        # Phase 8 (opt-in via C2HLS_PHASE8_BASELINE_ALIGN=1): if our Phase B
        # baseline is significantly worse than the reference baseline,
        # re-translate with metric-only feedback before optimization
        # starts. This stops a bad initial translation from poisoning
        # every downstream optimization step.
        baseline_alignment = self._baseline_alignment_loop(reference_report)
        self._record_phase_b_fast_candidate(reference_report)

        baseline_report = dict(self.synth_report) if self.synth_report else {}
        baseline_csim = _sanitize_test_summary(self.generated_csim)
        baseline_cosim = _sanitize_test_summary(self.generated_cosim)
        # Store on self so _optimization_step_attempt can diff per-loop
        # bottlenecks of any subsequent step against the baseline (Pillar 1).
        self._baseline_report = baseline_report
        baseline_comparison = self.run_phase_c(reference_report) if reference_report else {}
        step_results = []

        # Phase 6a: best-so-far history. Seed with the baseline so a
        # trajectory that finds nothing better still has a fallback.
        # Always-on (no env flag) — trivial overhead, big upside.
        best_so_far_history: list = []
        self._record_best_so_far(
            best_so_far_history, step_index=-1,
            step_name="baseline", source="baseline",
        )

        # Phase 3: seed the GT baseline so trajectory-alignment can
        # walk back to it when computing parent_gt_report.
        if reference_report:
            self._gt_baseline_report = dict(reference_report)

        # Phase 5b: pre-synthesize the FULL GT trajectory once up-front so
        # the trajectory-alignment check works regardless of step-firing
        # order. Without this, dynamic routing's first step (often
        # `coalescing` on this kernel set) finds the GT cache empty
        # because per-step GT synth only fired in canonical order. The
        # pre-pop is gated by C2HLS_PHASE5_GT_PREPOP=1 (default off so
        # legacy runs aren't affected) — but recommended-on for any run
        # that uses dynamic-routing or combo strategies.
        if (bool(int(os.getenv("C2HLS_PHASE5_GT_PREPOP", "0") or "0"))
                and gt_variants):
            for gt_step_name, gt_code in gt_variants.items():
                if not gt_code or gt_step_name in self._gt_step_reports:
                    continue
                gt_hdr = (gt_variant_headers or {}).get(gt_step_name) or self.header_code
                try:
                    gt_result = run_hls_synthesis(
                        gt_code, gt_hdr, header_name=self.header_name,
                        top_function=self.reference_hls_top,
                        part=self.part, clock_ns=self.clock_ns,
                        extra_files=self.extra_files,
                    )
                    if gt_result.get("success") and gt_result.get("report"):
                        self._gt_step_reports[gt_step_name] = gt_result["report"]
                        logging.info(
                            "[Phase 5b] pre-populated GT cache for step '%s' "
                            "(lat_cyc=%s)", gt_step_name,
                            gt_result["report"].get("latency_cycles"),
                        )
                except Exception as exc:  # pragma: no cover
                    logging.warning(
                        "[Phase 5b] GT pre-synth for '%s' failed: %s",
                        gt_step_name, exc,
                    )

        # Phase 3: combo strategies short-circuit the per-step loop.
        # combo_full asks the LLM to apply all techniques in one rewrite;
        # combo_progressive does it as a 2-step structural→parallel pair.
        if self.strategy in ("combo", "combo_full"):
            from prompt_c2hls import COMBO_FULL_STEPS
            steps = list(COMBO_FULL_STEPS)
        elif self.strategy == "flash":
            from prompt_c2hls import FLASH_STEPS
            steps = list(FLASH_STEPS)
        elif self.strategy == "combo_progressive":
            from prompt_c2hls import COMBO_PROGRESSIVE_STEPS
            steps = list(COMBO_PROGRESSIVE_STEPS)

        # When dynamic_routing is on, we walk the steps in
        # bottleneck-driven order rather than the configured static
        # order; the configured `steps` list still bounds the search
        # space (we never invent a step the caller didn't allow).
        # When off, we fall through the original loop unchanged.
        # Combo modes always use the static loop (the combo IS the
        # strategy; they typically have 1-2 steps).
        if self.dynamic_routing and self.strategy not in (
            *ONE_SHOT_STRATEGIES,
            "combo_progressive",
        ):
            from bottleneck_router import select_next_step
            from robustness import (
                trajectory_collapse_check,
                throughput_regression_check,
            )

            available = list(steps)
            completed: list = []
            effects: list = []
            prev_report: Optional[dict] = baseline_report

            while available:
                feedback = (self.synth_report or {}).get("feedback") or {}
                decision = select_next_step(
                    feedback=feedback,
                    library=self.skill_library,
                    completed_steps=completed,
                    available_steps=available,
                    vitis_version=self.vitis_version,
                    fpga=self.part,
                )
                if not decision.step_name:
                    logging.info("[Multistep:dynamic] no more steps to try")
                    break
                logging.info("[Multistep:dynamic] %s", decision.reason)
                self._append_history(
                    "system",
                    f"[Multistep:dynamic] selected '{decision.step_name}': {decision.reason}",
                )

                step_name = decision.step_name
                gt_code = gt_variants.get(step_name)
                gt_header = gt_variant_headers.get(step_name)
                # Phase 6b: forward_eval mode skips regression-revert; only
                # correctness gates (csynth/csim/cosim) apply. Best-so-far
                # tracking commits the peak state at the end.
                try:
                    if self.strategy == "forward_eval":
                        step_result = self.run_optimization_step_forward(
                            step_name, gt_code=gt_code, gt_header_code=gt_header,
                            skill_id=decision.skill_id,
                        )
                    else:
                        step_result = self.run_optimization_step(
                            step_name, gt_code=gt_code, gt_header_code=gt_header,
                            skill_id=decision.skill_id,
                        )
                except Exception as exc:
                    logging.warning("[Multistep] Step '%s' raised: %s", step_name, exc)
                    step_result = {
                        "step_name": step_name,
                        "success": False,
                        "error": str(exc),
                        "exception_type": type(exc).__name__,
                        "budget_exhausted": "budget_exhausted" in str(exc),
                        "profile_required": True,
                        "report": {},
                    }
                    step_result["routing_decision"] = {
                        "step_name": decision.step_name,
                        "reason": decision.reason,
                        "bottleneck_kind": decision.bottleneck_kind,
                        "skill_id": decision.skill_id,
                        "confidence": decision.confidence,
                        "fallback": decision.fallback,
                        "skills_loaded_count": (
                            len(self.skill_library.all())
                            if self.skill_library is not None else 0
                        ),
                        "skill_store": (
                            str(getattr(self.skill_library, "store_path", ""))
                            if self.skill_library is not None else None
                        ),
                    }
                    step_results.append(step_result)
                    completed.append(step_name)
                    available = [s for s in available if s != step_name]
                    self.robustness_log.append({
                        "step": step_name,
                        "kind": "step_exception",
                        "error": str(exc),
                        "exception_type": type(exc).__name__,
                    })
                    break
                step_result["routing_decision"] = {
                    "step_name": decision.step_name,
                    "reason": decision.reason,
                    "bottleneck_kind": decision.bottleneck_kind,
                    "skill_id": decision.skill_id,
                    "confidence": decision.confidence,
                    "fallback": decision.fallback,
                    "skills_loaded_count": (
                        len(self.skill_library.all())
                        if self.skill_library is not None else 0
                    ),
                    "skill_store": (
                        str(getattr(self.skill_library, "store_path", ""))
                        if self.skill_library is not None else None
                    ),
                }

                # Pillar 9 item 3: hidden throughput regression flag.
                tp = throughput_regression_check(
                    step_result.get("report") or step_result.get("rejected_report"),
                    prev_report,
                )
                if tp.flagged:
                    step_result.setdefault("warnings", []).extend(tp.reasons)
                    self.robustness_log.append({
                        "step": step_name, "kind": "throughput_regression",
                        "reasons": tp.reasons,
                    })

                step_results.append(step_result)
                completed.append(step_name)
                available = [s for s in available if s != step_name]
                if step_result.get("budget_exhausted"):
                    self._append_history(
                        "system",
                        "[Budget] Total synthesis-evaluation budget exhausted; "
                        "ending the trajectory before any further LLM calls.",
                    )
                    break

                # Per-step effect — pulled from the existing classifier so
                # the robustness checks have the same labels the dataset
                # pipeline records.
                from dataset_pipeline.recorder import classify_step_effect
                csim = step_result.get("csim")
                csim_passed = (
                    bool(csim.get("passed")) if isinstance(csim, dict) else None
                )
                effect = classify_step_effect(
                    step_result.get("report") or step_result.get("rejected_report"),
                    prev_report,
                    success=bool(step_result.get("success")),
                    csim_passed=csim_passed,
                    error=step_result.get("error"),
                )
                step_result["step_effect"] = effect
                effects.append(effect)

                if (
                    decision.skill_id
                    and self.skill_library is not None
                    and _skill_updates_enabled()
                ):
                    rel_adv = None
                    cur_report = step_result.get("report") or {}
                    try:
                        prev_lat = float((prev_report or {}).get("latency_ns") or 0.0)
                        cur_lat = float(cur_report.get("latency_ns") or 0.0)
                        if prev_lat > 0 and cur_lat > 0:
                            rel_adv = (prev_lat - cur_lat) / prev_lat
                    except (TypeError, ValueError):
                        rel_adv = None
                    updated = self.skill_library.update_skill_statistics(
                        decision.skill_id,
                        success=bool(step_result.get("success")) and csim_passed is not False,
                        relative_advantage=rel_adv,
                    )
                    if updated is not None:
                        self.skill_library.promote_demote(decision.skill_id)
                        step_result["skill_update"] = {
                            "skill_id": decision.skill_id,
                            "success": bool(step_result.get("success")) and csim_passed is not False,
                            "relative_advantage": rel_adv,
                            "occurrences": updated.occurrences,
                            "sec_pass": updated.sec_pass,
                            "mean_advantage": updated.mean_advantage,
                            "confidence": updated.confidence,
                        }

                # Pillar 9 item 2: trajectory-collapse abort.
                collapse = trajectory_collapse_check(effects)
                if collapse.should_abort:
                    self._append_history(
                        "system",
                        f"[Multistep:dynamic] aborting trajectory: {collapse.reason}",
                    )
                    self.robustness_log.append({
                        "kind": "trajectory_collapse",
                        "reason": collapse.reason,
                        "consecutive_no_ops": collapse.consecutive_no_ops,
                    })
                    break

                if step_result.get("success") and step_result.get("report"):
                    prev_report = step_result["report"]
                    # Phase 6a: snapshot the just-accepted state.
                    self._record_best_so_far(
                        best_so_far_history,
                        step_index=len(step_results) - 1,
                        step_name=step_name,
                        source=("step_forward"
                                if self.strategy == "forward_eval"
                                else "step"),
                    )
        else:
            for idx, step_name in enumerate(steps):
                gt_code = gt_variants.get(step_name)
                gt_header = gt_variant_headers.get(step_name)
                # Phase 6b: forward_eval skips regression-revert.
                try:
                    if self.strategy == "forward_eval":
                        step_result = self.run_optimization_step_forward(
                            step_name, gt_code=gt_code, gt_header_code=gt_header,
                        )
                    else:
                        step_result = self.run_optimization_step(
                            step_name, gt_code=gt_code, gt_header_code=gt_header,
                        )
                except Exception as exc:
                    logging.warning("[Multistep] Step '%s' raised: %s", step_name, exc)
                    step_result = {
                        "step_name": step_name,
                        "success": False,
                        "error": str(exc),
                        "exception_type": type(exc).__name__,
                        "budget_exhausted": "budget_exhausted" in str(exc),
                        "profile_required": True,
                        "report": {},
                    }
                    step_results.append(step_result)
                    self.robustness_log.append({
                        "step": step_name,
                        "kind": "step_exception",
                        "error": str(exc),
                        "exception_type": type(exc).__name__,
                    })
                    break
                step_results.append(step_result)
                if step_result.get("budget_exhausted"):
                    self._append_history(
                        "system",
                        "[Budget] Total synthesis-evaluation budget exhausted; "
                        "ending the trajectory before any further LLM calls.",
                    )
                    break
                if not step_result.get("success"):
                    logging.warning("[Multistep] Step '%s' failed: %s", step_name, step_result.get("error", "unknown"))
                if step_result.get("success") and step_result.get("report"):
                    # Phase 6a: snapshot the just-accepted state.
                    self._record_best_so_far(
                        best_so_far_history, step_index=idx,
                        step_name=step_name,
                        source=("step_forward"
                                if self.strategy == "forward_eval"
                                else "step"),
                    )

        if (
            self.skill_library is not None
            and not _env_flag(SKILL_LIBRARY_FROZEN_ENV)
            and not self.reference_blind
            and _env_flag("C2HLS_SKILL_LIBRARY_PERSIST", "1")
        ):
            try:
                self.skill_library.save()
            except OSError as exc:
                logging.warning("SkillLibrary persistence failed at trajectory end: %s", exc)

        # Phase 6a: at the end of the trajectory, promote whichever
        # snapshot has the best score (lowest latency_ns + tiny
        # resource-sum tiebreak). If the best is the current state, this
        # is a no-op. If a mid-trajectory state was better, snap back to
        # it and overwrite final_report / hls_code with that snapshot.
        promotion = self._promote_best_so_far(best_so_far_history)

        candidate_feasibility = _paper_candidate_feasibility(
            self.synth_report,
            csim=self.generated_csim,
            correctness_required=bool(self.testbench_code),
            part=self.part,
            clock_ns=self.clock_ns,
        )
        selected_cosim = self._run_selected_winner_cosim()
        selected_cosim_ok = not (
            _cosim_selected_only() and self.supports_cosim and self.testbench_code
        ) or _selected_cosim_measurement_ok(selected_cosim)
        run_success = (
            bool(self.synth_report)
            and (
                not _feasibility_selection_enabled()
                or candidate_feasibility.get("feasible")
            )
            and selected_cosim_ok
        )

        return run_success, {
            "phase": "flash" if self.strategy == "flash" else "multistep",
            "baseline_report": baseline_report,
            "baseline_comparison": baseline_comparison,
            "baseline_csim": baseline_csim,
            "baseline_cosim": baseline_cosim,
            "csim": self.generated_csim,
            "cosim": self.generated_cosim,
            "final_report": self.synth_report,
            "steps": step_results,
            "generated_step_history": [
                {
                    "step_name": "baseline",
                    "success": True,
                    "report": baseline_report,
                    "comparison": baseline_comparison,
                    "csim": baseline_csim,
                    "cosim": baseline_cosim,
                },
                *step_results,
            ],
            "hls_code": self.hls_code,
            "best_so_far_history": [
                {k: v for k, v in h.items() if k != "code"}  # drop heavy code blob from results JSON
                for h in best_so_far_history
            ],
            "best_so_far_promotion": (
                {"promoted": True,
                 "from_step_name": promotion.get("step_name"),
                 "from_step_index": promotion.get("step_index"),
                 "score": promotion.get("score")}
                if promotion is not None else
                {"promoted": False, "reason": "final state was already the best"}
            ),
            "baseline_alignment": baseline_alignment,
            "phase_b_mode": self.phaseb_mode,
            "preflight_patches": self.preflight_patches,
            "phase_b_fast_candidate": (
                {k: v for k, v in (self.phase_b_fast_candidate or {}).items() if k != "code"}
                if self.phase_b_fast_candidate else None
            ),
            "skill_library_provenance": self.skill_library_provenance,
            "llm_usage": self._llm_usage_summary(),
            "synthesis_evaluations": self._synthesis_evaluation_summary(),
            "selected_winner_cosim_count": getattr(
                self, "selected_winner_cosim_count", 0
            ),
            "total_synthesis_calls": self._total_synthesis_calls(),
            "selected_code_sha256": getattr(self, "selected_code_sha256", None),
            "cosim_target_code_sha256": getattr(
                self, "cosim_target_code_sha256", None
            ),
            "candidate_feasibility": candidate_feasibility,
            "selected_cosim_measurement_valid": selected_cosim_ok,
            "selected_cosim_measurement_error": (
                "selected winner lacks a passing executed RTL cycle measurement"
                if not selected_cosim_ok else ""
            ),
            "independent_golden": self.independent_golden_provenance,
            "correctness_status": (
                ((self.generated_csim or {}).get("correctness") or {}).get(
                    "correctness_status"
                )
                if isinstance(self.generated_csim, dict)
                else None
            ) or (
                "passed"
                if isinstance(self.generated_csim, dict)
                and self.generated_csim.get("passed")
                else "not_run"
            ),
        }

    def save_results(self, output_dir: str, bench_name: str):
        os.makedirs(output_dir, exist_ok=True)

        if self.hls_code:
            with open(os.path.join(output_dir, f"{bench_name}_generated.cpp"), "w") as f:
                f.write(self.hls_code)

        history_payload = {
            "model": self.gpt_model,
            "model_translator":     os.getenv(TRANSLATOR_MODEL_ENV)     or self.gpt_model,
            "model_synthesis":      os.getenv(SYNTHESIS_MODEL_ENV)      or self.gpt_model,
            "model_quality_repair": os.getenv(QUALITY_REPAIR_MODEL_ENV) or self.gpt_model,
            "llm_usage": self._llm_usage_summary(),
            "messages": self.llm_controller_transcript or self.history,
            "event_history": self.history,
        }
        with open(os.path.join(output_dir, f"{bench_name}_history.json"), "w") as f:
            json.dump(history_payload, f, indent=2)

        if self.synth_report:
            with open(os.path.join(output_dir, f"{bench_name}_synth_report.json"), "w") as f:
                json.dump(self.synth_report, f, indent=2)

    def save_multistep_results(self, output_dir: str, bench_name: str, results: dict):
        os.makedirs(output_dir, exist_ok=True)

        if self.hls_code:
            with open(os.path.join(output_dir, f"{bench_name}_final.cpp"), "w") as f:
                f.write(self.hls_code)

        steps_dir = os.path.join(output_dir, "steps")
        os.makedirs(steps_dir, exist_ok=True)
        for index, step in enumerate(results.get("steps", [])):
            step_name = step.get("step_name", f"step_{index}")
            if step.get("code"):
                with open(os.path.join(steps_dir, f"{index}_{step_name}.cpp"), "w") as f:
                    f.write(step["code"])
            step_save = {key: value for key, value in step.items() if key != "code"}
            with open(os.path.join(steps_dir, f"{index}_{step_name}_report.json"), "w") as f:
                json.dump(step_save, f, indent=2, default=str)

        results_save = {key: value for key, value in results.items() if key != "hls_code"}
        for step in results_save.get("steps", []):
            step.pop("code", None)
        with open(os.path.join(output_dir, f"{bench_name}_multistep_results.json"), "w") as f:
            json.dump(results_save, f, indent=2, default=str)

        history_payload = {
            "model": self.gpt_model,
            "model_translator":     os.getenv(TRANSLATOR_MODEL_ENV)     or self.gpt_model,
            "model_synthesis":      os.getenv(SYNTHESIS_MODEL_ENV)      or self.gpt_model,
            "model_quality_repair": os.getenv(QUALITY_REPAIR_MODEL_ENV) or self.gpt_model,
            "llm_usage": self._llm_usage_summary(),
            "messages": self.llm_controller_transcript or self.history,
            "event_history": self.history,
        }
        with open(os.path.join(output_dir, f"{bench_name}_history.json"), "w") as f:
            json.dump(history_payload, f, indent=2)

    def run(self, c_code: str, header_code: str = "", header_name: str = "kernel.h",
            ground_truth_report: dict = None):
        if getattr(self, "_candidate_stream_started_monotonic", None) is None:
            self._candidate_stream_started_monotonic = time.monotonic()
        if not _ground_truth_control_enabled():
            ground_truth_report = {}
            self.cosim_reference_cycle_info = {}
        if not self.run_phase_a(c_code, header_code, header_name):
            return False, {"phase": "A", "error": "C code validation failed"}

        if not self.run_phase_b(multistep=False):
            return False, {
                "phase": "B",
                "error": "HLS synthesis/correctness failed",
                "turn_history": self.turn_results,
                "csim": self.generated_csim,
                "cosim": self.generated_cosim,
                "preflight_patches": self.preflight_patches,
            }

        comparison = {}
        quality_repair = {
            "attempted": False,
            "applied": False,
            "attempts": [],
        }
        if ground_truth_report:
            comparison = self.run_phase_c(ground_truth_report)
            quality_repair = self.run_quality_repair(
                ground_truth_report,
                comparison.get("comparison") if comparison.get("success") else None,
            )
            if quality_repair.get("applied"):
                comparison = self.run_phase_c(ground_truth_report)

        candidate_feasibility = _paper_candidate_feasibility(
            self.synth_report,
            csim=self.generated_csim,
            correctness_required=bool(self.testbench_code),
            part=self.part,
            clock_ns=self.clock_ns,
        )
        selected_cosim = self._run_selected_winner_cosim()
        selected_cosim_ok = not (
            _cosim_selected_only() and self.supports_cosim and self.testbench_code
        ) or _selected_cosim_measurement_ok(selected_cosim)
        run_success = (
            not _feasibility_selection_enabled()
            or candidate_feasibility.get("feasible")
        ) and selected_cosim_ok

        return run_success, {
            "phase": "complete",
            "hls_code": self.hls_code,
            "synth_report": self.synth_report,
            "comparison": comparison,
            "csim": self.generated_csim,
            "cosim": self.generated_cosim,
            "quality_repair": quality_repair,
            "turn_history": self.turn_results,
            "preflight_patches": self.preflight_patches,
            "llm_usage": self._llm_usage_summary(),
            "synthesis_evaluations": self._synthesis_evaluation_summary(),
            "selected_winner_cosim_count": getattr(
                self, "selected_winner_cosim_count", 0
            ),
            "total_synthesis_calls": self._total_synthesis_calls(),
            "selected_code_sha256": getattr(self, "selected_code_sha256", None),
            "cosim_target_code_sha256": getattr(
                self, "cosim_target_code_sha256", None
            ),
            "candidate_feasibility": candidate_feasibility,
            "selected_cosim_measurement_valid": selected_cosim_ok,
            "selected_cosim_measurement_error": (
                "selected winner lacks a passing executed RTL cycle measurement"
                if not selected_cosim_ok else ""
            ),
            "independent_golden": self.independent_golden_provenance,
            "correctness_status": (
                ((self.generated_csim or {}).get("correctness") or {}).get(
                    "correctness_status"
                )
                if isinstance(self.generated_csim, dict)
                else None
            ) or (
                "passed"
                if isinstance(self.generated_csim, dict)
                and self.generated_csim.get("passed")
                else "not_run"
            ),
        }


def _load_benchmark_inputs(bench_dir: str) -> dict:
    bench_dir = Path(bench_dir)
    meta_path = bench_dir / "metadata.json"
    with open(meta_path, "r") as f:
        meta = json.load(f)

    bench_name = meta["benchmark"]
    header_name = meta.get("header_file") or "kernel.h"

    with open(bench_dir / "plain.cpp", "r") as f:
        c_code = f.read()

    header_code = ""
    header_path = bench_dir / header_name
    if header_name and header_path.exists():
        with open(header_path, "r") as f:
            header_code = f.read()

    ground_truth_code = None
    gt_file = meta.get("gold_hls_baseline_file", "hls_baseline.cpp")

    # GT-variant selection.
    # Order of preference:
    #   1. metadata["preferred_gt_file"]      — explicit override when a
    #                                           specific variant must be used
    #                                           (e.g. it's the only one whose
    #                                           top signature matches the
    #                                           testbench).
    #   2. variants[-1]                       — the last (most optimized)
    #                                           variant in the workflow.
    #   3. gold_hls_baseline_file             — plain baseline fallback.
    #
    # Vitis HLS is the sole validator. We deliberately do NOT run a g++
    # compile-check on GT variants: gold HLS code routinely uses HLS-only
    # headers (ap_int.h via mc.h) that g++ cannot compile, so a preflight
    # would false-reject valid variants. If Vitis later rejects the selected
    # variant, validate_gold_reference()'s fallback-to-baseline handles it.
    preferred = meta.get("preferred_gt_file")
    if preferred and (bench_dir / preferred).exists():
        gt_file = preferred
        logging.info(f"Using preferred GT variant '{preferred}'")
    else:
        variants = meta.get("variants", [])
        for variant in reversed(variants):
            vfile = variant["file"]
            if (bench_dir / vfile).exists():
                gt_file = vfile
                logging.info(f"Using last variant '{variant['name']}' as ground truth")
                break

    gt_path = bench_dir / gt_file
    if gt_path.exists():
        with open(gt_path, "r") as f:
            ground_truth_code = f.read()

    gold_hls_source_code = ""
    gold_src_file = meta.get("gold_hls_source_file", "gold_hls_source.cpp")
    gold_src_path = bench_dir / gold_src_file
    if gold_src_path.exists():
        with open(gold_src_path, "r") as f:
            gold_hls_source_code = f.read()

    # Per-step GT pairs. Each rodinia variant ships its own header with
    # variant-specific `#define`s (e.g. TILE_SIZE differs between tiling and
    # coalescing). The local cleaned header doesn't contain those, so
    # synthesising the GT cpp with the local header fails with "undeclared
    # identifier". Pair the variant cpp with its sibling header from upstream.
    use_upstream_variant_headers = (
        meta.get("source_repo") in {"rodinia-hls", "rodinia-hls-nova"}
        or bool(meta.get("use_upstream_variant_headers"))
    )
    gt_variants = {}
    gt_variant_headers = {}
    for variant in meta.get("variants", []):
        vname = variant["name"]
        vfile = variant["file"]
        vpath = bench_dir / vfile
        if not vpath.exists():
            continue
        step_key = _normalize_variant_step_name(vname)
        with open(vpath, "r") as f:
            gt_variants[step_key] = f.read()
        # Prefer the upstream variant's own header so per-variant `#define`s
        # (TILE_SIZE, COALESCING_5_512bit, etc.) survive into the synth tcl.
        # Falls back to the local header if the upstream copy is missing or
        # the metadata didn't record a source_path.
        upstream_src = variant.get("source_path") or ""
        if upstream_src and use_upstream_variant_headers:
            upstream_header = Path(upstream_src).with_name(header_name)
            if upstream_header.exists():
                gt_variant_headers[step_key] = _rewrite_source_includes_for_local_support(
                    upstream_header.read_text(), bench_dir,
                )

    testbench_code = ""
    tb_file = meta.get("testbench_file") or ""
    tb_path = bench_dir / tb_file if tb_file else None
    if tb_path and tb_path.exists():
        with open(tb_path, "r") as f:
            testbench_code = f.read()

    extra_files = []
    extra_file_paths = set()
    for rel_path in meta.get("support_files", []):
        file_path = bench_dir / rel_path
        if file_path.exists():
            extra_files.append({"path": rel_path, "content": file_path.read_text()})
            extra_file_paths.add(rel_path)

    support_dir = bench_dir / "support"
    if support_dir.exists():
        for file_path in sorted(support_dir.rglob("*")):
            if not file_path.is_file():
                continue
            rel_path = str(file_path.relative_to(bench_dir))
            if rel_path in extra_file_paths:
                continue
            extra_files.append({"path": rel_path, "content": file_path.read_text()})
            extra_file_paths.add(rel_path)

    # GT code is deliberately NOT passed here. _build_benchmark_context may
    # only look at plain C, the header, the testbench-visible signature, and
    # static policy hints — never the gold reference.
    benchmark_context = _build_benchmark_context(
        meta,
        header_name,
        header_code,
        c_code,
        testbench_code,
    )

    return {
        "meta": meta,
        "bench_dir": str(bench_dir),
        "bench_name": bench_name,
        "header_name": header_name,
        "c_code": c_code,
        "header_code": header_code,
        "ground_truth_code": ground_truth_code,
        "gold_hls_source_code": gold_hls_source_code,
        "gt_variants": gt_variants,
        "gt_variant_headers": gt_variant_headers,
        "testbench_code": testbench_code,
        "extra_files": extra_files,
        "benchmark_context": benchmark_context,
    }


_HLSFACTORY_SHAPE_REGISTRY = (
    Path(__file__).resolve().parent / "configs" / "hlsfactory_output_shapes.json"
)


def _authoritative_hlsfactory_output_specs(
    meta: dict, testbench_code: str
) -> tuple[dict, dict, dict]:
    """Return hash-bound output contracts audited from public testbenches.

    HLSFactory dumps are flattened text.  Inferring ``[value_count]`` from a
    golden execution only relabels the flat stream; it does not establish the
    public harness's dimensional contract.  The checked-in registry records
    ``print_array`` bounds and is bound to the exact testbench bytes, so a
    missing kernel, changed harness, or conflicting local override fails shut.
    """

    try:
        registry_bytes = _HLSFACTORY_SHAPE_REGISTRY.read_bytes()
        registry = json.loads(registry_bytes)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"HLSFactory shape registry unavailable: {exc}") from exc
    if registry.get("schema_version") != "c2hls.hlsfactory-output-shapes.v1":
        raise ValueError("unsupported HLSFactory shape-registry schema")

    benchmark = str(meta.get("benchmark") or "")
    entry = (registry.get("benchmarks") or {}).get(benchmark)
    if not isinstance(entry, dict):
        raise ValueError(f"no authoritative HLSFactory shape entry for {benchmark!r}")
    actual_testbench_sha = hashlib.sha256(
        (testbench_code or "").encode("utf-8")
    ).hexdigest()
    expected_testbench_sha = str(entry.get("testbench_sha256") or "")
    if actual_testbench_sha != expected_testbench_sha:
        raise ValueError(
            "HLSFactory testbench does not match audited shape contract: "
            f"expected {expected_testbench_sha}, got {actual_testbench_sha}"
        )

    declared_outputs = entry.get("outputs")
    if not isinstance(declared_outputs, dict) or not declared_outputs:
        raise ValueError(f"empty HLSFactory output contract for {benchmark!r}")
    policy = registry.get("policy") or {}
    default_atol = float(policy.get("default_float_atol", 1e-6))
    default_rtol = float(policy.get("default_float_rtol", 1e-5))
    comparison_specs: dict[str, dict] = {}
    declarations: dict[str, dict] = {}
    for raw_name, raw_spec in declared_outputs.items():
        name = str(raw_name)
        if not name or not isinstance(raw_spec, dict):
            raise ValueError(f"invalid output declaration in {benchmark!r}")
        shape = raw_spec.get("shape")
        logical_shape = raw_spec.get("logical_shape")
        for label, dimensions in (("emission", shape), ("logical", logical_shape)):
            if (
                not isinstance(dimensions, list)
                or not dimensions
                or any(
                    not isinstance(dim, int) or isinstance(dim, bool) or dim <= 0
                    for dim in dimensions
                )
            ):
                raise ValueError(f"invalid {label} shape for {benchmark}:{name}")
        kind = str(raw_spec.get("kind") or "")
        if kind not in {"float", "integer"}:
            raise ValueError(f"invalid numeric kind for {benchmark}:{name}")
        comparison = {"shape": list(shape), "kind": kind}
        if kind == "float":
            comparison.update({"atol": default_atol, "rtol": default_rtol})
        comparison_specs[name] = comparison
        declarations[name] = {
            "shape": list(shape),
            "logical_shape": list(logical_shape),
            "layout": str(raw_spec.get("layout") or ""),
            "kind": kind,
        }

    # Metadata may redundantly assert the registry, but may never silently
    # redefine a shape while preserving the same total element count.
    explicit = meta.get("golden_output_specs")
    if explicit:
        if not isinstance(explicit, dict) or set(explicit) != set(comparison_specs):
            raise ValueError("benchmark golden_output_specs output set conflicts with registry")
        for name, expected in comparison_specs.items():
            supplied = explicit.get(name) or {}
            if list(supplied.get("shape") or []) != expected["shape"]:
                raise ValueError(
                    f"benchmark golden_output_specs shape conflicts for {benchmark}:{name}"
                )
            if supplied.get("kind", expected["kind"]) != expected["kind"]:
                raise ValueError(
                    f"benchmark golden_output_specs kind conflicts for {benchmark}:{name}"
                )

    return comparison_specs, declarations, {
        "path": _HLSFACTORY_SHAPE_REGISTRY.relative_to(
            Path(__file__).resolve().parent
        ).as_posix(),
        "sha256": hashlib.sha256(registry_bytes).hexdigest(),
        "testbench_sha256": actual_testbench_sha,
    }


def _prepare_independent_golden(inputs: dict) -> dict:
    """Build and validate a CPU oracle for print-only benchmark testbenches.

    The output text is intentionally kept in the private ``output`` field for
    comparator calls.  Saved result records receive only the content hash and
    typed output inventory in ``provenance``.
    """
    meta = inputs.get("meta") or {}
    explicit = meta.get("independent_golden_required")
    required = (
        bool(explicit)
        if explicit is not None
        else str(meta.get("source_repo") or "").strip().lower() == "hlsfactory"
    )
    base_provenance = {
        "schema_version": "c2hls.independent-golden.v1",
        "required": required,
        "source": "pragma_stripped_plain_c_and_public_testbench",
    }
    if not required:
        return {
            "success": True,
            "required": False,
            "output": "",
            "specs": {},
            "provenance": {**base_provenance, "status": "not_required"},
        }
    if not inputs.get("testbench_code"):
        return {
            "success": False,
            "required": True,
            "error": "independent golden required but benchmark testbench is missing",
            "output": "",
            "specs": {},
            "provenance": {**base_provenance, "status": "invalid"},
        }

    native = run_native_testbench(
        inputs.get("c_code") or "",
        inputs.get("testbench_code") or "",
        inputs.get("header_code") or "",
        inputs.get("header_name") or "kernel.h",
        extra_files=inputs.get("extra_files") or [],
        timeout=int(os.getenv("C2HLS_CPU_GOLDEN_TIMEOUT", "180") or "180"),
    )
    if not native.get("success"):
        error = native.get("error") or "CPU-golden execution failed"
        return {
            "success": False,
            "required": True,
            "error": error,
            "output": "",
            "specs": {},
            "provenance": {
                **base_provenance,
                "status": "invalid",
                "stage": native.get("stage"),
                "error": error,
            },
        }

    output = native.get("output") or ""
    try:
        from golden_output import compare_hlsfactory_dumps, parse_hlsfactory_dumps

        parsed = parse_hlsfactory_dumps(output)
        is_hlsfactory = (
            str(meta.get("source_repo") or "").strip().lower() == "hlsfactory"
        )
        declarations: dict[str, dict] = {}
        shape_registry: dict[str, str] = {}
        if is_hlsfactory:
            specs, declarations, shape_registry = _authoritative_hlsfactory_output_specs(
                meta, inputs.get("testbench_code") or ""
            )
            if set(parsed) != set(specs):
                raise ValueError(
                    "CPU-golden output set does not match HLSFactory shape registry: "
                    f"expected {sorted(specs)}, got {sorted(parsed)}"
                )
        else:
            specs = copy.deepcopy(meta.get("golden_output_specs") or {})
        default_atol = float(meta.get("golden_output_atol", 1e-6))
        default_rtol = float(meta.get("golden_output_rtol", 1e-5))
        for name, parsed_output in parsed.items():
            spec = dict(specs.get(name) or {})
            if not is_hlsfactory:
                spec.setdefault("shape", [len(parsed_output.values)])
            spec.setdefault("kind", "auto")
            if not parsed_output.integer_tokens:
                spec.setdefault("atol", default_atol)
                spec.setdefault("rtol", default_rtol)
            specs[name] = spec
        self_check = compare_hlsfactory_dumps(output, output, specs=specs).to_dict()
        if not self_check.get("passed"):
            raise ValueError(
                "CPU-golden self-check failed: "
                f"{self_check.get('reason')}: {self_check.get('details')}"
            )
    except Exception as exc:
        return {
            "success": False,
            "required": True,
            "error": f"independent golden output is invalid: {type(exc).__name__}: {exc}",
            "output": "",
            "specs": {},
            "provenance": {
                **base_provenance,
                "status": "invalid",
                "stage": "parse_and_self_check",
                "error": str(exc),
            },
        }

    output_inventory = {
        name: {
            "count": len(value.values),
            "integer_tokens": value.integer_tokens,
            "declared_shape": (specs.get(name) or {}).get("shape"),
            "logical_shape": (declarations.get(name) or {}).get("logical_shape"),
            "layout": (declarations.get(name) or {}).get("layout"),
        }
        for name, value in parsed.items()
    }
    provenance = {
        **base_provenance,
        "status": "passed",
        "output_sha256": hashlib.sha256(output.encode("utf-8")).hexdigest(),
        "specs_sha256": _sha256_text(specs),
        "shape_registry": shape_registry,
        "outputs": output_inventory,
        "output_count": len(output_inventory),
        "value_count": sum(item["count"] for item in output_inventory.values()),
    }
    return {
        "success": True,
        "required": True,
        "output": output,
        "specs": specs,
        "provenance": provenance,
    }


def _normalize_variant_step_name(variant_name: str) -> str:
    match = re.search(r"_(\d+)_(.+)$", variant_name or "")
    step_name = match.group(2) if match else (variant_name or "baseline")
    step_name = step_name.replace("double_buffer", "doublebuffer")
    step_name = step_name.replace("doublebuffer", "doublebuffer")
    step_name = step_name.replace("unrolll", "unroll")
    step_name = step_name.replace("unrolling", "unroll")
    return step_name or "baseline"


def _rewrite_source_includes_for_local_support(code: str, bench_dir: Path) -> str:
    """Adapt upstream-header `#include` directives so the synthesis sandbox can
    resolve them.

    Two transforms:
      1. `#include "../../common/foo"` → `#include "support/common/foo"`
         when the rodinia common/ tree is mirrored under <bench>/support/common/.
      2. `#include "support.h"` → stripped. machsuite-style benchmarks pull
         this in for host-side scaffolding (driver, golden-data harness); it
         doesn't belong in HLS synthesis and Vitis can't resolve it from the
         work_dir, breaking GT validation for fft / sort_merge / viterbi.
    """
    support_common = bench_dir / "support" / "common"
    if support_common.exists():
        def _replace(match: re.Match) -> str:
            rel_name = match.group(1)
            if (support_common / rel_name).exists():
                return f'#include "support/common/{rel_name}"'
            return match.group(0)

        code = re.sub(r'#include\s+"(?:\.\./)+common/([^"]+)"', _replace, code)

    # Strip machsuite-style host-side support.h includes — only those that
    # reference an unqualified "support.h", which is never an HLS file.
    code = re.sub(r'^[ \t]*#include\s+"support\.h"\s*\n', '', code, flags=re.MULTILINE)

    return code


def _strip_contract_comments_and_literals(text: str) -> str:
    """Remove syntax that must not seed the public-contract identifier set."""
    text = re.sub(r"/\*.*?\*/", " ", text or "", flags=re.DOTALL)
    text = re.sub(r"//[^\n]*", " ", text)
    text = re.sub(r'"(?:\\.|[^"\\])*"', '""', text)
    text = re.sub(r"'(?:\\.|[^'\\])*'", "''", text)
    return text


def _normalize_contract_tokens(text: str) -> str:
    return re.sub(r"\s+", "", _strip_contract_comments_and_literals(text))


def _header_contract_definitions(header_code: str) -> dict[str, dict]:
    """Extract macro/constant definitions relevant to workload semantics.

    The paper contract deliberately ignores include paths and private helper
    declarations.  Public identifiers are selected separately from the plain
    input and golden testbench, then followed transitively through this map.
    """
    clean = _strip_contract_comments_and_literals(header_code)
    clean = re.sub(r"\\\n", "", clean)
    definitions: dict[str, dict] = {}
    macro_pattern = re.compile(
        r"^[ \t]*#[ \t]*define[ \t]+"
        r"(?P<name>[A-Za-z_]\w*)"
        r"(?P<params>\([^\n)]*\))?"
        r"[ \t]*(?P<body>[^\n]*)$",
        re.MULTILINE,
    )
    for match in macro_pattern.finditer(clean):
        name = match.group("name")
        params = match.group("params") or ""
        body = match.group("body") or ""
        definitions[name] = {
            "kind": "function_macro" if params else "macro",
            "normalized": _normalize_contract_tokens(params + body),
            "dependencies": sorted(
                set(re.findall(r"\b[A-Za-z_]\w*\b", body)) - {name}
            ),
        }

    constant_pattern = re.compile(
        r"\b(?:static\s+)?(?:constexpr\s+|const\s+)"
        r"(?P<type>[A-Za-z_]\w*(?:\s*::\s*[A-Za-z_]\w*)?"
        r"(?:\s*<[^;=]+?>)?)\s+"
        r"(?P<name>[A-Za-z_]\w*)\s*=\s*(?P<value>[^;]+);",
        re.MULTILINE,
    )
    for match in constant_pattern.finditer(clean):
        name = match.group("name")
        value = match.group("value")
        definitions[name] = {
            "kind": "constant",
            "normalized": _normalize_contract_tokens(
                match.group("type") + "=" + value
            ),
            "dependencies": sorted(
                set(re.findall(r"\b[A-Za-z_]\w*\b", value)) - {name}
            ),
        }
    return definitions


def _public_header_contract_audit(
    canonical_header: str,
    candidate_header: str,
    *,
    plain_code: str,
    testbench_code: str,
) -> dict:
    """Compare the transitive public macro/constant workload contract."""
    canonical_defs = _header_contract_definitions(canonical_header)
    candidate_defs = _header_contract_definitions(candidate_header)
    public_text = _strip_contract_comments_and_literals(
        (plain_code or "") + "\n" + (testbench_code or "")
    )
    referenced = set(re.findall(r"\b[A-Za-z_]\w*\b", public_text))
    pending = sorted(referenced.intersection(canonical_defs))
    contract_names: set[str] = set()
    while pending:
        name = pending.pop()
        if name in contract_names:
            continue
        contract_names.add(name)
        for dependency in canonical_defs[name].get("dependencies", []):
            if dependency in canonical_defs and dependency not in contract_names:
                pending.append(dependency)

    differences = []
    canonical_contract = {}
    candidate_contract = {}
    for name in sorted(contract_names):
        expected = canonical_defs[name]
        actual = candidate_defs.get(name)
        canonical_contract[name] = {
            "kind": expected["kind"],
            "normalized": expected["normalized"],
        }
        if actual is not None:
            candidate_contract[name] = {
                "kind": actual["kind"],
                "normalized": actual["normalized"],
            }
        if actual is None:
            differences.append({
                "identifier": name,
                "reason_code": "public_definition_missing",
                "expected_sha256": _sha256_text(expected["normalized"]),
                "actual_sha256": "",
            })
        elif (
            actual.get("kind") != expected.get("kind")
            or actual.get("normalized") != expected.get("normalized")
        ):
            differences.append({
                "identifier": name,
                "reason_code": "public_definition_mismatch",
                "expected_sha256": _sha256_text(expected["normalized"]),
                "actual_sha256": _sha256_text(actual.get("normalized", "")),
            })

    return {
        "schema_version": "c2hls.public-workload-contract.v1",
        "passed": not differences,
        "reason_codes": sorted({item["reason_code"] for item in differences}),
        "public_identifiers": sorted(contract_names),
        "canonical_contract_sha256": _sha256_text(canonical_contract),
        "candidate_contract_sha256": _sha256_text(candidate_contract),
        "differences": differences,
    }


def _ground_truth_candidates(inputs: dict) -> list[dict]:
    meta = inputs["meta"]
    bench_dir = Path(inputs["bench_dir"])
    candidates = []
    seen_files = set()
    default_header_name = meta.get("header_file") or inputs.get("header_name") or "kernel.h"
    default_header_code = inputs.get("header_code", "")
    use_upstream_variant_headers = (
        meta.get("source_repo") in {"rodinia-hls", "rodinia-hls-nova"}
        or bool(meta.get("use_upstream_variant_headers"))
    )

    for variant in meta.get("variants", []):
        variant_file = variant.get("file")
        if not variant_file or variant_file in seen_files:
            continue
        variant_path = bench_dir / variant_file
        if not variant_path.exists():
            continue
        source_path = variant.get("source_path", "")
        header_code = default_header_code
        if source_path and use_upstream_variant_headers:
            source_header = Path(source_path).with_name(default_header_name)
            if source_header.exists():
                header_code = _rewrite_source_includes_for_local_support(source_header.read_text(), bench_dir)
        variant_code = variant_path.read_text()
        contract_audit = _public_header_contract_audit(
            default_header_code,
            header_code,
            plain_code=inputs.get("c_code", ""),
            testbench_code=inputs.get("testbench_code", ""),
        )
        candidates.append(
            {
                "variant_name": variant.get("name", Path(variant_file).stem),
                "file": variant_file,
                "step_name": _normalize_variant_step_name(variant.get("name", variant_file)),
                "source_path": source_path,
                "header_name": default_header_name,
                "header_code": header_code,
                "code": variant_code,
                "public_contract_audit": contract_audit,
            }
        )
        seen_files.add(variant_file)

    if candidates:
        return candidates

    hls_code = inputs.get("ground_truth_code")
    if hls_code:
        source_path = inputs["meta"].get("gold_hls_source_path", "")
        header_code = default_header_code
        if source_path and use_upstream_variant_headers:
            source_header = Path(source_path).with_name(default_header_name)
            if source_header.exists():
                header_code = _rewrite_source_includes_for_local_support(source_header.read_text(), bench_dir)
        return [
            {
                "variant_name": "baseline",
                "file": inputs["meta"].get("gold_hls_baseline_file", "hls_baseline.cpp"),
                "step_name": "baseline",
                "source_path": source_path,
                "header_name": default_header_name,
                "header_code": header_code,
                "code": hls_code,
                "public_contract_audit": _public_header_contract_audit(
                    default_header_code,
                    header_code,
                    plain_code=inputs.get("c_code", ""),
                    testbench_code=inputs.get("testbench_code", ""),
                ),
            }
        ]
    return []


def _sha256_text(value: Any) -> str:
    text = value if isinstance(value, str) else json.dumps(value, sort_keys=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _reference_cache_descriptor(inputs: dict) -> dict:
    """Build an exact-input identity for reusable local Vitis evidence."""
    from evaluation_repro import _probe_vitis_version, sha256_file

    meta = inputs["meta"]
    vitis_probe = _probe_vitis_version(os.environ)
    settings_path = Path(os.getenv("C2HLS_VITIS_SETTINGS", "")).expanduser()
    toolchain_identity = {
        "configured_version": os.getenv("C2HLS_VITIS_VERSION", "unknown"),
        "probe": {
            key: vitis_probe.get(key)
            for key in (
                "ran",
                "returncode",
                "version",
                "executable",
                "executable_sha256",
                "output_sha256",
                "error",
            )
        },
        "settings_sha256": (
            sha256_file(settings_path) if settings_path.is_file() else None
        ),
    }
    candidates = _ground_truth_candidates(inputs)
    extra_files = sorted(
        (
            {
                "path": str(item.get("path") or ""),
                "sha256": _sha256_text(item.get("content") or ""),
            }
            for item in (inputs.get("extra_files") or [])
            if isinstance(item, dict)
        ),
        key=lambda item: item["path"],
    )
    candidate_inputs = [
        {
            "variant_name": candidate.get("variant_name") or "",
            "file": candidate.get("file") or "",
            "step_name": candidate.get("step_name") or "",
            "source_path": candidate.get("source_path") or "",
            "header_name": candidate.get("header_name") or "",
            "header_sha256": _sha256_text(candidate.get("header_code") or ""),
            "code_sha256": _sha256_text(candidate.get("code") or ""),
            "public_contract_audit": {
                "schema_version": (
                    candidate.get("public_contract_audit") or {}
                ).get("schema_version", ""),
                "passed": bool(
                    (candidate.get("public_contract_audit") or {}).get("passed")
                ),
                "canonical_contract_sha256": (
                    candidate.get("public_contract_audit") or {}
                ).get("canonical_contract_sha256", ""),
                "candidate_contract_sha256": (
                    candidate.get("public_contract_audit") or {}
                ).get("candidate_contract_sha256", ""),
                "reason_codes": (
                    candidate.get("public_contract_audit") or {}
                ).get("reason_codes", []),
            },
        }
        for candidate in candidates
    ]
    configuration = {
        "schema_version": REFERENCE_CACHE_SCHEMA_VERSION,
        "benchmark": meta.get("benchmark") or inputs.get("bench_name") or "",
        "source_repo": meta.get("source_repo") or "",
        "top_function": meta.get("hls_top", "workload"),
        "part": meta.get("part", DEFAULT_PART),
        "clock_ns": float(meta.get("clock_ns", DEFAULT_CLOCK_NS)),
        "flow_target": os.getenv("C2HLS_FLOW_TARGET", "vitis"),
        "vitis_version": os.getenv("C2HLS_VITIS_VERSION", "unknown"),
        "toolchain_identity": toolchain_identity,
        "validation_mode": (
            os.getenv("C2HLS_REFERENCE_VALIDATE_MODE", "all").strip().lower() or "all"
        ),
        "supports_csim": bool(meta.get("supports_csim") and inputs.get("testbench_code")),
        "supports_cosim": _generated_cosim_supported(
            meta, inputs.get("testbench_code", "")
        ),
        "reference_cosim_all_candidates": _env_flag("C2HLS_REFERENCE_COSIM", "1"),
        "reference_cosim_selected_only": _env_flag(
            REFERENCE_COSIM_SELECTED_ONLY_ENV
        ),
        "reference_cosim_baseline": _env_flag(REFERENCE_COSIM_BASELINE_ENV),
        "force_selected_cosim": _env_flag(FORCE_SELECTED_COSIM_ENV),
        "preferred_gt_file": meta.get("preferred_gt_file") or "",
        "cosim_depths": meta.get("cosim_depths") or {},
        "testbench_sha256": _sha256_text(inputs.get("testbench_code") or ""),
        "independent_golden_output_sha256": (
            (inputs.get("independent_golden_provenance") or {}).get("output_sha256")
            or ""
        ),
        "independent_golden_specs_sha256": _sha256_text(
            inputs.get("independent_golden_specs") or {}
        ),
        "extra_files": extra_files,
        "candidates": candidate_inputs,
    }
    serialized = json.dumps(configuration, sort_keys=True, separators=(",", ":"))
    fingerprint = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    return {"fingerprint": fingerprint, "configuration": configuration}


def _paper_reference_cache_identity_complete(descriptor: dict) -> bool:
    """Require an executed, content-bound toolchain identity in paper mode."""

    if not _reference_blind_enabled():
        return True
    identity = (descriptor.get("configuration") or {}).get("toolchain_identity") or {}
    probe = identity.get("probe") or {}
    return bool(
        probe.get("ran")
        and probe.get("returncode") == 0
        and not probe.get("error")
        and probe.get("version")
        == identity.get("configured_version")
        and probe.get("executable")
        and probe.get("executable_sha256")
        and identity.get("settings_sha256")
    )


def _reference_cache_path(descriptor: dict) -> Optional[Path]:
    raw_root = os.getenv(REFERENCE_CACHE_DIR_ENV, "").strip()
    if not raw_root:
        return None
    root = Path(raw_root).expanduser()
    if not root.is_absolute():
        root = REPO_ROOT / root
    benchmark = str(descriptor["configuration"].get("benchmark") or "benchmark")
    safe_benchmark = re.sub(r"[^A-Za-z0-9_.-]+", "_", benchmark).strip("_") or "benchmark"
    return root / f"{safe_benchmark}.{descriptor['fingerprint']}.json"


def _reference_validation_cacheable(inputs: dict, validation: Any) -> bool:
    if not isinstance(validation, dict) or not validation.get("benchmark_ready"):
        return False
    if (validation.get("synthesis") or {}).get("status") != "passed":
        return False
    supports_csim = bool(inputs["meta"].get("supports_csim") and inputs.get("testbench_code"))
    if supports_csim and (validation.get("csim") or {}).get("status") != "passed":
        return False
    if _env_flag(REFERENCE_COSIM_SELECTED_ONLY_ENV) and not validation.get(
        "selected_reference_cosim_measurement_valid"
    ):
        return False
    if _env_flag(REFERENCE_COSIM_BASELINE_ENV) and not validation.get(
        "baseline_reference_cosim_measurement_valid"
    ):
        return False
    return True


def _strip_cached_work_dirs(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _strip_cached_work_dirs(item)
            for key, item in value.items()
            if key != "work_dir"
        }
    if isinstance(value, list):
        return [_strip_cached_work_dirs(item) for item in value]
    return value


def _write_reference_validation_cache(
    inputs: dict,
    validation: dict,
    *,
    source_result_json: str = "",
) -> Optional[Path]:
    descriptor = _reference_cache_descriptor(inputs)
    cache_path = _reference_cache_path(descriptor)
    if (
        cache_path is None
        or not _paper_reference_cache_identity_complete(descriptor)
        or not _reference_validation_cacheable(inputs, validation)
    ):
        return None
    entry = {
        "schema_version": REFERENCE_CACHE_SCHEMA_VERSION,
        "fingerprint": descriptor["fingerprint"],
        "configuration": descriptor["configuration"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_result_json": source_result_json,
        "reference_validation": _strip_cached_work_dirs(validation),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = cache_path.with_name(f".{cache_path.name}.tmp.{os.getpid()}")
    temp_path.write_text(json.dumps(entry, indent=2, sort_keys=True, default=str) + "\n")
    os.replace(temp_path, cache_path)
    return cache_path


def _load_reference_validation_cache(inputs: dict) -> tuple[Optional[dict], dict]:
    descriptor = _reference_cache_descriptor(inputs)
    cache_path = _reference_cache_path(descriptor)
    provenance = {
        "schema_version": REFERENCE_CACHE_SCHEMA_VERSION,
        "enabled": cache_path is not None,
        "hit": False,
        "fingerprint": descriptor["fingerprint"],
        "cache_path": str(cache_path) if cache_path else "",
    }
    if not _paper_reference_cache_identity_complete(descriptor):
        provenance["enabled"] = False
        provenance["rejection_reason"] = "paper_cache_toolchain_identity_incomplete"
        return None, provenance
    if cache_path is None or not cache_path.is_file():
        return None, provenance
    try:
        entry = json.loads(cache_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        provenance["rejection_reason"] = f"cache_read_failed: {exc}"
        return None, provenance
    if (
        entry.get("schema_version") != REFERENCE_CACHE_SCHEMA_VERSION
        or entry.get("fingerprint") != descriptor["fingerprint"]
        or entry.get("configuration") != descriptor["configuration"]
    ):
        provenance["rejection_reason"] = "cache_identity_mismatch"
        return None, provenance
    validation = entry.get("reference_validation")
    if not _reference_validation_cacheable(inputs, validation):
        provenance["rejection_reason"] = "cached_reference_not_synth_csim_valid"
        return None, provenance
    require_cosim = os.getenv(REFERENCE_CACHE_REQUIRE_COSIM_ENV, "0").strip().lower() in {
        "1", "true", "yes", "on",
    }
    supports_cosim = _generated_cosim_supported(
        inputs["meta"], inputs.get("testbench_code", "")
    )
    if require_cosim and supports_cosim and (validation.get("cosim") or {}).get("status") != "passed":
        provenance["rejection_reason"] = "cached_reference_cosim_not_passed"
        return None, provenance

    cached = copy.deepcopy(validation)
    original_source = cached.get("reference_source") or "local_vitis"
    cached["reference_source"] = (
        "cached_local_vitis" if original_source in {"local_vitis", "cached_local_vitis"}
        else f"cached_{original_source}"
    )
    provenance.update({
        "hit": True,
        "source_result_json": entry.get("source_result_json") or "",
        "original_reference_source": original_source,
        "synthesis_status": (cached.get("synthesis") or {}).get("status"),
        "csim_status": (cached.get("csim") or {}).get("status"),
        "cosim_status": (cached.get("cosim") or {}).get("status"),
    })
    cached["reference_cache"] = provenance
    return cached, provenance


def _preferred_reference_file(meta: dict, workflow: list[dict]) -> str:
    if meta.get("source_repo") == "rodinia-hls":
        for entry in reversed(workflow):
            if entry.get("step_name") == "coalescing":
                return entry.get("file", "")
        optimized = [entry.get("file", "") for entry in workflow if entry.get("step_name") != "baseline"]
        if optimized:
            return optimized[-1]
    return meta.get("preferred_gt_file", "")


def _preferred_reference_candidate_file(meta: dict, candidates: list[dict]) -> str:
    preferred = meta.get("preferred_gt_file", "")
    if preferred:
        return preferred
    if meta.get("source_repo") == "rodinia-hls":
        for entry in reversed(candidates):
            if entry.get("step_name") == "coalescing":
                return entry.get("file", "")
        optimized = [entry.get("file", "") for entry in candidates if entry.get("step_name") != "baseline"]
        if optimized:
            return optimized[-1]
    return ""


def _reference_jsonl_paths() -> list[Path]:
    """Candidate direct-reference JSONL files, ordered from broad references
    to local repair/rerun artifacts. Passing duplicate records are preferred
    during indexing, so repair artifacts can improve an older fail/timeout
    without silently masking a pass.
    """
    defaults = [
        REPO_ROOT / "csynth_vitis_2023.2__device_xilinx_u280_gen3x16_xdma_1_202211_1.jsonl",
        REPO_ROOT / "sw_emu_vitis_2023.2__device_xilinx_u280_gen3x16_xdma_1_202211_1.jsonl",
        REPO_ROOT / "results" / "references_philip" / "sw_emu_vitis_2023.2__device_xilinx_u280_gen3x16_xdma_1_202211_1.jsonl",
        REPO_ROOT / "results" / "references_philip" / "hw_emu_vitis_2023.2__device_xilinx_u280_gen3x16_xdma_1_202211_1.jsonl",
        REPO_ROOT / "artifacts" / "requested_hwemu_matrix.jsonl",
        REPO_ROOT / "artifacts" / "requested_hwemu_mismatch_rerun.jsonl",
        REPO_ROOT / "artifacts" / "hw_emu_reference_candidate_after_mismatch_rerun.jsonl",
        REPO_ROOT / "artifacts" / "nw2_pipeline_hwemu_xrt_debug_off_after_agentic_20260506_001152.jsonl",
    ]
    extra = [
        Path(item)
        for item in os.getenv("C2HLS_REFERENCE_JSONL_PATHS", "").split(os.pathsep)
        if item.strip()
    ]
    paths = []
    seen = set()
    for path in [*defaults, *extra]:
        if path in seen:
            continue
        seen.add(path)
        if path.exists():
            paths.append(path)
    return paths


def _record_payload_status(record: dict) -> str:
    report_type = record.get("report_type")
    payload = record.get(report_type) if report_type else None
    if not isinstance(payload, dict):
        return "unknown"
    return str(payload.get("status") or "unknown").lower()


def _direct_status_is_pass(status: str) -> bool:
    return str(status or "").lower() in {"pass", "passed", "success", "ok"}


def _normal_status(status: str) -> str:
    lowered = str(status or "").lower()
    if _direct_status_is_pass(lowered):
        return "passed"
    if lowered in {"fail", "failed", "error"}:
        return "failed"
    if lowered == "timeout":
        return "timeout"
    return lowered or "unknown"


def _load_direct_reference_index() -> dict:
    global _DIRECT_REFERENCE_CACHE
    if _DIRECT_REFERENCE_CACHE is not None:
        return _DIRECT_REFERENCE_CACHE

    index: dict[tuple[str, tuple[str, ...], int], list[dict]] = {}
    paths = _reference_jsonl_paths()
    for path in paths:
        try:
            lines = path.read_text().splitlines()
        except OSError:
            continue
        for line_no, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            impl = record.get("implementation") or {}
            if impl.get("origin") != "rodinia_hls_benchmark":
                continue
            report_type = record.get("report_type")
            if report_type not in {"hls_synth", "sw_run", "rtl_sim"}:
                continue
            variant = impl.get("variant") or {}
            try:
                variant_index = int(variant.get("index"))
            except (TypeError, ValueError):
                continue
            group_path = tuple(str(part) for part in ((record.get("problem") or {}).get("group_path") or []))
            if not group_path:
                continue
            enriched = dict(record)
            enriched["_reference_artifact"] = str(path)
            enriched["_reference_line"] = line_no
            enriched["_variant_name_norm"] = _normalize_variant_token(variant.get("name"))
            key = (report_type, group_path, variant_index)
            bucket = index.setdefault(key, [])
            new_pass = _direct_status_is_pass(_record_payload_status(enriched))
            if new_pass and not any(_direct_status_is_pass(_record_payload_status(existing)) for existing in bucket):
                bucket.insert(0, enriched)
            elif not bucket:
                bucket.append(enriched)
            else:
                bucket.append(enriched)

    _DIRECT_REFERENCE_CACHE = {"index": index, "paths": [str(path) for path in paths]}
    return _DIRECT_REFERENCE_CACHE


def _normalize_variant_token(value: str | None) -> str:
    token = str(value or "").strip().lower()
    token = token.replace("double_buffer", "doublebuffer")
    token = token.replace("unrolll", "unroll")
    token = token.replace("unrolling", "unroll")
    return token


def _candidate_variant_index(candidate: dict) -> int | None:
    name = candidate.get("variant_name") or candidate.get("file") or ""
    match = re.search(r"_(\d+)_", name)
    if match:
        return int(match.group(1))
    if candidate.get("step_name") == "baseline":
        return 0
    return None


def _candidate_variant_aliases(candidate: dict) -> set[str]:
    aliases = {
        candidate.get("step_name", ""),
        _normalize_variant_step_name(candidate.get("variant_name", "")),
        candidate.get("variant_name", "").rsplit("_", 2)[-1],
    }
    normalized = {_normalize_variant_token(item) for item in aliases if item}
    if "doublebuffer" in normalized:
        normalized.add("double_buffer")
    if "unroll" in normalized:
        normalized.update({"unrolll", "unrolling"})
    return normalized


def _reference_group_path_candidates(meta: dict, candidates: list[dict] | None = None) -> list[tuple[str, ...]]:
    explicit = meta.get("group_path")
    paths: list[tuple[str, ...]] = []
    if isinstance(explicit, list) and explicit:
        paths.append(tuple(str(part) for part in explicit))
    elif isinstance(explicit, str) and explicit:
        paths.append(tuple(part for part in explicit.split("/") if part))

    bench = meta.get("benchmark") or ""
    if bench.startswith("cfd_"):
        paths.append(("cfd", bench))
    if bench.startswith("lc_"):
        paths.append(("leukocyte", bench))
    if bench:
        paths.append((bench,))

    source_candidates = candidates or []
    for candidate in source_candidates:
        source_path = candidate.get("source_path") or ""
        marker = "/Benchmarks/"
        if marker not in source_path:
            continue
        tail = source_path.split(marker, 1)[1]
        parts = [part for part in tail.split("/") if part]
        if not parts:
            continue
        if len(parts) >= 2 and parts[0] in {"cfd", "leukocyte", "backprop"}:
            paths.append((parts[0], parts[1]))
        else:
            paths.append((parts[0],))

    deduped = []
    seen = set()
    for path in paths:
        if path and path not in seen:
            seen.add(path)
            deduped.append(path)
    return deduped


def _find_direct_reference_record(report_type: str, meta: dict, candidate: dict,
                                  all_candidates: list[dict]) -> dict | None:
    variant_index = _candidate_variant_index(candidate)
    if variant_index is None:
        return None
    aliases = _candidate_variant_aliases(candidate)
    ref_index = _load_direct_reference_index()["index"]
    for group_path in _reference_group_path_candidates(meta, all_candidates):
        bucket = ref_index.get((report_type, group_path, variant_index), [])
        if not bucket:
            continue
        exact = [
            record for record in bucket
            if (record.get("_variant_name_norm") or "") in aliases
        ]
        candidates_to_rank = exact or bucket
        passing = [
            record for record in candidates_to_rank
            if _direct_status_is_pass(_record_payload_status(record))
        ]
        return (passing or candidates_to_rank)[0]
    return None


def _num(value) -> float | None:
    if value is None:
        return None
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def _int_num(value) -> int | None:
    parsed = _num(value)
    return int(parsed) if parsed is not None else None


def _time_to_ns(value: str | None) -> float | None:
    if value is None:
        return None
    raw = str(value).strip()
    match = re.match(r"^([-+]?\d+(?:\.\d+)?)\s*(sec|s|ms|us|ns)?$", raw, re.IGNORECASE)
    if not match:
        return _num(raw)
    amount = float(match.group(1))
    unit = (match.group(2) or "ns").lower()
    if unit in {"sec", "s"}:
        return amount * 1_000_000_000.0
    if unit == "ms":
        return amount * 1_000_000.0
    if unit == "us":
        return amount * 1_000.0
    return amount


def _report_from_hls_synth_payload(payload: dict) -> dict:
    perf = payload.get("PerformanceEstimates") or {}
    timing = perf.get("SummaryOfTimingAnalysis") or {}
    latency = perf.get("SummaryOfOverallLatency") or {}
    area = payload.get("AreaEstimates") or {}
    resources = area.get("Resources") or {}
    assignments = payload.get("UserAssignments") or {}

    estimated = _num(timing.get("EstimatedClockPeriod"))
    requested = _num(assignments.get("TargetClockPeriod"))
    report = {
        "latency_cycles": _int_num(latency.get("Average-caseLatency")) or _int_num(latency.get("Worst-caseLatency")),
        "latency_ns": _time_to_ns(latency.get("Average-caseRealTimeLatency")) or _time_to_ns(latency.get("Worst-caseRealTimeLatency")),
        "latency_cycles_worst": _int_num(latency.get("Worst-caseLatency")),
        "latency_ns_worst": _time_to_ns(latency.get("Worst-caseRealTimeLatency")),
        "interval": _int_num(latency.get("Interval-max")),
        "bram": _int_num(resources.get("BRAM_18K")),
        "dsp": _int_num(resources.get("DSP")),
        "ff": _int_num(resources.get("FF")),
        "lut": _int_num(resources.get("LUT")),
        "uram": _int_num(resources.get("URAM")),
        "estimated_clock_period_ns": estimated,
        "requested_clock_period_ns": requested,
        "fmax_mhz": round(1000.0 / estimated, 2) if estimated and estimated > 0 else None,
        "slack_ns": round(requested - estimated, 3) if requested is not None and estimated is not None else None,
    }
    return report


def _direct_record_summary(record: dict | None, report_type: str) -> dict:
    if not record:
        return {
            "status": "missing",
            "passed": False,
            "artifact": "",
            "line": None,
            "profile_required": True,
        }
    payload = record.get(report_type) or {}
    raw_status = _record_payload_status(record)
    return {
        "status": _normal_status(raw_status),
        "direct_status": raw_status,
        "passed": _direct_status_is_pass(raw_status),
        "artifact": record.get("_reference_artifact", ""),
        "line": record.get("_reference_line"),
        "run": record.get("run") or {},
        "variant": (record.get("implementation") or {}).get("variant") or {},
        "payload": {
            key: payload.get(key)
            for key in ("status", "kernel_runtime_cycles", "kernel_runtime_us", "kernel_clock_freq_mhz")
            if key in payload
        },
    }


def _validate_external_ground_truth_candidate(candidate: dict, inputs: dict,
                                              supports_csim: bool, supports_cosim: bool,
                                              all_candidates: list[dict]) -> dict:
    meta = inputs["meta"]
    synth_record = _find_direct_reference_record("hls_synth", meta, candidate, all_candidates)
    sw_record = _find_direct_reference_record("sw_run", meta, candidate, all_candidates)
    hw_record = _find_direct_reference_record("rtl_sim", meta, candidate, all_candidates)
    synth_summary_direct = _direct_record_summary(synth_record, "hls_synth")
    sw_summary = _direct_record_summary(sw_record, "sw_run")
    hw_summary = _direct_record_summary(hw_record, "rtl_sim")

    report = {}
    synth_summary = {
        "status": synth_summary_direct["status"],
        "success": synth_summary_direct["passed"],
        "external": True,
        "direct_record": synth_summary_direct,
        "report": report,
    }
    benchmark_ready = bool(synth_summary_direct["passed"])
    invalid_reason = ""
    if not synth_record:
        invalid_reason = (
            "Missing trusted external direct hls_synth record for "
            f"{meta.get('benchmark')} variant {candidate.get('variant_name')}"
        )
    elif not benchmark_ready:
        invalid_reason = (
            "Trusted external direct hls_synth record is not passing: "
            f"status={synth_summary_direct.get('direct_status')}"
        )
    else:
        report = _report_from_hls_synth_payload(synth_record.get("hls_synth") or {})
        synth_summary["report"] = report

    csim_summary = {
        "status": _test_status(supports_csim, False, False),
        "supported": supports_csim,
        "ran": False,
        "success": False,
        "passed": False,
        "skip_reason": "reference CSim skipped; trusted direct Vitis reference artifacts are authoritative",
        "profile_required": True,
    }
    cosim_summary = {
        "status": _test_status(supports_cosim, False, False),
        "supported": supports_cosim,
        "ran": False,
        "success": False,
        "passed": False,
        "skip_reason": "reference cosim skipped; direct hw_emu status is recorded under external_validation.hw_emu",
        "profile_required": True,
    }

    return {
        "variant_name": candidate.get("variant_name", "baseline"),
        "file": candidate.get("file", ""),
        "step_name": candidate.get("step_name", "baseline"),
        "source_path": candidate.get("source_path", ""),
        "benchmark_ready": benchmark_ready,
        "invalid_reason": invalid_reason,
        "synthesis": synth_summary,
        "csim": csim_summary,
        "cosim": cosim_summary,
        "report": report,
        "selected": False,
        "testbench_interface_mismatch": "",
        "reference_source": "direct_jsonl",
        "external_validation": {
            "mode": "trusted_external",
            "used": True,
            "jsonl_paths": _load_direct_reference_index().get("paths", []),
            "hls_synth": synth_summary_direct,
            "sw_emu": sw_summary,
            "hw_emu": hw_summary,
            "csim": {
                "status": "not_run",
                "skip_reason": "no standalone direct reference CSim artifact; not used to validate trusted references",
                "profile_required": True,
            },
            "cosim": {
                "status": "not_run",
                "skip_reason": "no standalone Rodinia/Nova direct cosim artifact; hw_emu is used for RTL-level direct evidence",
                "profile_required": True,
            },
        },
    }


def _trusted_external_gt_step_reports(inputs: dict) -> dict[str, dict]:
    meta = inputs["meta"]
    if meta.get("source_repo") not in TRUSTED_EXTERNAL_REFERENCE_REPOS:
        return {}
    candidates = _ground_truth_candidates(inputs)
    reports: dict[str, dict] = {}
    for candidate in candidates:
        record = _find_direct_reference_record("hls_synth", meta, candidate, candidates)
        if not record or not _direct_status_is_pass(_record_payload_status(record)):
            continue
        report = _report_from_hls_synth_payload(record.get("hls_synth") or {})
        if report:
            reports[candidate.get("step_name", "baseline")] = report
    return reports


def _validate_ground_truth_candidate(candidate: dict, inputs: dict,
                                     supports_csim: bool, supports_cosim: bool,
                                     run_csim_check: bool = True,
                                     run_cosim_check: bool = True) -> dict:
    meta = inputs["meta"]
    hls_code = candidate["code"]
    header_name = candidate.get("header_name") or inputs.get("header_name") or "kernel.h"
    header_code = candidate.get("header_code", inputs.get("header_code", ""))
    top_function = meta.get("hls_top", "workload")

    contract_audit = candidate.get("public_contract_audit")
    if not isinstance(contract_audit, dict):
        contract_audit = _public_header_contract_audit(
            inputs.get("header_code", ""),
            header_code,
            plain_code=inputs.get("c_code", ""),
            testbench_code=inputs.get("testbench_code", ""),
        )

    expected_signature = _expected_top_signature(
        inputs.get("header_code", ""),
        inputs.get("testbench_code", ""),
        top_function,
    )
    candidate_signature = _extract_function_signature(
        hls_code, top_function, definitions_only=True
    )
    if expected_signature is None:
        csim_signature_mismatch = (
            f"canonical `{top_function}` signature is missing or unparseable"
        )
    elif candidate_signature is None:
        csim_signature_mismatch = (
            f"candidate `{top_function}` definition is missing or unparseable"
        )
    else:
        csim_signature_mismatch = _top_signature_mismatch_reason(
            hls_code,
            inputs.get("header_code", ""),
            inputs.get("testbench_code", ""),
            top_function,
        )

    exclusion_reasons = []
    if not contract_audit.get("passed"):
        identifiers = [
            item.get("identifier", "unknown")
            for item in contract_audit.get("differences", [])
        ]
        exclusion_reasons.append(
            "public workload/header contract mismatch"
            + (f" ({', '.join(identifiers)})" if identifiers else "")
        )
    if csim_signature_mismatch:
        exclusion_reasons.append(csim_signature_mismatch)

    # Contract-incompatible upstream variants are not comparable experts.
    # Exclude them before any CSim, synthesis, or co-simulation call.
    if exclusion_reasons:
        excluded_reason = "; ".join(exclusion_reasons)
        return {
            "variant_name": candidate.get("variant_name", "baseline"),
            "file": candidate.get("file", ""),
            "step_name": candidate.get("step_name", "baseline"),
            "source_path": candidate.get("source_path", ""),
            "benchmark_ready": False,
            "invalid_reason": "Excluded incomparable reference variant: " + excluded_reason,
            "synthesis": {
                "status": "not_run",
                "ran": False,
                "success": False,
                "error": "reference contract exclusion",
                "report": {},
            },
            "csim": {
                **_summarize_test_result(None, supports_csim),
                "skip_reason": "reference contract exclusion",
            },
            "cosim": {
                **_summarize_test_result(None, supports_cosim),
                "skip_reason": "reference contract exclusion",
            },
            "report": {},
            "selected": False,
            "testbench_interface_mismatch": csim_signature_mismatch,
            "public_contract_audit": contract_audit,
            "reference_contract_status": "excluded",
            "feasibility": {
                "schema_version": "c2hls.candidate-feasibility.v1",
                "feasible": False,
                "reasons": ["reference_contract_incompatible"],
            },
        }

    outcome = _run_synth_csim_cosim(
        hls_code,
        header_code=header_code,
        header_name=header_name,
        top_function=top_function,
        part=meta.get("part", DEFAULT_PART),
        clock_ns=meta.get("clock_ns", DEFAULT_CLOCK_NS),
        extra_files=inputs.get("extra_files", []),
        testbench_code=inputs.get("testbench_code", ""),
        run_csim_check=supports_csim and run_csim_check and not csim_signature_mismatch,
        run_cosim_check=supports_cosim and run_cosim_check and not csim_signature_mismatch,
        cosim_depths=meta.get("cosim_depths", {}),
        cosim_requires_csim_pass=True,
        golden_output_text=inputs.get("independent_golden_output", ""),
        golden_output_specs=inputs.get("independent_golden_specs", {}),
    )
    synth_result = outcome["synth"]
    synth_summary = _summarize_synth_result(synth_result)

    # When csim is supported but was skipped for signature mismatch, surface
    # the reason in the summary instead of leaving it as "not_run".
    csim_summary = outcome["csim"]
    if csim_summary is None:
        csim_summary = _summarize_test_result(None, supports_csim)
    if csim_signature_mismatch and supports_csim and run_csim_check:
        csim_summary["error"] = f"Skipped CSim: {csim_signature_mismatch}"

    cosim_summary = outcome["cosim"]
    if cosim_summary is None:
        cosim_summary = _summarize_test_result(None, supports_cosim)
        if supports_cosim and not run_cosim_check:
            cosim_summary["skip_reason"] = "reference cosim disabled by C2HLS_REFERENCE_COSIM=0"
            cosim_summary["profile_required"] = True

    benchmark_ready = synth_summary["status"] == "passed"
    invalid_reason = ""
    if not benchmark_ready:
        invalid_reason = f"Gold HLS synthesis failed: {synth_summary.get('error', '')}".strip()
    elif csim_signature_mismatch and supports_csim and run_csim_check:
        benchmark_ready = False
        invalid_reason = f"Gold HLS CSim is incompatible with the benchmark testbench: {csim_signature_mismatch}"
    elif supports_csim and run_csim_check and not csim_summary.get("passed", False):
        benchmark_ready = False
        invalid_reason = f"Gold HLS csim failed: {csim_summary.get('error', '') or 'testbench did not pass'}"

    feasibility = _paper_candidate_feasibility(
        synth_summary.get("report", {}),
        csim=csim_summary,
        correctness_required=supports_csim and run_csim_check,
        part=meta.get("part", DEFAULT_PART),
        clock_ns=meta.get("clock_ns", DEFAULT_CLOCK_NS),
    )
    if _feasibility_selection_enabled() and benchmark_ready and not feasibility.get("feasible"):
        benchmark_ready = False
        invalid_reason = (
            "Gold HLS candidate is not feasible: "
            + ", ".join(feasibility.get("reasons") or ["unknown"])
        )

    return {
        "variant_name": candidate.get("variant_name", "baseline"),
        "file": candidate.get("file", ""),
        "step_name": candidate.get("step_name", "baseline"),
        "source_path": candidate.get("source_path", ""),
        "benchmark_ready": benchmark_ready,
        "invalid_reason": invalid_reason,
        "synthesis": synth_summary,
        "csim": csim_summary,
        "cosim": cosim_summary,
        "report": synth_summary.get("report", {}),
        "selected": False,
        "testbench_interface_mismatch": csim_signature_mismatch,
        "public_contract_audit": contract_audit,
        "reference_contract_status": "passed",
        "feasibility": feasibility,
    }


def _execute_reference_candidate_cosim(candidate: dict, inputs: dict) -> dict:
    """Run one offline reference RTL measurement with typed failure output.

    This preflight evidence is deliberately separate from every method's
    matched search budget.  It is used only after synthesis/CSim establish the
    designated baseline or selected expert as a valid reference candidate.
    """
    meta = inputs["meta"]
    try:
        raw_cosim = run_cosim(
            candidate["code"],
            inputs.get("testbench_code", ""),
            candidate.get("header_code", inputs.get("header_code", "")),
            header_name=(
                candidate.get("header_name")
                or inputs.get("header_name")
                or "kernel.h"
            ),
            top_function=meta.get("hls_top", "workload"),
            part=meta.get("part", DEFAULT_PART),
            clock_ns=meta.get("clock_ns", DEFAULT_CLOCK_NS),
            extra_files=inputs.get("extra_files", []),
            interface_depths=meta.get("cosim_depths", {}),
            golden_output_text=inputs.get("independent_golden_output", ""),
            golden_output_specs=inputs.get("independent_golden_specs", {}),
        )
        return _summarize_test_result(raw_cosim, True)
    except Exception as exc:  # pragma: no cover - defensive Vitis path
        return {
            "status": "tool_error",
            "supported": True,
            "ran": True,
            "success": False,
            "passed": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _validate_gold_reference_uncached(inputs: dict) -> dict:
    meta = inputs["meta"]
    supports_csim = bool(meta.get("supports_csim") and inputs.get("testbench_code"))
    supports_cosim = _generated_cosim_supported(
        meta, inputs.get("testbench_code", "")
    )
    run_reference_cosim = os.getenv("C2HLS_REFERENCE_COSIM", "1").strip().lower() in {
        "1", "true", "yes", "on",
    }
    candidates = _ground_truth_candidates(inputs)
    validation_mode = os.getenv("C2HLS_REFERENCE_VALIDATE_MODE", "all").strip().lower() or "all"
    if validation_mode not in {"all", "selected", "preferred", "baseline", "external", "trusted_external"}:
        validation_mode = "all"
    external_requested = validation_mode == "external"
    use_trusted_external = (
        validation_mode in {"external", "trusted_external"}
        and meta.get("source_repo") in TRUSTED_EXTERNAL_REFERENCE_REPOS
    )
    local_validation_mode = "selected" if validation_mode == "trusted_external" else validation_mode
    validation_scope = (
        "selected" if local_validation_mode in {"selected", "preferred", "baseline"} or use_trusted_external
        else "all"
    )

    if not candidates:
        return {
            "benchmark_ready": False,
            "invalid_reason": "Missing gold HLS workflow code",
            "synthesis": _summarize_synth_result(None),
            "csim": _summarize_test_result(None, supports_csim),
            "cosim": _summarize_test_result(None, supports_cosim),
            "report": {},
            "top_function": meta.get("hls_top", "workload"),
            "workflow": [],
            "selected_variant_name": "",
            "selected_variant_file": "",
            "selected_variant_step": "",
            "selection_reason": "",
            "validation_mode": validation_mode,
            "validation_scope": "none",
            "skipped_candidates": [],
            "reference_source": "none",
            "external_validation": {
                "used": False,
                "reason": "missing gold HLS workflow code",
                "profile_required": True,
            },
        }

    if external_requested and not use_trusted_external:
        return {
            "benchmark_ready": False,
            "invalid_reason": (
                "C2HLS_REFERENCE_VALIDATE_MODE=external requires a trusted "
                f"direct-reference source repo; got {meta.get('source_repo')!r}"
            ),
            "synthesis": _summarize_synth_result(None),
            "csim": _summarize_test_result(None, supports_csim),
            "cosim": _summarize_test_result(None, supports_cosim),
            "report": {},
            "top_function": meta.get("hls_top", "workload"),
            "workflow": [],
            "selected_variant_name": "",
            "selected_variant_file": "",
            "selected_variant_step": "",
            "selection_reason": "",
            "validation_mode": validation_mode,
            "validation_scope": "none",
            "skipped_candidates": [],
            "reference_source": "none",
            "external_validation": {
                "used": False,
                "reason": "source_repo_not_in_trusted_external_reference_set",
                "source_repo": meta.get("source_repo"),
                "profile_required": True,
            },
        }

    validation_candidates = list(candidates)
    skipped_candidates = []
    if validation_scope == "selected":
        preferred_file = _preferred_reference_candidate_file(meta, candidates)
        selected_candidate = None
        if local_validation_mode == "baseline":
            selected_candidate = next(
                (candidate for candidate in candidates if candidate.get("step_name") == "baseline"),
                None,
            )
        elif preferred_file:
            selected_candidate = next(
                (candidate for candidate in candidates if candidate.get("file") == preferred_file),
                None,
            )
        if selected_candidate is None:
            optimized = [candidate for candidate in candidates if candidate.get("step_name") != "baseline"]
            selected_candidate = optimized[-1] if optimized else candidates[-1]
        validation_candidates = [selected_candidate]
        skipped_candidates = [
            {
                "variant_name": candidate.get("variant_name", ""),
                "file": candidate.get("file", ""),
                "step_name": candidate.get("step_name", ""),
                "skip_reason": f"not validated in C2HLS_REFERENCE_VALIDATE_MODE={validation_mode}",
                "profile_required": True,
            }
            for candidate in candidates
            if candidate.get("file") != selected_candidate.get("file")
        ]

    if use_trusted_external:
        workflow = [
            _validate_external_ground_truth_candidate(
                candidate,
                inputs,
                supports_csim,
                supports_cosim,
                candidates,
            )
            for candidate in validation_candidates
        ]
        reference_source = "direct_jsonl"
    else:
        workflow = [
            _validate_ground_truth_candidate(
                candidate,
                inputs,
                supports_csim,
                supports_cosim,
                run_cosim_check=run_reference_cosim,
            )
            for candidate in validation_candidates
        ]
        reference_source = "local_vitis"

    baseline_report = None
    previous_valid_report = None
    for entry in workflow:
        report = entry.get("report", {})
        if entry.get("benchmark_ready") and entry.get("step_name") == "baseline" and baseline_report is None:
            baseline_report = report
        if report and previous_valid_report is not None:
            entry["vs_previous_valid"] = compare_reports(report, previous_valid_report)
        if report and baseline_report is not None and entry.get("step_name") != "baseline":
            entry["vs_baseline"] = compare_reports(report, baseline_report)
        if entry.get("benchmark_ready"):
            previous_valid_report = report

    preferred_file = _preferred_reference_file(meta, workflow)
    selected = None
    selection_reason = ""
    feasibility_required = _feasibility_selection_enabled()
    if feasibility_required:
        feasible_entries = [
            entry for entry in workflow
            if entry.get("benchmark_ready")
            and (entry.get("feasibility") or {}).get("feasible")
        ]
        if feasible_entries:
            selected = min(
                feasible_entries,
                key=lambda entry: C2HLSOrchestrator._best_so_far_score(
                    entry.get("report") or {}
                ),
            )
            selection_reason = (
                "selected fastest correct, device-fitting, target-timing "
                "variant under the matched Vitis target"
            )
    elif preferred_file:
        for entry in workflow:
            if entry.get("file") == preferred_file and entry.get("benchmark_ready"):
                selected = entry
                selection_reason = f"selected preferred validated variant `{preferred_file}`"
                break

    if selected is None and not feasibility_required:
        valid_entries = [entry for entry in workflow if entry.get("benchmark_ready")]
        optimized_entries = [entry for entry in valid_entries if entry.get("step_name") != "baseline"]
        if optimized_entries:
            selected = optimized_entries[-1]
            selection_reason = "selected latest validated optimized variant"
        elif valid_entries:
            selected = valid_entries[-1]
            selection_reason = "selected latest validated baseline-only variant"

    selected_reference_cosim_ran = False
    if (
        selected
        and reference_source == "local_vitis"
        and _env_flag(REFERENCE_COSIM_SELECTED_ONLY_ENV)
        and not run_reference_cosim
        and supports_cosim
    ):
        selected_candidate = next(
            (
                candidate
                for candidate in validation_candidates
                if candidate.get("file") == selected.get("file")
            ),
            None,
        )
        if selected_candidate is not None:
            selected_reference_cosim_ran = True
            selected["cosim"] = _execute_reference_candidate_cosim(
                selected_candidate, inputs
            )

    # The recovery metric needs executed RTL cycles for the designated
    # baseline as well as the selected expert.  Measure that baseline once in
    # common reference preflight; never charge it to a method's five-search
    # synthesis/candidate budget.
    baseline_entry = next(
        (
            entry
            for entry in workflow
            if entry.get("step_name") == "baseline"
            and entry.get("benchmark_ready")
        ),
        None,
    )
    baseline_reference_cosim_ran = bool(
        baseline_entry and (baseline_entry.get("cosim") or {}).get("ran")
    )
    if (
        baseline_entry
        and reference_source == "local_vitis"
        and _env_flag(REFERENCE_COSIM_BASELINE_ENV)
        and not run_reference_cosim
        and supports_cosim
    ):
        if selected and baseline_entry.get("file") == selected.get("file"):
            baseline_entry["cosim"] = selected.get("cosim") or {}
            baseline_reference_cosim_ran = selected_reference_cosim_ran
        else:
            baseline_candidate = next(
                (
                    candidate
                    for candidate in validation_candidates
                    if candidate.get("file") == baseline_entry.get("file")
                ),
                None,
            )
            if baseline_candidate is not None:
                baseline_reference_cosim_ran = True
                baseline_entry["cosim"] = _execute_reference_candidate_cosim(
                    baseline_candidate, inputs
                )

    for entry in workflow:
        entry["selected"] = bool(selected and entry.get("file") == selected.get("file"))

    if not selected:
        last_error = (
            "no correct, device-fitting, target-timing reference variant"
            if feasibility_required and any(
                entry.get("benchmark_ready") for entry in workflow
            )
            else workflow[-1].get("invalid_reason")
            if workflow
            else "Missing valid ground-truth workflow"
        )
        return {
            "benchmark_ready": False,
            "invalid_reason": last_error or "Missing valid ground-truth workflow",
            "top_function": meta.get("hls_top", "workload"),
            "synthesis": workflow[-1]["synthesis"] if workflow else _summarize_synth_result(None),
            "csim": workflow[-1]["csim"] if workflow else _summarize_test_result(None, supports_csim),
            "cosim": workflow[-1]["cosim"] if workflow else _summarize_test_result(None, supports_cosim),
            "report": workflow[-1].get("report", {}) if workflow else {},
            "workflow": workflow,
            "selected_variant_name": "",
            "selected_variant_file": "",
            "selected_variant_step": "",
            "selection_reason": "",
            "validation_mode": validation_mode,
            "validation_scope": validation_scope,
            "skipped_candidates": skipped_candidates,
            "reference_source": reference_source,
            "external_validation": {
                "used": bool(use_trusted_external),
                "reason": last_error or "Missing valid ground-truth workflow",
                "profile_required": True,
            },
            "selected_reference_cosim_measurement_valid": False,
        }

    selection_fallback = (
        selected.get("step_name") == "baseline"
        and any(entry.get("step_name") != "baseline" for entry in workflow)
    )
    selected_reference_cosim_valid = _selected_cosim_measurement_ok(
        selected.get("cosim")
    )
    baseline_reference_cosim_valid = _selected_cosim_measurement_ok(
        baseline_entry.get("cosim") if baseline_entry else None
    )
    selected_measurement_required = bool(
        supports_cosim and _env_flag(REFERENCE_COSIM_SELECTED_ONLY_ENV)
    )
    baseline_measurement_required = bool(
        supports_cosim and _env_flag(REFERENCE_COSIM_BASELINE_ENV)
    )
    missing_measurements = []
    if selected_measurement_required and not selected_reference_cosim_valid:
        missing_measurements.append("selected expert executed RTL cosim cycles")
    if baseline_measurement_required and not baseline_reference_cosim_valid:
        missing_measurements.append("designated baseline executed RTL cosim cycles")
    measurement_ready = not missing_measurements
    return {
        "benchmark_ready": measurement_ready,
        "frontier_synthesis_csim_valid": True,
        "rtl_measurement_pair_valid": bool(
            selected_reference_cosim_valid and baseline_reference_cosim_valid
        ),
        "invalid_reason": (
            "Missing required reference measurement: "
            + "; ".join(missing_measurements)
            if missing_measurements
            else ""
        ),
        "top_function": meta.get("hls_top", "workload"),
        "synthesis": selected["synthesis"],
        "csim": selected["csim"],
        "cosim": selected["cosim"],
        "report": selected.get("report", {}),
        "workflow": workflow,
        "selected_variant_name": selected.get("variant_name", ""),
        "selected_variant_file": selected.get("file", ""),
        "selected_variant_step": selected.get("step_name", ""),
        "selection_reason": selection_reason,
        "selection_fallback": selection_fallback,
        "selected_reference_cosim_policy": {
            "all_candidates": run_reference_cosim,
            "selected_only": _env_flag(REFERENCE_COSIM_SELECTED_ONLY_ENV),
            "selected_only_executed": selected_reference_cosim_ran,
            "baseline_required": _env_flag(REFERENCE_COSIM_BASELINE_ENV),
            "baseline_executed": baseline_reference_cosim_ran,
            "outside_method_search_budget": True,
        },
        "selected_reference_cosim_measurement_valid": selected_reference_cosim_valid,
        "baseline_reference_cosim_measurement_valid": baseline_reference_cosim_valid,
        "baseline_reference": {
            "variant_name": baseline_entry.get("variant_name", "") if baseline_entry else "",
            "file": baseline_entry.get("file", "") if baseline_entry else "",
            "step_name": baseline_entry.get("step_name", "") if baseline_entry else "",
            "report": baseline_entry.get("report", {}) if baseline_entry else {},
            "synthesis": baseline_entry.get("synthesis", {}) if baseline_entry else {},
            "csim": baseline_entry.get("csim", {}) if baseline_entry else {},
            "cosim": baseline_entry.get("cosim", {}) if baseline_entry else {},
        },
        "selection_fallback_reason": (
            "optimized GT variants were unavailable or invalid; selected baseline"
            if selection_fallback else ""
        ),
        "validation_mode": validation_mode,
        "validation_scope": validation_scope,
        "skipped_candidates": skipped_candidates,
        "reference_source": reference_source,
        "external_validation": selected.get("external_validation") if use_trusted_external else {
            "used": False,
            "reason": "local_vitis_reference_validation",
            "profile_required": False,
        },
    }


def validate_gold_reference(inputs: dict) -> dict:
    """Validate gold HLS evidence, reusing an exact-input cache when enabled."""
    cached, cache_provenance = _load_reference_validation_cache(inputs)
    benchmark = inputs.get("bench_name") or inputs.get("meta", {}).get("benchmark") or "benchmark"
    if cached is not None:
        logging.info(
            "Reference validation cache hit for %s: synth=%s csim=%s cosim=%s",
            benchmark,
            cache_provenance.get("synthesis_status"),
            cache_provenance.get("csim_status"),
            cache_provenance.get("cosim_status"),
        )
        return cached

    validation = _validate_gold_reference_uncached(inputs)
    cache_path = _write_reference_validation_cache(inputs, validation)
    if cache_provenance.get("enabled"):
        cache_provenance["written"] = cache_path is not None
        if cache_path is not None:
            cache_provenance["cache_path"] = str(cache_path)
        validation["reference_cache"] = cache_provenance
    return validation


def _reference_cycle_info(reference_validation: Optional[dict]) -> dict:
    """Select the best gold cycle count for the predictive cosim policy.

    Prefer measured RTL cycles from local cosim or a trusted external RTL
    record. Fall back to the selected gold csynth latency only when no runtime
    measurement exists. The source is retained so exported provenance states
    exactly which comparison was made.
    """
    validation = reference_validation or {}

    local_cosim = validation.get("cosim") or {}
    local_cycles = _positive_int(local_cosim.get("kernel_runtime_cycles"))
    if local_cycles is not None and (
        local_cosim.get("passed") is True
        or str(local_cosim.get("status") or "").lower() in {"pass", "passed"}
    ):
        return {
            "cycles": local_cycles,
            "source": "reference_validation.cosim.kernel_runtime_cycles",
            "metric": "rtl_runtime_cycles",
            "target": "vitis.cosim",
        }

    external = validation.get("external_validation") or {}
    external_rtl = external.get("hw_emu") or external.get("rtl_sim") or {}
    payload = external_rtl.get("payload") or {}
    external_cycles = _positive_int(payload.get("kernel_runtime_cycles"))
    if external_cycles is not None and (
        external_rtl.get("passed") is True
        or str(external_rtl.get("status") or "").lower() in {"pass", "passed"}
    ):
        run = external_rtl.get("run") or {}
        target = run.get("target") or "external.rtl_sim"
        return {
            "cycles": external_cycles,
            "source": "reference_validation.external_validation.rtl_sim",
            "metric": "rtl_runtime_cycles",
            "target": target,
            "artifact": external_rtl.get("artifact") or "",
            "line": external_rtl.get("line"),
        }

    report = validation.get("report") or {}
    synth_cycles = _positive_int(
        report.get("latency_cycles_worst") or report.get("latency_cycles")
    )
    if synth_cycles is not None:
        return {
            "cycles": synth_cycles,
            "source": "reference_validation.report.latency_cycles",
            "metric": "csynth_latency_cycles",
            "target": "vitis.csynth",
        }

    return {}


def _looks_like_reference_error(message: str) -> bool:
    if not message:
        return False
    lowered = str(message).lower()
    needles = [
        "gold hls",
        "gold reference",
        "invalid gold reference",
        "ground-truth",
        "ground truth",
        "reference invalid",
        "missing valid ground-truth workflow",
    ]
    return any(needle in lowered for needle in needles)


def _sanitize_test_summary(summary: dict | None) -> dict | None:
    if not isinstance(summary, dict):
        return summary
    cleaned = dict(summary)
    status = cleaned.get("status")
    if status == "passed" or cleaned.get("passed") is True:
        cleaned.pop("error", None)
        cleaned.pop("log_excerpt", None)
    elif cleaned.get("error") == "":
        cleaned.pop("error", None)
    if status in {"not_run", "not_supported"}:
        cleaned.pop("error", None)
        cleaned.pop("log_excerpt", None)
    return cleaned


def _normalize_saved_test_summary(summary: dict | None, available: bool, ran: bool) -> dict | None:
    if summary is None:
        if not available and not ran:
            return None
        return _sanitize_test_summary({
            "status": _test_status(available, ran, False),
            "supported": available,
            "ran": ran,
            "success": False,
            "passed": False,
            "error": "",
        })

    cleaned = dict(summary)
    explicit_status = str(cleaned.get("status") or "").strip().lower()
    supported = bool(cleaned.get("supported", available))
    if available and not supported and not cleaned.get("ran", False):
        supported = True
    ran_value = bool(cleaned.get("ran", False) or ran)
    passed = bool(cleaned.get("passed", False))
    cleaned["supported"] = supported
    cleaned["ran"] = ran_value
    cleaned["success"] = bool(cleaned.get("success", False)) if ran_value else False
    cleaned["passed"] = passed if ran_value else False
    cleaned["status"] = (
        "timeout"
        if explicit_status == "timeout"
        else _test_status(supported, ran_value, cleaned["passed"])
    )
    return _sanitize_test_summary(cleaned)


def _sanitize_comparison_payload(comparison: dict | None,
                                 reference_validation: dict | None = None,
                                 synth_report: dict | None = None) -> dict:
    reference_ready = bool(reference_validation and reference_validation.get("benchmark_ready"))
    ground_truth_report = (reference_validation or {}).get("report", {})

    if reference_validation and not reference_ready:
        return {
            "success": False,
            "valid_reference": False,
            "invalid_reference": True,
            "error": reference_validation.get("invalid_reason", "Invalid gold reference"),
        }

    if reference_ready and synth_report:
        return {
            "success": True,
            "valid_reference": True,
            "invalid_reference": False,
            "generated_report": synth_report,
            "ground_truth_report": ground_truth_report,
            "comparison": compare_reports(synth_report, ground_truth_report),
        }

    cleaned = dict(comparison or {})
    if reference_ready:
        cleaned["valid_reference"] = True
        cleaned["invalid_reference"] = False
        if ground_truth_report:
            cleaned["ground_truth_report"] = ground_truth_report
        if synth_report:
            cleaned["generated_report"] = synth_report
        if cleaned.get("error") and _looks_like_reference_error(cleaned.get("error")):
            cleaned.pop("error", None)
    if cleaned.get("success") is True:
        cleaned.pop("error", None)
    elif cleaned.get("error") == "":
        cleaned.pop("error", None)
    return cleaned


def _sanitize_attempt_entries(entries, overall_success: bool = False) -> list:
    if not isinstance(entries, list):
        return []

    cleaned_entries = []
    for entry in entries:
        if not isinstance(entry, dict):
            cleaned_entries.append(entry)
            continue

        item = dict(entry)
        entry_success = item.get("success")
        error = item.get("error")

        if item.get("csim") is not None:
            item["csim"] = _sanitize_test_summary(item.get("csim"))
        if item.get("cosim") is not None:
            item["cosim"] = _sanitize_test_summary(item.get("cosim"))

        if overall_success and entry_success is False and error:
            item["superseded_by_success"] = True
            item["attempt_error"] = error
            item.pop("error", None)
        elif entry_success is True or error == "":
            item.pop("error", None)

        if "comparison" in item and isinstance(item["comparison"], dict):
            comp = dict(item["comparison"])
            if comp.get("success") is True or comp.get("error") == "":
                comp.pop("error", None)
            item["comparison"] = comp

        cleaned_entries.append(item)
    return cleaned_entries


def sanitize_saved_result_record(result: dict, reference_validation: dict | None = None) -> dict:
    output = dict(result)
    if reference_validation is None and isinstance(output.get("reference_validation"), dict):
        reference_validation = output.get("reference_validation")
    overall_success = bool(output.get("success"))
    explicit_gt_valid = output.get("ground_truth_status") == "valid"
    coverage = output.get("coverage") or {}

    if output.get("csim") is not None:
        output["csim"] = _normalize_saved_test_summary(
            output.get("csim"),
            bool(coverage.get("generated_csim_available", False)),
            bool(coverage.get("generated_csim_ran", False)),
        )
    if output.get("cosim") is not None:
        output["cosim"] = _normalize_saved_test_summary(
            output.get("cosim"),
            bool(coverage.get("generated_cosim_available", False)),
            bool(coverage.get("generated_cosim_ran", False)),
        )

    for key in ["csim", "cosim", "baseline_csim", "baseline_cosim"]:
        if key in output:
            output[key] = _sanitize_test_summary(output.get(key))

    if "turn_history" in output:
        output["turn_history"] = _sanitize_attempt_entries(output.get("turn_history"), overall_success)
    if "optimization_history" in output:
        output["optimization_history"] = _sanitize_attempt_entries(output.get("optimization_history"), overall_success)
    if "generated_step_history" in output:
        output["generated_step_history"] = _sanitize_attempt_entries(output.get("generated_step_history"), overall_success)
    if "steps" in output:
        output["steps"] = _sanitize_attempt_entries(output.get("steps"), overall_success)

    quality_repair = output.get("quality_repair")
    if isinstance(quality_repair, dict):
        cleaned_quality = dict(quality_repair)
        cleaned_quality["attempts"] = _sanitize_attempt_entries(
            cleaned_quality.get("attempts", []),
            bool(cleaned_quality.get("applied")) or overall_success,
        )
        output["quality_repair"] = cleaned_quality

    synth_report = output.get("synth_report")
    if not synth_report:
        synth_report = ((output.get("comparison") or {}).get("generated_report")) or None

    output["comparison"] = _sanitize_comparison_payload(
        output.get("comparison"),
        reference_validation,
        synth_report,
    )

    if reference_validation is not None:
        normalized_reference = dict(reference_validation)
        normalized_reference["csim"] = _normalize_saved_test_summary(
            normalized_reference.get("csim"),
            bool(coverage.get("ground_truth_csim_available", False)),
            bool(coverage.get("ground_truth_csim_ran", False)),
        )
        normalized_reference["cosim"] = _normalize_saved_test_summary(
            normalized_reference.get("cosim"),
            bool(coverage.get("ground_truth_cosim_available", False)),
            bool(coverage.get("ground_truth_cosim_ran", False)),
        )
        workflow = []
        for stage in normalized_reference.get("workflow", []) or []:
            if not isinstance(stage, dict):
                workflow.append(stage)
                continue
            stage_copy = dict(stage)
            is_selected_stage = bool(stage_copy.get("selected")) or (
                stage_copy.get("file") == normalized_reference.get("selected_variant_file")
            )
            if is_selected_stage:
                stage_copy["csim"] = normalized_reference.get("csim")
                stage_copy["cosim"] = normalized_reference.get("cosim")
            else:
                stage_copy["csim"] = _sanitize_test_summary(stage_copy.get("csim"))
                stage_copy["cosim"] = _sanitize_test_summary(stage_copy.get("cosim"))
            workflow.append(stage_copy)
        normalized_reference["workflow"] = workflow

        output["reference_validation"] = normalized_reference
        output["ground_truth_report"] = normalized_reference.get("report", {})
        output["ground_truth_status"] = "valid" if normalized_reference.get("benchmark_ready") else "invalid"
        output["baseline_status"] = normalized_reference.get("synthesis", {}).get("status", "failed")
        output["invalid_reference_reason"] = "" if normalized_reference.get("benchmark_ready") else normalized_reference.get("invalid_reason", "")
        if "ground_truth_workflow" in output:
            output["ground_truth_workflow"] = workflow
        reference_validation = normalized_reference

    if output.get("generated_status") == "passed":
        output.pop("error", None)
    elif reference_validation is None and explicit_gt_valid and _looks_like_reference_error(output.get("error")):
        output.pop("error", None)
    elif reference_validation is not None and reference_validation.get("benchmark_ready"):
        if _looks_like_reference_error(output.get("error")):
            output.pop("error", None)
    elif reference_validation is not None and not reference_validation.get("benchmark_ready"):
        output["error"] = reference_validation.get("invalid_reason", output.get("error", "Invalid gold reference"))
    elif output.get("error") == "":
        output.pop("error", None)

    comparison = output.get("comparison")
    if isinstance(comparison, dict) and comparison.get("invalid_reference"):
        invalid_reason = (
            output.get("invalid_reference_reason")
            or output.get("error")
            or comparison.get("error")
        )
        if invalid_reason:
            output["invalid_reference_reason"] = invalid_reason
            output["comparison"]["error"] = invalid_reason
    if isinstance(output.get("comparison"), dict) and explicit_gt_valid:
        output["comparison"]["valid_reference"] = True
        output["comparison"]["invalid_reference"] = False
        if _looks_like_reference_error(output["comparison"].get("error")):
            output["comparison"].pop("error", None)
        output["invalid_reference_reason"] = ""

    if isinstance(output.get("comparison"), dict) and output["comparison"].get("error") == "":
        output["comparison"].pop("error", None)

    coverage = output.get("coverage") or {}
    output["csim_status"] = {
        "ground_truth": _summary_status(
            (output.get("reference_validation") or {}).get("csim"),
            bool(coverage.get("ground_truth_csim_available", False)),
        ),
        "generated": _summary_status(
            output.get("csim"),
            bool(coverage.get("generated_csim_available", False)),
        ),
    }
    output["cosim_status"] = {
        "ground_truth": _summary_status(
            (output.get("reference_validation") or {}).get("cosim"),
            bool(coverage.get("ground_truth_cosim_available", False)),
        ),
        "generated": _summary_status(
            output.get("cosim"),
            bool(coverage.get("generated_cosim_available", False)),
        ),
    }

    return output


def _build_run_attribution(orchestrator, meta: dict) -> dict:
    """Capture which model + tool config produced this run, for downstream
    consumers (JSONL exporter, evals, side-by-side comparisons)."""
    import hls_eval
    llm_summary = getattr(orchestrator, "_llm_usage_summary", lambda: {})()
    decoding_events = [
        {
            "provider": event.get("provider"),
            "model": event.get("model"),
            **dict(event.get("decoding") or {}),
        }
        for event in (llm_summary.get("events") or [])
    ]
    try:
        configured_decoding = orchestrator._configured_decoding()
    except (AttributeError, TypeError, ValueError):
        configured_decoding = {}
    return {
        "model": getattr(orchestrator, "gpt_model", None),
        "model_revision": os.getenv("C2HLS_MODEL_REVISION") or None,
        "model_translator":     os.getenv(TRANSLATOR_MODEL_ENV)     or getattr(orchestrator, "gpt_model", None),
        "model_synthesis":      os.getenv(SYNTHESIS_MODEL_ENV)      or getattr(orchestrator, "gpt_model", None),
        "model_quality_repair": os.getenv(QUALITY_REPAIR_MODEL_ENV) or getattr(orchestrator, "gpt_model", None),
        "vitis_version": _vitis_version(),
        "flow_target": getattr(hls_eval, "DEFAULT_FLOW_TARGET", "vitis"),
        "part": meta.get("part", DEFAULT_PART),
        "clock_ns": meta.get("clock_ns", DEFAULT_CLOCK_NS),
        "skill_mode": os.getenv("C2HLS_SKILL_MODE"),
        "skill_prompts": os.getenv("C2HLS_FORCE_SKILL_PROMPTS", "").strip().lower()
                         in {"1", "true", "yes", "on"},
        "skill_library_provenance": getattr(orchestrator, "skill_library_provenance", {}) or {},
        "skill_library_frozen": _env_flag(SKILL_LIBRARY_FROZEN_ENV),
        "skill_control_enabled": _skill_library_control_enabled(),
        "skill_online_updates_enabled": _skill_updates_enabled(),
        "skill_persistence_enabled": (
            not _env_flag(SKILL_LIBRARY_FROZEN_ENV)
            and not _reference_blind_enabled()
            and _env_flag("C2HLS_SKILL_LIBRARY_PERSIST", "1")
        ),
        "reference_blind": _reference_blind_enabled(),
        "ground_truth_control_enabled": _ground_truth_control_enabled(),
        "cosim_selected_only": _cosim_selected_only(),
        "feasibility_selection": _feasibility_selection_enabled(),
        "decoding": {
            "configured": configured_decoding,
            "effective": {
                "per_call": decoding_events,
                "all_calls_reported": len(decoding_events) == llm_summary.get("calls", 0),
            },
        },
        "llm_usage": llm_summary,
        "synthesis_evaluations": getattr(
            orchestrator, "_synthesis_evaluation_summary", lambda: {}
        )(),
        "cosim_skip_slower_than_gold": os.getenv(
            COSIM_SKIP_SLOWER_THAN_GOLD_ENV, "0"
        ).strip().lower() in {"1", "true", "yes", "on"},
        "cosim_skip_gold_ratio": _cosim_skip_gold_ratio(),
        "cosim_reference_cycle_info": getattr(
            orchestrator, "cosim_reference_cycle_info", {}
        ) or {},
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def _json_artifact_provenance(path: "str | Path") -> dict:
    path = Path(path)
    out = {
        "path": str(path),
        "exists": path.exists(),
    }
    if not path.exists():
        return out
    try:
        raw = path.read_bytes()
    except OSError as exc:
        out["error"] = str(exc)
        return out
    out["bytes"] = len(raw)
    out["sha256"] = hashlib.sha256(raw).hexdigest()
    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return out
    if isinstance(data, dict):
        skills = data.get("skills")
        out["schema"] = data.get("schema") or data.get("schema_version")
        out["saved_at"] = data.get("saved_at")
        if isinstance(skills, list):
            out["skill_count"] = len(skills)
    return {k: v for k, v in out.items() if v not in (None, "")}


def _skill_library_provenance(skill_library=None) -> dict:
    store_path = getattr(skill_library, "store_path", None) or (REPO_ROOT / "skills" / "skills.json")
    package_path = REPO_ROOT / "hls_full_optimization_skills_schema_1_1_package" / "skills.json"
    exact_frozen = bool(getattr(skill_library, "exact_frozen_snapshot", False))
    payload = {
        "loaded": skill_library is not None,
        "store": _json_artifact_provenance(store_path),
        "source_mode": "exact_frozen_snapshot" if exact_frozen else "default_merged_library",
        "package_merged": not exact_frozen,
    }
    if not exact_frozen:
        payload["package"] = _json_artifact_provenance(package_path)
    if skill_library is not None:
        try:
            skills = skill_library.all()
        except Exception:  # pragma: no cover - defensive telemetry only
            skills = []
        payload["loaded_skill_count"] = len(skills)
        payload["loaded_skill_ids"] = [getattr(sk, "id", "") for sk in skills if getattr(sk, "id", "")]
    return payload


def _vitis_version() -> str:
    explicit = os.getenv("C2HLS_VITIS_VERSION")
    if explicit:
        return explicit
    for env in ("C2HLS_VITIS_SETTINGS", "XILINX_VITIS", "XILINX_HLS"):
        for token in os.getenv(env, "").split("/"):
            if token.count(".") == 1 and token and token[0].isdigit():
                return token
    return "unknown"


def _finalize_singleshot_results(bench_name: str, meta: dict, success: bool,
                                 base_results: dict, reference_validation: dict,
                                 orchestrator=None) -> dict:
    output = dict(base_results)
    output["benchmark"] = bench_name
    output["success"] = success
    if orchestrator is not None:
        output["run"] = _build_run_attribution(orchestrator, meta)
    output["reference_validation"] = reference_validation
    output["ground_truth_report"] = reference_validation.get("report", {})
    output["ground_truth_status"] = "valid" if reference_validation.get("benchmark_ready") else "invalid"
    output["baseline_status"] = reference_validation.get("synthesis", {}).get("status", "failed")
    output["invalid_reference_reason"] = reference_validation.get("invalid_reason", "")
    output["ground_truth_variant"] = {
        "name": reference_validation.get("selected_variant_name", ""),
        "file": reference_validation.get("selected_variant_file", ""),
        "step": reference_validation.get("selected_variant_step", ""),
        "selection_reason": reference_validation.get("selection_reason", ""),
        "fallback_used": reference_validation.get("selection_fallback", False),
        "fallback_reason": reference_validation.get("selection_fallback_reason", ""),
    }
    output["ground_truth_workflow"] = reference_validation.get("workflow", [])
    output["optimization_history"] = output.get("turn_history", [])

    selected_measurement_required = bool(
        _cosim_selected_only() and _env_flag(FORCE_SELECTED_COSIM_ENV)
    )
    generated_measurement_ok = (
        not selected_measurement_required
        or _selected_cosim_measurement_ok(output.get("cosim"))
    )
    if (
        output.get("phase") == "complete"
        and output.get("synth_report")
        and generated_measurement_ok
    ):
        output["generated_status"] = "passed"
    else:
        output["generated_status"] = "failed"

    generated_csim = output.get("csim")
    generated_cosim = output.get("cosim")
    generated_csim_available = bool(meta.get("supports_csim") and meta.get("testbench_file"))
    generated_cosim_available = bool(
        meta.get("testbench_file")
        and (
            meta.get("supports_cosim")
            or (_cosim_selected_only() and _env_flag(FORCE_SELECTED_COSIM_ENV))
        )
    )
    output["csim_status"] = {
        "ground_truth": _summary_status(
            reference_validation.get("csim"),
            bool(meta.get("supports_csim") and meta.get("testbench_file")),
        ),
        "generated": _summary_status(generated_csim, generated_csim_available),
    }
    output["cosim_status"] = {
        "ground_truth": _summary_status(
            reference_validation.get("cosim"),
            generated_cosim_available,
        ),
        "generated": _summary_status(generated_cosim, generated_cosim_available),
    }
    output["coverage"] = _build_coverage(meta, reference_validation, generated_csim, generated_cosim)

    if not reference_validation.get("benchmark_ready"):
        output["comparison"] = {
            "success": False,
            "valid_reference": False,
            "invalid_reference": True,
            "error": reference_validation.get("invalid_reason", "Invalid gold reference"),
        }

    return sanitize_saved_result_record(output, reference_validation)


def _parse_variant_dir_name(name: str) -> dict:
    match = re.match(r"^(.+)_(\d+)_(.+)$", name or "")
    if not match:
        return {
            "index": None,
            "step": _normalize_variant_step_name(name),
            "name": name,
        }
    return {
        "index": int(match.group(2)),
        "step": _normalize_variant_step_name(match.group(3)),
        "name": name,
    }


def _rodinia_variant_parents(bench_name: str) -> list[tuple[Path, str]]:
    roots = [
        (Path("/home/luo00466/rodinia-hls-nova/Benchmarks"), "rodinia-hls-nova"),
        (Path("/home/luo00466/rodinia-hls/Benchmarks"), "rodinia-hls"),
    ]
    parents: list[tuple[Path, str]] = []
    for root, source_repo in roots:
        if not root.is_dir():
            continue
        for parent in (
            root / bench_name,
            root / "cfd" / bench_name,
            root / "leukocyte" / bench_name,
        ):
            if parent.is_dir():
                parents.append((parent, source_repo))
    return parents


def _resolve_rodinia_variant(bench_name: str,
                             requested_step: "str | None") -> tuple[dict | None, str]:
    """Find the upstream variant matching the generated/selected final step.

    Returns (variant_info, error). A missing requested variant is reported as
    an explicit skip instead of falling back to baseline.
    """
    parents = _rodinia_variant_parents(bench_name)
    if not parents:
        return None, f"no rodinia-hls-nova/rodinia-hls counterpart for {bench_name}"

    kernel_basename = bench_name
    variant_prefixes = {bench_name}
    meta_path = Path(__file__).resolve().parent / "benchmarks" / bench_name / "metadata.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
            kernel_file = meta.get("kernel_file") or ""
            if kernel_file:
                kernel_basename = Path(kernel_file).stem
                variant_prefixes.add(kernel_basename)
            baseline_variant = meta.get("baseline_variant") or ""
            match = re.match(r"^(.+)_0_.+$", baseline_variant)
            if match:
                variant_prefixes.add(match.group(1))
        except Exception:
            pass

    variants = []
    for parent, source_repo in parents:
        for child in sorted(parent.iterdir()):
            if not child.is_dir() or not any(
                child.name.startswith(f"{prefix}_") for prefix in variant_prefixes
            ):
                continue
            parsed = _parse_variant_dir_name(child.name)
            variants.append({
                "bench_dir": str(child),
                "kernel_basename": kernel_basename,
                "variant_name": child.name,
                "variant_index": parsed["index"],
                "variant_step": parsed["step"],
                "source_repo": source_repo,
            })

    if not variants:
        return None, f"no upstream variants found for {bench_name}"

    if not requested_step:
        return None, f"no final/selected variant step was provided for {bench_name}"
    normalized_step = _normalize_variant_step_name(requested_step)

    matches = [v for v in variants if v["variant_step"] == normalized_step]
    if not matches:
        available = ", ".join(v["variant_name"] for v in variants)
        return None, (
            f"requested hw_emu variant step '{requested_step}' for {bench_name} "
            f"does not match any upstream variant; available: {available}"
        )
    matches.sort(key=lambda v: (v["source_repo"] != "rodinia-hls-nova",
                               v["variant_index"] if v["variant_index"] is not None else 9999))
    return matches[0], ""


def _infer_final_variant_step(results: dict) -> str:
    if results.get("phase") == "multistep":
        promotion = results.get("best_so_far_promotion") or {}
        promoted_step = promotion.get("from_step_name")
        if promotion.get("promoted") and promoted_step:
            return promoted_step
        for step in reversed(results.get("steps", [])):
            if step.get("success") and step.get("step_name"):
                return step.get("step_name")
        return "baseline"
    return ""


def _wide_abi_markers(code: str) -> list[str]:
    markers = []
    checks = {
        "memcpy_wide_bus": "memcpy_wide_bus",
        "MARS_WIDE_BUS_TYPE": "MARS_WIDE_BUS_TYPE",
        "common/mc.h": "common/mc.h",
        "ap_uint_512": "ap_uint<512",
        "ap_uint_large_bus": "ap_uint<LARGE_BUS",
    }
    for name, token in checks.items():
        if token in (code or ""):
            markers.append(name)
    return markers


def _maybe_run_hw_emu_final(orchestrator, results: dict, bench_name: str,
                            timeout: int = 7200,
                            variant_step: "str | None" = None) -> None:
    """Optional post-completion step: run nova `make check TARGET=hw_emu` on
    the orchestrator's final HLS code. Authoritative kernel-cycle measurement
    via XSIM RTL simulation, complementing the predictive csynth latency.

    Adds `results['hw_emu']` with `{kernel_runtime_us, kernel_runtime_cycles,
    passed, success, error, work_dir}`. Mutates `results` in place; no return.

    Skipped (no-op) when:
      - C2HLS_HW_EMU_FINAL is not set / unset to "0"
      - bench has no upstream rodinia-hls-nova/rodinia-hls counterpart
      - requested final/selected variant cannot be staged exactly
      - orchestrator has no hls_code (failed before producing a kernel)
    """
    if os.getenv("C2HLS_HW_EMU_FINAL", "").lower() not in ("1", "true", "yes"):
        return
    if not orchestrator.hls_code:
        results["hw_emu"] = {
            "ran": False,
            "skip_reason": "no final HLS code available for hw_emu",
            "profile_required": True,
        }
        return
    try:
        timeout = int(os.getenv("C2HLS_HW_EMU_TIMEOUT", str(timeout)))
    except (TypeError, ValueError):
        results.setdefault("hw_emu_warnings", []).append({
            "kind": "invalid_hw_emu_timeout_env",
            "env": "C2HLS_HW_EMU_TIMEOUT",
            "value": os.getenv("C2HLS_HW_EMU_TIMEOUT"),
            "fallback_timeout_sec": timeout,
            "profile_required": True,
        })
    requested_step = variant_step or _infer_final_variant_step(results)
    variant, variant_error = _resolve_rodinia_variant(bench_name, requested_step)
    if not variant:
        results["hw_emu"] = {
            "ran": False,
            "skip_reason": variant_error,
            "profile_required": True,
            "requested_variant_step": requested_step,
        }
        return
    wide_markers = _wide_abi_markers(orchestrator.hls_code)
    allow_wide_abi = os.getenv("C2HLS_ALLOW_WIDE_ABI", "").lower() in ("1", "true", "yes")
    if wide_markers and not allow_wide_abi:
        results["hw_emu"] = {
            "ran": False,
            "skip_reason": (
                "generated kernel uses wide-bus ABI/helper markers but "
                "C2HLS_ALLOW_WIDE_ABI is not enabled; refusing to stage into "
                "a possibly narrow host/testbench contract"
            ),
            "profile_required": True,
            "requested_variant_step": requested_step,
            "variant_step": variant["variant_step"],
            "variant_name": variant["variant_name"],
            "variant_index": variant["variant_index"],
            "source_repo": variant["source_repo"],
            "interface_contract": "narrow_safe_default",
            "interface_mismatch": True,
            "wide_abi_markers": wide_markers,
        }
        return
    logging.info("[hw_emu_final] Running on %s step=%s via %s",
                 bench_name, requested_step, variant["bench_dir"])
    import hls_eval as _hls_eval
    hw = _hls_eval.run_hw_emu_via_nova(
        variant["bench_dir"],
        orchestrator.hls_code,
        kernel_basename=variant["kernel_basename"],
        timeout=timeout,
    )
    # Strip log to keep results.json from blowing up.
    results["hw_emu"] = {k: v for k, v in hw.items() if k != "log"}
    results["hw_emu"].update({
        "nova_bench_dir": variant["bench_dir"],
        "kernel_basename": variant["kernel_basename"],
        "variant_step": variant["variant_step"],
        "variant_name": variant["variant_name"],
        "variant_index": variant["variant_index"],
        "source_repo": variant["source_repo"],
        "requested_variant_step": requested_step,
    })
    if hw.get("kernel_runtime_us") is not None:
        logging.info("[hw_emu_final] kernel_runtime_us=%.3f cycles=%s passed=%s",
                     hw["kernel_runtime_us"], hw["kernel_runtime_cycles"], hw["passed"])
    else:
        logging.warning("[hw_emu_final] no kernel runtime: %s", hw.get("error", ""))


def run_benchmark(bench_dir: str, output_dir: str = None,
                  gpt_model: str = DEFAULT_MODEL_ID,
                  turns_limitation: int = 3,
                  quality_repair_turns: int = DEFAULT_QUALITY_REPAIR_TURNS) -> dict:
    inputs = _load_benchmark_inputs(bench_dir)
    bench_name = inputs["bench_name"]

    if output_dir is None:
        output_dir = _default_output_dir(bench_dir, bench_name)
    output_dir = str(output_dir)

    independent_golden = _prepare_independent_golden(inputs)
    inputs["independent_golden_output"] = independent_golden.get("output", "")
    inputs["independent_golden_specs"] = independent_golden.get("specs", {})
    inputs["independent_golden_provenance"] = independent_golden.get("provenance", {})
    if not independent_golden.get("success"):
        results = {
            "benchmark": bench_name,
            "success": False,
            "phase": "independent_golden",
            "error": independent_golden.get("error") or "independent golden invalid",
            "correctness_status": "invalid_oracle",
            "independent_golden": independent_golden.get("provenance", {}),
        }
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, f"{bench_name}_results.json"), "w") as f:
            json.dump(results, f, indent=2, default=str)
        return results

    orchestrator = C2HLSOrchestrator(
        gpt_model=gpt_model,
        turns_limitation=turns_limitation,
        quality_repair_turns=quality_repair_turns,
    )
    orchestrator.testbench_code = inputs.get("testbench_code", "")
    orchestrator.configure_benchmark(
        extra_files=inputs.get("extra_files", []),
        translated_hls_top=inputs["meta"].get("translated_hls_top", "workload"),
        reference_hls_top=inputs["meta"].get("hls_top", "workload"),
        part=inputs["meta"].get("part", DEFAULT_PART),
        clock_ns=inputs["meta"].get("clock_ns", DEFAULT_CLOCK_NS),
        supports_cosim=_generated_cosim_supported(
            inputs["meta"], inputs.get("testbench_code", "")
        ),
        cosim_depths=inputs["meta"].get("cosim_depths", {}),
        benchmark_name=bench_name,
        benchmark_context=inputs.get("benchmark_context", ""),
        independent_golden_output=inputs.get("independent_golden_output", ""),
        independent_golden_specs=inputs.get("independent_golden_specs", {}),
        independent_golden_provenance=inputs.get("independent_golden_provenance", {}),
    )

    reference_validation = validate_gold_reference(inputs)

    if not reference_validation.get("benchmark_ready"):
        results = _finalize_singleshot_results(
            bench_name,
            inputs["meta"],
            False,
            {
                "phase": "reference",
                "error": reference_validation.get("invalid_reason") or "Gold HLS reference invalid",
            },
            reference_validation,
            orchestrator=orchestrator,
        )
        orchestrator.save_results(output_dir, bench_name)
        with open(os.path.join(output_dir, f"{bench_name}_results.json"), "w") as f:
            json.dump(results, f, indent=2, default=str)
        return results

    orchestrator.cosim_reference_cycle_info = (
        _reference_cycle_info(reference_validation)
        if _ground_truth_control_enabled() else {}
    )
    if orchestrator.cosim_reference_cycle_info:
        logging.info(
            "Gold-relative cosim precheck reference for %s: %s cycles (%s)",
            bench_name,
            orchestrator.cosim_reference_cycle_info.get("cycles"),
            orchestrator.cosim_reference_cycle_info.get("source"),
        )

    success, results = orchestrator.run(
        inputs["c_code"],
        inputs["header_code"],
        inputs["header_name"] or "kernel.h",
        (
            reference_validation.get("report", {})
            if _ground_truth_control_enabled() else {}
        ),
    )

    # Optional: run nova hw_emu on the final kernel for an authoritative
    # cycle measurement. Gated on C2HLS_HW_EMU_FINAL=1; ~30 min per bench.
    _maybe_run_hw_emu_final(
        orchestrator,
        results,
        bench_name,
        variant_step=reference_validation.get("selected_variant_step"),
    )

    orchestrator.save_results(output_dir, bench_name)
    results = _finalize_singleshot_results(
        bench_name,
        inputs["meta"],
        success,
        results,
        reference_validation,
        orchestrator=orchestrator,
    )
    if orchestrator.synth_report and reference_validation.get("report"):
        results["offline_reference_comparison"] = compare_reports(
            orchestrator.synth_report,
            reference_validation.get("report", {}),
        )
    results["independent_golden"] = independent_golden.get("provenance", {})
    results["cosim_reference_cycle_info"] = orchestrator.cosim_reference_cycle_info

    with open(os.path.join(output_dir, f"{bench_name}_results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)

    return results


def run_benchmark_multistep(bench_dir: str, output_dir: str = None,
                            gpt_model: str = DEFAULT_MODEL_ID,
                            turns_limitation: int = 3,
                            steps: list = None,
                            quality_repair_turns: int = DEFAULT_QUALITY_REPAIR_TURNS) -> dict:
    total_started = time.monotonic()
    inputs = _load_benchmark_inputs(bench_dir)
    bench_name = inputs["bench_name"]

    if output_dir is None:
        output_dir = _default_output_dir(bench_dir, bench_name, multistep=True)
    output_dir = str(output_dir)

    independent_golden = _prepare_independent_golden(inputs)
    inputs["independent_golden_output"] = independent_golden.get("output", "")
    inputs["independent_golden_specs"] = independent_golden.get("specs", {})
    inputs["independent_golden_provenance"] = independent_golden.get("provenance", {})
    if not independent_golden.get("success"):
        preflight_elapsed = time.monotonic() - total_started
        return {
            "benchmark": bench_name,
            "success": False,
            "phase": "independent_golden",
            "error": independent_golden.get("error") or "independent golden invalid",
            "correctness_status": "invalid_oracle",
            "independent_golden": independent_golden.get("provenance", {}),
            "preflight_elapsed_seconds": preflight_elapsed,
            "search_elapsed_seconds": 0.0,
            "total_elapsed_seconds": preflight_elapsed,
        }

    available_gt = (
        set(inputs["gt_variants"].keys())
        if _ground_truth_control_enabled() else set()
    )
    if steps is None:
        env_strategy = os.getenv("C2HLS_STRATEGY", "").strip().lower()
        if env_strategy in ("combo", "combo_full"):
            from prompt_c2hls import COMBO_FULL_STEPS
            steps = list(COMBO_FULL_STEPS)
        elif env_strategy == "flash":
            from prompt_c2hls import FLASH_STEPS
            steps = list(FLASH_STEPS)
        elif env_strategy == "combo_progressive":
            from prompt_c2hls import COMBO_PROGRESSIVE_STEPS
            steps = list(COMBO_PROGRESSIVE_STEPS)
        else:
            steps = [step for step in DEFAULT_OPT_STEPS if step in available_gt or step in OPTIMIZATION_PROMPTS]

    logging.info(
        "Benchmark %s: running steps %s (oracle control: %s)",
        bench_name,
        steps,
        _ground_truth_control_enabled(),
    )

    orchestrator = C2HLSOrchestrator(
        gpt_model=gpt_model,
        turns_limitation=turns_limitation,
        quality_repair_turns=quality_repair_turns,
    )
    orchestrator.testbench_code = inputs.get("testbench_code", "")
    orchestrator.configure_benchmark(
        extra_files=inputs.get("extra_files", []),
        translated_hls_top=inputs["meta"].get("translated_hls_top", "workload"),
        reference_hls_top=inputs["meta"].get("hls_top", "workload"),
        part=inputs["meta"].get("part", DEFAULT_PART),
        clock_ns=inputs["meta"].get("clock_ns", DEFAULT_CLOCK_NS),
        supports_cosim=_generated_cosim_supported(
            inputs["meta"], inputs.get("testbench_code", "")
        ),
        cosim_depths=inputs["meta"].get("cosim_depths", {}),
        benchmark_name=bench_name,
        benchmark_context=inputs.get("benchmark_context", ""),
        independent_golden_output=inputs.get("independent_golden_output", ""),
        independent_golden_specs=inputs.get("independent_golden_specs", {}),
        independent_golden_provenance=inputs.get("independent_golden_provenance", {}),
    )

    reference_validation = validate_gold_reference(inputs)
    if not reference_validation.get("benchmark_ready"):
        preflight_elapsed = time.monotonic() - total_started
        return {
            "benchmark": bench_name,
            "success": False,
            "phase": "reference",
            "error": reference_validation.get("invalid_reason") or "Gold HLS reference invalid",
            "reference_validation": reference_validation,
            "ground_truth_status": "invalid",
            "baseline_status": reference_validation.get("synthesis", {}).get("status", "failed"),
            "invalid_reference_reason": reference_validation.get("invalid_reason", ""),
            "preflight_elapsed_seconds": preflight_elapsed,
            "search_elapsed_seconds": 0.0,
            "total_elapsed_seconds": preflight_elapsed,
        }

    orchestrator.cosim_reference_cycle_info = (
        _reference_cycle_info(reference_validation)
        if _ground_truth_control_enabled() else {}
    )
    if orchestrator.cosim_reference_cycle_info:
        logging.info(
            "Gold-relative cosim precheck reference for %s: %s cycles (%s)",
            bench_name,
            orchestrator.cosim_reference_cycle_info.get("cycles"),
            orchestrator.cosim_reference_cycle_info.get("source"),
        )

    if (
        _ground_truth_control_enabled()
        and reference_validation.get("reference_source") == "direct_jsonl"
    ):
        external_step_reports = _trusted_external_gt_step_reports(inputs)
        orchestrator._gt_step_reports.update(external_step_reports)
        if "baseline" in external_step_reports:
            orchestrator._gt_baseline_report = dict(external_step_reports["baseline"])
        logging.info(
            "Loaded %d trusted external GT step reports for %s",
            len(external_step_reports),
            bench_name,
        )

    preflight_elapsed = time.monotonic() - total_started
    search_started = time.monotonic()
    success, results = orchestrator.run_multistep(
        inputs["c_code"],
        inputs["header_code"],
        inputs["header_name"] or "kernel.h",
        steps=steps,
        gt_variants=(inputs["gt_variants"] if _ground_truth_control_enabled() else {}),
        gt_variant_headers=(
            inputs.get("gt_variant_headers", {})
            if _ground_truth_control_enabled() else {}
        ),
        reference_report=(
            reference_validation.get("report", {})
            if _ground_truth_control_enabled() else {}
        ),
    )
    orchestrator._seal_candidate_event_stream(
        reason="controller terminated before completing candidate evaluation"
    )
    results["llm_usage"] = orchestrator._llm_usage_summary()
    results["synthesis_evaluations"] = orchestrator._synthesis_evaluation_summary()
    results["selected_winner_cosim_count"] = getattr(
        orchestrator, "selected_winner_cosim_count", 0
    )
    results["total_synthesis_calls"] = orchestrator._total_synthesis_calls()
    results["selected_code_sha256"] = getattr(
        orchestrator, "selected_code_sha256", None
    )
    results["cosim_target_code_sha256"] = getattr(
        orchestrator, "cosim_target_code_sha256", None
    )
    search_elapsed = time.monotonic() - search_started

    # Optional hw_emu on the final-step kernel for authoritative cycle count.
    post_route_started = time.monotonic()
    _maybe_run_hw_emu_final(orchestrator, results, bench_name)
    post_route_elapsed = time.monotonic() - post_route_started

    results["benchmark"] = bench_name
    results["success"] = success
    results["run"] = _build_run_attribution(orchestrator, inputs["meta"])
    results["reference_validation"] = reference_validation
    results["ground_truth_status"] = "valid"
    results["baseline_status"] = reference_validation.get("synthesis", {}).get("status", "failed")
    results["invalid_reference_reason"] = ""
    results["cosim_reference_cycle_info"] = orchestrator.cosim_reference_cycle_info
    results["ground_truth_variant"] = {
        "name": reference_validation.get("selected_variant_name", ""),
        "file": reference_validation.get("selected_variant_file", ""),
        "step": reference_validation.get("selected_variant_step", ""),
        "selection_reason": reference_validation.get("selection_reason", ""),
        "fallback_used": reference_validation.get("selection_fallback", False),
        "fallback_reason": reference_validation.get("selection_fallback_reason", ""),
    }
    results["ground_truth_workflow"] = reference_validation.get("workflow", [])
    results["optimization_history"] = results.get("generated_step_history", [])
    if orchestrator.synth_report and reference_validation.get("report"):
        results["offline_reference_comparison"] = compare_reports(
            orchestrator.synth_report,
            reference_validation.get("report", {}),
        )
    results["independent_golden"] = independent_golden.get("provenance", {})
    results["preflight_elapsed_seconds"] = preflight_elapsed
    results["search_elapsed_seconds"] = search_elapsed
    results["post_route_elapsed_seconds"] = post_route_elapsed
    results["total_elapsed_seconds"] = time.monotonic() - total_started
    results["timing_scope"] = {
        "schema_version": "c2hls.timing-scope.v1",
        "paper_method_wall_time_field": "search_elapsed_seconds",
        "preflight_includes": [
            "public_input_load",
            "cpu_golden_generation",
            "reference_contract_audit",
            "reference_frontier_synthesis_csim",
            "baseline_and_selected_expert_cosim_or_cache",
        ],
        "search_includes": [
            "llm_candidates",
            "candidate_csim_golden",
            "candidate_synthesis",
            "selected_generated_cosim",
        ],
        "post_route_excluded_from_method_wall_time": True,
    }
    results["coverage"] = _build_coverage(
        inputs["meta"],
        reference_validation,
        results.get("csim"),
        results.get("cosim"),
    )
    results = sanitize_saved_result_record(results, reference_validation)
    orchestrator.save_multistep_results(output_dir, bench_name, results)
    return results


def _print_multistep_summary(results: dict):
    bench = results.get("benchmark", "?")
    print(f"\n{'='*70}")
    print(f"  {bench} - Multi-step Optimization Results")
    print(f"{'='*70}")

    baseline = results.get("baseline_report", {})
    if baseline:
        print(
            f"\n  Baseline: lat={baseline.get('latency_ns', '?')} ns, "
            f"BRAM={baseline.get('bram', '?')}, DSP={baseline.get('dsp', '?')}, "
            f"FF={baseline.get('ff', '?')}, LUT={baseline.get('lut', '?')}, "
            f"Fmax={baseline.get('fmax_mhz', '?')} MHz"
        )

    for step in results.get("steps", []):
        name = step.get("step_name", "?")
        status = "OK" if step.get("success") else "FAIL"
        report = step.get("report", {})
        print(f"\n  [{name}] {status}")
        if report:
            print(
                f"    lat={report.get('latency_ns', '?')} ns, "
                f"BRAM={report.get('bram', '?')}, DSP={report.get('dsp', '?')}, "
                f"FF={report.get('ff', '?')}, LUT={report.get('lut', '?')}, "
                f"Fmax={report.get('fmax_mhz', '?')} MHz"
            )

    final = results.get("final_report", {})
    if final and baseline:
        print("\n  Final vs Baseline:")
        for key in ["latency_ns", "bram", "dsp", "ff", "lut"]:
            final_value = final.get(key)
            baseline_value = baseline.get(key)
            if final_value is None or baseline_value is None:
                continue
            try:
                ratio = float(final_value) / float(baseline_value) if float(baseline_value) > 0 else None
            except (TypeError, ValueError):
                ratio = None
            if ratio is not None:
                print(f"    {key}: {final_value} / {baseline_value} = {ratio:.3f}x")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="C-to-HLS Translation Pipeline")
    parser.add_argument("--bench", type=str, default="nw", help="Benchmark name (from benchmarks/ directory)")
    parser.add_argument("--bench-dir", type=str, default=None, help="Direct path to benchmark directory")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for results")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_ID, help="LLM model ID")
    parser.add_argument("--turns", type=int, default=3, help="Max fix attempts per phase")
    parser.add_argument("--quality-repair-turns", type=int, default=DEFAULT_QUALITY_REPAIR_TURNS, help="Max post-synthesis quality repair attempts")
    parser.add_argument("--all", action="store_true", help="Run all benchmarks")
    parser.add_argument("--multistep", action="store_true", help="Run multi-step incremental optimization instead of single-shot")
    parser.add_argument(
        "--steps",
        type=str,
        default=None,
        help="Comma-separated optimization steps (e.g., 'tiling,pipeline,unroll'). Default: all available steps for the benchmark",
    )
    parser.add_argument(
        "--strategy",
        choices=["static", "dynamic", "combo", "combo_full",
                 "combo_progressive", "forward_eval", "flash"],
        default=None,
        help=(
            "Multistep strategy. "
            "static (default): tiling→pipeline→…→coalescing in order. "
            "dynamic: bottleneck-routed (Phase 2). "
            "combo / combo_full: ask the LLM to apply ALL techniques in a single "
            "rewrite then synth once. "
            "combo_progressive: 2-step structural→parallel combo. "
            "flash: functional Phase-B baseline followed by one aggressive "
            "all-in optimization step with normal candidate/attempt telemetry. "
            "**forward_eval (Phase 6b)**: run all steps without per-step "
            "regression-revert; correctness gates (csynth/csim/cosim) only. "
            "Best-so-far tracking commits the peak mid-trajectory state at the end. "
            "Sets C2HLS_STRATEGY for the orchestrator."
        ),
    )
    parser.add_argument(
        "--no-gt-aware-revert",
        action="store_true",
        help="Disable Phase-3 trajectory-alignment-aware revert tolerance. "
             "When set, regressions are reverted on shape regardless of GT "
             "trajectory shape — i.e., revert-the-old-way.",
    )
    parser.add_argument(
        "--candidates-per-step",
        type=str,
        default=None,
        help=(
            "Candidate search width for multistep optimization. Accepts an "
            "integer or JSON map such as '{\"coalescing\":5,\"default\":3}'."
        ),
    )
    parser.add_argument(
        "--attempts-per-candidate",
        type=int,
        default=None,
        help="Synth-tested attempts per candidate when exhaustive candidate attempts are enabled.",
    )
    parser.add_argument(
        "--exhaustive-candidate-attempts",
        action="store_true",
        help="Evaluate all attempts per candidate and select the best passing attempt.",
    )
    args = parser.parse_args()
    if args.strategy:
        os.environ["C2HLS_STRATEGY"] = args.strategy
    if args.no_gt_aware_revert:
        os.environ["C2HLS_GT_AWARE_REVERT"] = "0"
    if args.candidates_per_step is not None:
        os.environ[STEP_CANDIDATES_ENV] = args.candidates_per_step
    if args.attempts_per_candidate is not None:
        os.environ[CANDIDATE_ATTEMPTS_ENV] = str(args.attempts_per_candidate)
    if args.exhaustive_candidate_attempts:
        os.environ[EXHAUSTIVE_CANDIDATE_ATTEMPTS_ENV] = "1"

    steps = args.steps.split(",") if args.steps else None

    if args.all:
        index_path = REPO_ROOT / "benchmarks" / "index.json"
        with open(index_path, "r") as f:
            benchmarks = json.load(f)

        all_results = []
        for meta in benchmarks:
            bench_name = meta["benchmark"]
            bench_dir = REPO_ROOT / "benchmarks" / bench_name
            print(f"\n{'='*60}")
            print(f"Running: {bench_name}")
            print(f"{'='*60}")
            try:
                if args.multistep:
                    result = run_benchmark_multistep(
                        str(bench_dir),
                        gpt_model=args.model,
                        turns_limitation=args.turns,
                        steps=steps,
                        quality_repair_turns=args.quality_repair_turns,
                    )
                    _print_multistep_summary(result)
                else:
                    result = run_benchmark(
                        str(bench_dir),
                        gpt_model=args.model,
                        turns_limitation=args.turns,
                        quality_repair_turns=args.quality_repair_turns,
                    )
                all_results.append(result)
                print(f"  Result: {'SUCCESS' if result['success'] else 'FAIL'}")
            except Exception as exc:
                print(f"  ERROR: {exc}")
                all_results.append({"benchmark": bench_name, "success": False, "error": str(exc)})

        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        for result in all_results:
            status = "PASS" if result.get("success") else "FAIL"
            print(f"  {result.get('benchmark', '?'):20s} {status}")
        passed = sum(1 for result in all_results if result.get("success"))
        print(f"\n  Total: {passed}/{len(all_results)} passed")

        results_dir = REPO_ROOT / ("results_multistep" if args.multistep else "results")
        os.makedirs(results_dir, exist_ok=True)
        with open(results_dir / "all_results.json", "w") as f:
            json.dump(all_results, f, indent=2, default=str)
    else:
        bench_dir = args.bench_dir or str(REPO_ROOT / "benchmarks" / args.bench)
        if args.multistep:
            result = run_benchmark_multistep(
                bench_dir,
                output_dir=args.output_dir,
                gpt_model=args.model,
                turns_limitation=args.turns,
                steps=steps,
                quality_repair_turns=args.quality_repair_turns,
            )
            _print_multistep_summary(result)
        else:
            result = run_benchmark(
                bench_dir,
                output_dir=args.output_dir,
                gpt_model=args.model,
                turns_limitation=args.turns,
                quality_repair_turns=args.quality_repair_turns,
            )
            status = "SUCCESS" if result["success"] else "FAIL"
            print(f"\nResult: {status}")
            if result.get("synth_report"):
                print(f"Report:\n{format_report_summary(result['synth_report'])}")
            comparison = result.get("comparison") or {}
            if comparison.get("comparison"):
                print("\nComparison vs ground truth:")
                for metric, vals in comparison["comparison"].items():
                    if isinstance(vals, dict) and vals.get("ratio") is not None:
                        print(f"  {metric}: gen={vals['generated']} gt={vals['ground_truth']} ratio={vals['ratio']:.3f}")
