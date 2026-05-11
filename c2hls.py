"""
C-to-HLS Translation Pipeline.

Adapts the Fortran-to-C++ pipeline for translating plain C kernels
into Xilinx Vitis HLS optimized code.

Pipeline:
  Reference Gate: Validate the gold HLS baseline with local Vitis HLS
  Phase A: Validate input C code compiles with g++
  Phase B: LLM translates C -> HLS-C, validate with Vitis HLS synthesis
  Phase C: Compare synthesis reports against the validated gold baseline
"""

import json
import logging
import os
import re
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

from dotenv import load_dotenv
from openai import OpenAI

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
            "When timing is poor, move closer to the gold-baseline pragma style: keep the interface pragmas, but remove compute-side PIPELINE/ARRAY_PARTITION/INLINE directives unless they clearly help.",
            "For this benchmark, a simpler gold-like pragma set is preferable to an over-pragmatized kernel.",
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
}


def _policy(benchmark_name: str, field: str, default=None):
    """Look up a field of BENCHMARK_POLICIES with a fallback."""
    return (BENCHMARK_POLICIES.get(benchmark_name or "") or {}).get(field, default)


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
        work_dir = tempfile.mkdtemp(prefix="c2hls_compile_")
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
    summary = {
        "status": _test_status(True, True, passed),
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
    if "kernel_runtime_cycles" in result:
        summary["kernel_runtime_cycles"] = result.get("kernel_runtime_cycles")
    log_excerpt = _extract_failure_excerpt(result.get("log", ""))
    if log_excerpt and not passed:
        summary["log_excerpt"] = log_excerpt
    return summary


def _summary_status(summary: Optional[dict], available: bool) -> str:
    if isinstance(summary, dict):
        status = summary.get("status")
        if status in {"passed", "failed", "not_run", "not_supported"}:
            return status
        supported = bool(summary.get("supported", available))
        ran = bool(summary.get("ran", False))
        passed = bool(summary.get("passed", False))
        return _test_status(supported, ran, passed)
    return _test_status(available, False, False)


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
    synth_result = run_hls_synthesis(
        hls_code,
        header_code,
        header_name=header_name,
        top_function=top_function,
        part=part,
        clock_ns=clock_ns,
        extra_files=extra_files,
    )

    csim_summary = None
    if synth_result.get("success") and testbench_code and run_csim_check:
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
    return {
        "ground_truth_csim_available": bool(meta.get("supports_csim") and meta.get("testbench_file")),
        "ground_truth_csim_ran": bool(gt_csim.get("ran", False)),
        "ground_truth_cosim_available": bool(meta.get("supports_cosim") and meta.get("testbench_file")),
        "ground_truth_cosim_ran": bool(gt_cosim.get("ran", False)),
        "generated_csim_available": bool(meta.get("supports_csim") and meta.get("testbench_file")),
        "generated_csim_ran": bool(gen_csim.get("ran", False)),
        "generated_cosim_available": bool(meta.get("supports_cosim") and meta.get("testbench_file")),
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


def _classify_synth_error(error: str) -> str:
    """Coarse error-class buckets for streak detection. Distinct classes
    should drive distinct repair strategies; the same class repeating is
    a strong signal the LLM is stuck."""
    if not error:
        return "unknown"
    e = error.lower()
    if "timed out" in e or "timeout" in e:
        return "timeout"
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


# Per-step regression thresholds (Phase 5 follow-up tuning).
#
# The original single-threshold design (1.10x for everything) was too tight
# for steps that *legitimately* trade resources for throughput — unroll
# typically grows DSP/FF, doublebuffer doubles BRAM by definition, and
# coalescing widens the AXI port (often 8x DSP on knn-style kernels).
#
# These ceilings are calibrated against what philip's reference actually
# does (rodinia-hls upstream), with ~20-30% slack on top so the agent has
# room to land near the reference without false-positive reverts.
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
}


def _resolve_step_thresholds(step_name: str,
                              global_override: float = None) -> dict:
    """Return the threshold dict for ``step_name``. Order of precedence:

    1. ``C2HLS_STEP_REGRESSION_THRESHOLDS_JSON`` (per-step JSON) — full
       per-step override.
    2. ``C2HLS_STEP_REGRESSION_THRESHOLD`` env var **explicitly set** — a
       single number applies to everything (legacy behaviour preserved).
    3. ``STEP_REGRESSION_THRESHOLDS[step_name]`` — the new per-step default.
    4. ``STEP_REGRESSION_THRESHOLDS["_default"]`` — fallback.
    """
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

    Per-step thresholds come from STEP_REGRESSION_THRESHOLDS, which is
    calibrated against what philip's rodinia-hls reference actually does on
    each step. Pass ``threshold > 0`` for the legacy single-threshold path
    (used as a global override fallback). When ``step_name`` is empty, falls
    through to the ``_default`` entry.
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
# Phase B's translation is a single-shot rewrite of plain C → HLS. If the
# LLM lands a baseline that's significantly worse than the reference's
# baseline, every downstream optimization step compounds the bad
# starting point. The clearest example is knn_static's 72M-cycle
# baseline (vs philip's 1.05M reference baseline — 70× worse) — even
# with strong optimizations the trajectory could only recover to
# 5.32M, never close to the reference's 262K.
#
# Phase 8 adds an opt-in baseline-alignment loop that runs *between*
# Phase B (translate + synth) and Phase C (compare against gold). When
# our baseline is more than ``C2HLS_PHASE8_BASELINE_LATENCY_TOL`` (default
# 1.20×) over the reference's baseline cycles, or any single resource is
# more than ``C2HLS_PHASE8_BASELINE_RESOURCE_TOL`` (default 2.00×) over,
# we re-translate with **metric-only feedback** — no gold-code leak,
# just numeric gaps + per-loop diagnostics — and re-synth. Up to 3
# attempts.
#
# Critical constraint: the feedback must NOT include the reference HLS
# source. The translator agent should still be solving the
# C-to-HLS task, not regurgitating philip's code. We render only:
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
    feedback. **Never** includes the reference HLS source — just numeric
    gaps + per-loop diagnostics from our own report's
    ``feedback`` field (which the orchestrator already attaches).
    """
    if gap.get("within_tolerance"):
        return ""

    o, r = gap.get("ours_summary") or {}, gap.get("reference_summary") or {}

    lines: list = [
        f"Your previous translation has a baseline that's significantly "
        f"worse than the canonical reference baseline for this kernel.",
        "",
        "Per-axis comparison (your translation → reference baseline):",
    ]
    if o.get("latency_cycles") and r.get("latency_cycles"):
        ratio = o["latency_cycles"] / r["latency_cycles"]
        lines.append(
            f"  latency_cycles: {o['latency_cycles']:>10} → "
            f"{r['latency_cycles']:>10}  ({ratio:.2f}× slower)"
        )
    for k in ("bram", "dsp", "ff", "lut"):
        v_o, v_r = o.get(k), r.get(k)
        if v_o is not None and v_r is not None and v_r > 0:
            ratio = v_o / v_r
            arrow = "→"
            tag = "" if ratio <= 1.10 else f"  ({ratio:.2f}× more)"
            lines.append(f"  {k:<6}      : {int(v_o):>10} {arrow} {int(v_r):>10}{tag}")
    if o.get("fmax_mhz") and r.get("fmax_mhz"):
        lines.append(
            f"  Fmax (MHz)   : {o['fmax_mhz']:>10.1f} → "
            f"{r['fmax_mhz']:>10.1f}"
        )

    if gap.get("over_resources"):
        lines.append("")
        lines.append("Resources over the alignment tolerance:")
        for k, vo, vr, rr in gap["over_resources"]:
            lines.append(f"  - {k}: {int(vo)} vs {int(vr)} ({rr:.2f}×)")

    if gap.get("fmax_below_floor"):
        fmax_ratio = gap.get("fmax_ratio", 0)
        fmax_floor = gap.get("fmax_floor", 0.80)
        o_fmax = (gap.get("ours_summary") or {}).get("fmax_mhz")
        r_fmax = (gap.get("reference_summary") or {}).get("fmax_mhz")
        lines.append("")
        lines.append(
            f"Fmax is too low: your design achieves {o_fmax:.1f} MHz "
            f"({fmax_ratio:.2f}× of reference {r_fmax:.1f} MHz, floor "
            f"is {fmax_floor:.2f}×). A cycle-count match at much lower "
            f"Fmax indicates a long combinational critical path — the "
            f"translation has structural timing problems that will worsen "
            f"under optimization. Target at least {r_fmax * fmax_floor:.0f} MHz."
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
        "synth-report that fits within the reference's per-resource "
        "envelope. Do NOT add optimization pragmas (PIPELINE / UNROLL / "
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


def _build_quality_guidance(benchmark_name: str, report: dict, ground_truth_report: dict, comparison: dict) -> str:
    bench = benchmark_name or ""
    issues = []

    slack = _as_float((report or {}).get("slack_ns"))
    if slack is not None and slack < 0:
        issues.append(f"Current slack is {slack:.3f} ns, so reduce critical-path pressure and improve timing closure.")

    fmax_ratio = _comparison_ratio(comparison, "fmax_mhz")
    if fmax_ratio is not None and fmax_ratio < 0.8:
        issues.append(f"Current Fmax is only {fmax_ratio:.3f}x the gold baseline; improve clock frequency without breaking functionality.")

    latency_ratio = _comparison_ratio(comparison, "latency_ns")
    if latency_ratio is not None and latency_ratio > 2.0:
        issues.append(f"Latency is {latency_ratio:.3f}x the gold baseline in ns; reduce unnecessary serialization or buffering if possible.")

    for key, label, threshold in [
        ("bram", "BRAM", 1.15),
        ("dsp", "DSP", 1.15),
        ("ff", "FF", 1.25),
        ("lut", "LUT", 1.25),
    ]:
        ratio = _comparison_ratio(comparison, key)
        if ratio is not None and ratio > threshold:
            issues.append(f"{label} usage is {ratio:.3f}x the gold baseline; reduce over-parallelization or duplicated storage for this resource.")

    if bench == "spmv_crs" and latency_ratio is not None and latency_ratio > 1.5 and (slack is None or slack >= 0) and (fmax_ratio is None or fmax_ratio >= 1.0):
        issues.insert(0, "Timing is already healthy, so focus this repair on reducing latency while keeping slack non-negative.")

    if bench == "spmv_crs" and ((slack is not None and slack < 0) or (fmax_ratio is not None and fmax_ratio < 0.8)):
        issues.insert(0, "Timing is still poor on this benchmark; prefer a simpler, gold-like pragma set over additional aggressive compute-side directives.")

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


def _quality_score(benchmark_name: str, report: dict, comparison: dict) -> float:
    bench = benchmark_name or ""
    score = 0.0

    slack = _as_float((report or {}).get("slack_ns"))
    if slack is not None and slack < 0:
        score += abs(slack) * 25.0

    fmax_ratio = _comparison_ratio(comparison, "fmax_mhz")
    if fmax_ratio is not None and fmax_ratio < 1.0:
        score += (1.0 - fmax_ratio) * 40.0

    for key, weight in [
        ("latency_ns", 12.0),
        ("bram", 10.0),
        ("dsp", 8.0),
        ("ff", 6.0),
        ("lut", 6.0),
    ]:
        ratio = _comparison_ratio(comparison, key)
        if ratio is not None and ratio > 1.0:
            score += (ratio - 1.0) * weight

    if bench == "nw":
        if slack is not None and slack < 0:
            score += abs(slack) * 30.0
        if fmax_ratio is not None and fmax_ratio < 0.8:
            score += (0.8 - fmax_ratio) * 80.0
    elif bench == "spmv_crs":
        latency_focus = (slack is None or slack >= 0) and (fmax_ratio is None or fmax_ratio >= 1.0)
        if latency_focus:
            latency_ratio = _comparison_ratio(comparison, "latency_ns")
            if latency_ratio is not None and latency_ratio > 1.0:
                score += (latency_ratio - 1.0) * 35.0
        for key, weight in [("bram", 20.0), ("ff", 10.0), ("lut", 10.0), ("latency_ns", 14.0)]:
            ratio = _comparison_ratio(comparison, key)
            if ratio is not None and ratio > 1.0:
                score += (ratio - 1.0) * weight
    elif bench == "StreamCluster":
        if fmax_ratio is not None and fmax_ratio < 0.5:
            score += (0.5 - fmax_ratio) * 120.0
        for key, weight in [("ff", 20.0), ("lut", 20.0), ("dsp", 30.0), ("bram", 8.0)]:
            ratio = _comparison_ratio(comparison, key)
            if ratio is not None and ratio > 1.0:
                score += (ratio - 1.0) * weight

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
    dsp_ratio = _comparison_ratio(comparison, "dsp")

    if bench == "spmv_crs":
        if (slack is not None and slack < 0) or (fmax_ratio is not None and fmax_ratio < 0.8):
            return "timing"
        if latency_ratio is not None and latency_ratio > 1.5:
            return "latency"
        return "area"

    if bench == "StreamCluster":
        if (slack is not None and slack < 0) or (fmax_ratio is not None and fmax_ratio < 0.5):
            return "timing_dsp"
        if dsp_ratio is not None and dsp_ratio > 1.1:
            return "dsp"
        return "area"

    if bench == "nw":
        if (slack is not None and slack < 0) or (fmax_ratio is not None and fmax_ratio < 0.8):
            return "timing"
        return "area"

    return "general"


def _quality_focus_improved(benchmark_name: str, focus: str, current_report: dict, current_comparison: dict,
                            candidate_report: dict, candidate_comparison: dict) -> bool:
    current_slack = _as_float((current_report or {}).get("slack_ns"))
    candidate_slack = _as_float((candidate_report or {}).get("slack_ns"))
    current_fmax = _comparison_ratio(current_comparison, "fmax_mhz") or 0.0
    candidate_fmax = _comparison_ratio(candidate_comparison, "fmax_mhz") or 0.0
    current_latency = _comparison_ratio(current_comparison, "latency_ns") or float("inf")
    candidate_latency = _comparison_ratio(candidate_comparison, "latency_ns") or float("inf")
    current_dsp = _comparison_ratio(current_comparison, "dsp") or 1.0
    candidate_dsp = _comparison_ratio(candidate_comparison, "dsp") or 1.0

    timing_better = False
    if current_slack is not None and candidate_slack is not None and candidate_slack > current_slack + 0.5:
        timing_better = True
    if candidate_fmax > current_fmax + 0.05:
        timing_better = True

    if focus == "timing":
        return timing_better
    if focus == "latency":
        return candidate_latency < current_latency - 0.05
    if focus == "timing_dsp":
        return timing_better or (candidate_dsp < current_dsp - 0.05)
    if focus == "dsp":
        return (candidate_dsp < current_dsp - 0.05) or timing_better

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
        )

    def _request_code_revision(self, prompt: str) -> Optional[str]:
        """Append a user prompt to orch.messages, call this agent's LLM,
        record both turns in history, and return the extracted code."""
        self.orch.messages.append({"role": "user", "content": prompt})
        reply = self._call_llm(self.orch.messages)
        self.orch.messages.append({"role": "assistant", "content": reply})
        self.orch._append_history("user", prompt)
        self.orch._append_history("assistant", reply)
        return extract_cpp_code(reply)


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
                    logging.info("[Phase A] Fixed C code compiles (turn %d)", turn)
                    return True
                logging.warning("[Phase A] Still fails (turn %d): %s", turn, err[:200])

        orch._append_history(
            "system",
            f"[Phase A] FAIL: C code does not compile after {orch.turns_limitation} turns",
        )
        return False

    def translate_initial(self) -> Optional[str]:
        """Run Phase B's initial translate prompt and return the extracted
        HLS code. Does not synthesize — the SynthesisAgent does that."""
        orch = self.orch
        logging.info("=== [Phase B] Translating C to HLS ===")

        prompt = q_translate_c_to_hls.format(
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
            logging.error("[Phase B] No code block in LLM response")
            orch._append_history("system", "[Phase B] FAIL: no code in response")
            return None
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
        base_prompt = q_translate_c_to_hls.format(
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
                    else:
                        logging.warning(
                            "[Phase B] Cosim FAILED: %s",
                            (orch.generated_cosim.get("error") or "")[:200],
                        )

                # Snapshot for potential revert later (e.g. if a future
                # quality-repair-style chain breaks things). Currently
                # synthesize_with_repair returns on first success, but the
                # snapshot is cheap and future-proofs the agent.
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

        orch._append_history(
            "system",
            f"[Phase B] FAIL: HLS synthesis not achieved in {orch.turns_limitation} turns",
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
                ground_truth_report=format_report_summary(ground_truth_report),
                comparison_summary=json.dumps(current_comparison, indent=2),
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
            self.anthropic_client = anthropic.Anthropic(api_key=api_key)
        else:
            if self.use_hosted_openai:
                self.key = _load_openai_api_key()
                assert self.key, f"Missing OpenAI API key. Set OPENAI_API_KEY or populate {OPENAI_API_KEY_FILE}."
                self.base_url = OPENAI_HOSTED_BASE_URL
            else:
                self.key = os.getenv("OPENAI_API_KEY", "EMPTY")
                self.base_url = os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1")
            self.client = OpenAI(base_url=self.base_url, api_key=self.key)

        # Per-agent LLM cache. Populated lazily when an agent's model differs
        # from the orchestrator's default. Keys: model id; values: tuple
        # (kind, client) where kind is "anthropic" or "openai".
        self._extra_clients: dict = {}

        self.messages = []
        self.history = []
        self.c_code = None
        self.header_code = ""
        self.header_name = "kernel.h"
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
        self.benchmark_name = ""
        self.benchmark_context = ""
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
        # or by setting `orch.dynamic_routing = True` directly.
        self.dynamic_routing: bool = bool(int(os.getenv("C2HLS_DYNAMIC_ROUTING", "0")))
        self.skill_library = None  # SkillLibrary, lazy-loaded when dynamic_routing kicks in
        self.vitis_version: str = os.getenv("C2HLS_VITIS_VERSION", "")
        # Trajectory-collapse / throughput-regression telemetry, populated
        # by run_multistep so callers can inspect the new robustness
        # signals without enabling dynamic routing.
        self.robustness_log: list = []

        # ---- Phase 3 wiring: strategy + GT-aware revert tolerance ----
        # `strategy` is the source of truth. dynamic_routing flag
        # remains for backward compat (it implies strategy="dynamic").
        # Allowed values: "static" | "dynamic" | "combo" | "combo_progressive".
        # `gt_aware_revert` toggles the trajectory-alignment check that
        # tolerates regressions on enabling steps when the GT trajectory
        # also regresses there.
        env_strategy = os.getenv("C2HLS_STRATEGY", "").strip().lower()
        if env_strategy:
            self.strategy: str = env_strategy
        elif self.dynamic_routing:
            self.strategy = "dynamic"
        else:
            self.strategy = "static"
        self.gt_aware_revert: bool = bool(int(os.getenv("C2HLS_GT_AWARE_REVERT", "1")))
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

    def _call_llm(self, messages: list, max_tokens: int = None) -> str:
        """Default-model LLM call. Kept as the public interface so existing
        callers (multistep, run_optimization_step, anything outside the
        agent classes) continue to work unchanged."""
        return self._call_llm_with_model(messages, model=self.gpt_model,
                                         max_tokens=max_tokens)

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
            client = anthropic.Anthropic(api_key=api_key)
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
            entry = ("openai", OpenAI(base_url=base_url, api_key=api_key))
        self._extra_clients[model] = entry
        return entry

    def _call_llm_with_model(self, messages: list, model: str = None,
                             max_tokens: int = None) -> str:
        """Route an LLM call to the requested model's backend. Used by
        agents to support per-agent model overrides without forcing every
        caller to manage clients.
        """
        if max_tokens is None:
            max_tokens = self.max_completion_tokens
        if not model:
            model = self.gpt_model

        kind, client = self._client_for_model(model)
        if kind == "anthropic":
            system_text = ""
            conv_messages = []
            for message in messages:
                if message["role"] == "system":
                    system_text += message["content"] + "\n"
                else:
                    conv_messages.append({"role": message["role"],
                                          "content": message["content"]})
            response = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=system_text.strip() if system_text else "",
                messages=conv_messages,
            )
            return response.content[0].text

        kwargs = {"model": model, "messages": messages}
        if _is_hosted_openai_model(model):
            kwargs["max_completion_tokens"] = max_tokens
        else:
            kwargs["max_tokens"] = max_tokens
        if "qwen" in model.lower():
            kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
        response = client.chat.completions.create(**kwargs)
        return response.choices[0].message.content

    def _append_history(self, role: str, content: str):
        self.history.append({"role": role, "content": content})

    def _request_code_revision(self, prompt: str) -> Optional[str]:
        self.messages.append({"role": "user", "content": prompt})
        reply = self._call_llm(self.messages)
        self.messages.append({"role": "assistant", "content": reply})
        self._append_history("user", prompt)
        self._append_history("assistant", reply)
        return extract_cpp_code(reply)

    def _preflight_generated_hls_code(self, code: str, context: str) -> str:
        normalized, note = _align_generated_top_signature(
            code,
            self.header_code,
            self.testbench_code,
            self.translated_hls_top,
        )
        if note:
            logging.info("[%s] ABI preflight: %s", context, note)
            self._append_history("system", f"[{context}] ABI preflight: {note}")
        return normalized

    def _synth_and_test(self, code: str, log_prefix: str = "") -> dict:
        """Synthesize `code` with the orchestrator's current config and run
        csim/cosim if a testbench is available. Returns the same shape as
        _run_synth_csim_cosim: {synth, csim, cosim}."""
        return _run_synth_csim_cosim(
            code,
            header_code=self.header_code,
            header_name=self.header_name,
            top_function=self.translated_hls_top,
            part=self.part,
            clock_ns=self.clock_ns,
            extra_files=self.extra_files,
            testbench_code=self.testbench_code,
            run_csim_check=bool(self.testbench_code),
            run_cosim_check=bool(self.testbench_code and self.supports_cosim),
            cosim_depths=self.cosim_depths,
            log_prefix=log_prefix,
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

    def run_phase_b(self) -> bool:
        """Delegate to TranslatorAgent for the initial translate, then to
        SynthesisAgent for the synth+repair loop. Splitting these two lets
        future RL work plug in different policies for "produce code" vs
        "fix a synth failure" without restructuring the full pipeline."""
        hls_code = self.translator.translate_initial()
        if not hls_code:
            return False
        self.hls_code = hls_code
        return self.synthesis.synthesize_with_repair()

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
        logging.info("=== [Phase C] Comparing against validated gold baseline ===")

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
        logging.info("[Phase C] Gold baseline report:\n%s", format_report_summary(ground_truth_report))
        logging.info("[Phase C] Comparison:")
        for metric, vals in comparison.items():
            if isinstance(vals, dict) and vals.get("ratio") is not None:
                logging.info(
                    "  %s: gen=%s gt=%s ratio=%.3f",
                    metric,
                    vals["generated"],
                    vals["ground_truth"],
                    vals["ratio"],
                )

        self._append_history("system", f"[Phase C] Comparison: {json.dumps(comparison, indent=2)}")

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
                               gt_header_code: str = None) -> dict:
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
                                   gt_header_code: str = None) -> dict:
        """One optimization-step pass: LLM → synth + repair loop. Returns a
        step_result dict with success / code / report / vs_previous / vs_ground_truth.

        Does NOT commit self.hls_code / self.synth_report — the outer
        run_optimization_step decides whether to accept based on the
        regression check. This deferred-commit design is what makes the
        regression guard possible.
        """
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
        # name. Off when skill_library is None (static order, no router).
        if self.skill_library is not None and self.synth_report is not None:
            try:
                from skill_library import render_skill_set_for_prompt
                feedback = (self.synth_report or {}).get("feedback") or {}
                top_bottleneck_kind = None
                bns = feedback.get("bottlenecks") or []
                if bns:
                    top_bottleneck_kind = bns[0].get("kind")
                if top_bottleneck_kind:
                    matching = self.skill_library.query(
                        bottleneck_kind=top_bottleneck_kind,
                        vitis_version=self.vitis_version,
                        fpga=self.part,
                    )
                    skill_block = render_skill_set_for_prompt(matching, max_skills=2)
                    if skill_block and "No matching skills" not in skill_block:
                        extra_blocks.append(
                            "RELEVANT SKILLS from library (pattern → strategy → "
                            "template). Apply the highest-confidence one that "
                            f"addresses the bottleneck '{top_bottleneck_kind}' "
                            f"on the `{step_name}` step:\n\n" + skill_block
                        )
            except Exception as exc:  # pragma: no cover - skill injection best-effort
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
        self._append_history("user", f"[Step: {step_name}] {prompt[:200]}...")
        self._append_history("assistant", reply)
        self.messages.append({"role": "assistant", "content": reply})

        new_code = extract_cpp_code(reply)
        if not new_code:
            logging.error("[Step: %s] No code in LLM response", step_name)
            return {"success": False, "step_name": step_name,
                    "error": "No code in response"}

        step_turn_records: list[dict] = []  # per-step attempt history

        for turn in range(self.turns_limitation):
            logging.info("[Step: %s] Synthesis attempt %d", step_name, turn)
            new_code = self._preflight_generated_hls_code(
                new_code, f"Step {step_name} attempt {turn}",
            )

            ok, err = compile_check_cpp(
                new_code, self.header_code, self.header_name,
                extra_files=self.extra_files,
            )
            if not ok:
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
                continue

            outcome = self._synth_and_test(new_code, log_prefix=f"[Step: {step_name}]")
            result = outcome["synth"]

            if result["success"]:
                logging.info("[Step: %s] Synthesis SUCCESS!\n%s",
                             step_name, format_report_summary(result["report"]))

                step_result = {
                    "success": True,
                    "step_name": step_name,
                    "report": result["report"],
                    "code": new_code,
                }
                if self.synth_report:
                    step_result["vs_previous"] = compare_reports(
                        result["report"], self.synth_report,
                    )

                if gt_code:
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
                    isinstance(cosim_summary, dict)
                    and cosim_summary.get("ran")
                    and not cosim_summary.get("passed")
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
                    continue

                return step_result

            logging.warning("[Step: %s] Synthesis failed: %s",
                            step_name, result["error"][:300])
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
                f"{self.turns_limitation} attempts ({last_err[:160]})"
            )
        else:
            error_msg = f"Synthesis failed after {self.turns_limitation} attempts"
        return {
            "success": False,
            "step_name": step_name,
            "error": error_msg,
        }

    # ---- Phase 6a: best-so-far tracking helpers ----

    @staticmethod
    def _best_so_far_score(report: dict) -> float:
        """Lower-is-better trajectory score. Latency_ns is the headline;
        ties broken by total resource sum (BRAM+DSP+FF+LUT) so smaller
        designs win when latency is identical (the Sonnet pathfinder
        coalescing/doublebuffer/pipeline tied on 8.695M, but pipeline +
        unroll were strictly smaller)."""
        if not report:
            return float("inf")
        try:
            lat = float(report.get("latency_ns") or float("inf"))
        except (TypeError, ValueError):
            lat = float("inf")
        rsum = 0.0
        for k in ("bram", "dsp", "ff", "lut"):
            try:
                rsum += float(report.get(k) or 0)
            except (TypeError, ValueError):
                pass
        # 1e-6 weight on resources so resource ties only matter at
        # millions-of-cycles latency parity.
        return lat + rsum * 1e-6

    def _record_best_so_far(self, history: list, *, step_index: int,
                             step_name: str, source: str) -> None:
        """Append a snapshot of the current orchestrator state to the
        best-so-far history. ``source`` is one of {"baseline", "step",
        "step_forward", "alignment_kept"} for downstream attribution."""
        if not self.synth_report:
            return
        history.append({
            "step_index": step_index,
            "step_name": step_name,
            "source": source,
            "score": self._best_so_far_score(self.synth_report),
            "code": self.hls_code,
            "report": dict(self.synth_report),
        })

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
                                       gt_header_code: str = None) -> dict:
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
        attempt["forward_eval_committed"] = True
        return attempt

    def run_multistep(self, c_code: str, header_code: str = "",
                      header_name: str = "kernel.h",
                      steps: list = None,
                      gt_variants: dict = None,
                      gt_variant_headers: dict = None,
                      reference_report: dict = None):
        if steps is None:
            steps = list(DEFAULT_OPT_STEPS)
        if gt_variants is None:
            gt_variants = {}
        if gt_variant_headers is None:
            gt_variant_headers = {}

        if not self.run_phase_a(c_code, header_code, header_name):
            return False, {"phase": "A", "error": "C code validation failed"}

        if not self.run_phase_b():
            return False, {"phase": "B", "error": "Baseline HLS synthesis failed"}

        # Phase 8 (opt-in via C2HLS_PHASE8_BASELINE_ALIGN=1): if our Phase B
        # baseline is significantly worse than the reference baseline,
        # re-translate with metric-only feedback before optimization
        # starts. This stops a bad initial translation from poisoning
        # every downstream optimization step.
        baseline_alignment = self._baseline_alignment_loop(reference_report)

        baseline_report = dict(self.synth_report) if self.synth_report else {}
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
        elif self.strategy == "combo_progressive":
            from prompt_c2hls import COMBO_PROGRESSIVE_STEPS
            steps = list(COMBO_PROGRESSIVE_STEPS)

        # Phase 2: lazy-load the skill library when dynamic routing is on.
        if (self.dynamic_routing or self.strategy == "dynamic") and self.skill_library is None:
            from skill_library import make_default_library
            self.skill_library = make_default_library(persist=False)

        # When dynamic_routing is on, we walk the steps in
        # bottleneck-driven order rather than the configured static
        # order; the configured `steps` list still bounds the search
        # space (we never invent a step the caller didn't allow).
        # When off, we fall through the original loop unchanged.
        # Combo modes always use the static loop (the combo IS the
        # strategy; they typically have 1-2 steps).
        if self.dynamic_routing and self.strategy not in ("combo", "combo_full",
                                                            "combo_progressive"):
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
                if self.strategy == "forward_eval":
                    step_result = self.run_optimization_step_forward(
                        step_name, gt_code=gt_code, gt_header_code=gt_header,
                    )
                else:
                    step_result = self.run_optimization_step(
                        step_name, gt_code=gt_code, gt_header_code=gt_header,
                    )
                step_result["routing_decision"] = {
                    "step_name": decision.step_name,
                    "reason": decision.reason,
                    "bottleneck_kind": decision.bottleneck_kind,
                    "skill_id": decision.skill_id,
                    "confidence": decision.confidence,
                    "fallback": decision.fallback,
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
                if self.strategy == "forward_eval":
                    step_result = self.run_optimization_step_forward(
                        step_name, gt_code=gt_code, gt_header_code=gt_header,
                    )
                else:
                    step_result = self.run_optimization_step(
                        step_name, gt_code=gt_code, gt_header_code=gt_header,
                    )
                step_results.append(step_result)
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

        # Phase 6a: at the end of the trajectory, promote whichever
        # snapshot has the best score (lowest latency_ns + tiny
        # resource-sum tiebreak). If the best is the current state, this
        # is a no-op. If a mid-trajectory state was better, snap back to
        # it and overwrite final_report / hls_code with that snapshot.
        promotion = self._promote_best_so_far(best_so_far_history)

        return True, {
            "phase": "multistep",
            "baseline_report": baseline_report,
            "baseline_comparison": baseline_comparison,
            "baseline_csim": self.generated_csim,
            "baseline_cosim": self.generated_cosim,
            "final_report": self.synth_report,
            "steps": step_results,
            "generated_step_history": [
                {
                    "step_name": "baseline",
                    "success": True,
                    "report": baseline_report,
                    "comparison": baseline_comparison,
                    "csim": self.generated_csim,
                    "cosim": self.generated_cosim,
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
            "messages": self.history,
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
            "messages": self.history,
        }
        with open(os.path.join(output_dir, f"{bench_name}_history.json"), "w") as f:
            json.dump(history_payload, f, indent=2)

    def run(self, c_code: str, header_code: str = "", header_name: str = "kernel.h",
            ground_truth_report: dict = None):
        if not self.run_phase_a(c_code, header_code, header_name):
            return False, {"phase": "A", "error": "C code validation failed"}

        if not self.run_phase_b():
            return False, {"phase": "B", "error": "HLS synthesis failed"}

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

        return True, {
            "phase": "complete",
            "hls_code": self.hls_code,
            "synth_report": self.synth_report,
            "comparison": comparison,
            "csim": self.generated_csim,
            "cosim": self.generated_cosim,
            "quality_repair": quality_repair,
            "turn_history": self.turn_results,
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
    gt_variants = {}
    gt_variant_headers = {}
    for variant in meta.get("variants", []):
        vname = variant["name"]
        vfile = variant["file"]
        vpath = bench_dir / vfile
        if not vpath.exists():
            continue
        parts = vname.split("_", 2)
        if len(parts) < 3:
            continue
        step_key = parts[2]
        step_key = step_key.replace("double_buffer", "doublebuffer")
        step_key = step_key.replace("unrolll", "unroll")
        step_key = step_key.replace("unrolling", "unroll")
        with open(vpath, "r") as f:
            gt_variants[step_key] = f.read()
        # Prefer the upstream variant's own header so per-variant `#define`s
        # (TILE_SIZE, COALESCING_5_512bit, etc.) survive into the synth tcl.
        # Falls back to the local header if the upstream copy is missing or
        # the metadata didn't record a source_path.
        upstream_src = variant.get("source_path") or ""
        if upstream_src:
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


def _normalize_variant_step_name(variant_name: str) -> str:
    parts = (variant_name or "").split("_", 2)
    step_name = parts[2] if len(parts) >= 3 else (variant_name or "baseline")
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


def _ground_truth_candidates(inputs: dict) -> list[dict]:
    meta = inputs["meta"]
    bench_dir = Path(inputs["bench_dir"])
    candidates = []
    seen_files = set()
    default_header_name = meta.get("header_file") or inputs.get("header_name") or "kernel.h"
    default_header_code = inputs.get("header_code", "")

    for variant in meta.get("variants", []):
        variant_file = variant.get("file")
        if not variant_file or variant_file in seen_files:
            continue
        variant_path = bench_dir / variant_file
        if not variant_path.exists():
            continue
        source_path = variant.get("source_path", "")
        header_code = default_header_code
        if source_path:
            source_header = Path(source_path).with_name(default_header_name)
            if source_header.exists():
                header_code = _rewrite_source_includes_for_local_support(source_header.read_text(), bench_dir)
        candidates.append(
            {
                "variant_name": variant.get("name", Path(variant_file).stem),
                "file": variant_file,
                "step_name": _normalize_variant_step_name(variant.get("name", variant_file)),
                "source_path": source_path,
                "header_name": default_header_name,
                "header_code": header_code,
                "code": variant_path.read_text(),
            }
        )
        seen_files.add(variant_file)

    if candidates:
        return candidates

    hls_code = inputs.get("ground_truth_code")
    if hls_code:
        source_path = inputs["meta"].get("gold_hls_source_path", "")
        header_code = default_header_code
        if source_path:
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
            }
        ]
    return []


def _preferred_reference_file(meta: dict, workflow: list[dict]) -> str:
    if meta.get("source_repo") == "rodinia-hls":
        for entry in reversed(workflow):
            if entry.get("step_name") == "coalescing":
                return entry.get("file", "")
        optimized = [entry.get("file", "") for entry in workflow if entry.get("step_name") != "baseline"]
        if optimized:
            return optimized[-1]
    return meta.get("preferred_gt_file", "")


def _validate_ground_truth_candidate(candidate: dict, inputs: dict,
                                     supports_csim: bool, supports_cosim: bool,
                                     run_csim_check: bool = True,
                                     run_cosim_check: bool = True) -> dict:
    meta = inputs["meta"]
    hls_code = candidate["code"]
    header_name = candidate.get("header_name") or inputs.get("header_name") or "kernel.h"
    header_code = candidate.get("header_code", inputs.get("header_code", ""))
    top_function = meta.get("hls_top", "workload")

    csim_signature_mismatch = ""
    if supports_csim and run_csim_check:
        csim_signature_mismatch = _top_signature_mismatch_reason(
            hls_code,
            header_code,
            inputs.get("testbench_code", ""),
            top_function,
        )

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
    }


def validate_gold_reference(inputs: dict) -> dict:
    meta = inputs["meta"]
    supports_csim = bool(meta.get("supports_csim") and inputs.get("testbench_code"))
    supports_cosim = bool(meta.get("supports_cosim") and inputs.get("testbench_code"))
    candidates = _ground_truth_candidates(inputs)

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
        }

    workflow = [
        _validate_ground_truth_candidate(candidate, inputs, supports_csim, supports_cosim)
        for candidate in candidates
    ]

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
    if preferred_file:
        for entry in workflow:
            if entry.get("file") == preferred_file and entry.get("benchmark_ready"):
                selected = entry
                selection_reason = f"selected preferred validated variant `{preferred_file}`"
                break

    if selected is None:
        valid_entries = [entry for entry in workflow if entry.get("benchmark_ready")]
        optimized_entries = [entry for entry in valid_entries if entry.get("step_name") != "baseline"]
        if optimized_entries:
            selected = optimized_entries[-1]
            selection_reason = "selected latest validated optimized variant"
        elif valid_entries:
            selected = valid_entries[-1]
            selection_reason = "selected latest validated baseline-only variant"

    for entry in workflow:
        entry["selected"] = bool(selected and entry.get("file") == selected.get("file"))

    if not selected:
        last_error = workflow[-1].get("invalid_reason") if workflow else "Missing valid ground-truth workflow"
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
        }

    selection_fallback = (
        selected.get("step_name") == "baseline"
        and any(entry.get("step_name") != "baseline" for entry in workflow)
    )
    return {
        "benchmark_ready": True,
        "invalid_reason": "",
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
        "selection_fallback_reason": (
            "optimized GT variants were unavailable or invalid; selected baseline"
            if selection_fallback else ""
        ),
    }


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
    supported = bool(cleaned.get("supported", available))
    if available and not supported and not cleaned.get("ran", False):
        supported = True
    ran_value = bool(cleaned.get("ran", False) or ran)
    passed = bool(cleaned.get("passed", False))
    cleaned["supported"] = supported
    cleaned["ran"] = ran_value
    cleaned["success"] = bool(cleaned.get("success", False)) if ran_value else False
    cleaned["passed"] = passed if ran_value else False
    cleaned["status"] = _test_status(supported, ran_value, cleaned["passed"])
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
    return {
        "model": getattr(orchestrator, "gpt_model", None),
        "model_translator":     os.getenv(TRANSLATOR_MODEL_ENV)     or getattr(orchestrator, "gpt_model", None),
        "model_synthesis":      os.getenv(SYNTHESIS_MODEL_ENV)      or getattr(orchestrator, "gpt_model", None),
        "model_quality_repair": os.getenv(QUALITY_REPAIR_MODEL_ENV) or getattr(orchestrator, "gpt_model", None),
        "vitis_version": _vitis_version(),
        "flow_target": getattr(hls_eval, "DEFAULT_FLOW_TARGET", "vitis"),
        "part": meta.get("part", DEFAULT_PART),
        "clock_ns": meta.get("clock_ns", DEFAULT_CLOCK_NS),
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


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

    if output.get("phase") == "complete" and output.get("synth_report"):
        output["generated_status"] = "passed"
    else:
        output["generated_status"] = "failed"

    generated_csim = output.get("csim")
    generated_cosim = output.get("cosim")
    generated_csim_available = bool(meta.get("supports_csim") and meta.get("testbench_file"))
    generated_cosim_available = bool(meta.get("supports_cosim") and meta.get("testbench_file"))
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
            bool(meta.get("supports_cosim") and meta.get("testbench_file")),
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
        for step in reversed(results.get("steps", [])):
            if step.get("success") and step.get("step_name"):
                return step.get("step_name")
        return "baseline"
    return ""


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
        supports_cosim=bool(inputs["meta"].get("supports_cosim")),
        cosim_depths=inputs["meta"].get("cosim_depths", {}),
        benchmark_name=bench_name,
        benchmark_context=inputs.get("benchmark_context", ""),
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

    success, results = orchestrator.run(
        inputs["c_code"],
        inputs["header_code"],
        inputs["header_name"] or "kernel.h",
        reference_validation.get("report", {}),
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

    with open(os.path.join(output_dir, f"{bench_name}_results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)

    return results


def run_benchmark_multistep(bench_dir: str, output_dir: str = None,
                            gpt_model: str = DEFAULT_MODEL_ID,
                            turns_limitation: int = 3,
                            steps: list = None,
                            quality_repair_turns: int = DEFAULT_QUALITY_REPAIR_TURNS) -> dict:
    inputs = _load_benchmark_inputs(bench_dir)
    bench_name = inputs["bench_name"]

    if output_dir is None:
        output_dir = _default_output_dir(bench_dir, bench_name, multistep=True)
    output_dir = str(output_dir)

    available_gt = set(inputs["gt_variants"].keys())
    if steps is None:
        steps = [step for step in DEFAULT_OPT_STEPS if step in available_gt or step in OPTIMIZATION_PROMPTS]

    logging.info("Benchmark %s: running steps %s (GT available: %s)", bench_name, steps, list(available_gt))

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
        supports_cosim=bool(inputs["meta"].get("supports_cosim")),
        cosim_depths=inputs["meta"].get("cosim_depths", {}),
        benchmark_name=bench_name,
        benchmark_context=inputs.get("benchmark_context", ""),
    )

    reference_validation = validate_gold_reference(inputs)
    if not reference_validation.get("benchmark_ready"):
        return {
            "benchmark": bench_name,
            "success": False,
            "phase": "reference",
            "error": reference_validation.get("invalid_reason") or "Gold HLS reference invalid",
            "reference_validation": reference_validation,
            "ground_truth_status": "invalid",
            "baseline_status": reference_validation.get("synthesis", {}).get("status", "failed"),
            "invalid_reference_reason": reference_validation.get("invalid_reason", ""),
        }

    success, results = orchestrator.run_multistep(
        inputs["c_code"],
        inputs["header_code"],
        inputs["header_name"] or "kernel.h",
        steps=steps,
        gt_variants=inputs["gt_variants"],
        gt_variant_headers=inputs.get("gt_variant_headers", {}),
        reference_report=reference_validation.get("report", {}),
    )

    # Optional hw_emu on the final-step kernel for authoritative cycle count.
    _maybe_run_hw_emu_final(orchestrator, results, bench_name)

    results["benchmark"] = bench_name
    results["success"] = success
    results["run"] = _build_run_attribution(orchestrator, inputs["meta"])
    results["reference_validation"] = reference_validation
    results["ground_truth_status"] = "valid"
    results["baseline_status"] = reference_validation.get("synthesis", {}).get("status", "failed")
    results["invalid_reference_reason"] = ""
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
    results["coverage"] = _build_coverage(
        inputs["meta"],
        reference_validation,
        results.get("baseline_csim"),
        results.get("baseline_cosim"),
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
                 "combo_progressive", "forward_eval"],
        default=None,
        help=(
            "Multistep strategy. "
            "static (default): tiling→pipeline→…→coalescing in order. "
            "dynamic: bottleneck-routed (Phase 2). "
            "combo / combo_full: ask the LLM to apply ALL techniques in a single "
            "rewrite then synth once. "
            "combo_progressive: 2-step structural→parallel combo. "
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
    args = parser.parse_args()
    if args.strategy:
        os.environ["C2HLS_STRATEGY"] = args.strategy
    if args.no_gt_aware_revert:
        os.environ["C2HLS_GT_AWARE_REVERT"] = "0"

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
