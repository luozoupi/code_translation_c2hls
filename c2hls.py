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

REPO_ROOT = Path(__file__).resolve().parent
CLAUDE_API_KEY_FILE = Path("/home/luo00466/claude-api-key.txt")
OPENAI_API_KEY_FILE = Path("/home/luo00466/gpt-key.txt")
OPENAI_HOSTED_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL_ID = "nvidia/OpenCodeReasoning-Nemotron-1.1-32B"
TIMEOUT_LIMIT = 60
DEFAULT_QUALITY_REPAIR_TURNS = 2
QUALITY_SCORE_EPSILON = 0.25

BENCHMARK_HINTS = {
    "nw": [
        "The header owns `bench_args_t`; do not redeclare it in the source.",
        "Preserve the existing `needwun` helper structure from the plain input instead of inventing a new algorithm decomposition.",
        "Keep the workload wrapper very close to the plain input: one pair of local dynamic-programming arrays named `M` and `ptr`, then a simple loop over jobs that calls `needwun`.",
        "Do not remove or rename dynamic-programming buffers like `M` and `ptr` if the existing helper logic still requires them.",
        "Avoid aggressive optimization on this benchmark: do not completely partition `M` or `ptr`, do not fully unroll the DP loops, and prefer only light inner-loop pipelining if any.",
    ],
    "spmv_crs": [
        "Use the existing kernel interface `spmv(val, cols, rowDelimiters, vec, out)` from the header.",
        "Keep the workload wrapper very close to the plain input: preserve the existing local arrays `l_val`, `l_cols`, `l_rowDelimiters`, `l_vec`, and `l_out` plus their copy-in/copy-out loops.",
        "Do not invent new helper buffers beyond the existing plain-input locals unless they are clearly necessary and fully declared.",
        "Keep the wrapper ports aligned with the reference AXI-visible arrays: `val`, `cols`, `rowDelimiters`, `vec`, and `out`.",
        "Do not collapse the wrapper into a direct pointer call to `spmv`; the plain input already gives the intended wrapper structure.",
    ],
    "StreamCluster": [
        "Preserve the existing helper-call structure from the plain input instead of rewriting the whole benchmark around a new buffer scheme.",
    ],
}


BENCHMARK_QUALITY_HINTS = {
    "nw": [
        "Treat the large dynamic-programming arrays `M` and `ptr` as simple memories; reduce or remove large partition factors on them if timing is poor.",
        "Avoid over-pipelining loops that repeatedly read and write `M` and `ptr` if it hurts timing closure.",
        "Keep the workload wrapper simple; do not add extra buffering layers or duplicated helper logic just to chase throughput.",
    ],
    "spmv_crs": [
        "Minimize BRAM-heavy local buffering and avoid complete partitioning of `out`/`l_out` or other large arrays unless it clearly pays off.",
        "Prefer modest cyclic factors or no partitioning on large arrays over aggressive partitioning that inflates memory resources.",
        "Keep the copy-in/copy-out loops simple and do not introduce extra array copies unless they materially help timing.",
        "When timing is already healthy, it is acceptable to spend a little area to shrink the remaining latency gap, especially in the compute loop.",
        "When timing is poor, move closer to the gold-baseline pragma style: keep the interface pragmas, but remove compute-side PIPELINE/ARRAY_PARTITION/INLINE directives unless they clearly help.",
        "For this benchmark, a simpler gold-like pragma set is preferable to an over-pragmatized kernel.",
    ],
    "StreamCluster": [
        "Reduce FF/LUT blow-up by avoiding aggressive unrolling, inlining, or duplicated helper pipelines.",
        "Prefer shared buffers and sequential helper calls over dataflow-like rewrites that replicate large logic blocks.",
        "Remove unnecessary complete partitioning on large state arrays and keep the design closer to the original helper structure.",
        "This benchmark has large latency headroom, so it is acceptable to relax throughput-oriented pipelining if that improves slack/Fmax or reduces DSP pressure.",
        "Prefer one reusable arithmetic pipeline over DSP-heavy parallel scheduling when timing is poor.",
    ],
}


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
    normalized = " ".join((text or "").strip().split())
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
    current_linkage = bool(current.get("extern_c") or ('extern "C"' in (code or "")))
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
                             c_code: str, ground_truth_code: str,
                             testbench_code: str = "") -> str:
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

    reference_ports = _extract_interface_ports(ground_truth_code)
    if reference_ports:
        joined = ", ".join(f"`{name}`" for name in reference_ports)
        hints.append(f"Reference wrapper interface ports: {joined}.")

    for manual_hint in BENCHMARK_HINTS.get(bench, []):
        hints.append(manual_hint)

    return "\n".join(f"- {hint}" for hint in hints)


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
    if not hints:
        hints.append("- Preserve the existing helper/kernel structure and make the smallest change that fixes the reported error.")
    return "\n".join(hints)


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

    priorities = {
        "nw": "Primary objective: improve timing/slack/Fmax while keeping the simple workload wrapper and dynamic-programming structure intact.",
        "spmv_crs": "Primary objective: reduce resource usage, especially memory-heavy buffering, without making latency much worse.",
        "StreamCluster": "Primary objective: reduce FF/LUT blow-up from over-aggressive parallelism or duplicated logic.",
    }

    guidance = []
    if priorities.get(bench):
        guidance.append(priorities[bench])
    guidance.extend(issues)
    guidance.extend(BENCHMARK_QUALITY_HINTS.get(bench, []))
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


class C2HLSOrchestrator:
    """Pipeline orchestrator for C-to-HLS translation."""

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

    def configure_benchmark(
        self,
        extra_files=None,
        translated_hls_top: str = "workload",
        reference_hls_top: str = "workload",
        part: str = DEFAULT_PART,
        clock_ns: int = DEFAULT_CLOCK_NS,
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
        if max_tokens is None:
            max_tokens = self.max_completion_tokens

        if self.use_anthropic:
            system_text = ""
            conv_messages = []
            for message in messages:
                if message["role"] == "system":
                    system_text += message["content"] + "\n"
                else:
                    conv_messages.append({"role": message["role"], "content": message["content"]})
            response = self.anthropic_client.messages.create(
                model=self.gpt_model,
                max_tokens=max_tokens,
                system=system_text.strip() if system_text else "",
                messages=conv_messages,
            )
            return response.content[0].text

        kwargs = {
            "model": self.gpt_model,
            "messages": messages,
        }
        if self.use_hosted_openai:
            kwargs["max_completion_tokens"] = max_tokens
        else:
            kwargs["max_tokens"] = max_tokens
        if "qwen" in self.gpt_model.lower():
            kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
        response = self.client.chat.completions.create(**kwargs)
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

    def _evaluate_candidate_with_repairs(self, candidate_code: str, label: str) -> dict:
        current_code = candidate_code
        last_error = ""

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
                logging.warning("%s Compile error: %s", label, err[:200])
                fixed = self._request_code_revision(
                    c_compilation_fix.format(
                        compile_error=err,
                        hls_code=current_code,
                        benchmark_context=self.benchmark_context,
                        repair_guidance=_build_repair_guidance(err),
                    )
                )
                if fixed:
                    current_code = fixed
                continue

            result = run_hls_synthesis(
                current_code,
                self.header_code,
                header_name=self.header_name,
                top_function=self.translated_hls_top,
                part=self.part,
                clock_ns=self.clock_ns,
                extra_files=self.extra_files,
            )
            if result["success"]:
                candidate = {
                    "success": True,
                    "code": current_code,
                    "report": result["report"],
                    "csim": None,
                    "cosim": None,
                }
                if self.testbench_code:
                    logging.info("%s Running C-simulation (csim)...", label)
                    csim_result = run_csim(
                        current_code,
                        self.testbench_code,
                        self.header_code,
                        header_name=self.header_name,
                        top_function=self.translated_hls_top,
                        part=self.part,
                        clock_ns=self.clock_ns,
                        extra_files=self.extra_files,
                    )
                    candidate["csim"] = _summarize_test_result(csim_result, True)

                if self.testbench_code and self.supports_cosim:
                    logging.info("%s Running co-simulation (cosim)...", label)
                    cosim_result = run_cosim(
                        current_code,
                        self.testbench_code,
                        self.header_code,
                        header_name=self.header_name,
                        top_function=self.translated_hls_top,
                        part=self.part,
                        clock_ns=self.clock_ns,
                        extra_files=self.extra_files,
                        interface_depths=self.cosim_depths,
                    )
                    candidate["cosim"] = _summarize_test_result(cosim_result, True)

                return candidate

            last_error = result["error"]
            logging.warning("%s Synthesis failed: %s", label, result["error"][:300])
            is_timeout = "timed out" in result["error"].lower()
            if is_timeout:
                fix_prompt = hls_synthesis_timeout_fix.format(
                    timeout=600,
                    hls_code=current_code,
                    header_code=self.header_code,
                    benchmark_context=self.benchmark_context,
                    repair_guidance=_build_repair_guidance(result["error"]),
                )
            else:
                fix_prompt = hls_synthesis_fix.format(
                    synth_error=result["error"],
                    hls_code=current_code,
                    header_code=self.header_code,
                    benchmark_context=self.benchmark_context,
                    repair_guidance=_build_repair_guidance(result["error"]),
                )
            fixed = self._request_code_revision(fix_prompt)
            if fixed:
                current_code = fixed

        return {
            "success": False,
            "code": current_code,
            "error": last_error or f"{label} failed after {self.turns_limitation} attempts",
        }

    def run_quality_repair(self, ground_truth_report: dict, initial_comparison: Optional[dict] = None) -> dict:
        summary = {
            "attempted": False,
            "applied": False,
            "attempts": [],
        }
        self.quality_repair_result = summary

        if self.quality_repair_turns <= 0 or not ground_truth_report or not self.synth_report or not self.hls_code:
            summary["reason"] = "Quality repair disabled or missing reports"
            return summary

        current_comparison = initial_comparison or compare_reports(self.synth_report, ground_truth_report)
        quality_guidance = _build_quality_guidance(
            self.benchmark_name,
            self.synth_report,
            ground_truth_report,
            current_comparison,
        )
        if not quality_guidance:
            summary["reason"] = "No timing/resource issues triggered quality repair"
            return summary

        summary["attempted"] = True
        summary["initial_score"] = _quality_score(self.benchmark_name, self.synth_report, current_comparison)
        summary["initial_comparison"] = current_comparison
        current_score = summary["initial_score"]

        for turn in range(self.quality_repair_turns):
            current_focus = _quality_focus(self.benchmark_name, self.synth_report, current_comparison)
            prompt = hls_quality_repair.format(
                hls_code=self.hls_code,
                current_report=format_report_summary(self.synth_report),
                ground_truth_report=format_report_summary(ground_truth_report),
                comparison_summary=json.dumps(current_comparison, indent=2),
                benchmark_context=self.benchmark_context,
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

            candidate = self._evaluate_candidate_with_repairs(proposed_code, "[Quality Repair]")
            if not candidate.get("success"):
                attempt["status"] = "failed"
                attempt["error"] = candidate.get("error", "")
                summary["attempts"].append(attempt)
                continue

            candidate_comparison = compare_reports(candidate["report"], ground_truth_report)
            candidate_score = _quality_score(self.benchmark_name, candidate["report"], candidate_comparison)
            tests_preserved = (
                _preserves_passed_test(self.generated_csim, candidate.get("csim"))
                and _preserves_passed_test(self.generated_cosim, candidate.get("cosim"))
            )
            attempt.update(
                {
                    "candidate_score": candidate_score,
                    "comparison": candidate_comparison,
                    "tests_preserved": tests_preserved,
                }
            )

            focus_improved = _quality_focus_improved(
                self.benchmark_name,
                current_focus,
                self.synth_report,
                current_comparison,
                candidate["report"],
                candidate_comparison,
            )
            attempt["focus_improved"] = focus_improved

            if tests_preserved and focus_improved and candidate_score + QUALITY_SCORE_EPSILON < current_score:
                self.hls_code = candidate["code"]
                self.synth_report = candidate["report"]
                self.generated_csim = candidate.get("csim")
                self.generated_cosim = candidate.get("cosim")
                current_comparison = candidate_comparison
                current_score = candidate_score
                summary["applied"] = True
                attempt["status"] = "accepted"
                summary["attempts"].append(attempt)
                logging.info("[Quality Repair] Accepted improved candidate with score %.3f", candidate_score)

                quality_guidance = _build_quality_guidance(
                    self.benchmark_name,
                    self.synth_report,
                    ground_truth_report,
                    current_comparison,
                )
                if not quality_guidance:
                    break
                continue

            attempt["status"] = "rejected"
            if not tests_preserved:
                attempt["rejection_reason"] = "Functional checks regressed"
            elif not focus_improved:
                attempt["rejection_reason"] = f"Candidate did not improve the active {current_focus} focus"
            else:
                attempt["rejection_reason"] = (
                    f"Quality score did not improve enough ({candidate_score:.3f} vs {current_score:.3f})"
                )
            summary["attempts"].append(attempt)

        summary["final_score"] = current_score
        summary["final_comparison"] = current_comparison
        self.quality_repair_result = summary
        return summary

    def run_phase_a(self, c_code: str, header_code: str = "",
                    header_name: str = "kernel.h") -> bool:
        self.c_code = c_code
        self.header_code = header_code
        self.header_name = header_name

        logging.info("=== [Phase A] Validating C code compilation ===")
        self._append_history("system", Instruction_c2hls)

        ok, err = compile_check_cpp(
            c_code,
            header_code,
            header_name,
            extra_files=self.extra_files,
        )
        if ok:
            logging.info("[Phase A] C code compiles successfully")
            self._append_history("system", "[Phase A] Input C code compiles OK.")
            return True

        logging.warning("[Phase A] C code fails to compile: %s", err)

        for turn in range(self.turns_limitation):
            prompt = q_validate_c_code.format(
                c_code=self.c_code,
                header_code=self.header_code,
                benchmark_context=self.benchmark_context,
            )
            self.messages = [
                {"role": "system", "content": Instruction_c2hls},
                {"role": "user", "content": prompt},
            ]
            reply = self._call_llm(self.messages)
            self._append_history("assistant", reply)

            fixed = extract_cpp_code(reply)
            if fixed:
                self.c_code = fixed
                ok, err = compile_check_cpp(
                    self.c_code,
                    self.header_code,
                    self.header_name,
                    extra_files=self.extra_files,
                )
                if ok:
                    logging.info("[Phase A] Fixed C code compiles (turn %d)", turn)
                    return True
                logging.warning("[Phase A] Still fails (turn %d): %s", turn, err[:200])

        self._append_history(
            "system",
            f"[Phase A] FAIL: C code does not compile after {self.turns_limitation} turns",
        )
        return False

    def run_phase_b(self) -> bool:
        logging.info("=== [Phase B] Translating C to HLS ===")

        prompt = q_translate_c_to_hls.format(
            c_code=self.c_code,
            header_code=self.header_code,
            benchmark_context=self.benchmark_context,
        )
        self.messages = [
            {"role": "system", "content": Instruction_c2hls},
            {"role": "user", "content": prompt},
        ]

        reply = self._call_llm(self.messages)
        self._append_history("user", prompt)
        self._append_history("assistant", reply)
        self.messages.append({"role": "assistant", "content": reply})

        hls_code = extract_cpp_code(reply)
        if not hls_code:
            logging.error("[Phase B] No code block in LLM response")
            self._append_history("system", "[Phase B] FAIL: no code in response")
            return False

        self.hls_code = hls_code

        for turn in range(self.turns_limitation):
            logging.info("[Phase B] Synthesis attempt %d", turn)
            self.hls_code = self._preflight_generated_hls_code(self.hls_code, f"Phase B attempt {turn}")

            ok, err = compile_check_cpp(
                self.hls_code,
                self.header_code,
                self.header_name,
                extra_files=self.extra_files,
            )
            if not ok:
                logging.warning("[Phase B] HLS code doesn't compile: %s", err[:200])
                fix_prompt = c_compilation_fix.format(
                    compile_error=err,
                    hls_code=self.hls_code,
                    benchmark_context=self.benchmark_context,
                    repair_guidance=_build_repair_guidance(err),
                )
                self.messages.append({"role": "user", "content": fix_prompt})
                reply = self._call_llm(self.messages)
                self.messages.append({"role": "assistant", "content": reply})
                self._append_history("user", fix_prompt)
                self._append_history("assistant", reply)

                fixed = extract_cpp_code(reply)
                if fixed:
                    self.hls_code = fixed
                continue

            result = run_hls_synthesis(
                self.hls_code,
                self.header_code,
                header_name=self.header_name,
                top_function=self.translated_hls_top,
                part=self.part,
                clock_ns=self.clock_ns,
                extra_files=self.extra_files,
            )

            self.turn_results.append({
                "turn": turn,
                "phase": "B",
                "success": result["success"],
                "report": result.get("report", {}),
                "error": result.get("error", ""),
            })

            if result["success"]:
                self.synth_report = result["report"]
                logging.info("[Phase B] Synthesis SUCCESS!\n%s", format_report_summary(result["report"]))
                self._append_history(
                    "system",
                    f"[Phase B] Synthesis SUCCESS. Report:\n{format_report_summary(result['report'])}",
                )

                self.generated_csim = None
                if self.testbench_code:
                    logging.info("[Phase B] Running C-simulation (csim)...")
                    csim_result = run_csim(
                        self.hls_code,
                        self.testbench_code,
                        self.header_code,
                        header_name=self.header_name,
                        top_function=self.translated_hls_top,
                        part=self.part,
                        clock_ns=self.clock_ns,
                        extra_files=self.extra_files,
                    )
                    self.generated_csim = _summarize_test_result(csim_result, True)
                    if self.generated_csim.get("passed"):
                        logging.info("[Phase B] Csim PASSED")
                    else:
                        logging.warning("[Phase B] Csim FAILED: %s", self.generated_csim.get("error", "")[:200])

                self.generated_cosim = None
                if self.testbench_code and self.supports_cosim:
                    logging.info("[Phase B] Running co-simulation (cosim)...")
                    cosim_result = run_cosim(
                        self.hls_code,
                        self.testbench_code,
                        self.header_code,
                        header_name=self.header_name,
                        top_function=self.translated_hls_top,
                        part=self.part,
                        clock_ns=self.clock_ns,
                        extra_files=self.extra_files,
                        interface_depths=self.cosim_depths,
                    )
                    self.generated_cosim = _summarize_test_result(cosim_result, True)
                    if self.generated_cosim.get("passed"):
                        logging.info("[Phase B] Cosim PASSED")
                    else:
                        logging.warning("[Phase B] Cosim FAILED: %s", self.generated_cosim.get("error", "")[:200])

                return True

            logging.warning("[Phase B] Synthesis failed: %s", result["error"][:300])
            is_timeout = "timed out" in result["error"].lower()
            if is_timeout:
                fix_prompt = hls_synthesis_timeout_fix.format(
                    timeout=600,
                    hls_code=self.hls_code,
                    header_code=self.header_code,
                    benchmark_context=self.benchmark_context,
                    repair_guidance=_build_repair_guidance(result["error"]),
                )
            else:
                fix_prompt = hls_synthesis_fix.format(
                    synth_error=result["error"],
                    hls_code=self.hls_code,
                    header_code=self.header_code,
                    benchmark_context=self.benchmark_context,
                    repair_guidance=_build_repair_guidance(result["error"]),
                )
            self.messages.append({"role": "user", "content": fix_prompt})
            reply = self._call_llm(self.messages)
            self.messages.append({"role": "assistant", "content": reply})
            self._append_history("user", fix_prompt)
            self._append_history("assistant", reply)

            fixed = extract_cpp_code(reply)
            if fixed:
                self.hls_code = fixed

        self._append_history(
            "system",
            f"[Phase B] FAIL: HLS synthesis not achieved in {self.turns_limitation} turns",
        )
        return False

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

    def run_optimization_step(self, step_name: str, gt_code: str = None) -> dict:
        logging.info("=== [Step: %s] Applying optimization ===", step_name)

        if not self.hls_code:
            return {"success": False, "step_name": step_name, "error": "No HLS code to optimize"}

        report_str = format_report_summary(self.synth_report) if self.synth_report else "(no prior report)"
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
            return {"success": False, "step_name": step_name, "error": "No code in response"}

        for turn in range(self.turns_limitation):
            logging.info("[Step: %s] Synthesis attempt %d", step_name, turn)
            new_code = self._preflight_generated_hls_code(new_code, f"Step {step_name} attempt {turn}")

            ok, err = compile_check_cpp(
                new_code,
                self.header_code,
                self.header_name,
                extra_files=self.extra_files,
            )
            if not ok:
                logging.warning("[Step: %s] Compile error: %s", step_name, err[:200])
                fix_prompt = c_compilation_fix.format(
                    compile_error=err,
                    hls_code=new_code,
                    benchmark_context=self.benchmark_context,
                    repair_guidance=_build_repair_guidance(err),
                )
                self.messages.append({"role": "user", "content": fix_prompt})
                reply = self._call_llm(self.messages)
                self.messages.append({"role": "assistant", "content": reply})
                self._append_history("assistant", reply)
                fixed = extract_cpp_code(reply)
                if fixed:
                    new_code = fixed
                continue

            result = run_hls_synthesis(
                new_code,
                self.header_code,
                header_name=self.header_name,
                top_function=self.translated_hls_top,
                part=self.part,
                clock_ns=self.clock_ns,
                extra_files=self.extra_files,
            )

            if result["success"]:
                prev_report = self.synth_report
                self.hls_code = new_code
                self.synth_report = result["report"]
                logging.info("[Step: %s] Synthesis SUCCESS!\n%s", step_name, format_report_summary(result["report"]))

                step_result = {
                    "success": True,
                    "step_name": step_name,
                    "report": result["report"],
                    "code": new_code,
                }

                if prev_report:
                    step_result["vs_previous"] = compare_reports(result["report"], prev_report)

                if gt_code:
                    gt_result = run_hls_synthesis(
                        gt_code,
                        self.header_code,
                        header_name=self.header_name,
                        top_function=self.reference_hls_top,
                        part=self.part,
                        clock_ns=self.clock_ns,
                        extra_files=self.extra_files,
                    )
                    if gt_result["success"]:
                        step_result["vs_ground_truth"] = compare_reports(result["report"], gt_result["report"])
                        step_result["gt_report"] = gt_result["report"]
                    else:
                        step_result["gt_report_status"] = _summarize_synth_result(gt_result)

                if self.testbench_code:
                    logging.info("[Step: %s] Running C-simulation (csim)...", step_name)
                    csim_result = run_csim(
                        new_code,
                        self.testbench_code,
                        self.header_code,
                        header_name=self.header_name,
                        top_function=self.translated_hls_top,
                        part=self.part,
                        clock_ns=self.clock_ns,
                        extra_files=self.extra_files,
                    )
                    step_result["csim"] = _summarize_test_result(csim_result, True)

                if self.testbench_code and self.supports_cosim:
                    logging.info("[Step: %s] Running co-simulation (cosim)...", step_name)
                    cosim_result = run_cosim(
                        new_code,
                        self.testbench_code,
                        self.header_code,
                        header_name=self.header_name,
                        top_function=self.translated_hls_top,
                        part=self.part,
                        clock_ns=self.clock_ns,
                        extra_files=self.extra_files,
                        interface_depths=self.cosim_depths,
                    )
                    step_result["cosim"] = _summarize_test_result(cosim_result, True)

                return step_result

            logging.warning("[Step: %s] Synthesis failed: %s", step_name, result["error"][:300])
            is_timeout = "timed out" in result["error"].lower()
            if is_timeout:
                fix_prompt = hls_synthesis_timeout_fix.format(
                    timeout=600,
                    hls_code=new_code,
                    header_code=self.header_code,
                    benchmark_context=self.benchmark_context,
                    repair_guidance=_build_repair_guidance(result["error"]),
                )
            else:
                fix_prompt = hls_synthesis_fix.format(
                    synth_error=result["error"],
                    hls_code=new_code,
                    header_code=self.header_code,
                    benchmark_context=self.benchmark_context,
                    repair_guidance=_build_repair_guidance(result["error"]),
                )
            self.messages.append({"role": "user", "content": fix_prompt})
            reply = self._call_llm(self.messages)
            self.messages.append({"role": "assistant", "content": reply})
            self._append_history("assistant", reply)
            fixed = extract_cpp_code(reply)
            if fixed:
                new_code = fixed

        return {
            "success": False,
            "step_name": step_name,
            "error": f"Synthesis failed after {self.turns_limitation} attempts",
        }

    def run_multistep(self, c_code: str, header_code: str = "",
                      header_name: str = "kernel.h",
                      steps: list = None,
                      gt_variants: dict = None,
                      reference_report: dict = None):
        if steps is None:
            steps = list(DEFAULT_OPT_STEPS)
        if gt_variants is None:
            gt_variants = {}

        if not self.run_phase_a(c_code, header_code, header_name):
            return False, {"phase": "A", "error": "C code validation failed"}

        if not self.run_phase_b():
            return False, {"phase": "B", "error": "Baseline HLS synthesis failed"}

        baseline_report = dict(self.synth_report) if self.synth_report else {}
        baseline_comparison = self.run_phase_c(reference_report) if reference_report else {}
        step_results = []

        for step_name in steps:
            gt_code = gt_variants.get(step_name)
            step_result = self.run_optimization_step(step_name, gt_code=gt_code)
            step_results.append(step_result)
            if not step_result.get("success"):
                logging.warning("[Multistep] Step '%s' failed: %s", step_name, step_result.get("error", "unknown"))

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
        }

    def save_results(self, output_dir: str, bench_name: str):
        os.makedirs(output_dir, exist_ok=True)

        if self.hls_code:
            with open(os.path.join(output_dir, f"{bench_name}_generated.cpp"), "w") as f:
                f.write(self.hls_code)

        with open(os.path.join(output_dir, f"{bench_name}_history.json"), "w") as f:
            json.dump(self.history, f, indent=2)

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

        with open(os.path.join(output_dir, f"{bench_name}_history.json"), "w") as f:
            json.dump(self.history, f, indent=2)

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

    # Use preferred GT variant if explicitly set in metadata (pre-validated).
    preferred = meta.get("preferred_gt_file")
    if preferred and (bench_dir / preferred).exists():
        gt_file = preferred
        logging.info(f"Using preferred GT variant '{preferred}'")
    else:
        # Fall back: try from best to worst; skip variants that don't compile.
        extra_files_for_check = []
        for rel_path in meta.get("support_files", []):
            fp = bench_dir / rel_path
            if fp.exists():
                extra_files_for_check.append({"path": rel_path, "content": fp.read_text()})

        variants = meta.get("variants", [])
        if len(variants) > 1:
            for variant in reversed(variants):
                vfile = variant["file"]
                vpath = bench_dir / vfile
                if not vpath.exists():
                    continue
                vcode = vpath.read_text()
                ok, err = compile_check_cpp(vcode, header_code, header_name,
                                            extra_files=extra_files_for_check)
                if ok:
                    gt_file = vfile
                    logging.info(f"Using best variant '{variant['name']}' as ground truth")
                    break
                else:
                    logging.debug(f"Variant '{variant['name']}' failed compile check: {err[:120]}")

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

    gt_variants = {}
    for variant in meta.get("variants", []):
        vname = variant["name"]
        vfile = variant["file"]
        vpath = bench_dir / vfile
        if vpath.exists():
            parts = vname.split("_", 2)
            if len(parts) >= 3:
                step_key = parts[2]
                step_key = step_key.replace("double_buffer", "doublebuffer")
                step_key = step_key.replace("unrolll", "unroll")
                step_key = step_key.replace("unrolling", "unroll")
                with open(vpath, "r") as f:
                    gt_variants[step_key] = f.read()

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

    benchmark_context = _build_benchmark_context(
        meta,
        header_name,
        header_code,
        c_code,
        ground_truth_code or gold_hls_source_code,
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
    support_common = bench_dir / "support" / "common"
    if not support_common.exists():
        return code

    def _replace(match: re.Match) -> str:
        rel_name = match.group(1)
        if (support_common / rel_name).exists():
            return f'#include "support/common/{rel_name}"'
        return match.group(0)

    return re.sub(r'#include\s+"(?:\.\./)+common/([^"]+)"', _replace, code)


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
    synth_result = run_hls_synthesis(
        hls_code,
        header_code,
        header_name=header_name,
        top_function=meta.get("hls_top", "workload"),
        part=meta.get("part", DEFAULT_PART),
        clock_ns=meta.get("clock_ns", DEFAULT_CLOCK_NS),
        extra_files=inputs.get("extra_files", []),
    )
    synth_summary = _summarize_synth_result(synth_result)
    csim_signature_mismatch = ""
    if supports_csim and run_csim_check:
        csim_signature_mismatch = _top_signature_mismatch_reason(
            hls_code,
            header_code,
            inputs.get("testbench_code", ""),
            meta.get("hls_top", "workload"),
        )

    csim_result = None
    if synth_result.get("success") and supports_csim and run_csim_check and not csim_signature_mismatch:
        csim_result = run_csim(
            hls_code,
            inputs.get("testbench_code", ""),
            header_code,
            header_name=header_name,
            top_function=meta.get("hls_top", "workload"),
            part=meta.get("part", DEFAULT_PART),
            clock_ns=meta.get("clock_ns", DEFAULT_CLOCK_NS),
            extra_files=inputs.get("extra_files", []),
        )
    csim_summary = _summarize_test_result(csim_result, supports_csim)
    if csim_signature_mismatch and supports_csim and run_csim_check:
        csim_summary["error"] = f"Skipped CSim: {csim_signature_mismatch}"

    cosim_result = None
    if (
        synth_result.get("success")
        and supports_cosim
        and run_cosim_check
        and not csim_signature_mismatch
        and ((not supports_csim) or (not run_csim_check) or csim_summary.get("passed"))
    ):
        cosim_result = run_cosim(
            hls_code,
            inputs.get("testbench_code", ""),
            header_code,
            header_name=header_name,
            top_function=meta.get("hls_top", "workload"),
            part=meta.get("part", DEFAULT_PART),
            clock_ns=meta.get("clock_ns", DEFAULT_CLOCK_NS),
            extra_files=inputs.get("extra_files", []),
            interface_depths=meta.get("cosim_depths", {}),
        )
    cosim_summary = _summarize_test_result(cosim_result, supports_cosim)

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


def _finalize_singleshot_results(bench_name: str, meta: dict, success: bool,
                                 base_results: dict, reference_validation: dict) -> dict:
    output = dict(base_results)
    output["benchmark"] = bench_name
    output["success"] = success
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

    orchestrator.save_results(output_dir, bench_name)
    results = _finalize_singleshot_results(
        bench_name,
        inputs["meta"],
        success,
        results,
        reference_validation,
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
        reference_report=reference_validation.get("report", {}),
    )

    results["benchmark"] = bench_name
    results["success"] = success
    results["reference_validation"] = reference_validation
    results["ground_truth_status"] = "valid"
    results["baseline_status"] = reference_validation.get("synthesis", {}).get("status", "failed")
    results["invalid_reference_reason"] = ""
    results["ground_truth_variant"] = {
        "name": reference_validation.get("selected_variant_name", ""),
        "file": reference_validation.get("selected_variant_file", ""),
        "step": reference_validation.get("selected_variant_step", ""),
        "selection_reason": reference_validation.get("selection_reason", ""),
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
    args = parser.parse_args()

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
