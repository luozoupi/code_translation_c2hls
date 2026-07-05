"""Static audit helpers for tier_A_ready gold-gate corpus quality."""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

REPO = Path(__file__).resolve().parents[2]
TIER_A_READY_ROOT = REPO / "related_work/benchmarks/HLSFactory_benchmarks/tier_A_ready"

if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

_TB_LOAD_DATA_RE = re.compile(r"""load_txt_to_array\s*\(\s*["']([^"']+)["']""")
_DEFINE_SEMICOLON_RE = re.compile(r"^\s*#define\b.*;\s*$")
_NONSTATIC_STACK_ARRAY_RE = re.compile(
    r"(?m)^\s*(?!static\s)(?:int|float|double|char)\s+(\w+)\s*\[([^\]]+)\]"
)


@dataclass(frozen=True)
class AuditViolation:
    bench: str
    kind: str
    detail: str

    def key(self) -> tuple[str, str]:
        return (self.bench, self.kind)


def iter_tier_a_benches(root: Path | None = None) -> list[str]:
    base = root or TIER_A_READY_ROOT
    return sorted(
        p.name
        for p in base.iterdir()
        if p.is_dir() and (p / "metadata.json").is_file()
    )


def _bench_dir(bench: str, root: Path | None = None) -> Path:
    return (root or TIER_A_READY_ROOT) / bench


def audit_metadata_present(bench: str, root: Path | None = None) -> list[AuditViolation]:
    bench_dir = _bench_dir(bench, root)
    meta_path = bench_dir / "metadata.json"
    if not meta_path.is_file():
        return [AuditViolation(bench, "metadata_missing", "metadata.json not found")]
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return [AuditViolation(bench, "metadata_invalid", str(exc))]
    violations: list[AuditViolation] = []
    if not meta.get("testbench_file"):
        violations.append(AuditViolation(bench, "metadata_incomplete", "missing testbench_file"))
    tb_path = bench_dir / str(meta.get("testbench_file") or "")
    if meta.get("testbench_file") and not tb_path.is_file():
        violations.append(
            AuditViolation(bench, "metadata_incomplete", f"missing testbench: {meta['testbench_file']}")
        )
    if meta.get("supports_csim") and not meta.get("testbench_file"):
        violations.append(
            AuditViolation(bench, "metadata_incomplete", "supports_csim but no testbench_file")
        )
    return violations


def audit_forgebench_support_staging(bench: str, root: Path | None = None) -> list[AuditViolation]:
    bench_dir = _bench_dir(bench, root)
    meta_path = bench_dir / "metadata.json"
    if not meta_path.is_file():
        return []
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("dataset") != "forgebench":
        return []

    tb_file = meta.get("testbench_file") or ""
    tb_path = bench_dir / tb_file
    if not tb_path.is_file():
        return [AuditViolation(bench, "forgebench_support", f"missing testbench {tb_file}")]

    tb_text = tb_path.read_text(encoding="utf-8")
    bare_refs = sorted({m.group(1) for m in _TB_LOAD_DATA_RE.finditer(tb_text)})
    data_refs = bare_refs
    if not data_refs:
        return []

    violations: list[AuditViolation] = []
    for ref in data_refs:
        rel = ref if "/" in ref else f"support/{ref}"
        if not (bench_dir / rel).is_file() and not (bench_dir / Path(ref).name).is_file():
            violations.append(AuditViolation(bench, "forgebench_support", f"TB references missing file: {ref}"))

    csim_tcl = bench_dir / "dataset_hls_csim.tcl"
    if csim_tcl.is_file():
        tcl_text = csim_tcl.read_text(encoding="utf-8")
        for ref in data_refs:
            basename = Path(ref).name
            staged_candidates = [ref, f"support/{basename}", basename]
            if not any(s in tcl_text for s in staged_candidates):
                violations.append(
                    AuditViolation(
                        bench,
                        "forgebench_support",
                        f"dataset_hls_csim.tcl missing add_files for {ref}",
                    )
                )
            if not any(f"add_files -tb {s}" in tcl_text for s in staged_candidates):
                violations.append(
                    AuditViolation(
                        bench,
                        "forgebench_support",
                        f"dataset_hls_csim.tcl missing add_files -tb for {ref}",
                    )
                )
    return violations


def audit_signature_compatible(bench: str, root: Path | None = None) -> list[AuditViolation]:
    from c2hls import _load_benchmark_inputs, _top_signature_mismatch_reason

    bench_dir = _bench_dir(bench, root)
    if not bench_dir.is_dir():
        return [AuditViolation(bench, "signature_mismatch", "bench dir missing")]
    inputs = _load_benchmark_inputs(str(bench_dir))
    meta = inputs["meta"]
    gold = inputs["ground_truth_code"]
    mismatch = _top_signature_mismatch_reason(
        gold,
        inputs.get("header_code", ""),
        inputs.get("testbench_code", ""),
        meta.get("hls_top", "workload"),
    )
    if mismatch:
        return [AuditViolation(bench, "signature_mismatch", mismatch)]
    return []


def audit_params_h_semicolons(bench: str, root: Path | None = None) -> list[AuditViolation]:
    params_path = _bench_dir(bench, root) / "params.h"
    if not params_path.is_file():
        return []
    bad = [ln.strip() for ln in params_path.read_text(encoding="utf-8").splitlines() if _DEFINE_SEMICOLON_RE.match(ln)]
    if bad:
        return [
            AuditViolation(
                bench,
                "params_h_semicolon",
                f"{len(bad)} #define lines end with ';' (e.g. {bad[0][:80]})",
            )
        ]
    return []


def _params_defines(params_text: str) -> dict[str, int]:
    defines: dict[str, int] = {}
    for match in re.finditer(r"#\s*define\s+(\w+)\s+(\d+)", params_text):
        defines[match.group(1)] = int(match.group(2))
    return defines


def _eval_array_size(expr: str, defines: dict[str, int]) -> int | None:
    expr = expr.strip()
    substituted = re.sub(
        r"[A-Za-z_][A-Za-z0-9_]*",
        lambda m: str(defines[m.group(0)]) if m.group(0) in defines else m.group(0),
        expr,
    )
    if not re.fullmatch(r"[\d\s+\-*()]+", substituted):
        return None
    try:
        return int(eval(substituted, {"__builtins__": {}}, {}))  # noqa: S307
    except Exception:
        return None


def audit_params_h_include_guard(bench: str, root: Path | None = None) -> list[AuditViolation]:
    params_path = _bench_dir(bench, root) / "params.h"
    if not params_path.is_file():
        return []
    text = params_path.read_text(encoding="utf-8")
    has_guard = "#ifndef" in text[:200]
    needs_guard = bool(re.search(r"^\s*const\s+", text, re.MULTILINE)) or text.count("#if") >= 2
    if needs_guard and not has_guard:
        return [AuditViolation(bench, "params_h_guard", "params.h missing #ifndef include guard")]
    return []


def _estimate_stack_array_bytes(
    tb_text: str,
    *,
    defines: dict[str, int] | None = None,
    word_bytes: int = 4,
) -> tuple[int, str]:
    """Rough static estimate for simple ``T name[expr]`` locals."""
    defines = defines or {}
    total = 0
    worst = ""
    for match in _NONSTATIC_STACK_ARRAY_RE.finditer(tb_text):
        _name, size_expr = match.group(1), match.group(2).strip()
        count = _eval_array_size(size_expr, defines)
        if count is None:
            continue
        nbytes = max(0, count) * word_bytes
        total += nbytes
        if nbytes > 0 and (not worst or nbytes > int(worst.split()[0])):
            worst = f"{nbytes} bytes for {_name}[{size_expr}]"
    return total, worst


def audit_testbench_stack(bench: str, root: Path | None = None, *, max_stack_bytes: int = 2_000_000) -> list[AuditViolation]:
    meta_path = _bench_dir(bench, root) / "metadata.json"
    if not meta_path.is_file():
        return []
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    tb_file = meta.get("testbench_file") or ""
    tb_path = _bench_dir(bench, root) / tb_file
    if not tb_path.is_file():
        return []
    tb_text = tb_path.read_text(encoding="utf-8")
    main_match = re.search(r"int\s+main\s*\([^)]*\)\s*\{", tb_text)
    if not main_match:
        return []
    tb_body = tb_text[main_match.end() :]
    params_path = _bench_dir(bench, root) / "params.h"
    defines = _params_defines(params_path.read_text(encoding="utf-8")) if params_path.is_file() else {}
    total, worst = _estimate_stack_array_bytes(tb_body, defines=defines)
    if total > max_stack_bytes:
        return [
            AuditViolation(
                bench,
                "testbench_stack",
                f"estimated main() stack arrays ~{total} bytes (> {max_stack_bytes}); {worst}",
            )
        ]
    return []


def audit_hls_eval_csim_extra_files(bench: str, root: Path | None = None) -> list[AuditViolation]:
    """Ensure programmatic csim TCL would stage TB data files (bucket A)."""
    import hls_eval
    from c2hls import _load_benchmark_inputs

    bench_dir = _bench_dir(bench, root)
    inputs = _load_benchmark_inputs(str(bench_dir))
    tb_text = inputs.get("testbench_code") or ""
    support_refs = sorted({m.group(1) for m in _TB_LOAD_DATA_RE.finditer(tb_text)})
    if not support_refs:
        return []

    work_dir = "/tmp/tier_a_csim_audit"
    lines = hls_eval._tcl_tb_extra_add_lines(
        work_dir, inputs.get("extra_files") or [], relative=True,
    )
    staged = "\n".join(lines)
    violations: list[AuditViolation] = []
    for rel in support_refs:
        candidates = [rel, f"support/{rel}"]
        if not any(f"add_files -tb {path}" in staged for path in candidates):
            violations.append(
                AuditViolation(
                    bench,
                    "hls_eval_csim_tcl",
                    f"csim TCL builder missing add_files -tb for {rel}",
                )
            )
    return violations


AUDIT_CHECKS = (
    audit_metadata_present,
    audit_forgebench_support_staging,
    audit_signature_compatible,
    audit_params_h_semicolons,
    audit_params_h_include_guard,
    audit_testbench_stack,
    audit_hls_eval_csim_extra_files,
)

# Violations expected until later corpus-fix phases (mergesort gold synth, forgebench_mlp timeout).
ALLOWED_VIOLATIONS: set[tuple[str, str]] = set()


def run_tier_a_gold_gate_audit(
    benches: Iterable[str] | None = None,
    *,
    root: Path | None = None,
) -> list[AuditViolation]:
    names = list(benches) if benches is not None else iter_tier_a_benches(root)
    violations: list[AuditViolation] = []
    for bench in names:
        for check in AUDIT_CHECKS:
            violations.extend(check(bench, root))
    return violations


def unexpected_violations(
    violations: list[AuditViolation],
    allowed: set[tuple[str, str]] | None = None,
) -> list[AuditViolation]:
    allow = allowed if allowed is not None else ALLOWED_VIOLATIONS
    return [v for v in violations if v.key() not in allow]
