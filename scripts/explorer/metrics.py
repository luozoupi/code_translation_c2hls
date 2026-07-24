"""Shared csynth latency speedup metrics for explorer and Fir dashboard."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


def latency_cycles(report: dict[str, Any] | None, kind: str) -> int | None:
    if not isinstance(report, dict):
        return None
    if kind == "best":
        raw = report.get("latency_cycles_best")
        if raw is None:
            raw = report.get("latency_cycles")
    elif kind == "avg":
        raw = report.get("latency_cycles")
    elif kind == "worst":
        raw = report.get("latency_cycles_worst")
        if raw is None:
            raw = report.get("latency_cycles")
    else:
        return None
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def speedup(base: int | None, opt: int | None) -> float | None:
    if base is None or opt is None or opt <= 0:
        return None
    return base / opt


def geomean(values: list[float]) -> float | None:
    positive = [v for v in values if v is not None and v > 0]
    if not positive:
        return None
    return math.exp(sum(math.log(v) for v in positive) / len(positive))


def latency_triple(report: dict[str, Any] | None) -> dict[str, int | None]:
    return {
        "best": latency_cycles(report, "best"),
        "avg": latency_cycles(report, "avg"),
        "worst": latency_cycles(report, "worst"),
    }


def bench_csynth_latency_from_multistep_doc(doc: dict[str, Any]) -> dict[str, int | None] | None:
    """Final csynth latency cycles (best / avg / worst) from multistep results."""
    if bench_csynth_failed(doc):
        return None
    triple = latency_triple(doc.get("final_report") or {})
    if not any(triple.values()):
        return None
    return triple


def bench_csynth_latency_from_multistep(path: Path) -> dict[str, int | None] | None:
    if not path.is_file():
        return None
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return bench_csynth_latency_from_multistep_doc(doc)


def bench_speedup_from_multistep_doc(doc: dict[str, Any]) -> dict[str, float | None] | None:
    if bench_csynth_failed(doc):
        return None
    # Csynth speedup always uses shipped gold (hls_baseline.cpp) as numerator,
    # not in-run Phase B, so phaseb and direct variants are comparable.
    baseline = doc.get("ground_truth_report") or doc.get("baseline_report") or {}
    final = doc.get("final_report") or {}
    out: dict[str, float | None] = {}
    for kind in ("best", "avg", "worst"):
        out[kind] = speedup(
            latency_cycles(baseline, kind),
            latency_cycles(final, kind),
        )
    return out


def bench_speedup_from_multistep(path: Path) -> dict[str, float | None] | None:
    if not path.is_file():
        return None
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return bench_speedup_from_multistep_doc(doc)


_COSIM_CRASH_MARKERS = (
    "sigsegv",
    "segmentation fault",
    "signal named",
    "aborting co-simulation",
)


def cosim_latency_cycles(cosim: dict[str, Any] | None) -> int | None:
    """RTL kernel runtime cycles from a cosim summary dict."""
    if not isinstance(cosim, dict):
        return None
    raw = cosim.get("kernel_runtime_cycles")
    if raw is None:
        return None
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def classify_cosim_status(cosim: dict[str, Any] | None) -> str:
    """Classify cosim outcome: not_run | pass | fail | crash."""
    if not isinstance(cosim, dict) or not cosim.get("ran"):
        return "not_run"
    passed = cosim.get("passed") is True or str(cosim.get("status") or "").lower() in {
        "pass",
        "passed",
        "ok",
    }
    if passed and cosim_latency_cycles(cosim) is not None:
        return "pass"
    err = f"{cosim.get('error') or ''} {cosim.get('log_excerpt') or ''}".lower()
    if any(marker in err for marker in _COSIM_CRASH_MARKERS):
        return "crash"
    return "fail"


def find_winning_step_name(doc: dict[str, Any]) -> str | None:
    """Step whose csynth report matches ``final_report`` (lowest-latency selection)."""
    final_lat = latency_cycles(doc.get("final_report"), "avg")
    if final_lat is None:
        return None

    matches: list[str] = []
    if latency_cycles(doc.get("baseline_report"), "avg") == final_lat:
        matches.append("baseline")
    for step in doc.get("steps") or []:
        name = str(step.get("step_name") or "")
        if name and latency_cycles(step.get("report"), "avg") == final_lat:
            matches.append(name)

    if not matches:
        promo = doc.get("best_so_far_promotion") or {}
        promoted = promo.get("from_step_name")
        return str(promoted) if promoted else None

    order = ["baseline"] + [
        str(step.get("step_name") or "")
        for step in doc.get("steps") or []
        if step.get("step_name")
    ]
    for name in reversed(order):
        if name in matches:
            return name
    return matches[-1]


def cosim_summary_for_step(doc: dict[str, Any], step_name: str | None) -> dict[str, Any] | None:
    if not step_name:
        return None
    if step_name == "baseline":
        for step in doc.get("generated_step_history") or []:
            if step.get("step_name") == "baseline":
                cosim = step.get("cosim")
                if isinstance(cosim, dict):
                    return cosim
        baseline_cosim = doc.get("baseline_cosim")
        return baseline_cosim if isinstance(baseline_cosim, dict) else None
    for step in doc.get("steps") or []:
        if step.get("step_name") == step_name:
            cosim = step.get("cosim")
            return cosim if isinstance(cosim, dict) else None
    return None


def _flash_step(doc: dict[str, Any]) -> dict[str, Any]:
    return next(
        (step for step in doc.get("steps") or [] if step.get("step_name") == "flash"),
        {},
    )


_LLM_TIMEOUT_EXCEPTIONS = frozenset({"APITimeoutError", "TimeoutError"})
_LLM_CONNECTION_EXCEPTIONS = frozenset({"APIConnectionError", "ConnectError"})


def bench_run_issues_from_multistep_doc(doc: dict[str, Any]) -> list[str]:
    """Run issues: infra failures, csynth timeout, flash revert (slower code kept)."""
    issues: list[str] = []
    flash = _flash_step(doc)
    attempt_error = str(flash.get("attempt_error") or flash.get("error") or "")
    exception_type = str(flash.get("exception_type") or "")
    err_lower = attempt_error.lower()
    attempt_results = flash.get("attempt_results") or []
    synth_error = attempt_error
    if attempt_results:
        synth_error = str(attempt_results[-1].get("error") or synth_error)
    synth_lower = synth_error.lower()

    if exception_type in _LLM_TIMEOUT_EXCEPTIONS or "request timed out" in err_lower:
        issues.append("llm_timeout")
    elif (
        "connection error" in err_lower
        or exception_type in _LLM_CONNECTION_EXCEPTIONS
        or "connection refused" in err_lower
        or "connection reset" in err_lower
    ):
        issues.append("llm_connection_error")

    if "timed out after" in synth_lower and "csynth_timeout" not in issues:
        issues.append("csynth_timeout")

    if flash.get("reverted_to_prev"):
        issues.append("flash_reverted")

    if (
        flash
        and not flash.get("success")
        and not flash.get("reverted_to_prev")
        and "llm_timeout" not in issues
        and "llm_connection_error" not in issues
        and "csynth_timeout" not in issues
    ):
        if any(
            str(row.get("stage") or "").lower() in {"synthesis", "csynth", "synth"}
            and not row.get("success")
            for row in attempt_results
        ) or any(
            marker in synth_lower
            for marker in (
                "synthesizability",
                "synthesis failed",
                "synchk",
                "hls 200-",
                "csynth",
            )
        ):
            issues.append("flash_csynth_fail")

    return issues


# Infra / flash-step failures — exclude from geomeans; explorer shows fail (not 1.00×).
GEOMEAN_EXCLUDED_RUN_ISSUES = frozenset(
    {"llm_connection_error", "llm_timeout", "csynth_timeout", "flash_csynth_fail"}
)
CSYNTH_FAIL_RUN_ISSUES = GEOMEAN_EXCLUDED_RUN_ISSUES
RETRYABLE_RUN_ISSUES = frozenset({"llm_connection_error", "llm_timeout"})


def cosim_flow_requested(doc: dict[str, Any]) -> bool:
    """True when the multistep run recorded cosim fields (cosim-enabled campaign)."""
    if "baseline_cosim" in doc or "cosim" in doc:
        return True
    if isinstance(doc.get("baseline_cosim"), dict) or isinstance(doc.get("cosim"), dict):
        return True
    for step in doc.get("steps") or []:
        if isinstance(step.get("cosim"), dict):
            return True
    for step in doc.get("generated_step_history") or []:
        if isinstance(step.get("cosim"), dict):
            return True
    return False


def bench_csynth_failed(doc: dict[str, Any]) -> bool:
    """True when csynth outcome should display as fail (not flat 1.00×)."""
    if not doc.get("success"):
        return True
    issues = bench_run_issues_from_multistep_doc(doc)
    if set(issues) & CSYNTH_FAIL_RUN_ISSUES:
        return True
    flash = _flash_step(doc)
    if not flash or flash.get("success"):
        return False
    if flash.get("reverted_to_prev"):
        return False
    flash_report = flash.get("report") or {}
    flash_lat = latency_cycles(flash_report, "avg")
    if flash_lat is None or not flash_report:
        return True
    baseline = doc.get("ground_truth_report") or doc.get("baseline_report") or {}
    base_lat = latency_cycles(baseline, "avg")
    final_lat = latency_cycles(doc.get("final_report"), "avg")
    if (
        base_lat is not None
        and final_lat == base_lat
        and flash_lat > base_lat
    ):
        return False
    return False


def bench_excluded_from_geomean(info: dict[str, Any]) -> bool:
    """True when run_issues indicate flash did not complete (not a quality revert)."""
    issues = info.get("run_issues") or []
    return bool(set(issues) & GEOMEAN_EXCLUDED_RUN_ISSUES)


def bench_cosim_latency_from_multistep_doc(doc: dict[str, Any]) -> int | None:
    """Passed cosim cycles for the winning kernel only."""
    metrics = bench_cosim_metrics_from_multistep_doc(doc)
    if not metrics or metrics.get("status") != "pass":
        return None
    generated = metrics.get("generated")
    return int(generated) if generated is not None else None


def bench_cosim_latency_from_multistep(path: Path) -> int | None:
    if not path.is_file():
        return None
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return bench_cosim_latency_from_multistep_doc(doc)


def bench_cosim_speedup(baseline_cycles: int | None, generated_cycles: int | None) -> float | None:
    return speedup(baseline_cycles, generated_cycles)


def bench_cosim_metrics_from_multistep_doc(
    doc: dict[str, Any],
    *,
    baseline_map: dict[str, int] | None = None,
    bench_short_name: str | None = None,
) -> dict[str, Any] | None:
    """Cosim status and latency for the winning kernel (phase B or flash).

    Latency and speedup are recorded only when cosim ran, passed, and reported
    ``kernel_runtime_cycles``.
    """
    if not doc.get("success") and not doc.get("steps"):
        return None

    issues = bench_run_issues_from_multistep_doc(doc)
    infra_failed = bool(set(issues) & CSYNTH_FAIL_RUN_ISSUES)
    cosim_expected = cosim_flow_requested(doc)

    winning_step = find_winning_step_name(doc)
    summary = cosim_summary_for_step(doc, winning_step)
    status = classify_cosim_status(summary)
    generated = cosim_latency_cycles(summary) if status == "pass" else None

    baseline = None
    if baseline_map and bench_short_name:
        baseline = baseline_map.get(bench_short_name)

    flash = _flash_step(doc)
    flash_infra_failed = bool(
        flash
        and not flash.get("success")
        and not flash.get("reverted_to_prev")
        and infra_failed
    )

    if status == "pass" and winning_step == "baseline" and flash_infra_failed:
        status = "fail"
        generated = None

    if status == "not_run" and cosim_expected and (infra_failed or flash_infra_failed):
        status = "fail"

    if status == "not_run" and baseline is None and not issues and not cosim_expected:
        return None

    return {
        "status": status,
        "winning_step": winning_step,
        "generated": generated,
        "baseline": baseline,
        "speedup": bench_cosim_speedup(baseline, generated) if status == "pass" else None,
        "ran": bool(isinstance(summary, dict) and summary.get("ran")),
        "expected": cosim_expected,
    }


def cosim_status_counts_from_benches(
    benches: dict[str, dict[str, Any]],
    *,
    bench_filter: set[str] | None = None,
) -> dict[str, int]:
    counts: dict[str, int] = {
        "pass": 0,
        "fail": 0,
        "crash": 0,
        "not_run": 0,
    }
    for bench, info in benches.items():
        if bench_filter is not None and bench not in bench_filter:
            continue
        if str(info.get("status") or "") not in ("ok", "done"):
            continue
        status = str((info.get("cosim") or {}).get("status") or "not_run")
        counts[status] = counts.get(status, 0) + 1
    return counts


def geomean_cosim_speedup_from_benches(
    benches: dict[str, dict[str, Any]],
    *,
    bench_filter: set[str] | None = None,
) -> dict[str, Any]:
    values: list[float] = []
    n = 0
    for bench, info in benches.items():
        if bench_filter is not None and bench not in bench_filter:
            continue
        if str(info.get("status") or "") not in ("ok", "done"):
            continue
        if bench_excluded_from_geomean(info):
            continue
        cosim = info.get("cosim") or {}
        if cosim.get("status") != "pass":
            continue
        sp = cosim.get("speedup")
        if sp is None or sp <= 0:
            continue
        n += 1
        values.append(float(sp))
    return {
        "n": n,
        "geomean": geomean(values),
        "status_counts": cosim_status_counts_from_benches(
            benches,
            bench_filter=bench_filter,
        ),
    }


def mean_latency_from_benches(
    benches: dict[str, dict[str, Any]],
    *,
    bench_filter: set[str] | None = None,
) -> dict[str, Any]:
    """Arithmetic mean of per-bench final csynth latency cycles."""
    buckets: dict[str, list[int]] = {"best": [], "avg": [], "worst": []}
    n = 0
    for bench, info in benches.items():
        if bench_filter is not None and bench not in bench_filter:
            continue
        if str(info.get("status") or "") not in ("ok", "done"):
            continue
        lat = info.get("latency") or {}
        if not any(lat.get(kind) for kind in ("best", "avg", "worst")):
            continue
        n += 1
        for kind in ("best", "avg", "worst"):
            value = lat.get(kind)
            if value is not None and value > 0:
                buckets[kind].append(int(value))

    def _mean(values: list[int]) -> float | None:
        if not values:
            return None
        return sum(values) / len(values)

    return {
        "n": n,
        "best": _mean(buckets["best"]),
        "avg": _mean(buckets["avg"]),
        "worst": _mean(buckets["worst"]),
    }


def geomean_from_bench_speedups(
    benches: dict[str, dict[str, Any]],
    *,
    bench_filter: set[str] | None = None,
) -> dict[str, Any]:
    """Compute geomean speedups over bench map values with optional speedup dicts."""
    buckets: dict[str, list[float]] = {"best": [], "avg": [], "worst": []}
    n = 0
    for bench, info in benches.items():
        if bench_filter is not None and bench not in bench_filter:
            continue
        if str(info.get("status") or "") not in ("ok", "done"):
            continue
        if bench_excluded_from_geomean(info):
            continue
        sp = info.get("speedup") or {}
        if not sp:
            continue
        n += 1
        for kind in ("best", "avg", "worst"):
            value = sp.get(kind)
            if value is not None and value > 0:
                buckets[kind].append(float(value))
    return {
        "n": n,
        "best": geomean(buckets["best"]),
        "avg": geomean(buckets["avg"]),
        "worst": geomean(buckets["worst"]),
    }
