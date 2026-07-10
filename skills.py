"""
Skill loader + retrieval for the c2hls agentic flow.

Skills come from the hls_full_optimization_skills_schema_1_1_package format
(see <skills_root>/README.md). Each skill has fields like `id`, `pattern`,
`strategy`, `kind`, `required_steps`, `guards`, `template`, `bottleneck_kinds`,
`tags`, `confidence`.

This module is benchmark-neutral: it never embeds benchmark names. It returns
a list of skill dicts to be formatted into the LLM prompt, plus a helper that
formats them into a markdown block to append after a normal prompt.

Enable from c2hls.py by setting C2HLS_SKILLS_PATH=<path-to-skills.json>.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Iterable

_log = logging.getLogger(__name__)


# -- skills config introspection ------------------------------------------

def skills_config_path() -> str | None:
    """Return the raw C2HLS_SKILLS_PATH env var (colon-separated paths or a
    single path). None when the env var is unset or empty after stripping.
    """
    raw = os.environ.get("C2HLS_SKILLS_PATH", "")
    raw = raw.strip()
    return raw or None


def skills_config_sha1() -> str | None:
    """Return sha1 of the concatenated content of all paths listed in
    C2HLS_SKILLS_PATH (colon-separated), or None if no path is set or
    nothing readable.
    """
    raw = skills_config_path()
    if not raw:
        return None
    paths = [p.strip() for p in raw.split(":") if p.strip()]
    h = hashlib.sha1()
    any_read = False
    for p in paths:
        try:
            data = Path(p).read_bytes()
        except OSError:
            continue
        any_read = True
        h.update(data)
    if not any_read:
        return None
    return h.hexdigest()


# -- per-callsite input digest --------------------------------------------

def _inputs_digest(callsite: str, **kwargs) -> dict:
    """Build a small dict describing the inputs that drove a retrieval call.

    Shape depends on `callsite` (one of "translation", "error", "quality"):
      - translation: {"target_part"}
      - error:       {"target_part", "error_excerpt"}
      - quality:     {"target_part", "report_keys", "comparison_keys",
                      and (when present) "slack_ns", "latency_ratio",
                      "fmax_ratio"}
    """
    target_part = kwargs.get("target_part")
    if callsite == "translation":
        return {"target_part": target_part}
    if callsite == "error":
        error_text = kwargs.get("error_text") or ""
        return {
            "target_part": target_part,
            "error_excerpt": error_text[:120],
        }
    if callsite == "quality":
        report = kwargs.get("report") or {}
        comparison = kwargs.get("comparison") or {}
        digest = {
            "target_part": target_part,
            "report_keys": sorted(list(report.keys()))[:8],
            "comparison_keys": sorted(list(comparison.keys()))[:8] if comparison else [],
        }
        # Surface a few key quality signals when present.
        slack = report.get("slack_ns") if isinstance(report, dict) else None
        if slack is not None:
            digest["slack_ns"] = slack
        if isinstance(comparison, dict):
            def _ratio(key: str):
                v = (comparison.get(key) or {}) if isinstance(comparison.get(key), dict) else {}
                return v.get("ratio")
            lat_r = _ratio("latency_ns")
            if lat_r is not None:
                digest["latency_ratio"] = lat_r
            fmax_r = _ratio("fmax_mhz")
            if fmax_r is not None:
                digest["fmax_ratio"] = fmax_r
        return digest
    return {"target_part": target_part}


def load_skills(path: str | Path) -> list[dict]:
    """Parse skills.json. Returns [] on any error.

    `path` may be a single file path or a colon-separated list of paths
    (POSIX-style). When multiple paths are given, the skills lists are
    concatenated in order; later files extend the earlier ones. This lets
    callers layer an extension (e.g. skills_extension.json with hard_guard
    additions) on top of the base curated skills.
    """
    paths = [p.strip() for p in str(path).split(":") if p.strip()]
    all_skills: list[dict] = []
    for p in paths:
        try:
            data = json.loads(Path(p).read_text())
            skills = data.get("skills", [])
            _log.info("Loaded %d skills from %s", len(skills), p)
            all_skills.extend(skills)
        except Exception as e:
            _log.warning("Failed to load skills from %s: %s", p, e)
    return all_skills


# -- formatting -----------------------------------------------------------

def _format_skill(s: dict, verbose: bool = True) -> str:
    """Format one skill as a Markdown stanza.

    Phase 6 (memo 2026-05-27): `verbose=False` drops the Required-steps
    list, which is the bulkiest field on most skills. Use the compact form
    for the initial-translation prompt where the agent gets the broad
    intent but doesn't need every imperative step.
    """
    lines: list[str] = [f"### {s['id']}"]
    kind = s.get("kind")
    confidence = s.get("confidence")
    if kind or confidence:
        bits = []
        if kind:
            bits.append(f"kind={kind}")
        if confidence:
            bits.append(f"confidence={confidence}")
        lines.append(f"({', '.join(bits)})")
    if s.get("pattern"):
        lines.append(f"When applicable: {s['pattern']}")
    if s.get("strategy"):
        lines.append(f"Strategy: {s['strategy']}")
    if verbose:
        steps = s.get("required_steps") or []
        if steps:
            lines.append("Required steps:")
            for x in steps:
                lines.append(f"  - {x}")
    guards = s.get("guards") or []
    if guards:
        lines.append("Guards:")
        for x in guards:
            lines.append(f"  - {x}")
    return "\n".join(lines)


def render_skill_block(skills_list: list[dict],
                       header: str = "Optimization skills you may reference",
                       verbose: bool = True) -> str:
    """Render the retrieved skills into a Markdown block.

    Phase 6: `verbose=False` produces a more compact block (no required-steps
    bullets) for translation prompts that need pattern/strategy/guards but
    not the full step recipe.
    """
    if not skills_list:
        return ""
    body = "\n\n".join(_format_skill(s, verbose=verbose) for s in skills_list)
    instructions = (
        "Apply only the skills whose `When applicable` matches the current state. "
        + ("Treat `Required steps` as a checklist and `Guards` as hard constraints. "
           if verbose
           else "Treat `Guards` as hard constraints. ")
        + "Avoid-rules (kind=avoid_rule) describe things NOT to do."
    )
    return (
        f"\n\n## {header} (curated, benchmark-neutral)\n\n"
        f"{instructions}\n\n"
        f"{body}\n"
    )


# -- retrieval ------------------------------------------------------------

def _by_ids(skills: list[dict], ids: Iterable[str]) -> list[dict]:
    wanted = set(ids)
    return [s for s in skills if s.get("id") in wanted]


def _with_bottleneck(skills: list[dict], kinds: Iterable[str]) -> list[dict]:
    wanted = set(kinds)
    out = []
    for s in skills:
        bk = set(s.get("bottleneck_kinds") or [])
        if bk & wanted:
            out.append(s)
    return out


def _avoid_rules(skills: list[dict]) -> list[dict]:
    return [s for s in skills if s.get("kind") == "avoid_rule" or s.get("confidence") == "avoid"]


def _dedup_keep_order(skills: list[dict]) -> list[dict]:
    seen: set[str] = set()
    out: list[dict] = []
    for s in skills:
        sid = s.get("id", "")
        if sid in seen:
            continue
        seen.add(sid)
        out.append(s)
    return out


def _platform_ok(s: dict, target_part: str | None) -> bool:
    """Phase 4 (memo 2026-05-27): respect the `applicable_fpgas` field that
    most skills set to []. A non-empty list means "this skill is specific to
    these parts" — anything else should drop it from retrieval to avoid
    misfires like the Artix-7-on-coalescing case the reviewer flagged.

    An empty list (or missing field) = platform-neutral, always kept.
    When `target_part` is None, no filtering happens (back-compat).
    """
    if not target_part:
        return True
    fpgas = s.get("applicable_fpgas") or []
    return not fpgas or target_part in fpgas


def _filter_platform(skills: list[dict], target_part: str | None) -> list[dict]:
    return [s for s in skills if _platform_ok(s, target_part)]


# --- "all positive skills" sourcing mode --------------------------------
# When C2HLS_SKILLS_ALL_POSITIVE is set, every retrieve_for_* callsite returns
# the FULL set of positive (constructive, non-preventative) skills instead of
# a curated top_k subset. "Preventative" = avoid-rules + guards (kind in
# {avoid_rule, hard_guard, conditional_guard} or confidence == 'avoid').
# This is an ablation of the skill-SOURCING strategy: breadth (all positives,
# every turn) vs the default curated subset. Read at import (per-process).
_ALL_POSITIVE_MODE = os.getenv("C2HLS_SKILLS_ALL_POSITIVE", "").strip().lower() \
    not in ("", "0", "false", "no")
_PREVENTATIVE_KINDS = {"avoid_rule", "hard_guard", "conditional_guard"}


def _all_positive_skills(skills: list[dict], target_part: str | None) -> list[dict]:
    """All constructive skills (drop avoid-rules + guards), platform-filtered,
    deterministic catalog order, no top_k cap."""
    pos = [s for s in skills
           if s.get("kind") not in _PREVENTATIVE_KINDS
           and s.get("confidence") != "avoid"]
    return _dedup_keep_order(_filter_platform(pos, target_part))


def _hard_guards(skills: list[dict]) -> list[dict]:
    """Always-included guardrail skills. Never filtered out by top_k caps.
    Excludes conditional_guard (those are state-dependent, see _conditional_guards_for_report)."""
    return [s for s in skills if s.get("kind") == "hard_guard"]


def _conditional_guards_for_report(skills: list[dict], report: dict | None,
                                   error_text: str | None = None) -> list[dict]:
    """Return any conditional_guard skills whose trigger condition is satisfied
    by the current synth report or error text. Conservative: returns empty list
    when no report is available (i.e. at translation time before Phase B).
    """
    if not report and not error_text:
        return []
    triggered: list[dict] = []
    for s in skills:
        if s.get("kind") != "conditional_guard":
            continue
        trigger = s.get("trigger") or {}

        # Resource-pressure trigger: any_resource_ratio_above_limit_pct
        thr_pct = trigger.get("any_resource_ratio_above_limit_pct")
        limits = trigger.get("device_limits") or {}
        if thr_pct is not None and limits and report:
            for res, limit in limits.items():
                try:
                    used = float(report.get(res) or 0)
                except (TypeError, ValueError):
                    used = 0.0
                if limit > 0 and (used / limit) * 100.0 >= thr_pct:
                    triggered.append(s)
                    break  # next skill

        # Error-keyword trigger: error_keywords (substring match)
        kws = trigger.get("error_keywords") or []
        if kws and error_text:
            err_l = error_text.lower()
            if any(kw.lower() in err_l for kw in kws):
                if s not in triggered:
                    triggered.append(s)
    return triggered


def retrieve_for_translation(skills: list[dict], top_k: int = 4,
                             target_part: str | None = None) -> list[dict]:
    """Initial C->HLS translation: hard_guards first, then 'prompt-*' + key avoids.

    Phase 4: when `target_part` is set, drop skills whose `applicable_fpgas`
    is non-empty and excludes the target. Empty `applicable_fpgas` = neutral.

    Phase 6 (memo 2026-05-27): default `top_k` lowered from 8 to 4. The
    initial-translation prompt was ~9 KB of skills text and the agent
    largely skimmed it. Hard_guards still always pass through (they're
    appended after the top_k cap). Combined with the verbose=False rendering
    used by the translation callsite, the block shrinks by roughly half.
    """
    if _ALL_POSITIVE_MODE:
        return _all_positive_skills(skills, target_part)
    skills = _filter_platform(skills, target_part)
    guards = _hard_guards(skills)
    prompts = [s for s in skills if (s.get("id") or "").startswith("prompt-")]
    avoids = _by_ids(
        skills,
        [
            "hls-avoid-pipeline-pragma-only",
            "hls-avoid-superficial-tiling",
            "hls-avoid-coalescing-interface-only",
            "avoid-over-unroll-axi-dep",
        ],
    )
    # hard_guards always included, in addition to top_k selected
    return _dedup_keep_order(guards + prompts + avoids)[: top_k + len(guards)]


def retrieve_for_error(skills: list[dict], error_text: str, top_k: int = 5,
                       target_part: str | None = None) -> list[dict]:
    """Synth/compile error: route to relevant avoid-rules + targeted transformation skills."""
    if _ALL_POSITIVE_MODE:
        return _all_positive_skills(skills, target_part)
    skills = _filter_platform(skills, target_part)
    err = (error_text or "").lower()
    picked: list[dict] = list(_hard_guards(skills))  # hard guards always included
    # Conditional guards: triggered by error keywords (e.g. "resource", "DSP", "BRAM")
    picked.extend(_conditional_guards_for_report(skills, report=None, error_text=error_text))

    if "pragma" in err and ("scope" in err or "function" in err):
        picked += _by_ids(skills, ["hls-avoid-pipeline-pragma-only"])
    if "ii" in err or "interval" in err or "pipeline" in err:
        picked += _with_bottleneck(skills, ["ii_target_miss", "non_pipelined_hot_loop", "pipeline_blocked"])
    if "memory" in err or "axi" in err or "burst" in err or "bandwidth" in err:
        picked += _with_bottleneck(skills, ["memory_bandwidth", "axi_burst_failed", "port_conflict"])
    if "resource" in err or "lut" in err or "ff" in err or "bram" in err:
        picked += _by_ids(skills, ["avoid-over-unroll-axi-dep", "hls-avoid-unroll-resource-explosion"])
    if "timed out" in err or "timeout" in err:
        picked += _by_ids(
            skills,
            [
                "hls-avoid-unroll-resource-explosion",
                "hls-avoid-superficial-tiling",
                "hls-pipeline-realistic-ii-selection",
            ],
        )

    if not picked:
        picked = _avoid_rules(skills)[:3]

    return _dedup_keep_order(picked)[:top_k]


def retrieve_for_quality(skills: list[dict], report: dict, comparison: dict | None,
                         top_k: int = 6, target_part: str | None = None) -> list[dict]:
    """Quality repair: pick skills based on which metric is off vs the gold baseline."""
    if _ALL_POSITIVE_MODE:
        return _all_positive_skills(skills, target_part)
    skills = _filter_platform(skills, target_part)
    report = report or {}
    comparison = comparison or {}
    picked: list[dict] = list(_hard_guards(skills))  # hard guards always included
    # Conditional guards: state-aware (e.g. device-budget fires only when usage > 50% of limit)
    picked.extend(_conditional_guards_for_report(skills, report=report))

    def _ratio(key: str) -> float | None:
        v = (comparison.get(key) or {}).get("ratio")
        try:
            return float(v) if v is not None else None
        except (TypeError, ValueError):
            return None

    # Timing
    slack = report.get("slack_ns")
    try:
        slack_v = float(slack) if slack is not None else None
    except (TypeError, ValueError):
        slack_v = None
    fmax_ratio = _ratio("fmax_mhz")
    if (slack_v is not None and slack_v < 0) or (fmax_ratio is not None and fmax_ratio < 0.8):
        picked += _by_ids(
            skills,
            [
                "hls-pipeline-realistic-ii-selection",
                "hls-pipeline-hot-loop-achieve-ii",
                "partition-cyclic-on-port-conflict",
            ],
        )

    # Latency / II
    latency_ratio = _ratio("latency_ns")
    if latency_ratio is not None and latency_ratio > 1.5:
        picked += _by_ids(
            skills,
            [
                "hls-pipeline-hot-loop-achieve-ii",
                "hls-tile-compute-inner-parallelism",
                "hls-tile-doublebuffer-load-compute",
            ],
        )

    # FF / LUT explosion (over-unrolled)
    if (_ratio("ff") or 0) > 1.3 or (_ratio("lut") or 0) > 1.3:
        picked += _by_ids(
            skills,
            [
                "avoid-over-unroll-axi-dep",
                "hls-avoid-unroll-resource-explosion",
                "hls-pipeline-bank-local-buffers",
            ],
        )

    # BRAM blow-up (superficial tiling)
    if (_ratio("bram") or 0) > 1.5:
        picked += _by_ids(
            skills,
            [
                "hls-avoid-superficial-tiling",
                "hls-avoid-tiling-without-reuse",
                "hls-tile-compute-inner-parallelism",
            ],
        )

    # DSP under-use vs GT
    dsp_r = _ratio("dsp")
    if dsp_r is not None and dsp_r < 0.5:
        picked += _by_ids(
            skills,
            [
                "prompt-unroll",
                "hls-coalescing-compute-lane-parallelism",
            ],
        )

    # Always include core guardrails
    picked += _by_ids(
        skills,
        [
            "hls-avoid-pipeline-pragma-only",
            "hls-avoid-coalescing-interface-only",
        ],
    )

    return _dedup_keep_order(picked)[:top_k]


# -- metadata-returning wrappers ------------------------------------------
# These mirror the retrieve_for_X functions but also return a per-call meta
# dict describing the inputs and the chosen top_k. They keep behavior
# identical by delegating to the existing functions.

def retrieve_for_translation_with_meta(
    skills: list[dict],
    *,
    target_part: str | None = None,
    top_k: int | None = None,
) -> tuple[list[dict], dict]:
    effective_top_k = 4 if top_k is None else top_k
    selected = retrieve_for_translation(skills, top_k=effective_top_k, target_part=target_part)
    meta = {
        "target_part": target_part,
        "top_k": effective_top_k,
        "inputs_digest": _inputs_digest("translation", target_part=target_part),
    }
    return selected, meta


def retrieve_for_error_with_meta(
    skills: list[dict],
    error_text: str,
    *,
    target_part: str | None = None,
    top_k: int | None = None,
) -> tuple[list[dict], dict]:
    effective_top_k = 5 if top_k is None else top_k
    selected = retrieve_for_error(skills, error_text, top_k=effective_top_k, target_part=target_part)
    meta = {
        "target_part": target_part,
        "top_k": effective_top_k,
        "inputs_digest": _inputs_digest(
            "error", target_part=target_part, error_text=error_text
        ),
    }
    return selected, meta


def retrieve_for_quality_with_meta(
    skills: list[dict],
    report: dict,
    comparison: dict | None,
    *,
    target_part: str | None = None,
    top_k: int | None = None,
) -> tuple[list[dict], dict]:
    effective_top_k = 6 if top_k is None else top_k
    selected = retrieve_for_quality(
        skills, report, comparison, top_k=effective_top_k, target_part=target_part
    )
    meta = {
        "target_part": target_part,
        "top_k": effective_top_k,
        "inputs_digest": _inputs_digest(
            "quality",
            target_part=target_part,
            report=report,
            comparison=comparison,
        ),
    }
    return selected, meta
