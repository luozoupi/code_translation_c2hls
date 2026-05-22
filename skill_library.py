"""Confidence-tagged HLS optimization skill library (Pillar 3).

A skill is one (pattern, strategy) pair plus a transformation template, the
versions/FPGAs it has been validated on, and running statistics so the
agent can pick reliable transformations over speculative ones.

The library is authoritative across all Phase-2 pillars:

- **Pillar 3** owns the storage + statistics. Skills live in
  `skills/skills.yaml` (or .json — see `_DEFAULT_STORE`) so they survive
  across runs and can be hand-edited.
- **Pillar 5** consults it via `query_skills_for_bottleneck()` when the
  bottleneck-router needs to decide what to apply next.
- **Pillar 6** filters by `(vitis_version, fpga_target)` — a skill that
  worked on 2023.2 may be redundant on 2025.2 (auto-applied) or have
  been deprecated.
- **Pillar 7** uses the `Avoid` confidence tier for "Vitis already does
  this" entries — the negative-knowledge half of the library.

Updates use group-relative advantage (Dr. RTL's signal): a skill's
running mean advantage is updated whenever it appears in a candidate that
was promoted, with the relative score against the same-iteration peers.

The bootstrap path uses
[prompt_c2hls.OPTIMIZATION_PROMPTS](prompt_c2hls.py) so the library is
non-empty even before the first agent run. Prompt-derived skills start
at confidence `medium` until trajectories provide statistics.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# Confidence tiers. Order matters — used for ranking when multiple skills
# match a bottleneck.
TIER_HIGH = "high"
TIER_MEDIUM = "medium"
TIER_LOW = "low"
TIER_AVOID = "avoid"
CONFIDENCE_TIERS = (TIER_HIGH, TIER_MEDIUM, TIER_LOW, TIER_AVOID)
_TIER_RANK = {TIER_HIGH: 0, TIER_MEDIUM: 1, TIER_LOW: 2, TIER_AVOID: 3}

# Default store path — relative to the repo root (the parent of this file).
SCHEMA_VERSION = "1.1"
_REPO_ROOT = Path(__file__).resolve().parent
_DEFAULT_STORE = _REPO_ROOT / "skills" / "skills.json"
_PACKAGED_SKILLS = (
    _REPO_ROOT
    / "hls_full_optimization_skills_schema_1_1_package"
    / "skills.json"
)


@dataclass
class Skill:
    """One reusable optimization recipe.

    Fields are deliberately flat (no nested objects) so the JSON wire
    format is easy to diff and hand-edit. Identifier conventions:

    - `id` is a stable handle; we keep it kebab-case
      (e.g. ``pipeline-inner-loop-ii1``).
    - `bottleneck_kinds` is the list of `hls_feedback`
      `BottleneckRecord.kind` values this skill addresses; empty means
      the skill is generic / applicable to any bottleneck.
    - `applicable_versions` / `applicable_fpgas` are include-lists. Empty
      = applicable everywhere.
    - `tags` is freeform metadata (e.g. ``stencil``, ``dataflow``,
      ``synth-absorbs``).
    - Statistics (`occurrences`, `sec_pass`, `mean_advantage`,
      `last_used_at`) are accumulated by `update_skill_statistics()`.
    """
    id: str
    pattern: str           # human-readable bottleneck description
    strategy: str          # human-readable transformation principle
    template: str = ""     # before/after code template (may be empty)
    confidence: str = TIER_MEDIUM
    kind: str = ""         # optional schema-1.1 category, e.g. avoid_rule
    bottleneck_kinds: List[str] = field(default_factory=list)
    applicable_versions: List[str] = field(default_factory=list)
    applicable_fpgas: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    guards: List[str] = field(default_factory=list)
    required_steps: List[str] = field(default_factory=list)
    # Running statistics.
    occurrences: int = 0
    sec_pass: int = 0          # csim+cosim-passing applications
    mean_advantage: float = 0.0
    last_used_at: Optional[str] = None
    # Source provenance — `prompt`, `paper`, `agent`, `manual`.
    origin: str = "manual"


_LIST_FIELDS = {
    "bottleneck_kinds",
    "applicable_versions",
    "applicable_fpgas",
    "tags",
    "guards",
    "required_steps",
}
_INT_FIELDS = {"occurrences", "sec_pass"}
_FLOAT_FIELDS = {"mean_advantage"}
_STATS_FIELDS = {"occurrences", "sec_pass", "mean_advantage", "last_used_at"}


def _as_str_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if item is not None]
    return [str(value)]


def _coerce_skill_entry(entry: Any) -> Optional[Skill]:
    """Accept schema-1.0 and schema-1.1 skill JSON without silent loss.

    Older stores do not contain `kind`, `guards`, or `required_steps`; the
    professionally curated package does. Unknown future fields are ignored so
    a strict dataclass constructor does not discard whole skill entries.
    """
    if not isinstance(entry, dict):
        logging.warning("skipping malformed skill entry: expected object, got %s", type(entry).__name__)
        return None
    known = {f.name for f in fields(Skill)}
    data = {k: v for k, v in entry.items() if k in known}
    missing = [name for name in ("id", "pattern", "strategy") if not data.get(name)]
    if missing:
        logging.warning("skipping malformed skill entry %s: missing %s", entry.get("id"), ", ".join(missing))
        return None
    for key in _LIST_FIELDS:
        data[key] = _as_str_list(data.get(key))
    for key in ("id", "pattern", "strategy", "template", "confidence", "kind", "origin"):
        if data.get(key) is None:
            data[key] = ""
        else:
            data[key] = str(data.get(key))
    for key in _INT_FIELDS:
        try:
            data[key] = int(data.get(key, 0) or 0)
        except (TypeError, ValueError):
            data[key] = 0
    for key in _FLOAT_FIELDS:
        try:
            data[key] = float(data.get(key, 0.0) or 0.0)
        except (TypeError, ValueError):
            data[key] = 0.0
    if data.get("last_used_at") is not None:
        data["last_used_at"] = str(data["last_used_at"])
    if data.get("confidence") not in CONFIDENCE_TIERS:
        data["confidence"] = TIER_LOW
    if not data.get("origin"):
        data["origin"] = "manual"
    try:
        return Skill(**data)
    except TypeError as exc:
        logging.warning("skipping malformed skill entry %s: %s", entry.get("id"), exc)
        return None


def _preserve_observed_stats(incoming: Skill, existing: Skill) -> Skill:
    data = asdict(incoming)
    for key in _STATS_FIELDS:
        data[key] = getattr(existing, key)
    return Skill(**data)


# === Storage ==============================================================


class SkillLibrary:
    """In-memory + on-disk skill collection. Thread-safe for the simple
    "load → query → update → save" pattern; not intended for high-
    concurrency parallel writers."""

    def __init__(self, store_path: Optional[Path] = None):
        self.store_path = Path(store_path or _DEFAULT_STORE)
        self._skills: Dict[str, Skill] = {}
        self._lock = threading.Lock()

    # ---- IO --------------------------------------------------------------

    def load(self) -> "SkillLibrary":
        if not self.store_path.exists():
            self._skills = {}
            return self
        try:
            data = json.loads(self.store_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logging.warning("SkillLibrary load failed (%s); starting empty", exc)
            self._skills = {}
            return self
        skills_raw = data.get("skills", []) if isinstance(data, dict) else data
        out: Dict[str, Skill] = {}
        for entry in skills_raw or []:
            sk = _coerce_skill_entry(entry)
            if sk is None:
                continue
            out[sk.id] = sk
        self._skills = out
        return self

    def save(self) -> None:
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": SCHEMA_VERSION,
            "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "skills": [asdict(sk) for sk in self._skills.values()],
        }
        tmp = self.store_path.with_suffix(self.store_path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(tmp, self.store_path)

    # ---- Mutators --------------------------------------------------------

    def add(self, skill: Skill, *, overwrite: bool = False,
            preserve_stats: bool = False) -> None:
        with self._lock:
            existing = self._skills.get(skill.id)
            if existing is not None and not overwrite:
                return
            if existing is not None and preserve_stats:
                skill = _preserve_observed_stats(skill, existing)
            self._skills[skill.id] = skill

    def remove(self, skill_id: str) -> bool:
        with self._lock:
            return self._skills.pop(skill_id, None) is not None

    def update_skill_statistics(
        self,
        skill_id: str,
        *,
        success: bool,
        relative_advantage: Optional[float] = None,
    ) -> Optional[Skill]:
        """Accumulate one observation. `relative_advantage` is the
        Dr. RTL-style group-relative score used by the optimization
        agent; pass None when not available (the running mean stays
        unchanged)."""
        with self._lock:
            sk = self._skills.get(skill_id)
            if sk is None:
                return None
            sk.occurrences += 1
            if success:
                sk.sec_pass += 1
            if relative_advantage is not None:
                # Welford's running mean update.
                n = max(1, sk.occurrences)
                sk.mean_advantage += (relative_advantage - sk.mean_advantage) / n
            sk.last_used_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
            return sk

    def promote_demote(self, skill_id: str) -> Optional[Skill]:
        """Auto-tier based on observed statistics. Conservative
        thresholds — see the inline policy."""
        with self._lock:
            sk = self._skills.get(skill_id)
            if sk is None or sk.occurrences < 3:
                return sk
            pass_rate = sk.sec_pass / max(1, sk.occurrences)
            adv = sk.mean_advantage
            new_tier = sk.confidence
            if pass_rate >= 0.6 and adv >= 0.0:
                new_tier = TIER_HIGH
            elif pass_rate >= 0.4 and adv > -0.2:
                new_tier = TIER_MEDIUM
            elif pass_rate < 0.2 or adv < -0.5:
                new_tier = TIER_LOW
            sk.confidence = new_tier
            return sk

    def mark_avoid(self, skill_id: str, *, reason: str = "absorbed-by-synth") -> Optional[Skill]:
        """Move a skill into the Avoid band — Pillar 7's
        'Vitis already does this'."""
        with self._lock:
            sk = self._skills.get(skill_id)
            if sk is None:
                return None
            sk.confidence = TIER_AVOID
            if reason and reason not in sk.tags:
                sk.tags = list(sk.tags) + [reason]
            return sk

    # ---- Queries ---------------------------------------------------------

    def all(self) -> List[Skill]:
        with self._lock:
            return list(self._skills.values())

    def get(self, skill_id: str) -> Optional[Skill]:
        with self._lock:
            return self._skills.get(skill_id)

    def query(
        self,
        *,
        bottleneck_kind: Optional[str] = None,
        vitis_version: Optional[str] = None,
        fpga: Optional[str] = None,
        include_avoid: bool = False,
    ) -> List[Skill]:
        """Return all skills matching the (bottleneck, version, fpga)
        filter, ordered by (confidence tier, mean_advantage descending)."""
        with self._lock:
            cands: List[Skill] = []
            for sk in self._skills.values():
                if not include_avoid and sk.confidence == TIER_AVOID:
                    continue
                if bottleneck_kind and sk.bottleneck_kinds and bottleneck_kind not in sk.bottleneck_kinds:
                    continue
                if vitis_version and sk.applicable_versions and vitis_version not in sk.applicable_versions:
                    continue
                if fpga and sk.applicable_fpgas and fpga not in sk.applicable_fpgas:
                    continue
                cands.append(sk)
        cands.sort(key=lambda s: (_TIER_RANK[s.confidence], -s.mean_advantage, s.id))
        return cands


# === Bootstrap from existing OPTIMIZATION_PROMPTS =========================


def _bootstrap_skills_from_prompts() -> List[Skill]:
    """Seed the library with one Skill per entry in
    [prompt_c2hls.OPTIMIZATION_PROMPTS](prompt_c2hls.py).

    These skills carry no statistics yet (occurrences=0, mean_advantage=0)
    and start at `medium` confidence. Once trajectory data arrives,
    `promote_demote()` will move them up or down.
    """
    skills: List[Skill] = []

    # Mapping from existing step name → bottleneck kinds the step typically
    # addresses. Conservative — based on hls_feedback's bottleneck taxonomy.
    step_bottleneck_kinds = {
        "tiling": ["non_pipelined_hot_loop", "interval_exceeds_latency"],
        "pipeline": ["non_pipelined_hot_loop", "ii_target_miss"],
        "unroll": ["ii_target_miss", "non_pipelined_hot_loop"],
        "doublebuffer": ["interval_exceeds_latency", "dataflow_blocked"],
        "coalescing": ["port_conflict", "interval_exceeds_latency"],
    }
    # One-line description of each step's pattern → strategy. The full
    # prompt body lives in prompt_c2hls; we keep the skill record compact
    # so the agent's prompt context stays short.
    step_pattern_strategy = {
        "tiling": (
            "outer loop with high trip count over a large array, no on-chip buffer",
            "split iteration space into tiles, load tile to a local buffer, "
            "compute on the tile, store result back",
        ),
        "pipeline": (
            "scalar loop body without #pragma HLS PIPELINE",
            "annotate the innermost feasible loop with PIPELINE II=1 and "
            "supporting array_partition/dependence pragmas",
        ),
        "unroll": (
            "data-parallel inner loop pipelined at II=1 but bottlenecked by "
            "single arithmetic unit",
            "apply UNROLL factor=2/4/8 to the inner loop, keeping the partition "
            "factor on consumed arrays in lockstep",
        ),
        "doublebuffer": (
            "interval exceeds latency — load and compute serialize",
            "create two ping-pong buffer copies; alternate load/compute across "
            "iterations to overlap DRAM read with kernel compute",
        ),
        "coalescing": (
            "narrow per-element AXI burst on a contiguous read/write loop",
            "preserve the public kernel ABI by default; first improve burst "
            "lengths, outstanding transactions, local staging, and tail-safe "
            "contiguous access. Only use a wide-bus ABI when the active "
            "variant explicitly supports it.",
        ),
    }

    for step, kinds in step_bottleneck_kinds.items():
        pattern, strategy = step_pattern_strategy[step]
        skills.append(Skill(
            id=f"prompt-{step}",
            pattern=pattern,
            strategy=strategy,
            confidence=TIER_MEDIUM,
            bottleneck_kinds=kinds,
            applicable_versions=[],   # broad
            applicable_fpgas=[],
            tags=["prompt-derived", step],
            origin="prompt",
        ))

    # A small set of high-confidence "always safe" skills extracted from
    # the prior knowledge of HLS engineers — lets the router have wins on
    # day zero before any trajectories arrive.
    skills.extend(_default_high_confidence_skills())

    # Avoid-band entries: things newer Vitis already does. Empty ranges
    # for `applicable_versions` mean "applies to all observed versions",
    # which is conservative; populate as Pillar 7 detects more.
    skills.extend(_default_avoid_skills())

    return skills


def _default_high_confidence_skills() -> List[Skill]:
    return [
        Skill(
            id="axi-burst-coalescing-narrow-safe",
            pattern=(
                "m_axi port feeds a contiguous streaming loop, but the active "
                "host/testbench contract expects the original narrow pointer "
                "types"
            ),
            strategy=(
                "keep the top-level workload signature unchanged; add AXI "
                "burst/outstanding pragmas, stage contiguous data into local "
                "buffers, pipeline the local compute/store loops, and preserve "
                "scalar tail handling"
            ),
            template=(
                "#pragma HLS INTERFACE m_axi port=in  offset=slave bundle=gmem "
                "max_read_burst_length=64 num_read_outstanding=16\n"
                "#pragma HLS INTERFACE m_axi port=out offset=slave bundle=gmem "
                "max_write_burst_length=64 num_write_outstanding=16\n"
                "// Keep pointer element types unchanged unless metadata marks the variant wide_bus."
            ),
            confidence=TIER_HIGH,
            bottleneck_kinds=[
                "memory_bandwidth",
                "axi_burst_failed",
                "port_conflict",
                "interval_exceeds_latency",
            ],
            applicable_versions=["2023.2"],
            applicable_fpgas=["xcu280-fsvh2892-2L-e"],
            origin="manual",
            tags=["coalescing", "narrow-abi", "u280", "safe-default"],
        ),
        Skill(
            id="axi-burst-widening-512",
            pattern=(
                "m_axi port feeds a pipelined loop and latency remains "
                "bandwidth dominated; burst.xml shows narrow or failed AXI "
                "burst inference"
            ),
            strategy=(
                "widen contiguous AXI transfers to 512 bits, unpack into a "
                "narrow local buffer, compute locally, and pack stores with "
                "tail handling"
            ),
            template=(
                "#include <ap_int.h>\n"
                "typedef ap_uint<512> wide_t;\n"
                "#pragma HLS INTERFACE m_axi port=in  offset=slave bundle=gmem "
                "max_read_burst_length=64 num_read_outstanding=16\n"
                "#pragma HLS INTERFACE m_axi port=out offset=slave bundle=gmem "
                "max_write_burst_length=64 num_write_outstanding=16\n"
                "// WIDTH_FACTOR = 512 / (8 * sizeof(element)); handle scalar tail explicitly"
            ),
            confidence=TIER_HIGH,
            bottleneck_kinds=[
                "memory_bandwidth",
                "axi_burst_failed",
                "port_conflict",
                "interval_exceeds_latency",
            ],
            applicable_versions=["2023.2"],
            applicable_fpgas=["xcu280-fsvh2892-2L-e"],
            origin="manual",
            tags=["coalescing", "wide-bus", "u280", "validated"],
        ),
        Skill(
            id="local-axi-staging-for-ii",
            pattern=(
                "AXI read dependency causes high II but the loop consumes a "
                "reusable contiguous tile"
            ),
            strategy=(
                "load a bounded tile into a local buffer first, pipeline the "
                "local compute loop, and avoid DSP-heavy unroll unless later "
                "steps prove arithmetic-bound"
            ),
            template=(
                "float tile[TILE];\n"
                "load_tile: for (int i = 0; i < TILE; ++i) tile[i] = in[base + i];\n"
                "compute_tile: for (int i = 0; i < TILE; ++i) {\n"
                "#pragma HLS PIPELINE II=1\n"
                "    ... tile[i] ...\n"
                "}\n"
            ),
            confidence=TIER_HIGH,
            bottleneck_kinds=["ii_target_miss", "loop_carried_dep", "port_conflict"],
            applicable_versions=["2023.2"],
            applicable_fpgas=["xcu280-fsvh2892-2L-e"],
            origin="manual",
            tags=["tiling", "local-buffer", "u280", "validated"],
        ),
        Skill(
            id="avoid-over-unroll-axi-dep",
            pattern=(
                "II miss is caused by AXI bandwidth/port dependency rather "
                "than arithmetic throughput"
            ),
            strategy=(
                "do not respond with large UNROLL or complete partitioning; "
                "route to local staging or 512-bit burst widening instead"
            ),
            confidence=TIER_AVOID,
            bottleneck_kinds=["memory_bandwidth", "axi_burst_failed", "ii_target_miss"],
            applicable_versions=["2023.2"],
            applicable_fpgas=["xcu280-fsvh2892-2L-e"],
            origin="manual",
            tags=["avoid", "over-unroll", "axi"],
        ),
        Skill(
            id="partition-cyclic-on-port-conflict",
            pattern="multiple parallel reads to a single BRAM port causing II>1",
            strategy="cyclic array_partition with factor matching the unroll factor",
            template=(
                "// BEFORE\n"
                "float buf[N];\n"
                "// AFTER\n"
                "float buf[N];\n"
                "#pragma HLS array_partition variable=buf cyclic factor=4 dim=1\n"
            ),
            confidence=TIER_HIGH,
            bottleneck_kinds=["port_conflict", "ii_target_miss"],
            origin="manual",
            tags=["partition", "always-safe"],
        ),
        Skill(
            id="dependence-inter-false-on-accum",
            pattern=(
                "loop pipelined at II>1 because Vitis pessimistically assumes a "
                "loop-carried dependence on an accumulator that the user knows is independent"
            ),
            strategy="add #pragma HLS dependence variable=<acc> inter false",
            template=(
                "for (int i = 0; i < N; ++i) {\n"
                "#pragma HLS pipeline II=1\n"
                "#pragma HLS dependence variable=accum inter false\n"
                "    accum[i] = ...;\n"
                "}\n"
            ),
            confidence=TIER_HIGH,
            bottleneck_kinds=["loop_carried_dep", "ii_target_miss", "pipeline_blocked"],
            origin="manual",
            tags=["dependence", "always-safe"],
        ),
        Skill(
            id="loop-tripcount-when-bound-runtime",
            pattern=(
                "Vitis cannot bound a loop's trip count statically and reports "
                "implausible latency"
            ),
            strategy="annotate the loop with #pragma HLS loop_tripcount min=N avg=N max=N",
            template=(
                "for (int i = 0; i < runtime_n; ++i) {\n"
                "#pragma HLS loop_tripcount min=N avg=N max=N\n"
                "    ...\n"
                "}\n"
            ),
            confidence=TIER_HIGH,
            bottleneck_kinds=["interval_exceeds_latency"],
            origin="manual",
            tags=["loop_tripcount", "diagnostic"],
        ),
    ]


def _default_avoid_skills() -> List[Skill]:
    return [
        Skill(
            id="avoid-manual-loop-interchange-on-perfect-nest",
            pattern="perfectly nested loops where Vitis can interchange automatically",
            strategy="rewriting the loop nesting order by hand (Vitis 2022.2+ already does this)",
            confidence=TIER_AVOID,
            applicable_versions=["2023.2", "2025.2"],
            tags=["absorbed-by-synth"],
            origin="manual",
        ),
        Skill(
            id="avoid-manual-cse-inside-pipelined-loop",
            pattern="repeated identical sub-expressions inside a pipelined loop body",
            strategy="manual common-subexpression elimination via temporary variables",
            confidence=TIER_AVOID,
            applicable_versions=["2023.2", "2025.2"],
            tags=["absorbed-by-synth"],
            origin="manual",
        ),
    ]


def _load_packaged_skills(path: Path = _PACKAGED_SKILLS) -> List[Skill]:
    """Load the schema-1.1 curated skill package when present.

    The package is benchmark-independent and intentionally lives outside the
    mutable `skills/` store. It updates the built-in recipes while preserving
    observed trajectory statistics from the active store.
    """
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logging.warning("packaged skill load failed (%s); using built-ins only", exc)
        return []
    skills_raw = data.get("skills", []) if isinstance(data, dict) else data
    out: List[Skill] = []
    for entry in skills_raw or []:
        sk = _coerce_skill_entry(entry)
        if sk is not None:
            out.append(sk)
    return out


def make_default_library(store_path: Optional[Path] = None,
                          *, persist: bool = True) -> SkillLibrary:
    """Initialize a SkillLibrary and merge any missing built-in skills.

    A hand-edited store remains authoritative for existing ids, while new
    prompt/manual defaults are added when code upgrades introduce them. The
    curated schema-1.1 package, when present, refreshes recipe text,
    guardrails, and required-step checklists while keeping observed pass-rate
    statistics from the mutable store.
    """
    lib = SkillLibrary(store_path or _DEFAULT_STORE).load()
    before = {sk.id: asdict(sk) for sk in lib.all()}
    for sk in _bootstrap_skills_from_prompts():
        lib.add(sk, overwrite=False)
    for sk in _load_packaged_skills():
        lib.add(sk, overwrite=True, preserve_stats=True)
    after = {sk.id: asdict(sk) for sk in lib.all()}
    if after != before and persist:
        try:
            lib.save()
        except OSError as exc:
            logging.warning("SkillLibrary persistence failed: %s", exc)
    return lib


# === Render helpers (for prompts) ========================================


def render_skill_for_prompt(sk: Skill) -> str:
    """Compact render with schema-1.1 guardrails/checklists."""
    bullets = [
        f"[skill {sk.id}] confidence={sk.confidence} pass={sk.sec_pass}/{sk.occurrences}",
        f"  pattern: {sk.pattern}",
        f"  strategy: {sk.strategy}",
    ]
    if sk.kind:
        bullets.insert(1, f"  kind: {sk.kind}")
    if sk.required_steps:
        bullets.append("  required steps:\n" + "\n".join(
            f"    - {item}" for item in sk.required_steps[:10]
        ))
    if sk.guards:
        bullets.append("  guards:\n" + "\n".join(
            f"    - {item}" for item in sk.guards[:8]
        ))
    if sk.template:
        template_lines = sk.template.strip().splitlines()
        if len(template_lines) > 16:
            template_lines = template_lines[:16] + ["..."]
        bullets.append("  template/example:\n" + "\n".join(
            f"    {line}" for line in template_lines
        ))
    return "\n".join(bullets)


def render_skill_set_for_prompt(skills: Iterable[Skill],
                                 max_skills: int = 5) -> str:
    skills_list = list(skills)[:max_skills]
    if not skills_list:
        return "No matching skills in library — fall back to your own reasoning."
    return "\n".join(render_skill_for_prompt(sk) for sk in skills_list)
