"""Post-run audit for expert-reference leakage into controller transcripts.

The audit artifact never copies leaked expert source.  Each finding records a
rule, message index, offset and a one-way digest of the matched material.  It
is therefore safe to publish as provenance without itself becoming a second
leak channel.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping, Sequence


AUDIT_SCHEMA = "c2hls.reference-isolation-audit.v1"
_IDENTIFIER_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")
_C_TOKEN_RE = re.compile(
    r'''(?:u8|u|U|L)?"(?:\\.|[^"\\])*"'''
    r'''|(?:u|U|L)?'(?:\\.|[^'\\])*' '''
    r"|[A-Za-z_][A-Za-z0-9_]*"
    r"|(?:0[xX][0-9A-Fa-f]+|(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?)[uUlLfF]*"
    r"|>>=|<<=|->\*|::|\+\+|--|->|==|!=|<=|>=|&&|\|\||<<|>>|\+=|-=|\*=|/=|%=|&=|\|=|\^="
    r"|[^\s]",
    re.VERBOSE,
)
_NUMERIC_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9_.])(?:[+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?)(?![A-Za-z0-9_.])"
)
_PATH_BOUNDARY_CHARS = r"A-Za-z0-9_.-"
_CODE_SIGNATURE_K = 8
_GENERIC_IDENTIFIERS = {
    "pipeline",
    "unroll",
    "tiling",
    "coalescing",
    "doublebuffer",
    "workload",
    "kernel",
    "latency",
    "cycles",
    "baseline",
    "reference",
    "ground_truth",
    # Standard HLS vocabulary is intentionally present in the generic system
    # prompt and therefore cannot identify a particular expert design.
    "interface",
    "s_axilite",
    "m_axi",
    "ap_memory",
    "ap_none",
    "ap_ctrl_hs",
    "array_partition",
    "array_reshape",
    "dataflow",
    "inline",
    "loop_tripcount",
    "dependence",
    "bind_storage",
    "bind_op",
    "allocation",
    "unrolling",
    "pragma",
    "hls",
    "ii",
    "rewind",
    "off",
    "include",
    "define",
    "ifdef",
    "ifndef",
    "endif",
    "ap_uint",
    "ap_int",
    "hls_stream",
}


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _strip_code_comments(value: str) -> str:
    return re.sub(
        r"//.*?$|/\*.*?\*/",
        lambda match: re.sub(r"[^\n]", " ", match.group(0)),
        value,
        flags=re.MULTILINE | re.DOTALL,
    )


def _code_tokens(value: str, *, spans: bool = False) -> list[Any]:
    """Return a formatting-insensitive C/C++ token stream.

    Keeping token spans for transcript text lets the audit report offsets and
    hash only the matched bytes without persisting any expert source.
    """

    stripped = _strip_code_comments(value)
    matches = list(_C_TOKEN_RE.finditer(stripped))
    if spans:
        return [(match.group(0), match.start(), match.end()) for match in matches]
    return [match.group(0) for match in matches]


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _metadata(benchmark_dir: Path) -> dict[str, Any]:
    try:
        data = json.loads((benchmark_dir / "metadata.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _reference_paths(benchmark_dir: Path, metadata: Mapping[str, Any]) -> list[Path]:
    names: set[str] = set()
    for key in ("gold_hls_source_file", "gold_hls_baseline_file", "preferred_gt_file"):
        value = metadata.get(key)
        if isinstance(value, str) and value:
            names.add(value)
    for variant in metadata.get("variants") or []:
        if isinstance(variant, Mapping) and isinstance(variant.get("file"), str):
            names.add(variant["file"])
    # Metadata conventions vary across imported suites; these conservative
    # filename patterns only discover source files, never logs/results.
    for path in benchmark_dir.iterdir() if benchmark_dir.is_dir() else []:
        low = path.name.lower()
        if path.is_file() and path.suffix.lower() in {".c", ".cc", ".cpp", ".h", ".hpp"}:
            if low.startswith("hls_") or "gold" in low or "reference" in low:
                names.add(path.name)
    return sorted(
        {benchmark_dir / name for name in names if (benchmark_dir / name).is_file()},
        key=lambda item: item.name,
    )


def _plain_paths(benchmark_dir: Path, metadata: Mapping[str, Any]) -> list[Path]:
    names = {
        str(metadata.get("plain_c_file") or "plain.cpp"),
        str(metadata.get("header_file") or ""),
    }
    return [benchmark_dir / name for name in sorted(names) if name and (benchmark_dir / name).is_file()]


def _expert_tokens(
    benchmark_dir: Path,
    metadata: Mapping[str, Any],
    references: Sequence[Path],
    plain_text: str,
) -> dict[str, Any]:
    paths: set[str] = set()
    identifiers: set[str] = set()

    def add_path(value: Any) -> None:
        if not isinstance(value, str) or not value:
            return
        paths.add(value)
        paths.add(Path(value).name)

    for key in ("gold_hls_source_path", "gold_hls_source_file", "gold_hls_baseline_file", "preferred_gt_file"):
        add_path(metadata.get(key))
    for value in (metadata.get("variant_source_paths") or {}).values():
        add_path(value)
    for variant in metadata.get("variants") or []:
        if not isinstance(variant, Mapping):
            continue
        add_path(variant.get("file"))
        add_path(variant.get("source_path"))
        name = variant.get("name")
        # Variant names are private expert identities even when deliberately
        # terse (for example ``v3``).  Their explicit metadata provenance is
        # stronger evidence than the length heuristics used for identifiers
        # mined from source text.
        if (
            isinstance(name, str)
            and len(name) >= 2
            and name.lower() not in _GENERIC_IDENTIFIERS
        ):
            identifiers.add(name)

    # Reference files discovered from metadata conventions are private even
    # when metadata omitted their explicit spelling.  Retain both the local
    # path and its basename; a short relative filename such as ``gt.c`` is a
    # useful high-confidence leak token, not noise.
    for path in references:
        add_path(str(path))
        add_path(path.name)

    plain_ids = set(_IDENTIFIER_RE.findall(plain_text))
    reference_text = "\n".join(_read_text(path) for path in references)
    benchmark_identifier = str(metadata.get("benchmark") or benchmark_dir.name)
    public_identifiers = {
        benchmark_identifier,
        str(metadata.get("kernel_top") or ""),
        str(metadata.get("hls_top") or ""),
        str(metadata.get("translated_hls_top") or ""),
    }
    for identifier in _IDENTIFIER_RE.findall(reference_text):
        structurally_specific = (
            "_" in identifier
            or any(char.isdigit() for char in identifier)
            or any(char.isupper() for char in identifier[1:])
        )
        if (
            len(identifier) >= 9
            and structurally_specific
            and identifier not in plain_ids
            and identifier not in public_identifiers
            and identifier.lower() not in _GENERIC_IDENTIFIERS
            and not identifier.startswith("VITIS_LOOP")
        ):
            identifiers.add(identifier)

    # Formatting-insensitive token k-grams catch copied *parts* of an expert
    # implementation after line wrapping or whitespace changes.  Excluding
    # every gram in the public plain-C input avoids flagging the common ABI or
    # unchanged source.  A gram must also contain an expert-only identifier or
    # literal; this keeps generic HLS vocabulary and boilerplate prompts usable.
    plain_tokens = _code_tokens(plain_text)
    plain_grams = {
        tuple(plain_tokens[index : index + _CODE_SIGNATURE_K])
        for index in range(max(0, len(plain_tokens) - _CODE_SIGNATURE_K + 1))
    }
    plain_token_set = set(plain_tokens)
    signatures: set[tuple[str, ...]] = set()
    for ref_path in references:
        ref_tokens = _code_tokens(_read_text(ref_path))
        for index in range(max(0, len(ref_tokens) - _CODE_SIGNATURE_K + 1)):
            gram = tuple(ref_tokens[index : index + _CODE_SIGNATURE_K])
            if gram in plain_grams:
                continue
            novel = [token for token in gram if token not in plain_token_set]
            informative_novel = {
                token
                for token in novel
                if (
                    _IDENTIFIER_RE.fullmatch(token)
                    and len(token) >= 3
                    and token.lower() not in _GENERIC_IDENTIFIERS
                )
                or (
                    re.fullmatch(r"\d+(?:\.\d+)?(?:[eE][+-]?\d+)?[uUlLfF]*", token)
                    and len(re.sub(r"\D", "", token)) >= 3
                )
            }
            distinctive = any(
                (
                    _IDENTIFIER_RE.fullmatch(token)
                    and len(token) >= 4
                    and token.lower() not in _GENERIC_IDENTIFIERS
                )
                or (
                    re.fullmatch(r"\d+(?:\.\d+)?(?:[eE][+-]?\d+)?[uUlLfF]*", token)
                    and len(re.sub(r"\D", "", token)) >= 3
                )
                for token in novel
            )
            if distinctive or len(informative_novel) >= 2:
                signatures.add(gram)

    return {
        "paths": sorted(paths, key=lambda item: (-len(item), item)),
        "identifiers": sorted(identifiers, key=lambda item: (-len(item), item)),
        "signatures": signatures,
        "signature_k": _CODE_SIGNATURE_K,
    }


_REFERENCE_METRIC_ALIASES = {
    "latency_cycles": "cycles",
    "latency_cycles_best": "cycles",
    "latency_cycles_worst": "cycles",
    "kernel_runtime_cycles": "cycles",
    "estimated_latency_cycles": "cycles",
    "latency_ns": "latency",
    "interval": "interval",
    "initiation_interval": "interval",
    "fmax_mhz": "fmax",
    "estimated_clock_period_ns": "clock",
    "slack_ns": "slack",
    "bram": "bram",
    "bram_18k": "bram",
    "dsp": "dsp",
    "ff": "ff",
    "lut": "lut",
    "uram": "uram",
}
_REFERENCE_METRIC_LABEL_PATTERNS = {
    "cycles": r"(?:cycles|latency[_\s-]*cycles|kernel[_\s-]*runtime[_\s-]*cycles)",
    "latency": r"(?:latency|latency[_\s-]*ns)",
    "interval": r"(?:interval|initiation[_\s-]*interval|ii)",
    "fmax": r"(?:fmax|frequency)",
    "clock": r"(?:clock|estimated[_\s-]*clock[_\s-]*period)",
    "slack": r"(?:slack|slack[_\s-]*ns)",
    "bram": r"(?:bram|bram[_\s-]*18k)",
    "dsp": r"dsp",
    "ff": r"ff",
    "lut": r"lut",
    "uram": r"uram",
}


def _unlabeled_metric_is_distinctive(value: float, rendered: str) -> bool:
    """Whether an exact number is safe enough to flag without a gold label.

    Tiny values and round powers/multiples are ubiquitous in compiler prompts,
    target descriptions, and C source.  Large non-round integers and precise
    decimal measurements are substantially more identifying.  Labeled expert
    metrics use a separate rule and do not need this stricter threshold.
    """

    magnitude = abs(value)
    if not math.isfinite(value) or magnitude < 10:
        return False
    if value.is_integer():
        integer = abs(int(value))
        common_power_of_two = integer > 0 and (integer & (integer - 1)) == 0
        common_year = 1900 <= integer <= 2100
        return (
            integer >= 1000
            and integer % 10 != 0
            and not common_power_of_two
            and not common_year
        )
    digits = re.sub(r"\D", "", rendered)
    return magnitude >= 10 and len(digits.lstrip("0")) >= 3


def _reference_metrics(
    reference_data: Mapping[str, Any] | None,
    *,
    public_text: str = "",
) -> list[tuple[str, str, bool]]:
    """Collect metric renderings recursively from the entire expert frontier.

    Reference validation contains selected, baseline, and alternate variants
    under different nested structures.  Walking all mappings/lists prevents an
    unselected variant's exact QoR from escaping the transcript audit.
    """

    if not isinstance(reference_data, Mapping):
        return []
    public_numbers = {match.group(0) for match in _NUMERIC_TOKEN_RE.finditer(public_text)}
    values: set[tuple[str, str, bool]] = set()

    def visit(node: Any) -> None:
        if isinstance(node, Mapping):
            for raw_key, child in node.items():
                key = str(raw_key).lower()
                alias = _REFERENCE_METRIC_ALIASES.get(key)
                if (
                    alias is not None
                    and not isinstance(child, bool)
                    and isinstance(child, (int, float))
                    and math.isfinite(float(child))
                    and abs(float(child)) >= 2
                ):
                    numeric = float(child)
                    renderings = {str(child), format(numeric, ".12g")}
                    if numeric.is_integer():
                        renderings.add(str(int(numeric)))
                    for rendered in renderings:
                        distinctive = (
                            rendered not in public_numbers
                            and _unlabeled_metric_is_distinctive(numeric, rendered)
                        )
                        values.add((alias, rendered, distinctive))
                visit(child)
        elif isinstance(node, Sequence) and not isinstance(node, (str, bytes, bytearray)):
            for child in node:
                visit(child)

    visit(reference_data)
    return sorted(values)


def _finding(rule: str, index: int, role: str, offset: int, matched: str) -> dict[str, Any]:
    return {
        "rule": rule,
        "message_index": index,
        "role": role,
        "offset": offset,
        "match_characters": len(matched),
        "match_sha256": _digest(matched),
    }


def audit_messages(
    messages: Sequence[Mapping[str, Any]],
    *,
    benchmark_dir: Path,
    reference_data: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Audit controller-visible messages against offline expert material."""

    benchmark_dir = Path(benchmark_dir)
    metadata = _metadata(benchmark_dir)
    references = _reference_paths(benchmark_dir, metadata)
    plain_text = "\n".join(
        _read_text(path) for path in _plain_paths(benchmark_dir, metadata)
    )
    tokens = _expert_tokens(benchmark_dir, metadata, references, plain_text)
    metrics = _reference_metrics(reference_data, public_text=plain_text)
    findings: list[dict[str, Any]] = []

    for index, message in enumerate(messages):
        if not isinstance(message, Mapping):
            continue
        role = str(message.get("role") or "unknown").lower()
        text = str(message.get("content") or "")
        # Assistant code can legitimately recover an expert-like block.  The
        # isolation claim concerns what the controller/LLM was shown, so code
        # signatures and expert-only identifiers are strict for system/user
        # turns.  Explicit paths and gold metrics are suspicious in any role.
        controller_visible = role in {"system", "user", "tool"}
        occupied_path_spans: list[tuple[int, int]] = []
        for token in tokens["paths"]:
            pattern = re.compile(
                rf"(?<![{_PATH_BOUNDARY_CHARS}]){re.escape(token)}"
                rf"(?![{_PATH_BOUNDARY_CHARS}])"
            )
            match = pattern.search(text)
            if match and not any(
                match.start() >= start and match.end() <= end
                for start, end in occupied_path_spans
            ):
                findings.append(
                    _finding("expert_path", index, role, match.start(), match.group(0))
                )
                occupied_path_spans.append((match.start(), match.end()))
        if controller_visible:
            for token in tokens["identifiers"]:
                match = re.search(rf"(?<![A-Za-z0-9_]){re.escape(token)}(?![A-Za-z0-9_])", text)
                if match:
                    findings.append(
                        _finding("expert_identifier", index, role, match.start(), match.group(0))
                    )
            message_tokens = _code_tokens(text, spans=True)
            signature_k = int(tokens["signature_k"])
            for start in range(max(0, len(message_tokens) - signature_k + 1)):
                gram = tuple(
                    item[0] for item in message_tokens[start : start + signature_k]
                )
                if gram in tokens["signatures"]:
                    begin = message_tokens[start][1]
                    end = message_tokens[start + signature_k - 1][2]
                    findings.append(
                        _finding(
                            "expert_code_signature",
                            index,
                            role,
                            begin,
                            text[begin:end],
                        )
                    )
                    # A copied block creates many overlapping k-grams; one
                    # finding per message establishes the leak without
                    # inflating the published audit artifact.
                    break

        labeled_metric_spans: list[tuple[int, int]] = []
        for alias, value, _distinctive in metrics:
            label_pattern = _REFERENCE_METRIC_LABEL_PATTERNS[alias]
            pattern = re.compile(
                rf"(?i)\b(?:gold(?:en)?|reference|expert|oracle|ground[_\s-]*truth)\b"
                rf"[^\n]{{0,120}}\b{label_pattern}"
                rf"(?:[_\s-]*(?:cycles|mhz|ns))?\s*[:=<>~]*\s*"
                rf"(?<![A-Za-z0-9_.]){re.escape(value)}"
                rf"(?![A-Za-z0-9_])(?!\.\d)"
            )
            match = pattern.search(text)
            if match:
                findings.append(
                    _finding("absolute_reference_metric", index, role, match.start(), match.group(0))
                )
                labeled_metric_spans.append((match.start(), match.end()))

        # A distinctive exact metric remains a leak without words such as
        # "expert" or "gold".  Scan unique renderings longest-first and avoid
        # duplicating a finding already explained by the labeled rule.
        unlabeled_values = sorted(
            {value for _alias, value, distinctive in metrics if distinctive},
            key=lambda value: (-len(value), value),
        )
        occupied_metric_spans = list(labeled_metric_spans)
        for value in unlabeled_values:
            pattern = re.compile(
                rf"(?<![A-Za-z0-9_.]){re.escape(value)}"
                rf"(?![A-Za-z0-9_])(?!\.\d)"
            )
            for match in pattern.finditer(text):
                if any(
                    match.start() >= start and match.end() <= end
                    for start, end in occupied_metric_spans
                ):
                    continue
                findings.append(
                    _finding(
                        "unlabeled_reference_metric",
                        index,
                        role,
                        match.start(),
                        match.group(0),
                    )
                )
                occupied_metric_spans.append((match.start(), match.end()))

    counts: dict[str, int] = {}
    for finding in findings:
        counts[finding["rule"]] = counts.get(finding["rule"], 0) + 1
    corpus_manifest = {
        "reference_file_count": len(references),
        "reference_files_sha256": _digest(
            "\n".join(f"{path.name}:{_digest(_read_text(path))}" for path in references)
        ),
        "expert_path_token_count": len(tokens["paths"]),
        "expert_identifier_count": len(tokens["identifiers"]),
        "expert_code_signature_count": len(tokens["signatures"]),
        "reference_metric_count": len(metrics),
        "distinctive_unlabeled_metric_count": sum(
            1 for _alias, _value, distinctive in metrics if distinctive
        ),
    }
    return {
        "schema_version": AUDIT_SCHEMA,
        "passed": not findings,
        "finding_count": len(findings),
        "finding_counts": counts,
        "findings": findings,
        "corpus": corpus_manifest,
    }


def audit_history_file(
    history_path: Path,
    *,
    benchmark_dir: Path,
    reference_data: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    path = Path(history_path)
    try:
        # Read once, hash those exact bytes, and parse the same in-memory
        # buffer.  The digest therefore binds the audit to the persisted
        # transcript rather than to a separately serialized message object.
        raw = path.read_bytes()
    except OSError as exc:
        return {
            "schema_version": AUDIT_SCHEMA,
            "passed": False,
            "finding_count": 0,
            "finding_counts": {},
            "findings": [],
            "error": f"transcript unavailable: {type(exc).__name__}",
        }
    transcript_sha256 = hashlib.sha256(raw).hexdigest()
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        return {
            "schema_version": AUDIT_SCHEMA,
            "passed": False,
            "finding_count": 0,
            "finding_counts": {},
            "findings": [],
            "transcript_sha256": transcript_sha256,
            "transcript_bytes": len(raw),
            "error": f"transcript unavailable: {type(exc).__name__}",
        }
    messages = payload.get("messages") if isinstance(payload, Mapping) else None
    if not isinstance(messages, list):
        return {
            "schema_version": AUDIT_SCHEMA,
            "passed": False,
            "finding_count": 0,
            "finding_counts": {},
            "findings": [],
            "transcript_sha256": transcript_sha256,
            "transcript_bytes": len(raw),
            "error": "transcript has no messages list",
        }
    audit = audit_messages(
        messages,
        benchmark_dir=benchmark_dir,
        reference_data=reference_data,
    )
    audit["transcript_sha256"] = transcript_sha256
    audit["transcript_bytes"] = len(raw)
    return audit
