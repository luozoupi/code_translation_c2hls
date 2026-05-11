"""AST-hash candidate cache (Pillar 4 cost-aware speedup).

Hashes a candidate kernel by a *canonical* form of its source so two
runs that produce structurally-identical code (whitespace, comment, and
pragma-order tweaks aside) reuse the cached synthesis result. Backed by
sqlite so the cache survives restarts.

Key shape:

    cache_key = sha256(
        canonicalize_source(hls_code)
        ||
        canonicalize_source(header_code or "")
        ||
        f"::part={part}::clock_ns={clock_ns}::vitis={version}"
    )

Canonicalization:

- strip trailing whitespace per line
- collapse runs of blank lines
- normalize CRLF → LF
- drop comments (line // and block /* */) — these never affect synthesis
- alphabetize consecutive `#pragma HLS …` lines that share the same
  scope (so reordering pragmas in a candidate doesn't bust the cache)
- preserve everything else verbatim

We deliberately do NOT do a full C++ AST parse — that would require a
Clang dependency and the conservatism gain isn't worth it for cache
hits. The canonicalization above is sufficient to deduplicate the
common LLM cosmetic-rewrite class.

Cached values are the full report dict from `run_hls_synthesis`
(top-level metrics + Pillar 1 feedback) plus a small metadata header.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Optional, Tuple


_DEFAULT_CACHE = Path.home() / ".cache" / "c2hls" / "candidate_cache.sqlite"

_BLOCK_COMMENT_RE = re.compile(r"/\*.*?\*/", re.DOTALL)
_LINE_COMMENT_RE = re.compile(r"//[^\n]*")
_TRAILING_WS_RE = re.compile(r"[ \t]+$", re.MULTILINE)
_BLANK_RUN_RE = re.compile(r"\n{3,}")
_PRAGMA_HLS_RE = re.compile(r"^\s*#pragma\s+HLS\b", re.IGNORECASE)


def canonicalize_source(text: str) -> str:
    """Apply all the canonicalization rules listed in the module
    docstring. Returns the canonical form."""
    if not text:
        return ""
    s = text.replace("\r\n", "\n").replace("\r", "\n")
    s = _BLOCK_COMMENT_RE.sub("", s)
    s = _LINE_COMMENT_RE.sub("", s)
    s = _TRAILING_WS_RE.sub("", s)
    s = _alphabetize_pragma_runs(s)
    s = _BLANK_RUN_RE.sub("\n\n", s).strip() + "\n"
    return s


def _alphabetize_pragma_runs(text: str) -> str:
    """Sort consecutive `#pragma HLS …` lines (in the same indentation
    block) so reordering pragmas doesn't bust the hash. We only sort
    runs of length ≥ 2."""
    lines = text.split("\n")
    out: list[str] = []
    i = 0
    while i < len(lines):
        if not _PRAGMA_HLS_RE.match(lines[i]):
            out.append(lines[i])
            i += 1
            continue
        j = i
        while j < len(lines) and _PRAGMA_HLS_RE.match(lines[j]):
            j += 1
        run = lines[i:j]
        if len(run) >= 2:
            out.extend(sorted(run))
        else:
            out.extend(run)
        i = j
    return "\n".join(out)


def hash_candidate(
    *,
    hls_code: str,
    header_code: str = "",
    part: str = "",
    clock_ns: float = 0.0,
    vitis_version: str = "",
) -> str:
    """Stable cache key for one (kernel, header, target, version)
    combination."""
    payload = (
        canonicalize_source(hls_code)
        + "\n----\n"
        + canonicalize_source(header_code)
        + f"\n::part={part}::clock_ns={clock_ns}::vitis={vitis_version}\n"
    )
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


# === sqlite-backed store =================================================


class CandidateCache:
    """Thread-safe sqlite-backed cache. The connection is local to each
    instance; methods take care of `WITH IMMEDIATE` transactions so
    multiple instances on the same DB don't trample one another."""

    _SCHEMA = """
    CREATE TABLE IF NOT EXISTS candidates (
        key            TEXT PRIMARY KEY,
        report_json    TEXT NOT NULL,
        success        INTEGER NOT NULL,
        latency_cycles INTEGER,
        latency_ns     REAL,
        bram           INTEGER,
        dsp            INTEGER,
        ff             INTEGER,
        lut            INTEGER,
        fmax_mhz       REAL,
        part           TEXT,
        clock_ns       REAL,
        vitis_version  TEXT,
        inserted_at    REAL NOT NULL,
        last_hit_at    REAL,
        hit_count      INTEGER NOT NULL DEFAULT 0
    );
    CREATE INDEX IF NOT EXISTS idx_inserted_at ON candidates(inserted_at);
    """

    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path or _DEFAULT_CACHE)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_schema()

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(self._SCHEMA)
            conn.commit()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(str(self.path), timeout=30.0,
                                isolation_level=None)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            yield conn
        finally:
            conn.close()

    # ---- API -----------------------------------------------------------

    def lookup(self, key: str) -> Optional[Dict[str, Any]]:
        """Return the cached report dict, or None on miss. Updates
        hit_count + last_hit_at as a side effect."""
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT report_json FROM candidates WHERE key=?", (key,),
            ).fetchone()
            if row is None:
                return None
            now = time.time()
            conn.execute(
                "UPDATE candidates SET hit_count=hit_count+1, last_hit_at=? WHERE key=?",
                (now, key),
            )
            conn.commit()
            try:
                return json.loads(row[0])
            except json.JSONDecodeError:
                return None

    def store(self, key: str, report: Dict[str, Any], *,
              success: bool, part: str = "",
              clock_ns: float = 0.0,
              vitis_version: str = "") -> None:
        """Insert (or replace) a cache entry."""
        # Strip the work_dir before serialization — it's a tempdir path
        # that won't exist on retrieval and bloats the cache.
        report_clean = {k: v for k, v in report.items() if k != "work_dir"}
        report_json = json.dumps(report_clean, separators=(",", ":"))
        with self._lock, self._connect() as conn:
            conn.execute(
                "REPLACE INTO candidates (key, report_json, success, "
                "latency_cycles, latency_ns, bram, dsp, ff, lut, fmax_mhz, "
                "part, clock_ns, vitis_version, inserted_at, hit_count) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)",
                (
                    key,
                    report_json,
                    1 if success else 0,
                    _coerce_int(report.get("latency_cycles")),
                    _coerce_float(report.get("latency_ns")),
                    _coerce_int(report.get("bram")),
                    _coerce_int(report.get("dsp")),
                    _coerce_int(report.get("ff")),
                    _coerce_int(report.get("lut")),
                    _coerce_float(report.get("fmax_mhz")),
                    part,
                    clock_ns,
                    vitis_version,
                    time.time(),
                ),
            )
            conn.commit()

    def stats(self) -> Dict[str, Any]:
        """Aggregate counts for monitoring / tests."""
        with self._lock, self._connect() as conn:
            count_row = conn.execute("SELECT COUNT(*) FROM candidates").fetchone()
            hits_row = conn.execute("SELECT SUM(hit_count) FROM candidates").fetchone()
        return {
            "path": str(self.path),
            "entries": count_row[0] if count_row else 0,
            "total_hits": (hits_row[0] if hits_row and hits_row[0] is not None else 0),
        }

    def clear(self) -> None:
        with self._lock, self._connect() as conn:
            conn.execute("DELETE FROM candidates")
            conn.commit()


def _coerce_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _coerce_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# === Convenience wrapper around run_hls_synthesis ========================
# Callers that want cache transparency can swap a single line:
#
#   from candidate_cache import cached_run_hls_synthesis as run_hls_synthesis
#
# The wrapper computes the key, looks up, falls through to the real
# `hls_eval.run_hls_synthesis` on miss, and stores the result.
# ========================================================================


_GLOBAL_CACHE: Optional[CandidateCache] = None
_GLOBAL_CACHE_LOCK = threading.Lock()


def get_cache() -> CandidateCache:
    """Return a process-wide shared cache instance."""
    global _GLOBAL_CACHE
    with _GLOBAL_CACHE_LOCK:
        if _GLOBAL_CACHE is None:
            _GLOBAL_CACHE = CandidateCache()
        return _GLOBAL_CACHE


def cached_run_hls_synthesis(
    hls_code: str, header_code: str = "", header_name: str = "kernel.h",
    *, top_function: str = "workload",
    part: str = "", clock_ns: float = 4.0,
    vitis_version: str = "",
    work_dir: Optional[str] = None,
    extra_files: Optional[list] = None,
    cache: Optional[CandidateCache] = None,
    bypass_cache: bool = False,
) -> Dict[str, Any]:
    """Drop-in replacement for `hls_eval.run_hls_synthesis` with caching."""
    cache = cache or get_cache()
    key = hash_candidate(
        hls_code=hls_code, header_code=header_code,
        part=part, clock_ns=clock_ns, vitis_version=vitis_version,
    )

    if not bypass_cache:
        cached = cache.lookup(key)
        if cached is not None:
            logging.info("candidate_cache HIT key=%s", key[:24])
            cached["_from_cache"] = True
            return {"success": bool(cached.get("success", True)),
                    "error": cached.get("error", ""),
                    "report": cached.get("report") or cached,
                    "report_raw": cached.get("report_raw", ""),
                    "log": cached.get("log", "")}

    # Miss: import lazily so unit tests of the cache itself don't pull
    # the heavy hls_eval dependency tree.
    from hls_eval import run_hls_synthesis  # noqa: WPS433
    result = run_hls_synthesis(
        hls_code, header_code, header_name=header_name,
        top_function=top_function, part=part, clock_ns=clock_ns,
        work_dir=work_dir, extra_files=extra_files,
    )
    if result.get("success"):
        # Cache the report dict only — keep raw log out of sqlite.
        to_store = {
            "success": True,
            "report": result.get("report") or {},
            "error": result.get("error", ""),
        }
        cache.store(key, to_store, success=True,
                    part=part, clock_ns=clock_ns, vitis_version=vitis_version)
    return result
