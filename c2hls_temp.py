"""c2hls_temp — temp-directory management for the c2hls pipeline.

Scratch dirs default to ``<C2HLS_TMP_ROOT>/`` (repo ``c2hls_tmp/`` on PC2).
Names are human-readable when ``C2HLS_TMP_TAG`` is set (benchmark, step, phase).

When ``C2HLS_TMP_RUN`` is set, dirs nest as::

    <C2HLS_TMP_ROOT>/<run>/<bench>/hls_synth__flash_synth

``C2HLS_TMP_BENCH`` (preferred) or the leading ``<bench>_`` prefix of the tag
supplies the bench folder. When ``C2HLS_TMP_RUN`` is unset, names stay flat.
"""
from __future__ import annotations

import os
import re
import tempfile
from contextlib import contextmanager
from pathlib import Path

C2HLS_TMP_ROOT_ENV = "C2HLS_TMP_ROOT"
C2HLS_TMP_TAG_ENV = "C2HLS_TMP_TAG"
C2HLS_TMP_RUN_ENV = "C2HLS_TMP_RUN"
C2HLS_TMP_BENCH_ENV = "C2HLS_TMP_BENCH"


def _default_root() -> Path:
    raw = os.getenv(C2HLS_TMP_ROOT_ENV, "").strip()
    if raw:
        return Path(raw).expanduser()
    base = os.getenv("TMPDIR", "").strip() or tempfile.gettempdir()
    return Path(base).expanduser() / "c2hls_tmp"


def sanitize_temp_tag(raw: str, max_len: int = 96) -> str:
    """Filesystem-safe slug for temp directory names."""
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", (raw or "").strip())
    slug = re.sub(r"_+", "_", slug).strip("_")
    if len(slug) > max_len:
        slug = slug[:max_len].rstrip("_")
    return slug


def join_temp_tag(*parts: str) -> str:
    return sanitize_temp_tag("__".join(str(part) for part in parts if str(part).strip()))


def get_temp_tag() -> str:
    return os.getenv(C2HLS_TMP_TAG_ENV, "").strip()


def get_temp_run() -> str:
    return sanitize_temp_tag(os.getenv(C2HLS_TMP_RUN_ENV, "").strip())


def get_temp_bench() -> str:
    return sanitize_temp_tag(os.getenv(C2HLS_TMP_BENCH_ENV, "").strip())


def set_temp_tag(tag: str) -> None:
    cleaned = sanitize_temp_tag(tag)
    if cleaned:
        os.environ[C2HLS_TMP_TAG_ENV] = cleaned
    else:
        os.environ.pop(C2HLS_TMP_TAG_ENV, None)


def set_temp_bench(bench: str) -> None:
    cleaned = sanitize_temp_tag(bench)
    if cleaned:
        os.environ[C2HLS_TMP_BENCH_ENV] = cleaned
    else:
        os.environ.pop(C2HLS_TMP_BENCH_ENV, None)


@contextmanager
def temp_tag_scope(*parts: str):
    """Temporarily set ``C2HLS_TMP_TAG`` for nested HLS work dirs."""
    prev = os.environ.get(C2HLS_TMP_TAG_ENV)
    tag = join_temp_tag(*parts)
    if tag:
        os.environ[C2HLS_TMP_TAG_ENV] = tag
    try:
        yield
    finally:
        if prev is None:
            os.environ.pop(C2HLS_TMP_TAG_ENV, None)
        else:
            os.environ[C2HLS_TMP_TAG_ENV] = prev


def configure_temp_env(create: bool = True) -> Path:
    """Resolve the c2hls scratch root and, by default, create it.

    Returns the base ``C2HLS_TMP_ROOT`` (not the per-run nested path).
    """
    root = _default_root()
    if create:
        root.mkdir(parents=True, exist_ok=True)
    os.environ[C2HLS_TMP_ROOT_ENV] = str(root)
    return root


def _infer_bench_from_tag(tag_slug: str) -> str:
    """Best-effort bench slug when ``C2HLS_TMP_BENCH`` is unset.

    Tags from ``join_temp_tag(bench, ...)`` become ``bench_phase_...`` after
    sanitize (``__`` collapses to ``_``). Prefer common ``chathls_*`` /
    ``hlsfactory_*`` / ``autosa_*`` prefixes; else first underscore segment.
    """
    if not tag_slug:
        return ""
    for prefix in ("chathls_", "hlsfactory_", "autosa_", "gnnbuilder_"):
        if tag_slug.startswith(prefix):
            rest = tag_slug[len(prefix) :]
            # bench is prefix + first remaining token
            first = rest.split("_", 1)[0]
            return sanitize_temp_tag(prefix + first)
    return tag_slug.split("_", 1)[0]


def _strip_bench_prefix(tag_slug: str, bench: str) -> str:
    if not bench:
        return tag_slug
    if tag_slug.startswith(bench + "_"):
        return tag_slug[len(bench) + 1 :]
    if tag_slug == bench:
        return ""
    return tag_slug


def _resolve_bench_and_leaf_tag(tag_slug: str) -> tuple[str, str]:
    """Return (bench_slug, leaf_tag) when nesting under a run."""
    explicit = get_temp_bench()
    if explicit:
        return explicit, _strip_bench_prefix(tag_slug, explicit)
    bench = _infer_bench_from_tag(tag_slug)
    if bench:
        return bench, _strip_bench_prefix(tag_slug, bench)
    return "", tag_slug


def resolve_temp_parent(tag: str | None = None) -> Path:
    """Directory under which the next temp leaf should be created."""
    base = configure_temp_env(create=True)
    run_slug = get_temp_run()
    if not run_slug:
        return base

    tag_slug = sanitize_temp_tag(tag if tag is not None else get_temp_tag())
    bench_slug, _ = _resolve_bench_and_leaf_tag(tag_slug)
    parent = base / run_slug
    if bench_slug:
        parent = parent / bench_slug
    parent.mkdir(parents=True, exist_ok=True)
    return parent


def make_tempdir(prefix: str = "c2hls_", tag: str | None = None) -> str:
    """Create a fresh directory under the scratch root.

    Flat (no ``C2HLS_TMP_RUN``)::

        hls_synth__hlsfactory_trmm_flash_synth

    Nested (``C2HLS_TMP_RUN`` set)::

        <run>/<bench>/hls_synth__flash_synth

    A numeric suffix is appended on collision (``_001``, ``_002``, ...).
    """
    tag_slug = sanitize_temp_tag(tag if tag is not None else get_temp_tag())
    run_slug = get_temp_run()
    leaf_tag = tag_slug
    if run_slug:
        _bench, leaf_tag = _resolve_bench_and_leaf_tag(tag_slug)
        parent = resolve_temp_parent(tag=tag_slug)
    else:
        parent = configure_temp_env(create=True)

    stem = prefix.rstrip("_")
    if leaf_tag:
        stem = f"{stem}__{leaf_tag}"

    for seq in range(10000):
        name = stem if seq == 0 else f"{stem}_{seq:03d}"
        path = parent / name
        try:
            path.mkdir(parents=False)
            return str(path)
        except FileExistsError:
            continue
    raise RuntimeError(f"could not allocate temp dir under {parent} for stem {stem}")
