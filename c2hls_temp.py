"""c2hls_temp — temp-directory management for the c2hls pipeline.

Scratch dirs default to ``<C2HLS_TMP_ROOT>/`` (repo ``c2hls_tmp/`` on PC2).
Names are human-readable when ``C2HLS_TMP_TAG`` is set (benchmark, step, phase).
"""
from __future__ import annotations

import os
import re
import tempfile
from contextlib import contextmanager
from pathlib import Path

C2HLS_TMP_ROOT_ENV = "C2HLS_TMP_ROOT"
C2HLS_TMP_TAG_ENV = "C2HLS_TMP_TAG"


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


def set_temp_tag(tag: str) -> None:
    cleaned = sanitize_temp_tag(tag)
    if cleaned:
        os.environ[C2HLS_TMP_TAG_ENV] = cleaned
    else:
        os.environ.pop(C2HLS_TMP_TAG_ENV, None)


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
    """Resolve the c2hls scratch root and, by default, create it."""
    root = _default_root()
    if create:
        root.mkdir(parents=True, exist_ok=True)
    os.environ[C2HLS_TMP_ROOT_ENV] = str(root)
    return root


def make_tempdir(prefix: str = "c2hls_", tag: str | None = None) -> str:
    """Create a fresh directory under the scratch root.

    When *tag* or ``C2HLS_TMP_TAG`` is set, names look like::

        hls_synth__hlsfactory_trmm__flash__synth
        hls_csim__hlsfactory_trmm__ref__baseline__csim

    A numeric suffix is appended on collision (``_001``, ``_002``, ...).
    """
    root = configure_temp_env(create=True)
    tag_slug = sanitize_temp_tag(tag if tag is not None else get_temp_tag())
    stem = prefix.rstrip("_")
    if tag_slug:
        stem = f"{stem}__{tag_slug}"

    for seq in range(10000):
        name = stem if seq == 0 else f"{stem}_{seq:03d}"
        path = root / name
        try:
            path.mkdir(parents=False)
            return str(path)
        except FileExistsError:
            continue
    raise RuntimeError(f"could not allocate temp dir under {root} for stem {stem}")
