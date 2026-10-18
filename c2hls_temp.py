"""Shared temporary-directory policy for C2HLS runs.

The sweep and Vitis flows can create hundreds of gigabytes of intermediates.
Keep those under a large, explicit scratch root instead of Python's default
`/tmp`, and make failures visible rather than silently falling back.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


C2HLS_TMP_ROOT_ENV = "C2HLS_TMP_ROOT"
DEFAULT_C2HLS_TMP_ROOT = Path("/mnt/data/luo00466/tmp")


def configured_temp_root() -> Path:
    raw = os.getenv(C2HLS_TMP_ROOT_ENV, str(DEFAULT_C2HLS_TMP_ROOT)).strip()
    root = Path(raw or str(DEFAULT_C2HLS_TMP_ROOT)).expanduser()
    if not root.is_absolute():
        raise RuntimeError(
            f"{C2HLS_TMP_ROOT_ENV} must be an absolute path, got: {root}"
        )
    return root


def ensure_temp_root() -> Path:
    root = configured_temp_root()
    try:
        root.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise RuntimeError(
            f"Could not create {C2HLS_TMP_ROOT_ENV}={root}. "
            "Set C2HLS_TMP_ROOT to a writable scratch filesystem."
        ) from exc
    if not root.is_dir():
        raise RuntimeError(f"{C2HLS_TMP_ROOT_ENV}={root} is not a directory")
    return root


def configure_temp_env(*, create: bool = False) -> Path:
    root = ensure_temp_root() if create else configured_temp_root()
    root_s = str(root)
    os.environ[C2HLS_TMP_ROOT_ENV] = root_s
    os.environ["TMPDIR"] = root_s
    os.environ["TEMP"] = root_s
    os.environ["TMP"] = root_s
    return root


def make_tempdir(prefix: str) -> str:
    root = configure_temp_env(create=True)
    return tempfile.mkdtemp(prefix=prefix, dir=str(root))
