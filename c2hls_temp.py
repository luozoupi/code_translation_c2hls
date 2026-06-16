"""c2hls_temp — temp-directory management for the c2hls pipeline.

RECONSTRUCTED (2026-06-16): the original module was gitignored on the
c2hls_enhanced branch (.gitignore lists `c2hls_temp.py`), so a fresh checkout
of that branch lacks it and c2hls.py / hls_eval.py fail at import with
`ModuleNotFoundError: No module named 'c2hls_temp'`. This reconstruction
provides the same public API the rest of the codebase imports:

    C2HLS_TMP_ROOT_ENV            env var name selecting the scratch root
    configure_temp_env(create)    -> Path : resolve (and optionally create)
                                    the scratch root
    make_tempdir(prefix=...)      -> str  : mkdtemp under the scratch root

Purpose: keep Vitis/Vivado intermediates (which can be GBs) on a writable,
roomy filesystem rather than scattering them in the default system temp.
The scratch root is chosen as:
    1. $C2HLS_TMP_ROOT if set
    2. else $TMPDIR/c2hls_tmp  (launchers export TMPDIR=/tmp)
    3. else <system-temp>/c2hls_tmp

This is pure scratch-location plumbing — it does not affect synthesis,
simulation, or any HLS metric; only where transient files are written.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

# Env var consumers (hls_eval._vitis_shell_exports) re-export so the same
# scratch root propagates into the Vitis subprocess shell.
C2HLS_TMP_ROOT_ENV = "C2HLS_TMP_ROOT"


def _default_root() -> Path:
    raw = os.getenv(C2HLS_TMP_ROOT_ENV, "").strip()
    if raw:
        return Path(raw).expanduser()
    base = os.getenv("TMPDIR", "").strip() or tempfile.gettempdir()
    return Path(base).expanduser() / "c2hls_tmp"


def configure_temp_env(create: bool = True) -> Path:
    """Resolve the c2hls scratch root and, by default, create it.

    Also pins the resolved root back into C2HLS_TMP_ROOT so any child
    process / later call agrees on the same location within one run.
    Returns the root as a Path (callers do `root / "vitis_user_home"`).
    """
    root = _default_root()
    if create:
        root.mkdir(parents=True, exist_ok=True)
    # Make the choice sticky for subprocesses and subsequent calls.
    os.environ[C2HLS_TMP_ROOT_ENV] = str(root)
    return root


def make_tempdir(prefix: str = "c2hls_") -> str:
    """Create and return a fresh temp directory under the scratch root."""
    root = configure_temp_env(create=True)
    return tempfile.mkdtemp(prefix=prefix, dir=str(root))
