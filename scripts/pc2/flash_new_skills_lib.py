"""PC2 re-export — canonical definitions live in ``scripts/flash_shared/``."""

from __future__ import annotations

import sys
from pathlib import Path

_SHARED = Path(__file__).resolve().parents[1] / "flash_shared"
if str(_SHARED.parent) not in sys.path:
    sys.path.insert(0, str(_SHARED.parent))

from flash_shared.new_skills_lib import *  # noqa: F401,F403,E402


def configure_new_skills_env(variant, *, inference: str = "vllm") -> None:  # type: ignore[no-redef]
    from flash_shared.new_skills_lib import configure_new_skills_env as _configure

    _configure(variant, inference="vllm")
