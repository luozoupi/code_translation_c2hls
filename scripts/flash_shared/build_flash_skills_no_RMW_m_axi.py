#!/usr/bin/env python3
"""DEPRECATED — flash no longer merges overlay skills into packaged JSON.

Flash skills-enabled runs use:
  - C2HLS_PACKAGED_SKILLS_JSON → skills_ii_target_miss_solutions_added(90skills).json
  - C2HLS_FLASH_SKILL_ENTRIES_JSON → flash_no_RMW_m_axi_skill_entries.json

Edit flash_no_RMW_m_axi_skill_entries.json only; do not rebuild *_no_RMW_m_axi.json.
"""

from __future__ import annotations

import sys


def main() -> int:
    print(__doc__.strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
