"""Zero-shot + skills: the bare zero-shot flash prompt plus the Expertise rung's skill block, verbatim.

c2hls's zero_shot flavor never synthesizes a baseline, so its skill injection (which needs a skill library and a
csynth report) never fires. apply(block_file) appends a fixed text block to the flash guidance blocks instead. The
block file is the exact `injected_prompt_text` recorded by a stock skills run (90-skill + gemm-flatten v1 pack,
all_skills_avoids_global), so the zero-shot prompt carries the same skills, byte for byte. c2hls stays read only.
"""
from __future__ import annotations

import hashlib
from pathlib import Path


def apply(block_file: str, log=print) -> dict:
    import c2hls

    text = Path(block_file).read_text()
    original_blocks = c2hls.flash_step_guidance_extra_blocks

    def blocks():
        return list(original_blocks()) + [text]

    c2hls.flash_step_guidance_extra_blocks = blocks
    info = {"zs_skills_block": str(block_file), "chars": len(text), "sha1": hashlib.sha1(text.encode()).hexdigest()}
    log(f"zs_skills applied: {info}")
    return info
