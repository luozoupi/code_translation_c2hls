"""Pinned C2HLS_SKILL_PROMPT_ORDER_JSON overrides baseline-first dump order."""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

from c2hls_paths import FLASH_SKILL_ORDER_20260830_MMFLOW_JSON

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "pc2"))
from skill_library import (
    TIER_HIGH,
    Skill,
    SkillLibrary,
    global_skills_for_prompt,
)

def _lib_with(ids: list[str], store: Path) -> SkillLibrary:
    lib = SkillLibrary(store)
    lib._skills = {}
    for sid in ids:
        lib.add(Skill(id=sid, pattern="p", strategy="s", confidence=TIER_HIGH), overwrite=True)
    return lib


def test_default_order_still_hoists_burst_after_lcst(tmp_path, monkeypatch):
    monkeypatch.delenv("C2HLS_SKILL_PROMPT_ORDER_JSON", raising=False)
    lib = _lib_with(
        [
            "z-tail",
            "hls-avoid-zero-pipeline-submit",
            "hls-baseline-load-compute-store-gate",
            "axi-burst-coalescing-narrow-safe",
        ],
        tmp_path / "skills.json",
    )
    ids = [sk.id for sk in global_skills_for_prompt(lib, include_avoids=True)]
    assert ids[0] == "hls-baseline-load-compute-store-gate"
    assert ids[1] == "axi-burst-coalescing-narrow-safe"


def test_order_json_pins_historical_dump(tmp_path, monkeypatch):
    pin = tmp_path / "order.json"
    pin.write_text(
        json.dumps(
            {
                "ids": [
                    "hls-baseline-load-compute-store-gate",
                    "hls-avoid-zero-pipeline-submit",
                    "z-tail",
                    "axi-burst-coalescing-narrow-safe",
                ]
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("C2HLS_SKILL_PROMPT_ORDER_JSON", str(pin))
    lib = _lib_with(
        [
            "z-tail",
            "hls-avoid-zero-pipeline-submit",
            "hls-baseline-load-compute-store-gate",
            "axi-burst-coalescing-narrow-safe",
        ],
        tmp_path / "skills.json",
    )
    ids = [sk.id for sk in global_skills_for_prompt(lib, include_avoids=True)]
    assert ids == [
        "hls-baseline-load-compute-store-gate",
        "hls-avoid-zero-pipeline-submit",
        "z-tail",
        "axi-burst-coalescing-narrow-safe",
    ]


def test_campaign_restores_skill_prompt_order_env(monkeypatch):
    from batch_parallel_config import apply_autosa_flow_from_campaign

    monkeypatch.delenv("C2HLS_SKILL_PROMPT_ORDER_JSON", raising=False)
    apply_autosa_flow_from_campaign(
        {"skill_prompt_order_json": "/tmp/flash_skill_order_20260830_mmflow.json"}
    )
    assert (
        os.environ["C2HLS_SKILL_PROMPT_ORDER_JSON"]
        == "/tmp/flash_skill_order_20260830_mmflow.json"
    )


def test_init_campaign_json_copies_skill_prompt_order(tmp_path, monkeypatch):
    from batch_parallel_config import BatchParallelConfig, init_campaign_json

    pin = "/tmp/flash_skill_order_20260830_mmflow.json"
    monkeypatch.setenv("C2HLS_SKILL_PROMPT_ORDER_JSON", pin)
    monkeypatch.setenv("C2HLS_FLASH_ONLY", "1")
    monkeypatch.setenv("C2HLS_AUTOSA_FLOW", "1")
    doc = init_campaign_json(tmp_path, BatchParallelConfig(job_prefix="mmf16t"), stamp="t")
    assert doc.get("skill_prompt_order_json") == pin
    assert doc.get("flash_only") is True
    assert doc.get("post_flash_dse") is False
    assert doc.get("post_flash_stream") is False
    assert doc.get("dse_v2") is False


def test_frozen_20260830_order_file_matches_mmflow_flash():
    pin = json.loads(FLASH_SKILL_ORDER_20260830_MMFLOW_JSON.read_text(encoding="utf-8"))
    ids = pin["ids"]
    assert pin["skill_count"] == 129
    assert len(ids) == 129
    assert ids[0] == "hls-baseline-load-compute-store-gate"
    assert ids[1] == "hls-avoid-zero-pipeline-submit"
    assert ids.index("axi-burst-coalescing-narrow-safe") == 18
    hist = (
        REPO
        / "artifacts/pc2/batch_parallel_autosa_mm_flow_aav_n_gf_20260830_mmflow"
        / "variants/autosa_aav_n_gf/autosa_mm"
        / "deepseek-v4-flash__flash__autosa__aav_n_gf"
        / "autosa_mm_flash_skills.json"
    )
    if not hist.is_file():
        return
    frozen = json.loads(hist.read_text(encoding="utf-8"))["flash_opt"]["injected_skills"]
    frozen_ids = [item["id"] if isinstance(item, dict) else item for item in frozen]
    assert ids == frozen_ids
    hist_json = hist.parent / "autosa_mm_history.json"
    if not hist_json.is_file():
        return

    messages = json.loads(hist_json.read_text(encoding="utf-8")).get("messages") or []
    user = next(
        m["content"]
        for m in messages
        if m.get("role") == "user" and "[Step: flash]" in (m.get("content") or "")
    )
    prompt_ids = re.findall(r"\[skill ([^\]]+)\]", user)
    assert prompt_ids == ids
