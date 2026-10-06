"""gemm_flatten_v2 is a derived copy; v1 and the 90 pack stay frozen."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "pc2"))
sys.path.insert(0, str(REPO / "scripts" / "flash_shared"))

from autosa_flash_lib import (  # noqa: E402
    SKILLS_90_GEMM_JSON,
    SKILLS_90_GEMM_V2_JSON,
    SKILLS_90_JSON,
)
from build_gemm_flatten_v2_preventive import (  # noqa: E402
    NEW_IDS,
    PATCH_IDS,
    V1_PATH,
    V2_PATH,
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _ids(doc: dict) -> list[str]:
    return [s["id"] for s in doc["skills"]]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_v1_default_pointers_unchanged() -> None:
    assert SKILLS_90_GEMM_JSON.resolve() == V1_PATH.resolve()
    assert SKILLS_90_GEMM_V2_JSON.resolve() == V2_PATH.resolve()
    assert SKILLS_90_JSON.name == "skills_ii_target_miss_solutions_added(90skills).json"
    assert V1_PATH.is_file()
    assert V2_PATH.is_file()
    assert V1_PATH != V2_PATH


def test_v2_is_derived_from_untouched_v1() -> None:
    v1 = _load(V1_PATH)
    v2 = _load(V2_PATH)
    assert v2["derived_from"] == V1_PATH.name
    assert v2["derived_from_sha256"] == _sha(V1_PATH)
    assert v2["schema"] == "1.1"
    v1_ids = _ids(v1)
    v2_ids = _ids(v2)
    assert v1["skill_count"] == 99
    assert set(v1_ids).isdisjoint(NEW_IDS)
    for sid in NEW_IDS:
        assert sid in v2_ids
    for sid in v1_ids:
        assert sid in v2_ids
    assert v2["skill_count"] == len(v1_ids) + len(NEW_IDS)
    assert v2["skill_count"] == len(v2_ids)


def test_v1_recurrence_still_accepts_legal_ii() -> None:
    v1 = _load(V1_PATH)
    rec = next(s for s in v1["skills"] if s["id"] == "hls-pipeline-handle-true-recurrence")
    assert "accept the legal II" in rec["strategy"]


def test_v2_preventive_patches_and_new_skills() -> None:
    v2 = _load(V2_PATH)
    by_id = {s["id"]: s for s in v2["skills"]}
    for sid in PATCH_IDS:
        assert sid in by_id
    rec = by_id["hls-pipeline-handle-true-recurrence"]
    assert "do not accept II=4" in rec["strategy"]
    assert "switch(k&3)" in rec["strategy"]
    coal = by_id["hls-coalescing-512-compound-transform"]
    blob = coal["strategy"] + " " + " ".join(coal["required_steps"])
    assert "k+=4" in blob or "k += 4" in blob
    load = by_id["hls-axi-load-step-matches-widen-lanes"]
    assert "LANES" in load["strategy"]
    assert "16" in load["strategy"]
    assert by_id["avoid-switch-k-mod-named-acc-banks"]["kind"] == "avoid_rule"
    acc = by_id["hls-fp-mac-k-step-independent-acc-banks"]
    acc_blob = acc["strategy"] + " " + " ".join(acc["required_steps"])
    assert "k += 4" in acc_blob or "k+=4" in acc_blob
    assert "switch" in acc["strategy"]
    assert v2["metadata"]["gemm_flatten_v2_preventive"] is True
    assert v2["metadata"]["added_skill_ids"] == list(NEW_IDS)
