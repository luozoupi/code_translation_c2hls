from __future__ import annotations

import json
from pathlib import Path

import pytest

from skill_library import Skill, load_frozen_library


def _write(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_frozen_loader_uses_only_snapshot_entries(tmp_path: Path) -> None:
    path = tmp_path / "skills.json"
    _write(
        path,
        {
            "schema": "1.1",
            "skills": [
                {"id": "validated-only", "pattern": "p", "strategy": "s"}
            ],
        },
    )

    library = load_frozen_library(path)

    assert [skill.id for skill in library.all()] == ["validated-only"]
    assert library.store_path == path.resolve()
    assert library.exact_frozen_snapshot is True


def test_frozen_library_rejects_every_mutator_and_returns_defensive_copies(
    tmp_path: Path,
) -> None:
    path = tmp_path / "skills.json"
    _write(
        path,
        {
            "schema": "1.1",
            "skills": [
                {
                    "id": "validated-only",
                    "pattern": "original pattern",
                    "strategy": "original strategy",
                    "tags": ["validated"],
                }
            ],
        },
    )
    library = load_frozen_library(path)

    exposed = library.get("validated-only")
    assert exposed is not None
    exposed.strategy = "mutated by caller"
    exposed.tags.append("mutated")
    assert library.get("validated-only").strategy == "original strategy"
    assert library.get("validated-only").tags == ["validated"]

    new_skill = Skill(id="new", pattern="p", strategy="s")
    mutators = [
        lambda: library.load(),
        lambda: library.save(),
        lambda: library.add(new_skill),
        lambda: library.remove("validated-only"),
        lambda: library.update_skill_statistics("validated-only", success=True),
        lambda: library.promote_demote("validated-only"),
        lambda: library.mark_avoid("validated-only"),
    ]
    for mutate in mutators:
        with pytest.raises(RuntimeError, match="immutable"):
            mutate()


@pytest.mark.parametrize(
    "payload",
    [
        {"schema": "1.0", "skills": [{"id": "x", "pattern": "p", "strategy": "s"}]},
        {"schema": "1.1", "skills": []},
        {"schema": "1.1", "skills": [{"id": "x"}]},
        {
            "schema": "1.1",
            "saved_at": "mutable-store-marker",
            "skills": [{"id": "x", "pattern": "p", "strategy": "s"}],
        },
        {
            "schema": "1.1",
            "skills": [
                {"id": "x", "pattern": "p", "strategy": "s", "unknown": 1}
            ],
        },
        {
            "schema": "1.1",
            "skills": [
                {"id": "x", "pattern": "p", "strategy": "s"},
                {"id": "x", "pattern": "q", "strategy": "t"},
            ],
        },
    ],
)
def test_frozen_loader_rejects_non_exact_or_malformed_snapshot(
    tmp_path: Path, payload: object
) -> None:
    path = tmp_path / "skills.json"
    _write(path, payload)

    with pytest.raises(ValueError):
        load_frozen_library(path)
