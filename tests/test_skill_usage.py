from pathlib import Path

from skill_library import Skill, SkillLibrary
from skill_usage import (
    parse_skill_usage_declaration,
    usage_declaration_instruction,
    verify_declared_skill_usage,
)


def _library(tmp_path: Path) -> SkillLibrary:
    library = SkillLibrary(tmp_path / "skills.json")
    library.add(
        Skill(
            id="pipeline-hot-loop",
            pattern="hot loop is not pipelined",
            strategy="pipeline the hot loop",
            tags=["pipeline", "hot-loop"],
        )
    )
    return library


def test_declaration_accepts_only_rendered_ids() -> None:
    parsed = parse_skill_usage_declaration(
        'SKILL_USAGE: ["pipeline-hot-loop", "not-rendered"]\n```cpp\nx\n```',
        rendered_skill_ids=["pipeline-hot-loop"],
    )

    assert parsed["declared_applied_skill_ids"] == ["pipeline-hot-loop"]
    assert parsed["undeclared_or_unrendered_skill_ids"] == ["not-rendered"]
    assert parsed["skill_declaration_status"] == "contains_unrendered_ids"


def test_static_verification_requires_new_family_marker(
    tmp_path: Path,
) -> None:
    verified = verify_declared_skill_usage(
        parent_code="void f() { for (;;) {} }",
        candidate_code=(
            "void f() {\n#pragma HLS PIPELINE II=1\nfor (;;) {}\n}"
        ),
        declared_skill_ids=["pipeline-hot-loop"],
        library=_library(tmp_path),
    )

    assert verified["verified_applied_skill_ids"] == ["pipeline-hot-loop"]
    assert verified["introduced_transformation_families"] == ["pipeline"]


def test_usage_instruction_lists_exact_allowed_ids() -> None:
    instruction = usage_declaration_instruction(["a", "b"])

    assert "SKILL_USAGE: []" in instruction
    assert 'Allowed IDs: ["a", "b"]' in instruction
