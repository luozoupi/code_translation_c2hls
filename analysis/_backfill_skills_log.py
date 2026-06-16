"""Backfill skills_log sidecar files for cells that ran BEFORE the skills_log
feature landed (2026-06-05).

For each cell missing a skills_log field in its *_results.json:
  1. Determine whether skills were ACTUALLY applied in the run by grepping the
     cell's *_history.json for "## Optimization skills" markers. This is the
     source of truth — if the prompt didn't carry the rendered block, no
     skills reached the LLM no matter what env vars were set.
  2. If skills WERE applied: replay the deterministic retrieve_for_translation
     against the pinned skills.json to reconstruct selected_ids. (Multistep
     cells with truncated history get the same replay but tagged with weaker
     provenance.)
  3. If skills were NOT applied: emit a skills_log with enabled=False and
     empty selected_ids — accurately reflecting reality.
  4. Write the result to a sidecar file:
       <bench>_skills_log.backfilled.json
     Do NOT mutate the original *_results.json (already-shipped data is
     immutable).

Usage:
  python3 _backfill_skills_log.py <results_dir> --skills-path <path[:path...]>
                                  [--target-part xcu280-fsvh2892-2L-e]

The --skills-path picks which skills config to replay against. For phase 8 use
the base skills.json only; for the bugged phase 9 cells the actual loaded
config was empty (bug), so passing the multi-path here is a "what would have
been retrieved if the bug hadn't fired" hypothetical — the script detects
this via history.json and records enabled=False with provenance
'inferred_no_skills_from_history' regardless of what --skills-path says.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

# Module path setup so we can import skills.py + reuse its retrievers
HERE = Path(__file__).resolve().parent.parent  # repo root (script lives in analysis/)
sys.path.insert(0, str(HERE))


def _file_sha1(paths: list[Path]) -> str:
    h = hashlib.sha1()
    for p in paths:
        if p.is_file():
            h.update(p.read_bytes())
    return h.hexdigest()


def _has_skill_marker(history_path: Path) -> bool:
    """True if any captured prompt in history.json carries the rendered skill
    block header. Authoritative source of truth: what the LLM actually saw."""
    if not history_path.exists():
        return False
    text = history_path.read_text(encoding="utf-8", errors="ignore")
    return "## Optimization skills" in text


def _has_partial_history_truncation(history_path: Path) -> bool:
    """Returns True if any user-prompt entry is a truncated f-string of the
    form '[Step: ...] <first 200 chars>...'. Indicates multistep optimization
    steps where the skill block (concatenated at the END) was dropped from
    capture. For these cells we cannot use grep on history; we MUST trust the
    deterministic replay."""
    if not history_path.exists():
        return False
    try:
        entries = json.loads(history_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    for e in entries:
        if isinstance(e, dict) and e.get("role") == "user":
            c = e.get("content") or ""
            if isinstance(c, str) and c.startswith("[Step:") and c.endswith("..."):
                return True
    return False


def _build_entry_from_translation(skills_list: list[dict], target_part: str,
                                   top_k: int = 4) -> dict:
    """Mirror the structure that c2hls._skills_block_for_translation produces
    via _log_skills_entry. We only backfill the Phase B / opt translation
    entries here — quality-repair and error-repair entries would require
    per-turn synth-report inputs from optimization_history (a future v2)."""
    # Import lazily so the script works even if skills.py changes signature
    from skills import retrieve_for_translation, render_skill_block
    selected = retrieve_for_translation(skills_list, top_k=top_k, target_part=target_part)
    rendered = render_skill_block(selected, verbose=False) if selected else ""
    return {
        "turn": 0,
        "phase": "B",
        "callsite": "translation",
        "retriever": "retrieve_for_translation",
        "target_part": target_part,
        "top_k": top_k,
        "inputs_digest": {"target_part": target_part},
        "selected_ids": [s["id"] for s in selected],
        "rendered_chars": len(rendered),
    }


def _process_cell(cell_dir: Path, bench: str, skills_list: list[dict],
                  config_path: str, config_sha1: str, target_part: str,
                  force: bool) -> tuple[str, dict | None]:
    """Returns (status, payload). Status one of:
      'wrote', 'skipped_has_native', 'skipped_already_backfilled',
      'skipped_no_results', 'wrote_no_skills'.
    """
    rj = cell_dir / f"{bench}_results.json"
    history = cell_dir / f"{bench}_history.json"
    if not rj.exists():
        return "skipped_no_results", None
    data = json.loads(rj.read_text())
    if data.get("skills_log") is not None:
        return "skipped_has_native", None

    sidecar = cell_dir / f"{bench}_skills_log.backfilled.json"
    if sidecar.exists() and not force:
        return "skipped_already_backfilled", None

    # Step 1: determine if skills were actually loaded based on cell tag
    # (the dir name encodes intent: __skills means env var was set)
    intended_on = cell_dir.name.endswith("__skills")

    # Step 2: confirm via history grep that the rendered skill block was
    # ACTUALLY in the prompt the LLM saw (catches the c2hls.py:88
    # multi-path bug case where intent=on but actual=off)
    marker_present = _has_skill_marker(history)
    history_truncated = _has_partial_history_truncation(history)

    if intended_on and marker_present:
        # Real skills run: replay deterministic retriever
        entry = _build_entry_from_translation(skills_list, target_part)
        payload = {
            "enabled": True,
            "skills_config_path": config_path,
            "skills_config_sha1": config_sha1,
            "n_entries": 1,
            "unique_skill_ids": sorted(set(entry["selected_ids"])),
            "entries": [entry],
            "provenance": "replayed_translation_only",
            "provenance_note": (
                "Phase B translation skill list reconstructed via deterministic "
                "replay against the pinned skills config. Verified by "
                "'## Optimization skills' marker in *_history.json. Does NOT "
                "include quality_repair/error_repair entries — those depend on "
                "per-turn synth/error inputs not captured here."
            ),
        }
        return "wrote", payload

    if intended_on and not marker_present and history_truncated:
        # Multistep cells truncate the user prompt to 200 chars in history.
        # We can't grep-verify, but we can replay; tag with weaker provenance.
        entry = _build_entry_from_translation(skills_list, target_part)
        payload = {
            "enabled": True,
            "skills_config_path": config_path,
            "skills_config_sha1": config_sha1,
            "n_entries": 1,
            "unique_skill_ids": sorted(set(entry["selected_ids"])),
            "entries": [entry],
            "provenance": "replayed_multistep_unverified",
            "provenance_note": (
                "Multistep cell: *_history.json has truncated '[Step: ...]' "
                "user entries that drop the skill block, so we cannot grep-"
                "verify that skills were actually applied. The replay assumes "
                "the multi-path loader worked (no bug). If skills_log.enabled "
                "is False on contemporaneous native cells, the backfilled "
                "skills here are HYPOTHETICAL not actual."
            ),
        }
        return "wrote", payload

    if intended_on and not marker_present and not history_truncated:
        # The bugged case: cell tagged skills=on, but history shows no skill
        # block was rendered. Means skills loader returned empty (the
        # c2hls.py:88 multi-path bug). Record enabled=False to match reality.
        payload = {
            "enabled": False,
            "skills_config_path": config_path,
            "skills_config_sha1": config_sha1,
            "n_entries": 0,
            "unique_skill_ids": [],
            "entries": [],
            "provenance": "inferred_no_skills_from_history",
            "provenance_note": (
                "Cell directory tagged __skills (intent=on) but history.json "
                "contains no '## Optimization skills' marker — skills did "
                "NOT reach the LLM despite the env var being set. Almost "
                "certainly the c2hls.py:88 multi-path Path.is_file() bug "
                "(fixed 2026-06-05). The backfilled skills_log records "
                "reality: enabled=False, no skills applied."
            ),
        }
        return "wrote_no_skills", payload

    # intent=off: skills env var was deliberately empty; record accordingly
    payload = {
        "enabled": False,
        "skills_config_path": None,
        "skills_config_sha1": None,
        "n_entries": 0,
        "unique_skill_ids": [],
        "entries": [],
        "provenance": "inferred_skills_off_from_cell_dir",
        "provenance_note": (
            "Cell directory tagged __noskills — C2HLS_SKILLS_PATH was unset "
            "for this run by the sweep driver. No skills were retrieved or "
            "rendered. This is the no-skills baseline."
        ),
    }
    return "wrote", payload


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results_dir", type=Path)
    ap.add_argument("--skills-path", required=True,
                    help="Colon-separated list of skill JSON files to replay against (e.g. base.json or base.json:extension.json)")
    ap.add_argument("--target-part", default="xcu280-fsvh2892-2L-e")
    ap.add_argument("--force", action="store_true",
                    help="Overwrite existing *_skills_log.backfilled.json sidecars")
    args = ap.parse_args()

    if not args.results_dir.is_dir():
        print(f"not a dir: {args.results_dir}", file=sys.stderr)
        return 2

    # Load skills via the production loader so the replay uses identical
    # parsing behavior + accepts the same colon-multi-path format.
    os.environ["C2HLS_SKILLS_PATH"] = args.skills_path
    from skills import load_skills
    skills = load_skills(args.skills_path)
    path_objs = [Path(p) for p in args.skills_path.split(":") if p.strip()]
    config_sha1 = _file_sha1(path_objs)
    print(f"Loaded {len(skills)} skills from {args.skills_path}")
    print(f"  config_sha1: {config_sha1}")
    print(f"  target_part: {args.target_part}")
    print()

    counts = {"wrote": 0, "wrote_no_skills": 0, "skipped_has_native": 0,
              "skipped_already_backfilled": 0, "skipped_no_results": 0}
    per_status: dict[str, list[str]] = {k: [] for k in counts}

    for bench_dir in sorted(args.results_dir.iterdir()):
        if not bench_dir.is_dir():
            continue
        bench = bench_dir.name
        for cell_dir in sorted(bench_dir.iterdir()):
            if not cell_dir.is_dir():
                continue
            status, payload = _process_cell(
                cell_dir, bench, skills, args.skills_path, config_sha1,
                args.target_part, args.force,
            )
            counts[status] += 1
            per_status[status].append(f"{bench}/{cell_dir.name}")
            if payload is not None:
                sidecar = cell_dir / f"{bench}_skills_log.backfilled.json"
                sidecar.write_text(json.dumps(payload, indent=2))

    print("Summary:")
    for k, n in counts.items():
        print(f"  {k:<30}: {n}")
    if counts["wrote_no_skills"] > 0:
        print()
        print(f"Note: {counts['wrote_no_skills']} cells recorded enabled=False "
              "(skills did not reach the LLM despite intent=on — see "
              "individual sidecar provenance for details).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
