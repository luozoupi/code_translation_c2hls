"""Demo: exercise FeedbackAgent.compose_with_llm() on a real regression
case from the static-knn run.

This is the LLM-aided composition path. With C2HLS_FEEDBACK_LLM=1 the
FeedbackAgent reads the LLM's actual edit + the typed failure record and
composes a *strategic* retry prompt — instead of the deterministic
template that just echoes the regression numbers.

Reads the unroll step from results_phase2/knn_static/. That step
regressed 6.88x in latency (240M → 1651M ns) with LUT 1.80x, FF 2.65x,
DSP 2.40x — a real failure mode, NOT synthetic.

Output: artifacts/phase4_feedback_llm_demo_<ts>.md with side-by-side
deterministic-template vs LLM-aided composition.

    cd /home/luo00466/code_translation-c2hls
    python tests/test_phase4_feedback_llm_demo.py
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def _build_orch():
    """Spin up a real C2HLSOrchestrator (requires Anthropic API key for
    the LLM-aided path)."""
    import c2hls
    return c2hls.C2HLSOrchestrator(gpt_model="claude-haiku-4-5-20251001")


def _load_unroll_regression_case() -> Dict[str, Any]:
    history_path = REPO_ROOT / "results_phase2" / "knn_static" / "knn_history.json"
    results_path = REPO_ROOT / "results_phase2" / "knn_static" / "knn_multistep_results.json"
    if not history_path.exists() or not results_path.exists():
        raise FileNotFoundError(
            "Need both knn_history.json and knn_multistep_results.json from the "
            "static-order knn run. Run tests/test_phase2_e2e_knn.py first."
        )
    history = json.loads(history_path.read_text())
    results = json.loads(results_path.read_text())

    # Find the unroll step
    unroll_step = next(
        (s for s in results.get("steps", []) if s.get("step_name") == "unroll"),
        None,
    )
    if unroll_step is None:
        raise RuntimeError("No 'unroll' step found in static knn results")

    # Pull the LLM's actual edit (assistant reply to the unroll prompt)
    msgs = history.get("messages", [])
    llm_edit = ""
    for i, m in enumerate(msgs):
        if (m.get("role") == "user"
                and "[Step: unroll]" in m.get("content", "")[:200]):
            if i + 1 < len(msgs) and msgs[i + 1].get("role") == "assistant":
                llm_edit = msgs[i + 1].get("content", "")
                break

    return {
        "step_name": "unroll",
        "rejected_report": unroll_step.get("rejected_report") or unroll_step.get("report") or {},
        "regression_reasons": unroll_step.get("regression_reasons") or [
            "latency_ns regressed 6.88x (240000000 -> 1651000000)",
            "resource_growth: lut 1.80x, ff 2.65x, dsp 2.40x",
        ],
        "llm_edit": llm_edit,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    timestamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    parser.add_argument("--out", type=Path,
                        default=REPO_ROOT / "artifacts" / f"phase4_feedback_llm_demo_{timestamp}.md")
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    case = _load_unroll_regression_case()
    print(f"Loaded regression case: step={case['step_name']}, "
          f"reasons={case['regression_reasons'][:1]}, "
          f"edit_len={len(case['llm_edit'])} chars")

    orch = _build_orch()
    fa = orch.feedback

    deterministic = fa.render(
        "regression",
        step_name=case["step_name"],
        reasons=case["regression_reasons"],
    )

    # Force the LLM-aided path
    os.environ["C2HLS_FEEDBACK_LLM"] = "1"
    bottleneck_record = {
        "kind": "resource_growth_with_latency_regression",
        "regressed_metrics": {
            "latency_ns": "+6.88x (240M → 1651M)",
            "lut": "+1.80x",
            "ff": "+2.65x",
            "dsp": "+2.40x",
        },
        "step": "unroll",
        "kernel": "knn",
    }
    llm_aided = fa.compose_with_llm(
        "regression",
        kernel_diff=case["llm_edit"][:6000],
        prior_template=deterministic,
        bottleneck_record=bottleneck_record,
        step_name=case["step_name"],
        reasons=case["regression_reasons"],
    )

    # Render markdown
    md: List[str] = []
    md.append(f"# FeedbackAgent: deterministic vs LLM-aided composition\n")
    md.append(f"_generated {_dt.datetime.now().isoformat(timespec='seconds')}_\n")
    md.append("")
    md.append("Real regression case from the static-knn run: the LLM's "
              "`unroll` step produced a kernel that synthesized 6.88x slower "
              "latency than the parent step (240M → 1651M ns) with "
              "LUT 1.80x, FF 2.65x, DSP 2.40x. Phase 1's regression-revert "
              "kicked in.")
    md.append("")
    md.append("This script asks the FeedbackAgent for a retry-guidance "
              "prompt twice:")
    md.append("- **Deterministic template** (default, no LLM call): "
              "`fa.render('regression', ...)`")
    md.append("- **LLM-aided composition** (Phase 4 stretch path): "
              "`fa.compose_with_llm('regression', kernel_diff=..., "
              "prior_template=..., bottleneck_record=...)` with "
              "`C2HLS_FEEDBACK_LLM=1`. Routes through the FeedbackAgent's "
              "own model (Haiku 4.5 in this run).")
    md.append("")

    md.append("## Deterministic template")
    md.append("")
    md.append("```")
    md.append(deterministic)
    md.append("```")
    md.append("")
    md.append(f"_{len(deterministic)} chars; emitted with zero LLM cost._")
    md.append("")

    md.append("## LLM-aided composition")
    md.append("")
    md.append("```")
    md.append(llm_aided)
    md.append("```")
    md.append("")
    md.append(f"_{len(llm_aided)} chars._")
    md.append("")

    md.append("## Compare")
    md.append("")
    md.append("| dimension | deterministic | LLM-aided |")
    md.append("|-----------|---------------|-----------|")
    md.append(f"| length | {len(deterministic)} chars | {len(llm_aided)} chars |")
    md.append(f"| reads kernel diff? | no | yes ({len(case['llm_edit'])} chars in) |")
    md.append(f"| names specific construct (loop label / pragma) | no (just numbers) | inspect output above |")
    md.append("| cost | 0 LLM calls | 1 small-model LLM call |")
    md.append("| latency | µs | ~1-3s |")
    md.append("")

    args.out.write_text("\n".join(md), encoding="utf-8")
    print(f"\nartifact: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
