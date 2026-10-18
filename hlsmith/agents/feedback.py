"""FeedbackAgent: phase-specific agent control flow."""
import json
import logging
import os
from typing import List, Optional

from hlsmith import pipeline as core
from .base import _AgentBase


class FeedbackAgent(_AgentBase):
    """Phase 4: single owner of "given a typed failure record, produce an
    LLM prompt fragment."

    Consolidates the seven feedback renderers that used to live as free
    module-level functions and one SynthesisAgent method. Two consumers
    (SynthesisAgent's repair loop and the orchestrator's multistep loop)
    now delegate here so feedback shape stays consistent and a future
    LLM-aided variant can plug in at one call site.

    Default behavior is **deterministic templates** (zero LLM calls) —
    same on-disk output as Phase 1+2+3 today, just re-homed. The
    optional ``compose_with_llm()`` path (gated by
    ``C2HLS_FEEDBACK_LLM=1``) reads the LLM's actual edit + the typed
    failure record and routes through this agent's own model
    (``C2HLS_FEEDBACK_MODEL``, defaults to a cheap Haiku) to compose a
    more strategic prompt. Off by default; not exercised in production
    yet.
    """
    AGENT_NAME = "feedback"
    MODEL_ENV = core.FEEDBACK_MODEL_ENV

    # ---- compile / synth error feedback (ex-SynthesisAgent helpers) ----

    def render_for_compile_error(self, err: str) -> str:
        """Hint block for a g++ pre-flight or Vitis compile error."""
        return core._build_repair_guidance(err)

    def render_for_synth_error(self, err: str,
                                report: Optional[dict] = None) -> str:
        """Hint block for a Vitis synth failure. Combines the error-text
        guidance with a structured profile-bottleneck summary when a
        partial report is available."""
        parts = [core._build_repair_guidance(err)]
        if report:
            profile = core._build_profile_signal(
                report,
                part=self.orch.part,
                requested_clock_ns=self.orch.clock_ns,
            )
            if profile:
                parts.append("")
                parts.append(profile)
        return "\n".join(parts)

    # ---- multistep regression / no-op / alignment feedback (Phase 1+2+3) ----

    def render_for_regression(self, step_name: str,
                                reasons: List[str]) -> str:
        """Phase 1 regression-revert prompt fragment."""
        return core._render_regression_guidance(step_name, reasons)

    def render_for_no_op(self, step_name: str,
                          reasons: List[str]) -> str:
        """Phase 9 no-op-trap prompt fragment."""
        return core._render_no_op_guidance(step_name, reasons)

    def render_for_alignment(self, step_name: str,
                              decision) -> str:
        """Phase 3 trajectory-alignment "this step is an enabler" block.
        ``decision`` is a TrajectoryAlignmentDecision; rendering returns
        empty string when no message is needed."""
        from trajectory_alignment import render_alignment_for_prompt
        return render_alignment_for_prompt(decision, step_name)

    def render_for_throughput_regression(
        self, step_name: str, check) -> str:
        """Phase 9 throughput-regression hint. ``check`` is a
        robustness.ThroughputCheck."""
        if not check or not getattr(check, "flagged", False):
            return ""
        bullets = "\n".join(f"  - {r}" for r in (check.reasons or []))
        return (
            f"NOTE on the `{step_name}` step: throughput regression "
            f"flagged on the just-synthesized variant:\n{bullets}\n"
            f"Common causes: top-level kernel cannot start a new "
            f"transaction until the previous one drains (no DATAFLOW); "
            f"or unroll factor exceeds memory-port partition factor; "
            f"or pipeline depth was extended without rebalancing. Aim "
            f"for `Interval ≤ Latency` on the workload top function."
        )

    # ---- quality-gap feedback (ex-_build_quality_guidance) ----

    def render_for_quality_gap(
        self,
        bench_name: str,
        report: dict,
        ground_truth_report: dict,
        comparison: dict,
    ) -> str:
        """QualityRepairAgent's gap-closing prompt fragment."""
        return core._build_quality_guidance(
            bench_name, report, ground_truth_report, comparison,
        )

    # ---- top-level dispatch ----

    def render(
        self,
        kind: str,
        *,
        step_name: str = "",
        err: str = "",
        report: Optional[dict] = None,
        reasons: Optional[List[str]] = None,
        decision=None,
        check=None,
        bench_name: str = "",
        ground_truth_report: Optional[dict] = None,
        comparison: Optional[dict] = None,
    ) -> str:
        """Single dispatcher by failure ``kind``. Returns the prompt
        fragment (or empty string when nothing actionable)."""
        if kind == "compile_error":
            return self.render_for_compile_error(err)
        if kind == "synth_error":
            return self.render_for_synth_error(err, report)
        if kind == "regression":
            return self.render_for_regression(step_name, reasons or [])
        if kind == "no_op":
            return self.render_for_no_op(step_name, reasons or [])
        if kind == "alignment":
            return self.render_for_alignment(step_name, decision)
        if kind == "throughput_regression":
            return self.render_for_throughput_regression(step_name, check)
        if kind == "quality_gap":
            return self.render_for_quality_gap(
                bench_name, report or {}, ground_truth_report or {},
                comparison or {},
            )
        return ""

    # ---- Phase-4 stretch: optional LLM-aided composition ----

    def compose_with_llm(
        self,
        kind: str,
        *,
        kernel_diff: str = "",
        prior_template: str = "",
        bottleneck_record: Optional[dict] = None,
        **render_kwargs,
    ) -> str:
        """Call this agent's own model (cheap by default) to compose a
        more strategic retry prompt by reading the LLM's actual edit.
        Off by default — only fires when ``C2HLS_FEEDBACK_LLM=1`` AND
        the deterministic template (``prior_template``) has already
        been tried at least once.

        Returns the LLM-composed prompt fragment, or falls back to the
        deterministic template on any error."""
        if not int(os.getenv("C2HLS_FEEDBACK_LLM", "0") or "0"):
            return prior_template or self.render(kind, **render_kwargs)

        try:
            Instruction_c2hls = core.Instruction_c2hls  # noqa
        except ImportError:
            Instruction_c2hls = ""  # type: ignore

        prompt_lines = [
            "You are a senior FPGA engineer reviewing an LLM-generated "
            "HLS edit that just failed in synthesis. Read the failure "
            "record and the diff of the LLM's edit, then produce a "
            "concise (≤ 8 sentences) retry guidance that names the "
            "specific code construct (loop label, function, array) to "
            "fix and the pragma syntax to use.",
            "",
            f"Failure kind: {kind}",
        ]
        if bottleneck_record:
            prompt_lines.append(
                f"Bottleneck record: {json.dumps(bottleneck_record, indent=2)}"
            )
        if prior_template:
            prompt_lines.append("")
            prompt_lines.append("Prior deterministic feedback (already tried, didn't work):")
            prompt_lines.append(prior_template)
        if kernel_diff:
            prompt_lines.append("")
            prompt_lines.append("LLM's edit (unified diff or full new code):")
            prompt_lines.append("```")
            prompt_lines.append(kernel_diff[:6000])
            prompt_lines.append("```")
        prompt_lines.append("")
        prompt_lines.append("Output: just the retry guidance text. No code fences.")

        try:
            messages = [
                {"role": "user", "content": "\n".join(prompt_lines)},
            ]
            return self._call_llm(messages, max_tokens=800)
        except Exception as exc:  # pragma: no cover - LLM-aided is best-effort
            logging.warning("FeedbackAgent.compose_with_llm failed (%s); "
                             "falling back to deterministic template", exc)
            return prior_template or self.render(kind, **render_kwargs)
