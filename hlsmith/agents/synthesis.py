"""SynthesisAgent: phase-specific agent control flow."""
import logging
import os
from typing import Optional

from hlsmith import pipeline as core
from .base import _AgentBase


class SynthesisAgent(_AgentBase):
    """Phase B synth + test loop with LLM-driven repair on compile or synth
    failures. Mutates orch.hls_code, orch.synth_report, orch.generated_csim,
    orch.generated_cosim, orch.turn_results.

    P5 enhancements:
      - **Profile-feedback loop.** Each repair prompt now carries a
        structured bottleneck summary (TIMING_VIOLATION / *_OVERFLOW /
        LATENCY_HIGH / FMAX_BELOW_TARGET) derived from the synth report,
        not just the raw error log. Empty when nothing is actionable.
      - **Best-state tracking + revert-on-streak.** If `C2HLS_SYNTH_REVERT_THRESHOLD`
        consecutive same-class errors hit, we either revert to the last
        successful synth (if any) and exit early, or — when no good run
        exists yet — abort the loop early instead of burning more LLM
        budget on the same dead end. Default 0 = disabled (preserves
        current behavior).
    """
    AGENT_NAME = "synthesis"
    MODEL_ENV = core.SYNTHESIS_MODEL_ENV

    @property
    def revert_threshold(self) -> int:
        """Read at access time so tests / env-overrides don't have to
        re-instantiate the orchestrator."""
        try:
            return int(os.getenv("C2HLS_SYNTH_REVERT_THRESHOLD", "0"))
        except ValueError:
            return 0

    def _compose_repair_guidance(self, err: str, report: dict = None) -> str:
        """Combine error-text hints with structured profile signals."""
        parts = [core._build_repair_guidance(err)]
        if report:
            profile = core._build_profile_signal(
                report,
                part=self.orch.part,
                requested_clock_ns=self.orch.clock_ns,
            )
            if profile:
                parts.append("")  # blank line between sections
                parts.append(profile)
        return "\n".join(parts)

    def _record_best(self, hls_code: str, result: dict, outcome: dict) -> dict:
        """Capture a synth-success snapshot for revert-on-streak."""
        return {
            "code": hls_code,
            "report": result.get("report", {}),
            "csim": outcome.get("csim"),
            "cosim": outcome.get("cosim"),
        }

    def _correctness_gate_failure(self, outcome: dict) -> "tuple[str, str]":
        """Return (gate_name, error_text) when Phase B synth passes but a
        generated csim/cosim check that actually ran fails.

        Phase B is only a valid baseline when it is functionally correct under
        the available generated testbench. Reference csim can be externally
        trusted, but generated code must still pass its own correctness gate.
        """
        if os.getenv("C2HLS_DISABLE_CORRECTNESS_REPAIR", "0").lower() in ("1", "true", "yes"):
            return "", ""
        for gate_name in ("csim", "cosim"):
            if gate_name == "cosim" and not core._cosim_required_for_correctness():
                continue
            summary = outcome.get(gate_name)
            if not isinstance(summary, dict):
                continue
            # A timeout, including a gold-relative predicted timeout, is an
            # inconclusive performance outcome rather than evidence that the
            # generated algorithm is functionally wrong. Do not spend repair
            # turns regenerating the same finite design.
            if core._test_summary_is_timeout(summary):
                continue
            if not summary.get("ran") or summary.get("passed"):
                continue
            gate_error = (
                (summary.get("error") or "").strip() + "\n"
                + (summary.get("log_excerpt") or "").strip()
            ).strip() or "(testbench reported a mismatch but did not capture an error message)"
            return gate_name, gate_error
        return "", ""

    def _restore_from_best(self, best: dict) -> None:
        orch = self.orch
        orch.hls_code = best["code"]
        orch.synth_report = best["report"]
        orch.generated_csim = best.get("csim")
        orch.generated_cosim = best.get("cosim")

    def synthesize_with_repair(self) -> bool:
        orch = self.orch
        best_state: Optional[dict] = None
        error_class_history: list[str] = []
        threshold = self.revert_threshold

        for turn in range(orch.turns_limitation):
            logging.info("[Phase B] Synthesis attempt %d", turn)
            orch.hls_code = orch._preflight_generated_hls_code(
                orch.hls_code, f"Phase B attempt {turn}",
            )

            ok, err = core.compile_check_cpp(
                orch.hls_code, orch.header_code, orch.header_name,
                extra_files=orch.extra_files,
            )
            if not ok:
                logging.warning("[Phase B] HLS code doesn't compile: %s", err[:200])
                orch._finalize_candidate_evaluation(
                    code=orch.hls_code,
                    correctness_status="not_run",
                    synthesis_status="not_run",
                    failure_class="compile_or_interface_failure",
                )
                error_class_history.append(core._classify_synth_error(err))
                if self._should_revert(error_class_history, best_state, threshold):
                    return self._revert_and_exit(error_class_history, best_state, threshold)
                fix_prompt = core.c_compilation_fix.format(
                    compile_error=err,
                    hls_code=orch.hls_code,
                    benchmark_context=orch.benchmark_context,
                    repair_guidance=self._compose_repair_guidance(err, report=None),
                    attempt_history=core._format_attempt_history(orch.turn_results, "B"),
                )
                orch.messages.append({"role": "user", "content": fix_prompt})
                reply = self._call_llm(orch.messages)
                orch.messages.append({"role": "assistant", "content": reply})
                orch._append_history("user", fix_prompt)
                orch._append_history("assistant", reply)
                fixed = core.extract_cpp_code(reply)
                if fixed:
                    orch.hls_code = fixed
                else:
                    orch._finalize_candidate_evaluation(
                        correctness_status="not_run",
                        synthesis_status="not_run",
                        failure_class="malformed_output",
                    )
                continue

            outcome = orch._synth_and_test(orch.hls_code, log_prefix="[Phase B]")
            result = outcome["synth"]
            orch.turn_results.append({
                "turn": turn,
                "phase": "B",
                "success": result["success"],
                "report": result.get("report", {}),
                "error": result.get("error", ""),
            })

            if result.get("budget_exhausted"):
                logging.warning("[Phase B] %s", result.get("error"))
                return False

            if result.get("skip_reason") == "csim_correctness_gate_failed":
                gate_summary = outcome.get("csim") or {}
                gate_error = (
                    (gate_summary.get("error") or "").strip()
                    + "\n"
                    + (gate_summary.get("log_excerpt") or "").strip()
                ).strip() or "pre-synthesis CSim/golden comparison failed"
                failure = f"csim_failed: {gate_error[:300]}"
                orch.turn_results.append({
                    "turn": turn,
                    "phase": "B",
                    "success": False,
                    "stage": "csim",
                    "csim": gate_summary,
                    "error": failure,
                })
                if turn >= orch.turns_limitation - 1:
                    continue
                fix_prompt = core.hls_correctness_repair_fix.format(
                    step_name="initial translation",
                    gate_name="csim",
                    gate_error=gate_error[:2000],
                    hls_code=orch.hls_code,
                    header_code=orch.header_code,
                    benchmark_context=orch.benchmark_context,
                    attempt_history=core._format_attempt_history(orch.turn_results, "B"),
                )
                orch.messages.append({"role": "user", "content": fix_prompt})
                reply = self._call_llm(orch.messages)
                orch.messages.append({"role": "assistant", "content": reply})
                orch._append_history("user", fix_prompt)
                orch._append_history("assistant", reply)
                fixed = core.extract_cpp_code(reply)
                if fixed:
                    orch.hls_code = fixed
                else:
                    orch._finalize_candidate_evaluation(
                        correctness_status="not_run",
                        synthesis_status="not_run",
                        failure_class="malformed_output",
                    )
                continue

            if result["success"]:
                orch.synth_report = result["report"]
                logging.info("[Phase B] Synthesis SUCCESS!\n%s",
                             core.format_report_summary(result["report"]))
                orch._append_history(
                    "system",
                    f"[Phase B] Synthesis SUCCESS. Report:\n"
                    f"{core.format_report_summary(result['report'])}",
                )

                orch.generated_csim = outcome["csim"]
                if orch.generated_csim is not None:
                    if orch.generated_csim.get("passed"):
                        logging.info("[Phase B] Csim PASSED")
                    else:
                        logging.warning(
                            "[Phase B] Csim FAILED: %s",
                            (orch.generated_csim.get("error") or "")[:200],
                        )

                orch.generated_cosim = outcome["cosim"]
                if orch.generated_cosim is not None:
                    if orch.generated_cosim.get("passed"):
                        logging.info("[Phase B] Cosim PASSED")
                    elif core._test_summary_is_timeout(orch.generated_cosim):
                        qualifier = "predicted; not run" if not orch.generated_cosim.get("ran") else "wall-clock"
                        logging.warning(
                            "[Phase B] Cosim TIMEOUT (%s): %s",
                            qualifier,
                            (orch.generated_cosim.get("error") or "")[:240],
                        )
                    else:
                        logging.warning(
                            "[Phase B] Cosim FAILED: %s",
                            (orch.generated_cosim.get("error") or "")[:200],
                        )

                gate_name, gate_error = self._correctness_gate_failure(outcome)
                if gate_name:
                    logging.warning(
                        "[Phase B] %s FAILED on attempt %d — asking translator/synthesis "
                        "thread to repair functionality before optimization",
                        gate_name,
                        turn,
                    )
                    failure = f"{gate_name}_failed: {gate_error[:300]}"
                    orch.turn_results.append({
                        "turn": turn,
                        "phase": "B",
                        "success": False,
                        "stage": gate_name,
                        "report": result.get("report", {}),
                        "csim": outcome.get("csim"),
                        "cosim": outcome.get("cosim"),
                        "error": failure,
                    })
                    error_class_history.append(f"{gate_name}_failed")
                    if self._should_revert(error_class_history, best_state, threshold):
                        return self._revert_and_exit(error_class_history, best_state, threshold)
                    if turn >= orch.turns_limitation - 1:
                        continue
                    fix_prompt = core.hls_correctness_repair_fix.format(
                        step_name="initial translation",
                        gate_name=gate_name,
                        gate_error=gate_error[:2000],
                        hls_code=orch.hls_code,
                        header_code=orch.header_code,
                        benchmark_context=orch.benchmark_context,
                        attempt_history=core._format_attempt_history(orch.turn_results, "B"),
                    )
                    orch.messages.append({"role": "user", "content": fix_prompt})
                    reply = self._call_llm(orch.messages)
                    orch.messages.append({"role": "assistant", "content": reply})
                    orch._append_history("user", fix_prompt)
                    orch._append_history("assistant", reply)
                    fixed = core.extract_cpp_code(reply)
                    if fixed:
                        orch.hls_code = fixed
                    else:
                        orch._finalize_candidate_evaluation(
                            correctness_status="not_run",
                            synthesis_status="not_run",
                            failure_class="malformed_output",
                        )
                    continue

                # Snapshot only after correctness passes (or no generated
                # testbench was available). A synth-only success with failing
                # csim is not a usable Phase B baseline.
                best_state = self._record_best(orch.hls_code, result, outcome)
                error_class_history.clear()
                return True

            logging.warning("[Phase B] Synthesis failed: %s", result["error"][:300])
            error_class_history.append(core._classify_synth_error(result["error"]))
            if self._should_revert(error_class_history, best_state, threshold):
                return self._revert_and_exit(error_class_history, best_state, threshold)

            is_timeout = "timed out" in result["error"].lower()
            guidance = self._compose_repair_guidance(
                result["error"], report=result.get("report"),
            )
            history_block = core._format_attempt_history(orch.turn_results, "B")
            if is_timeout:
                fix_prompt = core.hls_synthesis_timeout_fix.format(
                    timeout=600,
                    hls_code=orch.hls_code,
                    header_code=orch.header_code,
                    benchmark_context=orch.benchmark_context,
                    repair_guidance=guidance,
                    attempt_history=history_block,
                )
            else:
                fix_prompt = core.hls_synthesis_fix.format(
                    synth_error=result["error"],
                    hls_code=orch.hls_code,
                    header_code=orch.header_code,
                    target_context=core._target_context_for_prompt(orch.part, orch.clock_ns),
                    benchmark_context=orch.benchmark_context,
                    repair_guidance=guidance,
                    attempt_history=history_block,
                )
            orch.messages.append({"role": "user", "content": fix_prompt})
            reply = self._call_llm(orch.messages)
            orch.messages.append({"role": "assistant", "content": reply})
            orch._append_history("user", fix_prompt)
            orch._append_history("assistant", reply)
            fixed = core.extract_cpp_code(reply)
            if fixed:
                orch.hls_code = fixed
            else:
                orch._finalize_candidate_evaluation(
                    correctness_status="not_run",
                    synthesis_status="not_run",
                    failure_class="malformed_output",
                )

        orch._append_history(
            "system",
            f"[Phase B] FAIL: HLS synthesis/correctness not achieved in {orch.turns_limitation} turns",
        )
        return False

    def _should_revert(self, history: list, best_state: Optional[dict],
                       threshold: int) -> bool:
        """True when we've seen `threshold` consecutive errors of the same
        class. threshold<=0 disables the check entirely."""
        if threshold <= 0 or len(history) < threshold:
            return False
        recent = history[-threshold:]
        return len(set(recent)) == 1

    def _revert_and_exit(self, history: list, best_state: Optional[dict],
                         threshold: int) -> bool:
        """Either restore the last successful state or abort the loop early.
        Returning True means "Phase B succeeded (via the restored state)";
        False means "Phase B is giving up early because the LLM is stuck"."""
        orch = self.orch
        stuck_class = history[-1] if history else "unknown"
        if best_state is not None:
            self._restore_from_best(best_state)
            logging.info(
                "[Phase B] Stuck on %s for %d turns; reverting to last good synth",
                stuck_class, threshold,
            )
            orch._append_history(
                "system",
                f"[Phase B] Reverted to last good synth after "
                f"{threshold} consecutive {stuck_class} errors",
            )
            return True
        logging.warning(
            "[Phase B] Stuck on %s for %d turns and no good state to revert to; "
            "exiting Phase B early",
            stuck_class, threshold,
        )
        orch._append_history(
            "system",
            f"[Phase B] Early exit: {threshold} consecutive {stuck_class} "
            f"errors with no successful synth to revert to",
        )
        return False
