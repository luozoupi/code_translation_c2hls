"""TranslatorAgent: phase-specific agent control flow."""
import logging
from typing import Optional

from hlsmith import pipeline as core
from .base import _AgentBase


class TranslatorAgent(_AgentBase):
    """Phase A (validate plain C compiles) + initial Phase B translate.

    Owns the conversation initialization (system prompt + first user prompt)
    so the SynthesisAgent's repair turns layer on top of a coherent thread.
    """
    AGENT_NAME = "translator"
    MODEL_ENV = core.TRANSLATOR_MODEL_ENV

    def run_phase_a(self, c_code: str, header_code: str = "",
                    header_name: str = "kernel.h") -> bool:
        orch = self.orch
        orch.c_code = c_code
        orch.header_code = header_code
        orch.header_name = header_name

        logging.info("=== [Phase A] Validating C code compilation ===")
        orch._append_history("system", core.Instruction_c2hls)

        ok, err = core.compile_check_cpp(
            c_code, header_code, header_name, extra_files=orch.extra_files,
        )
        if ok:
            logging.info("[Phase A] C code compiles successfully")
            orch._append_history("system", "[Phase A] Input C code compiles OK.")
            return True

        logging.warning("[Phase A] C code fails to compile: %s", err)
        for turn in range(orch.turns_limitation):
            prompt = core.q_validate_c_code.format(
                c_code=orch.c_code,
                header_code=orch.header_code,
                benchmark_context=orch.benchmark_context,
            )
            orch.messages = [
                {"role": "system", "content": core.Instruction_c2hls},
                {"role": "user", "content": prompt},
            ]
            reply = self._call_llm(orch.messages)
            orch._append_history("assistant", reply)

            fixed = core.extract_cpp_code(reply)
            if fixed:
                orch.c_code = fixed
                ok, err = core.compile_check_cpp(
                    orch.c_code, orch.header_code, orch.header_name,
                    extra_files=orch.extra_files,
                )
                if ok:
                    orch._finalize_candidate_evaluation(
                        code=orch.c_code,
                        correctness_status="not_run",
                        synthesis_status="not_run",
                        failure_class="other",
                        failure_detail=(
                            "LLM response repaired the plain-C input preflight; "
                            "it is not an HLS candidate"
                        ),
                    )
                    logging.info("[Phase A] Fixed C code compiles (turn %d)", turn)
                    return True
                orch._finalize_candidate_evaluation(
                    code=orch.c_code,
                    correctness_status="not_run",
                    synthesis_status="not_run",
                    failure_class="compile_or_interface_failure",
                )
                logging.warning("[Phase A] Still fails (turn %d): %s", turn, err[:200])
            else:
                orch._finalize_candidate_evaluation(
                    correctness_status="not_run",
                    synthesis_status="not_run",
                    failure_class="malformed_output",
                )

        orch._append_history(
            "system",
            f"[Phase A] FAIL: C code does not compile after {orch.turns_limitation} turns",
        )
        return False

    def translate_initial(self) -> Optional[str]:
        """Run Phase B's initial translate prompt and return the extracted
        HLS code. Does not synthesize — the SynthesisAgent does that."""
        orch = self.orch
        mode = core._normalize_phaseb_mode(
            getattr(orch, "phaseb_mode", ""),
            multistep=bool(getattr(orch, "_phaseb_multistep_context", False)),
        )
        orch.phaseb_mode = mode
        logging.info("=== [Phase B] Translating C to HLS (mode=%s) ===", mode)

        prompt_template = (
            core.q_translate_c_to_hls_functional
            if mode == "functional"
            else core.q_translate_c_to_hls
        )
        prompt = prompt_template.format(
            c_code=orch.c_code,
            header_code=orch.header_code,
            benchmark_context=orch.benchmark_context,
        )
        orch.messages = [
            {"role": "system", "content": core.Instruction_c2hls},
            {"role": "user", "content": prompt},
        ]

        reply = self._call_llm(orch.messages)
        orch._append_history("user", prompt)
        orch._append_history("assistant", reply)
        orch.messages.append({"role": "assistant", "content": reply})

        hls_code = core.extract_cpp_code(reply)
        if not hls_code:
            orch._finalize_candidate_evaluation(
                correctness_status="not_run",
                synthesis_status="not_run",
                failure_class="malformed_output",
            )
            logging.error("[Phase B] No code block in LLM response")
            orch._append_history("system", "[Phase B] FAIL: no code in response")
            return None
        orch._append_history("system", f"[Phase B] Translation mode: {mode}.")
        return hls_code

    def retranslate_with_guidance(self, guidance: str,
                                    *, attempt: int = 1) -> Optional[str]:
        """Phase 8: re-run the translator prompt with a metric-only
        feedback block appended. Used by the baseline-alignment loop
        when our Phase B baseline is significantly worse than the
        reference baseline. ``guidance`` is a metric-only string
        produced by `_render_baseline_alignment_guidance` — it MUST
        NOT include the reference HLS source.
        """
        orch = self.orch
        logging.info(
            "=== [Phase 8] Re-translating with baseline-alignment "
            "guidance (attempt %d) ===", attempt,
        )
        # Baseline alignment always uses the functional prompt in multistep:
        # the goal is a clean starting point, not accidental optimization.
        prompt_template = (
            core.q_translate_c_to_hls_functional
            if core._normalize_phaseb_mode(getattr(orch, "phaseb_mode", ""), multistep=True) == "functional"
            else core.q_translate_c_to_hls
        )
        base_prompt = prompt_template.format(
            c_code=orch.c_code,
            header_code=orch.header_code,
            benchmark_context=orch.benchmark_context,
        )
        prompt = base_prompt + "\n\n=== BASELINE ALIGNMENT FEEDBACK ===\n" + guidance
        orch.messages = [
            {"role": "system", "content": core.Instruction_c2hls},
            {"role": "user", "content": prompt},
        ]
        reply = self._call_llm(orch.messages)
        orch._append_history("user", "[Phase 8 retranslate] " + prompt[:200] + "...")
        orch._append_history("assistant", reply)
        orch.messages.append({"role": "assistant", "content": reply})
        hls_code = core.extract_cpp_code(reply)
        if not hls_code:
            orch._finalize_candidate_evaluation(
                correctness_status="not_run",
                synthesis_status="not_run",
                failure_class="malformed_output",
            )
            logging.warning("[Phase 8] No code in LLM retranslation response")
            return None
        return hls_code
