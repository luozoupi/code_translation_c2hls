"""_AgentBase: phase-specific agent control flow."""
import os
from typing import Optional

from hlsmith import pipeline as core


class _AgentBase:
    """Base class for phase-specific agents.

    Each agent owns its phase's prompts and control flow. State (messages,
    history, hls_code, synth_report, etc.) lives on the parent orchestrator;
    agents read and mutate via `self.orch.<field>`. Helper methods that are
    shared across phases (compile_check_cpp via the orchestrator's
    _synth_and_test, the candidate-with-repairs evaluator) stay on the
    orchestrator so multistep mode and quality repair can keep reusing them.

    The only thing each agent owns independently is its LLM model id. The
    orchestrator's _call_llm_with_model() routes to the right backend.
    """
    AGENT_NAME = "base"
    MODEL_ENV = ""

    def __init__(self, orch: "core.C2HLSOrchestrator"):
        self.orch = orch
        env_model = os.getenv(self.MODEL_ENV) if self.MODEL_ENV else ""
        self.model = env_model or orch.gpt_model

    def _call_llm(self, messages: list, max_tokens: int = None) -> str:
        return self.orch._call_llm_with_model(
            messages, model=self.model, max_tokens=max_tokens,
            agent_name=self.AGENT_NAME,
        )

    def _request_code_revision(self, prompt: str) -> Optional[str]:
        """Append a user prompt to orch.messages, call this agent's LLM,
        record both turns in history, and return the extracted code."""
        self.orch.messages.append({"role": "user", "content": prompt})
        reply = self._call_llm(self.orch.messages)
        self.orch.messages.append({"role": "assistant", "content": reply})
        self.orch._append_history("user", prompt)
        self.orch._append_history("assistant", reply)
        code = core.extract_cpp_code(reply)
        if not code:
            self.orch._finalize_candidate_evaluation(
                correctness_status="not_run",
                synthesis_status="not_run",
                failure_class="malformed_output",
            )
        return code
