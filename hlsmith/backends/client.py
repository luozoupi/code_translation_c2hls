"""Provider clients, request routing, decoding and LLM usage accounting.

The controller owns candidate/evaluation state. Access to shared provider
helpers remains explicit through ``core`` so existing patches and versioned
runtime settings continue to apply to inherited methods.
"""
import hashlib
import json
import os
from datetime import datetime, timezone
from typing import Optional

from hlsmith import pipeline as core


class LLMBackendMixin:
    """Backend behavior shared by the C2HLS orchestrator's agent roles."""

    def _initialize_llm_backend(self, gpt_model: str):
        """Initialize provider clients and their cache before controller state."""
        self.use_anthropic = gpt_model.lower().startswith("claude")
        self.use_hosted_openai = core._is_hosted_openai_model(gpt_model)
        self.use_deepseek = core._is_deepseek_model(gpt_model)
        self.use_xai = core._is_xai_model(gpt_model)
        if self.use_anthropic:
            assert core.HAS_ANTHROPIC, "anthropic package required for Claude models: pip install anthropic"
            api_key = core._load_anthropic_api_key()
            assert api_key, f"Missing Anthropic API key. Set ANTHROPIC_API_KEY or populate {core.CLAUDE_API_KEY_FILE}."
            self.anthropic_client = core.anthropic.Anthropic(
                api_key=api_key,
                timeout=core._llm_timeout_seconds(),
                max_retries=0 if os.getenv("C2HLS_API_BUDGET_LEDGER") else 2,
            )
        else:
            if self.use_hosted_openai:
                self.key = core._load_openai_api_key()
                assert self.key, f"Missing OpenAI API key. Set OPENAI_API_KEY or populate {core.OPENAI_API_KEY_FILE}."
                self.base_url = core.OPENAI_HOSTED_BASE_URL
            elif self.use_deepseek:
                self.key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY", "EMPTY")
                self.base_url = os.getenv("DEEPSEEK_BASE_URL") or os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com")
            else:
                self.key = os.getenv("OPENAI_API_KEY", "EMPTY")
                self.base_url = os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1")
            self.client = core.OpenAI(
                base_url=self.base_url,
                api_key=self.key,
                timeout=core._llm_timeout_seconds(),
                max_retries=0 if os.getenv("C2HLS_API_BUDGET_LEDGER") else 2,
            )

        # Per-agent LLM cache. Populated lazily when an agent's model differs
        # from the orchestrator's default. Keys: model id; values: tuple
        # (kind, client) where kind is "anthropic" or "openai".
        self._extra_clients: dict = {}

    def _call_llm(self, messages: list, max_tokens: int = None) -> str:
        """Default-model LLM call. Kept as the public interface so existing
        callers (multistep, run_optimization_step, anything outside the
        agent classes) continue to work unchanged."""
        return self._call_llm_with_model(messages, model=self.gpt_model,
                                         max_tokens=max_tokens,
                                         agent_name="orchestrator")

    @staticmethod
    def _usage_value(obj, name: str, default: int = 0) -> int:
        if obj is None:
            return default
        if isinstance(obj, dict):
            value = obj.get(name, default)
        else:
            value = getattr(obj, name, default)
        try:
            return int(value or 0)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _configured_decoding() -> dict:
        def parse_float(name: str) -> Optional[float]:
            raw = os.getenv(name, "").strip()
            if not raw:
                return None
            value = float(raw)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {raw!r}")
            return value

        raw_seed = os.getenv(core.LLM_SEED_ENV, "").strip()
        return {
            "temperature": parse_float(core.LLM_TEMPERATURE_ENV),
            "top_p": parse_float(core.LLM_TOP_P_ENV),
            "seed": int(raw_seed) if raw_seed else None,
        }

    def _record_llm_usage(self, *, provider: str, model: str, agent_name: str,
                          usage, messages: list, max_tokens: int,
                          decoding: Optional[dict] = None,
                          candidate_evaluation_index: Optional[int] = None) -> None:
        """Record provider-reported token usage for bench-level accounting."""
        if candidate_evaluation_index is None:
            candidate_evaluation_index = len(self.llm_usage_events)
        prompt_payload = json.dumps(
            messages or [], sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        ).encode("utf-8")
        common = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "provider": provider,
            "model": model,
            "model_revision": os.getenv("C2HLS_MODEL_REVISION") or None,
            "response_model": getattr(self, "_last_response_metadata", {}).get("model"),
            "request_id": getattr(self, "_last_response_metadata", {}).get("request_id"),
            "system_fingerprint": getattr(self, "_last_response_metadata", {}).get("system_fingerprint"),
            "prompt_profile": getattr(self, "prompt_profile", dict(core.load_profile().metadata)),
            "agent": agent_name or "unknown",
            "message_count": len(messages or []),
            "max_tokens": max_tokens,
            "prompt_sha256": hashlib.sha256(prompt_payload).hexdigest(),
            "decoding": dict(decoding or {}),
            "candidate_evaluation_index": candidate_evaluation_index,
        }
        if usage is None:
            event = {
                **common,
                "usage_available": False,
            }
            self.llm_usage_events.append(event)
            self._start_candidate_evaluation_event(
                candidate_evaluation_index,
                agent_name=agent_name,
                model=model,
            )
            return

        if provider == "anthropic":
            input_tokens = self._usage_value(usage, "input_tokens")
            output_tokens = self._usage_value(usage, "output_tokens")
            cache_creation = self._usage_value(usage, "cache_creation_input_tokens")
            cache_read = self._usage_value(usage, "cache_read_input_tokens")
            normalized = {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
                "cache_creation_input_tokens": cache_creation,
                "cache_read_input_tokens": cache_read,
            }
        else:
            prompt_tokens = self._usage_value(usage, "prompt_tokens")
            completion_tokens = self._usage_value(usage, "completion_tokens")
            total_tokens = self._usage_value(
                usage, "total_tokens", prompt_tokens + completion_tokens
            )
            prompt_details = getattr(usage, "prompt_tokens_details", None)
            completion_details = getattr(usage, "completion_tokens_details", None)
            normalized = {
                "input_tokens": prompt_tokens,
                "output_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "cached_tokens": self._usage_value(prompt_details, "cached_tokens", self._usage_value(usage, "prompt_cache_hit_tokens")),
                "reasoning_tokens": self._usage_value(completion_details, "reasoning_tokens"),
            }

        event = {
            **common,
            "usage_available": True,
            **normalized,
        }
        self.llm_usage_events.append(event)
        self._start_candidate_evaluation_event(
            candidate_evaluation_index,
            agent_name=agent_name,
            model=model,
        )

    def _llm_usage_summary(self) -> dict:
        fields = [
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "cache_creation_input_tokens",
            "cache_read_input_tokens",
            "cached_tokens",
            "reasoning_tokens",
        ]
        totals = {field: 0 for field in fields}
        by_agent: dict[str, dict] = {}
        by_model: dict[str, dict] = {}

        def add(bucket: dict, key: str, event: dict) -> None:
            item = bucket.setdefault(key or "unknown", {"calls": 0, **{field: 0 for field in fields}})
            item["calls"] += 1
            for field in fields:
                item[field] += int(event.get(field) or 0)

        for event in self.llm_usage_events:
            for field in fields:
                totals[field] += int(event.get(field) or 0)
            add(by_agent, event.get("agent"), event)
            add(by_model, event.get("model"), event)

        return {
            "schema_version": "1.0",
            "calls": len(self.llm_usage_events),
            "candidate_requests": self.llm_candidate_request_count,
            "candidate_budget": self.llm_candidate_budget,
            **totals,
            "by_agent": by_agent,
            "by_model": by_model,
            "usage_missing_calls": sum(
                1 for event in self.llm_usage_events
                if not event.get("usage_available")
            ),
            "events": self.llm_usage_events,
        }

    def _client_for_model(self, model: str):
        """Get or create the right backend client for a given model id.

        Returns a tuple (kind, client) where kind identifies the provider.
        The orchestrator's default-model client is reused; non-
        default models populate self._extra_clients on first use.
        """
        if model == self.gpt_model:
            if self.use_anthropic:
                return ("anthropic", self.anthropic_client)
            if self.use_deepseek:
                return ("deepseek", self.client)
            if self.use_xai:
                return ("xai", self.client)
            return ("openai", self.client)
        cached = self._extra_clients.get(model)
        if cached is not None:
            return cached

        is_claude = model.lower().startswith("claude")
        if is_claude:
            assert core.HAS_ANTHROPIC, (
                "anthropic package required for Claude models: pip install anthropic"
            )
            api_key = core._load_anthropic_api_key()
            assert api_key, (
                f"Missing Anthropic API key for agent model {model!r}; "
                f"set ANTHROPIC_API_KEY or populate {core.CLAUDE_API_KEY_FILE}."
            )
            client = core.anthropic.Anthropic(
                api_key=api_key,
                timeout=core._llm_timeout_seconds(),
                max_retries=0 if os.getenv("C2HLS_API_BUDGET_LEDGER") else 2,
            )
            entry = ("anthropic", client)
        else:
            if core._is_hosted_openai_model(model):
                api_key = core._load_openai_api_key()
                assert api_key, (
                    f"Missing OpenAI API key for agent model {model!r}."
                )
                base_url = core.OPENAI_HOSTED_BASE_URL
            elif core._is_deepseek_model(model):
                api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY", "EMPTY")
                base_url = os.getenv("DEEPSEEK_BASE_URL") or os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com")
            else:
                api_key = os.getenv("OPENAI_API_KEY", "EMPTY")
                base_url = os.getenv("OPENAI_BASE_URL",
                                     "http://127.0.0.1:8000/v1")
            provider = (
                "deepseek"
                if core._is_deepseek_model(model)
                else "xai"
                if core._is_xai_model(model)
                else "openai"
            )
            entry = (provider, core.OpenAI(
                base_url=base_url,
                api_key=api_key,
                timeout=core._llm_timeout_seconds(),
                max_retries=0 if os.getenv("C2HLS_API_BUDGET_LEDGER") else 2,
            ))
        self._extra_clients[model] = entry
        return entry

    def _call_llm_with_model(self, messages: list, model: str = None,
                             max_tokens: int = None,
                             agent_name: str = "orchestrator") -> str:
        """Route an LLM call to the requested model's backend. Used by
        agents to support per-agent model overrides without forcing every
        caller to manage clients.
        """
        selected_profile = getattr(self, "prompt_profile", None)
        active_profile = getattr(core, "_ACTIVE_PROMPT_PROFILE", selected_profile)
        if selected_profile is not None and active_profile != selected_profile:
            raise ValueError("Prompt profiles are process-wide; use a separate process for each profile")
        if max_tokens is None:
            max_tokens = self.max_completion_tokens
        if not model:
            model = self.gpt_model

        if (
            self.llm_candidate_budget is not None
            and self.llm_candidate_request_count >= self.llm_candidate_budget
        ):
            raise RuntimeError(
                "llm_candidate_budget_exhausted: used "
                f"{self.llm_candidate_request_count}/{self.llm_candidate_budget}"
            )
        self.llm_candidate_request_count += 1
        if not hasattr(self, "llm_controller_transcript"):
            self.llm_controller_transcript = []
        call_index = self.llm_candidate_request_count - 1
        for message_index, message in enumerate(messages or []):
            self.llm_controller_transcript.append({
                "call_index": call_index,
                "message_index": message_index,
                "agent": agent_name or "orchestrator",
                "model": model,
                "role": message.get("role", "unknown"),
                "content": str(message.get("content") or ""),
            })

        kind, client = self._client_for_model(model)
        configured_decoding = self._configured_decoding()
        if kind == "anthropic":
            system_text = ""
            conv_messages = []
            for message in messages:
                if message["role"] == "system":
                    system_text += message["content"] + "\n"
                else:
                    conv_messages.append({"role": message["role"],
                                          "content": message["content"]})
            anthropic_kwargs = {
                "model": model,
                "max_tokens": max_tokens,
                "system": system_text.strip() if system_text else "",
                "messages": conv_messages,
            }
            omitted_decoding = None
            if configured_decoding["temperature"] is not None:
                anthropic_kwargs["temperature"] = configured_decoding[
                    "temperature"
                ]
                if configured_decoding["top_p"] is not None:
                    # Sonnet 4.6 rejects requests containing both controls.
                    # Prefer temperature and record the provider-constrained
                    # omission rather than silently claiming top_p was sent.
                    omitted_decoding = "top_p"
            elif configured_decoding["top_p"] is not None:
                anthropic_kwargs["top_p"] = configured_decoding["top_p"]
            effective_decoding = {
                "temperature": anthropic_kwargs.get("temperature"),
                "top_p": anthropic_kwargs.get("top_p"),
                "seed": None,
                "seed_supported": False,
                "requested_temperature": configured_decoding["temperature"],
                "requested_top_p": configured_decoding["top_p"],
                "mutually_exclusive_omission": omitted_decoding,
            }
            response = core.send_budgeted_request(self, client.messages.create,
                model=model, messages=messages, max_tokens=max_tokens, kwargs=anthropic_kwargs)
            self._record_llm_usage(
                provider="anthropic",
                model=model,
                agent_name=agent_name,
                usage=getattr(response, "usage", None),
                messages=messages,
                max_tokens=max_tokens,
                decoding=effective_decoding,
                candidate_evaluation_index=call_index,
            )
            return response.content[0].text

        kwargs = {"model": model, "messages": messages}
        if core._is_hosted_openai_model(model):
            kwargs["max_completion_tokens"] = max_tokens
        else:
            kwargs["max_tokens"] = max_tokens
        effective_decoding = {
            **configured_decoding,
            "seed_supported": True,
        }
        if kind == "deepseek":
            # DeepSeek's OpenAI-compatible API does not expose seed control
            # and recommends setting temperature or top_p, not both.
            omitted_decoding = None
            if configured_decoding["temperature"] is not None:
                kwargs["temperature"] = configured_decoding["temperature"]
                if configured_decoding["top_p"] is not None:
                    omitted_decoding = "top_p"
            elif configured_decoding["top_p"] is not None:
                kwargs["top_p"] = configured_decoding["top_p"]
            thinking = os.getenv(core.DEEPSEEK_THINKING_ENV, "enabled").strip().lower()
            if thinking not in {"enabled", "disabled"}:
                raise ValueError(
                    f"{core.DEEPSEEK_THINKING_ENV} must be enabled or disabled"
                )
            reasoning_effort = os.getenv(
                core.DEEPSEEK_REASONING_EFFORT_ENV, "high"
            ).strip().lower()
            if reasoning_effort not in {"low", "high", "max"}:
                raise ValueError(
                    f"{core.DEEPSEEK_REASONING_EFFORT_ENV} must be low, high, or max"
                )
            kwargs["extra_body"] = {
                "thinking": {"type": thinking},
                "reasoning_effort": reasoning_effort,
            }
            effective_decoding = {
                "temperature": kwargs.get("temperature"),
                "top_p": kwargs.get("top_p"),
                "seed": None,
                "seed_supported": False,
                "requested_temperature": configured_decoding["temperature"],
                "requested_top_p": configured_decoding["top_p"],
                "requested_seed": configured_decoding["seed"],
                "mutually_exclusive_omission": omitted_decoding,
                "thinking": thinking,
                "reasoning_effort": reasoning_effort,
            }
        elif kind == "xai":
            for key in ("temperature", "top_p", "seed"):
                if configured_decoding[key] is not None:
                    kwargs[key] = configured_decoding[key]
            reasoning_effort = os.getenv(
                core.XAI_REASONING_EFFORT_ENV, "low"
            ).strip().lower()
            if reasoning_effort not in {"low", "medium", "high"}:
                raise ValueError(
                    f"{core.XAI_REASONING_EFFORT_ENV} must be low, medium, or high"
                )
            kwargs["reasoning_effort"] = reasoning_effort
            effective_decoding = {
                **configured_decoding,
                "seed_supported": True,
                "reasoning": "required",
                "reasoning_effort": reasoning_effort,
            }
        elif kind == "openai" and model.lower().startswith("gpt-5.6"):
            effort = os.getenv("C2HLS_OPENAI_REASONING_EFFORT", "none").strip().lower()
            if effort not in {"none", "low", "medium", "high", "xhigh", "max"}:
                raise ValueError("invalid C2HLS_OPENAI_REASONING_EFFORT")
            kwargs["reasoning_effort"] = effort
            if effort == "none":
                for key in ("temperature", "top_p"):
                    if configured_decoding[key] is not None:
                        kwargs[key] = configured_decoding[key]
            effective_decoding = {
                "temperature": kwargs.get("temperature"), "top_p": kwargs.get("top_p"),
                "seed": None, "seed_supported": False,
                "seed_control": "not_requested; no reproducibility guarantee assumed",
                "requested_seed": configured_decoding["seed"], "reasoning_effort": effort,
            }
        else:
            for key in ("temperature", "top_p", "seed"):
                if configured_decoding[key] is not None:
                    kwargs[key] = configured_decoding[key]
        if "qwen" in model.lower():
            kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
        response = core.send_budgeted_request(self, client.chat.completions.create,
            model=model, messages=messages, max_tokens=max_tokens, kwargs=kwargs)
        self._record_llm_usage(
            provider=kind,
            model=model,
            agent_name=agent_name,
            usage=getattr(response, "usage", None),
            messages=messages,
            max_tokens=max_tokens,
            decoding=effective_decoding,
            candidate_evaluation_index=call_index,
        )
        return response.choices[0].message.content
