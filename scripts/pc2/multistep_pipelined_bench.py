"""Per-bench pipelined multistep driver (phase_b → 5 opt steps)."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from multistep_pipelined_queue import MultistepPipelinedQueue, PipelinedJob

REPO = Path(__file__).resolve().parents[2]

DEFAULT_OPT_STEPS = ["tiling", "pipeline", "unroll", "doublebuffer", "coalescing"]


def _load_benchmark_inputs(bench_dir: str) -> dict:
    from c2hls import _load_benchmark_inputs as load_inputs

    return load_inputs(bench_dir)


def _validate_reference(inputs: dict) -> dict:
    from c2hls import validate_gold_reference

    return validate_gold_reference(inputs)


def _build_run_attribution(orchestrator, meta: dict) -> dict:
    from c2hls import _build_run_attribution

    return _build_run_attribution(orchestrator, meta)


def _build_coverage(meta, reference_validation, generated_csim, generated_cosim) -> dict:
    from c2hls import _build_coverage

    return _build_coverage(meta, reference_validation, generated_csim, generated_cosim)


def _sanitize_saved_result_record(results, reference_validation) -> dict:
    from c2hls import sanitize_saved_result_record

    return sanitize_saved_result_record(results, reference_validation)


class MultistepPipelinedBenchSession:
    """Resume-able bench session with artifacts under ``cell/pipelined/``."""

    STATE_NAME = "orchestrator_state.json"
    META_NAME = "session_meta.json"

    def __init__(
        self,
        *,
        variant_key: str,
        bench: str,
        bench_dir: Path,
        cell_dir: Path,
        model_id: str,
        turns: int,
        opt_steps: list[str] | None = None,
    ) -> None:
        self.variant_key = variant_key
        self.bench = bench
        self.bench_dir = bench_dir
        self.cell_dir = cell_dir
        self.model_id = model_id
        self.turns = turns
        self.opt_steps = list(opt_steps or DEFAULT_OPT_STEPS)
        self.pipelined_dir = cell_dir / "pipelined"
        self.pipelined_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.pipelined_dir / self.STATE_NAME
        self.meta_path = self.pipelined_dir / self.META_NAME
        self.inputs = _load_benchmark_inputs(str(bench_dir))
        self.reference_validation = _validate_reference(self.inputs)
        self.orchestrator = None

    def _ensure_orchestrator(self):
        if self.orchestrator is not None:
            return self.orchestrator
        from c2hls import C2HLSOrchestrator

        orch = C2HLSOrchestrator(
            gpt_model=self.model_id,
            turns_limitation=self.turns,
        )
        meta = self.inputs["meta"]
        orch.testbench_code = self.inputs.get("testbench_code", "")
        orch.configure_benchmark(
            extra_files=self.inputs.get("extra_files", []),
            translated_hls_top=meta.get("translated_hls_top", "workload"),
            reference_hls_top=meta.get("hls_top", "workload"),
            part=meta.get("part", os.getenv("C2HLS_PART", "xcu280-fsvh2892-2L-e")),
            clock_ns=meta.get("clock_ns", 4.0),
            supports_cosim=bool(meta.get("supports_cosim")),
            cosim_depths=meta.get("cosim_depths", {}),
            benchmark_name=self.bench,
            benchmark_context=self.inputs.get("benchmark_context", ""),
        )
        force_skill_prompts = os.getenv("C2HLS_FORCE_SKILL_PROMPTS", "").strip().lower() in {
            "1", "true", "yes", "on",
        }
        if force_skill_prompts and orch.skill_library is None:
            from skill_library import make_default_library

            persist_skills = bool(int(os.getenv("C2HLS_SKILL_LIBRARY_PERSIST", "1") or "1"))
            orch.skill_library = make_default_library(persist=persist_skills)

        if self.state_path.is_file():
            state = json.loads(self.state_path.read_text(encoding="utf-8"))
            orch.pipelined_import_state(state)
        else:
            if not self.reference_validation.get("benchmark_ready"):
                raise RuntimeError(
                    self.reference_validation.get("invalid_reason") or "reference invalid"
                )
            if not orch.run_phase_a(
                self.inputs["c_code"],
                self.inputs["header_code"],
                self.inputs["header_name"] or "kernel.h",
            ):
                raise RuntimeError("Phase A failed")
            orch._pipelined_ctx = {
                "phase_b_attempt": 0,
                "opt_steps": self.opt_steps,
                "current_step_index": -1,
                "step_results": [],
            }
            self._save_state(orch)

        self.orchestrator = orch
        return orch

    def _save_state(self, orch) -> None:
        payload = orch.pipelined_export_state()
        self.state_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    def _next_step_after(self, phase: str) -> str | None:
        if phase == "phase_b":
            return self.opt_steps[0] if self.opt_steps else None
        if phase in self.opt_steps:
            idx = self.opt_steps.index(phase)
            if idx + 1 < len(self.opt_steps):
                return self.opt_steps[idx + 1]
        return None

    def handle_job(self, job: PipelinedJob, queue: MultistepPipelinedQueue) -> None:
        orch = self._ensure_orchestrator()
        try:
            if job.kind == "codegen":
                followups = self._run_codegen(job)
            else:
                followups = self._run_synth(job)
            self._save_state(orch)
            for spec in followups:
                phase = spec.get("phase")
                if phase == "finalize":
                    self._finalize_success()
                    queue.set_bench_status(self.variant_key, self.bench, "done")
                    continue
                if phase == "failed":
                    self._finalize_failure(spec.get("error", "failed"))
                    queue.set_bench_status(self.variant_key, self.bench, "failed")
                    continue
                queue.enqueue(
                    variant=self.variant_key,
                    bench=self.bench,
                    kind=spec["kind"],
                    phase=spec["phase"],
                    attempt=int(spec.get("attempt") or 0),
                    stage=spec.get("stage") or "",
                    meta=dict(spec.get("meta") or {}),
                )
        except Exception as exc:
            logging.exception("pipelined multistep bench %s failed on job %s", self.bench, job.id)
            self._finalize_failure(str(exc))
            queue.set_bench_status(self.variant_key, self.bench, "failed")
            raise

    def _run_codegen(self, job: PipelinedJob) -> list[dict[str, Any]]:
        orch = self.orchestrator
        ctx = getattr(orch, "_pipelined_ctx", {})
        repair = job.meta.get("repair")

        if job.phase == "phase_b":
            if job.stage == "translate" and not repair:
                result = orch.pipelined_phase_b_translate()
                if not result.get("ok"):
                    return [{"kind": "finalize", "phase": "failed", "attempt": 0, "stage": "phase_b", "error": result.get("error")}]
                attempt = int(ctx.get("phase_b_attempt") or 0)
                return [{
                    "kind": "synth",
                    "phase": "phase_b",
                    "attempt": attempt,
                    "stage": "synth",
                }]
            result = orch.pipelined_phase_b_repair_codegen(repair or job.meta.get("repair") or {})
            if not result.get("ok"):
                return [{"kind": "finalize", "phase": "failed", "attempt": job.attempt, "stage": "phase_b", "error": result.get("error")}]
            attempt = int(getattr(orch, "_pipelined_ctx", {}).get("phase_b_attempt") or job.attempt + 1)
            return [{
                "kind": "synth",
                "phase": "phase_b",
                "attempt": attempt,
                "stage": "synth",
            }]

        if job.phase in self.opt_steps:
            result = orch.pipelined_multistep_step_codegen(job.phase, repair)
            if not result.get("ok"):
                return [{"kind": "finalize", "phase": "failed", "attempt": job.attempt, "stage": job.phase, "error": result.get("error")}]
            attempt_key = f"{job.phase}_attempt"
            attempt = int(job.meta.get("next_attempt") or ctx.get(attempt_key) or 0)
            return [{
                "kind": "synth",
                "phase": job.phase,
                "attempt": attempt,
                "stage": "synth",
            }]

        raise ValueError(f"unknown codegen phase {job.phase}")

    def _run_synth(self, job: PipelinedJob) -> list[dict[str, Any]]:
        orch = self.orchestrator
        ctx = getattr(orch, "_pipelined_ctx", {})

        if job.phase == "phase_b":
            outcome = orch.pipelined_phase_b_synth_once(job.attempt)
            if outcome.get("status") == "phase_b_done":
                if not outcome.get("success"):
                    return [{
                        "kind": "finalize",
                        "phase": "failed",
                        "attempt": job.attempt,
                        "stage": "phase_b",
                        "error": outcome.get("error") or "Phase B failed",
                    }]
                next_step = self._next_step_after("phase_b")
                if not next_step:
                    return [{"kind": "finalize", "phase": "finalize", "attempt": job.attempt, "stage": "done"}]
                return [{
                    "kind": "codegen",
                    "phase": next_step,
                    "attempt": 0,
                    "stage": "optimize",
                }]
            repair = outcome.get("repair") or {}
            return [{
                "kind": "codegen",
                "phase": "phase_b",
                "attempt": job.attempt,
                "stage": "repair",
                "meta": {"repair": repair},
            }]

        if job.phase in self.opt_steps:
            outcome = orch.pipelined_multistep_step_synth_once(job.phase, job.attempt)
            done_status = orch._pipelined_step_done_status(job.phase)
            if outcome.get("status") == done_status:
                result_key = orch._pipelined_step_result_key(job.phase)
                step_result = ctx.get(result_key) or outcome.get("step_result")
                if outcome.get("success") and step_result:
                    step_results = list(ctx.get("step_results") or [])
                    step_results.append(step_result)
                    ctx["step_results"] = step_results
                    orch._pipelined_ctx = ctx
                    next_step = self._next_step_after(job.phase)
                    if next_step:
                        return [{
                            "kind": "codegen",
                            "phase": next_step,
                            "attempt": 0,
                            "stage": "optimize",
                        }]
                    return [{"kind": "finalize", "phase": "finalize", "attempt": job.attempt, "stage": "done"}]
                return [{
                    "kind": "finalize",
                    "phase": "failed",
                    "attempt": job.attempt,
                    "stage": job.phase,
                    "error": outcome.get("error") or f"{job.phase} step failed",
                }]
            repair = outcome.get("repair") or {}
            next_attempt = int(repair.get("attempt", job.attempt)) + 1
            ctx[f"{job.phase}_attempt"] = next_attempt
            orch._pipelined_ctx = ctx
            return [{
                "kind": "codegen",
                "phase": job.phase,
                "attempt": next_attempt,
                "stage": "repair",
                "meta": {"repair": repair, "next_attempt": next_attempt},
            }]

        raise ValueError(f"unknown synth phase {job.phase}")

    @staticmethod
    def _latency_cycles(report: dict | None) -> float | None:
        if not isinstance(report, dict):
            return None
        lat = report.get("latency_cycles")
        if lat is None:
            lat = report.get("latency_cycles_worst")
        try:
            return float(lat) if lat is not None else None
        except (TypeError, ValueError):
            return None

    def _promote_pipelined_best_so_far(
        self,
        orch,
        baseline_report: dict,
        step_results: list[dict],
    ) -> dict | None:
        """Pick lowest-latency snapshot among phase_b and successful opt steps."""
        candidates: list[tuple[str, dict, str | None]] = []
        pb_lat = self._latency_cycles(baseline_report)
        if pb_lat is not None:
            code = getattr(orch, "_flow_phase_b_code", None) or orch.hls_code
            candidates.append(("phase_b", baseline_report, code))
        for step in step_results:
            if not step.get("success"):
                continue
            rep = step.get("report") or {}
            lat = self._latency_cycles(rep)
            if lat is None:
                continue
            candidates.append((step.get("step_name") or "?", rep, step.get("code")))
        if not candidates:
            return None
        best_name, best_rep, best_code = min(
            candidates, key=lambda item: self._latency_cycles(item[1]) or float("inf")
        )
        cur_lat = self._latency_cycles(orch.synth_report)
        best_lat = self._latency_cycles(best_rep)
        if best_lat is None or (cur_lat is not None and best_lat >= cur_lat):
            return None
        if best_code:
            orch.hls_code = best_code
        orch.synth_report = dict(best_rep)
        return {
            "promoted": True,
            "from_step_name": best_name,
            "from_latency_cycles": best_lat,
            "previous_latency_cycles": cur_lat,
        }

    def _finalize_success(self) -> None:
        orch = self._ensure_orchestrator()
        ctx = getattr(orch, "_pipelined_ctx", {})
        baseline_report = dict(
            orch._flow_phase_b_report or getattr(orch, "_baseline_report", None) or {}
        )
        step_results = list(ctx.get("step_results") or [])
        promotion = self._promote_pipelined_best_so_far(orch, baseline_report, step_results)
        results = {
            "benchmark": self.bench,
            "success": bool(step_results) or bool(orch.synth_report),
            "phase": "multistep",
            "baseline_report": baseline_report,
            "baseline_comparison": {},
            "baseline_csim": orch.generated_csim,
            "baseline_cosim": orch.generated_cosim,
            "final_report": orch.synth_report,
            "steps": step_results,
            "best_so_far_promotion": promotion,
            "generated_step_history": [
                {
                    "step_name": "baseline",
                    "success": True,
                    "report": baseline_report,
                    "comparison": {},
                    "csim": orch.generated_csim,
                    "cosim": orch.generated_cosim,
                },
                *step_results,
            ],
            "hls_code": orch.hls_code,
            "phase_b_mode": orch.phaseb_mode,
            "preflight_patches": orch.preflight_patches,
            "llm_usage": orch._llm_usage_summary(),
            "pipelined": True,
        }
        results["run"] = _build_run_attribution(orch, self.inputs["meta"])
        results["reference_validation"] = self.reference_validation
        results["ground_truth_status"] = "valid"
        results["baseline_status"] = self.reference_validation.get("synthesis", {}).get("status", "failed")
        results["invalid_reference_reason"] = ""
        results["coverage"] = _build_coverage(
            self.inputs["meta"],
            self.reference_validation,
            results.get("baseline_csim"),
            results.get("baseline_cosim"),
        )
        results = _sanitize_saved_result_record(results, self.reference_validation)
        orch.save_multistep_results(str(self.cell_dir), self.bench, results)
        result_json = self.cell_dir / f"{self.bench}_multistep_results.json"
        result_json.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    def _finalize_failure(self, error: str) -> None:
        orch = self.orchestrator
        results = {
            "benchmark": self.bench,
            "success": False,
            "phase": "pipelined_multistep",
            "error": error,
            "turn_results": getattr(orch, "turn_results", []) if orch else [],
            "steps": list((getattr(orch, "_pipelined_ctx", {}) or {}).get("step_results") or []),
            "pipelined": True,
        }
        result_json = self.cell_dir / f"{self.bench}_multistep_results.json"
        result_json.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


def execute_job(
    *,
    job: PipelinedJob,
    queue: MultistepPipelinedQueue,
    bench_dir: Path,
    cell_dir: Path,
    variant_key: str,
    model_id: str,
    turns: int,
) -> None:
    session = MultistepPipelinedBenchSession(
        variant_key=variant_key,
        bench=job.bench,
        bench_dir=bench_dir,
        cell_dir=cell_dir,
        model_id=model_id,
        turns=turns,
    )
    session.handle_job(job, queue)
