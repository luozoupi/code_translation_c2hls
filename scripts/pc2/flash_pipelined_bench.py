"""Per-bench pipelined flash driver (codegen / synth steps)."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from flash_pipelined_queue import FlashPipelinedQueue, PipelinedJob

REPO = Path(__file__).resolve().parents[2]


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


class FlashPipelinedBenchSession:
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
    ) -> None:
        self.variant_key = variant_key
        self.bench = bench
        self.bench_dir = bench_dir
        self.cell_dir = cell_dir
        self.model_id = model_id
        self.turns = turns
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
            orch._pipelined_ctx = {"phase_b_attempt": 0, "flash_attempt": 0}
            self._save_state(orch)

        self.orchestrator = orch
        return orch

    def _save_state(self, orch) -> None:
        payload = orch.pipelined_export_state()
        self.state_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    def handle_job(self, job: PipelinedJob, queue: FlashPipelinedQueue) -> None:
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
            logging.exception("pipelined bench %s failed on job %s", self.bench, job.id)
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

        if job.phase == "flash":
            result = orch.pipelined_flash_codegen(repair)
            if not result.get("ok"):
                return [{"kind": "finalize", "phase": "failed", "attempt": job.attempt, "stage": "flash", "error": result.get("error")}]
            attempt = int(job.meta.get("next_attempt") or ctx.get("flash_attempt") or 0)
            return [{
                "kind": "synth",
                "phase": "flash",
                "attempt": attempt,
                "stage": "synth",
            }]

        raise ValueError(f"unknown codegen phase {job.phase}")

    def _run_synth(self, job: PipelinedJob) -> list[dict[str, Any]]:
        orch = self.orchestrator
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
                return [{
                    "kind": "codegen",
                    "phase": "flash",
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

        if job.phase == "flash":
            outcome = orch.pipelined_flash_synth_once(job.attempt)
            if outcome.get("status") == "flash_done":
                ctx = getattr(orch, "_pipelined_ctx", {})
                if outcome.get("success"):
                    ctx["flash_step_result"] = outcome.get("step_result")
                    orch._pipelined_ctx = ctx
                    return [{"kind": "finalize", "phase": "finalize", "attempt": job.attempt, "stage": "done"}]
                return [{
                    "kind": "finalize",
                    "phase": "failed",
                    "attempt": job.attempt,
                    "stage": "flash",
                    "error": outcome.get("error") or "flash step failed",
                }]
            repair = outcome.get("repair") or {}
            next_attempt = int(repair.get("attempt", job.attempt)) + 1
            return [{
                "kind": "codegen",
                "phase": "flash",
                "attempt": next_attempt,
                "stage": "repair",
                "meta": {"repair": repair, "next_attempt": next_attempt},
            }]

        raise ValueError(f"unknown synth phase {job.phase}")

    def _finalize_success(self) -> None:
        orch = self._ensure_orchestrator()
        ctx = getattr(orch, "_pipelined_ctx", {})
        baseline_report = dict(
            orch._flow_phase_b_report or getattr(orch, "_baseline_report", None) or {}
        )
        flash_step = ctx.get("flash_step_result") or {
            "success": True,
            "step_name": "flash",
            "report": orch.synth_report,
            "code": orch.hls_code,
        }
        step_results = [flash_step] if flash_step else []
        results = {
            "benchmark": self.bench,
            "success": bool(flash_step.get("success")),
            "phase": "flash",
            "baseline_report": baseline_report,
            "baseline_comparison": {},
            "baseline_csim": orch.generated_csim,
            "baseline_cosim": orch.generated_cosim,
            "final_report": orch.synth_report,
            "steps": step_results,
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
            "phase": "pipelined",
            "error": error,
            "turn_results": getattr(orch, "turn_results", []) if orch else [],
            "steps": [],
            "pipelined": True,
        }
        result_json = self.cell_dir / f"{self.bench}_multistep_results.json"
        result_json.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


def execute_job(
    *,
    job: PipelinedJob,
    queue: FlashPipelinedQueue,
    bench_dir: Path,
    cell_dir: Path,
    variant_key: str,
    model_id: str,
    turns: int,
) -> None:
    session = FlashPipelinedBenchSession(
        variant_key=variant_key,
        bench=job.bench,
        bench_dir=bench_dir,
        cell_dir=cell_dir,
        model_id=model_id,
        turns=turns,
    )
    session.handle_job(job, queue)
