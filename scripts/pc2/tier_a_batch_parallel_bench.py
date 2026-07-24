"""Tier A flash batch_parallel bench session (parallel gold gate, synth+csim, no RTL cosim)."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from batch_parallel_bench import BatchParallelBenchSession
from batch_parallel_env import configure_synth_env
from batch_parallel_queue import BatchParallelJob, BatchParallelQueue
from flash_pipelined_bench import _load_benchmark_inputs


def _sanitize_saved_result_record(results: dict, reference_validation: dict | None) -> dict:
    from c2hls import sanitize_saved_result_record

    return sanitize_saved_result_record(results, reference_validation)


class TierABatchParallelBenchSession(BatchParallelBenchSession):
    REFERENCE_VALIDATION_NAME = "reference_validation.json"

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
        self.reference_validation = self._load_reference_validation()
        self.orchestrator = None

    @staticmethod
    def _validate_gold_reference(inputs: dict) -> dict:
        from c2hls import validate_gold_reference

        return validate_gold_reference(inputs)

    def _reference_validation_path(self) -> Path:
        return self.cell_dir / self.REFERENCE_VALIDATION_NAME

    def _load_reference_validation(self) -> dict[str, Any]:
        path = self._reference_validation_path()
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
        return {
            "benchmark_ready": False,
            "invalid_reason": "reference gate not run",
        }

    def _save_reference_validation(self, payload: dict[str, Any]) -> None:
        self.cell_dir.mkdir(parents=True, exist_ok=True)
        self._reference_validation_path().write_text(
            json.dumps(payload, indent=2) + "\n",
            encoding="utf-8",
        )
        self.reference_validation = payload

    def _apply_bench_synth_timeout(self) -> None:
        from tier_a_flash_lib import apply_bench_synth_timeout_from_meta

        apply_bench_synth_timeout_from_meta(self.inputs.get("meta") or {})

    def handle_job(self, job: BatchParallelJob, queue: BatchParallelQueue) -> None:
        self._apply_bench_synth_timeout()
        if job.kind == "synth" and job.phase == "reference":
            configure_synth_env(cosim_timeout_s=int(os.getenv("C2HLS_COSIM_TIMEOUT", "43200")))
            followups = self._run_reference_synth(job)
            self._apply_followups(followups, queue)
            return
        super().handle_job(job, queue)

    def _ensure_orchestrator(self):
        self.reference_validation = self._load_reference_validation()
        if not self.reference_validation.get("benchmark_ready"):
            reason = self.reference_validation.get("invalid_reason") or "reference invalid"
            raise RuntimeError(reason)
        return super()._ensure_orchestrator()

    def _run_reference_synth(self, job: BatchParallelJob) -> list[dict[str, Any]]:
        logging.info("[Reference] Gold gate for %s", self.bench)
        ref = self._validate_gold_reference(self.inputs)
        self._save_reference_validation(ref)
        if not ref.get("benchmark_ready"):
            error = ref.get("invalid_reason") or "Gold HLS reference invalid"
            return [{
                "phase": "failed",
                "kind": "finalize",
                "attempt": job.attempt,
                "stage": "reference",
                "error": error,
            }]
        return [{
            "kind": "codegen",
            "phase": "phase_b",
            "attempt": 0,
            "stage": "translate",
        }]

    def _synth_csim_only(self, code: str, *, log_prefix: str, temp_suffix: str) -> dict:
        from c2hls import _run_synth_csim_cosim, join_temp_tag

        orch = self._ensure_orchestrator()
        tag = join_temp_tag(self.bench, temp_suffix)
        return _run_synth_csim_cosim(
            code,
            header_code=orch.header_code,
            header_name=orch.header_name,
            top_function=orch.translated_hls_top,
            part=orch.part,
            clock_ns=orch.clock_ns,
            extra_files=orch.extra_files,
            testbench_code=orch.testbench_code,
            run_csim_check=bool(orch.testbench_code),
            run_cosim_check=False,
            cosim_depths=orch.cosim_depths,
            log_prefix=log_prefix,
            temp_tag=tag,
        )

    @staticmethod
    def _compile_check_cpp(hls_code, header_code, header_name, *, extra_files):
        from c2hls import compile_check_cpp

        return compile_check_cpp(
            hls_code, header_code, header_name, extra_files=extra_files,
        )

    @staticmethod
    def _csim_link_error(error: str) -> bool:
        low = (error or "").lower()
        return (
            "ld returned 1 exit status" in low
            or ("csim.exe" in low and "error 1" in low)
            or "undefined reference" in low
        )

    def _csim_failed(self, outcome: dict) -> tuple[bool, str]:
        csim = outcome.get("csim") or {}
        if not isinstance(csim, dict) or not csim.get("ran"):
            return False, ""
        if csim.get("passed"):
            return False, ""
        error = (
            (csim.get("error") or "").strip()
            + "\n"
            + (csim.get("log_excerpt") or "").strip()
        ).strip() or "csim failed"
        return True, error

    def _run_synth(self, job: BatchParallelJob) -> list[dict[str, Any]]:
        if job.phase == "reference":
            return self._run_reference_synth(job)
        if job.phase == "phase_b":
            return self._run_synth_phase_b(job)
        if job.phase == "flash":
            return self._run_synth_flash(job)
        raise ValueError(f"unknown synth phase {job.phase}")

    @staticmethod
    def _classify_synth_error(err: str) -> str:
        from c2hls import _classify_synth_error

        return _classify_synth_error(err)

    @staticmethod
    def _record_flow_enabled() -> bool:
        return os.getenv("C2HLS_RECORD_FLOW", "1") == "1"

    def _run_synth_phase_b(self, job: BatchParallelJob) -> list[dict[str, Any]]:
        orch = self._ensure_orchestrator()
        ctx = getattr(orch, "_pipelined_ctx", {})
        if ctx.get("phase_b_best_state") is None:
            ctx["phase_b_best_state"] = None
        if ctx.get("phase_b_error_history") is None:
            ctx["phase_b_error_history"] = []
        best_state = ctx.get("phase_b_best_state")
        error_class_history = ctx["phase_b_error_history"]
        threshold = orch.synthesis.revert_threshold
        attempt = job.attempt

        logging.info("[Phase B] Synthesis+csim attempt %d (tier_a batch_parallel)", attempt)
        orch.hls_code = orch._preflight_generated_hls_code(
            orch.hls_code, f"Phase B attempt {attempt}",
        )

        ok, err = self._compile_check_cpp(
            orch.hls_code, orch.header_code, orch.header_name,
            extra_files=orch.extra_files,
        )
        if not ok:
            error_class_history.append(self._classify_synth_error(err))
            orch.turn_results.append({
                "turn": attempt, "phase": "B", "success": False, "error": err,
            })
            if orch.synthesis._should_revert(error_class_history, best_state, threshold):
                restored = orch.synthesis._revert_and_exit(error_class_history, best_state, threshold)
                ctx["phase_b_done"] = restored
                ctx["phase_b_success"] = restored
                orch._pipelined_ctx = ctx
                return [{
                    "kind": "finalize",
                    "phase": "failed",
                    "attempt": attempt,
                    "stage": "phase_b",
                    "error": err if not restored else "Phase B reverted",
                }]
            if attempt >= orch.turns_limitation - 1:
                return [{
                    "kind": "finalize",
                    "phase": "failed",
                    "attempt": attempt,
                    "stage": "phase_b",
                    "error": err,
                }]
            return [{
                "kind": "codegen",
                "phase": "phase_b",
                "attempt": attempt,
                "stage": "repair",
                "meta": {"repair": {"kind": "compile", "error": err, "attempt": attempt}},
            }]

        outcome = self._synth_csim_only(orch.hls_code, log_prefix="[Phase B]", temp_suffix="phase_b")
        result = outcome["synth"]
        orch.turn_results.append({
            "turn": attempt,
            "phase": "B",
            "success": result["success"],
            "report": result.get("report", {}),
            "error": result.get("error", ""),
        })

        if not result["success"]:
            error_class_history.append(self._classify_synth_error(result.get("error", "")))
            if orch.synthesis._should_revert(error_class_history, best_state, threshold):
                restored = orch.synthesis._revert_and_exit(error_class_history, best_state, threshold)
                ctx["phase_b_done"] = restored
                ctx["phase_b_success"] = restored
                orch._pipelined_ctx = ctx
                return [{
                    "kind": "finalize",
                    "phase": "failed",
                    "attempt": attempt,
                    "stage": "phase_b",
                    "error": result.get("error", "") or "Phase B reverted",
                }]
            if attempt >= orch.turns_limitation - 1:
                return [{
                    "kind": "finalize",
                    "phase": "failed",
                    "attempt": attempt,
                    "stage": "phase_b",
                    "error": result.get("error", ""),
                }]
            return [{
                "kind": "codegen",
                "phase": "phase_b",
                "attempt": attempt,
                "stage": "repair",
                "meta": {
                    "repair": {
                        "kind": "synth",
                        "error": result.get("error", ""),
                        "report": result.get("report"),
                        "attempt": attempt,
                    },
                },
            }]

        orch.synth_report = result["report"]
        ctx["phase_b_best_state"] = orch.synthesis._record_best(orch.hls_code, result, outcome)
        orch._pipelined_ctx = ctx

        csim_failed, csim_error = self._csim_failed(outcome)
        if csim_failed:
            if attempt >= orch.turns_limitation - 1:
                return [{
                    "kind": "finalize",
                    "phase": "failed",
                    "attempt": attempt,
                    "stage": "phase_b",
                    "error": csim_error,
                }]
            return [{
                "kind": "codegen",
                "phase": "phase_b",
                "attempt": attempt,
                "stage": "repair",
                "meta": {
                    "repair": {
                        "kind": "compile" if self._csim_link_error(csim_error) else "csim",
                        "error": csim_error,
                        "attempt": attempt,
                    },
                },
            }]

        ctx["phase_b_done"] = True
        ctx["phase_b_success"] = True
        orch._pipelined_ctx = ctx
        orch.generated_csim = outcome.get("csim")
        if self._record_flow_enabled():
            orch._flow_phase_b_code = orch.hls_code
            orch._flow_phase_b_report = dict(orch.synth_report or {})
        orch._baseline_report = dict(orch.synth_report or {})
        return [{
            "kind": "codegen",
            "phase": "flash",
            "attempt": 0,
            "stage": "optimize",
        }]

    def _run_synth_flash(self, job: BatchParallelJob) -> list[dict[str, Any]]:
        orch = self._ensure_orchestrator()
        ctx = getattr(orch, "_pipelined_ctx", {})
        step_name = "flash"
        new_code = ctx.get("flash_pending_code")
        if not new_code:
            return [{
                "kind": "finalize",
                "phase": "failed",
                "attempt": job.attempt,
                "stage": "flash",
                "error": "missing flash code",
            }]

        step_turn_records = list(ctx.get("flash_step_turn_records") or [])
        attempt_results = list(ctx.get("flash_attempt_results") or [])
        attempt = job.attempt

        logging.info("[Step: %s] Synthesis+csim attempt %d (tier_a batch_parallel)", step_name, attempt)
        new_code = orch._preflight_generated_hls_code(
            new_code, f"Step {step_name} attempt {attempt}",
        )
        ctx["flash_pending_code"] = new_code

        ok, err = self._compile_check_cpp(
            new_code, orch.header_code, orch.header_name,
            extra_files=orch.extra_files,
        )
        if not ok:
            attempt_results.append({
                "attempt_index": attempt,
                "success": False,
                "stage": "compile_check",
                "error": err,
            })
            step_turn_records.append({"turn": attempt, "phase": "B", "success": False, "error": err})
            ctx["flash_step_turn_records"] = step_turn_records
            ctx["flash_attempt_results"] = attempt_results
            orch._pipelined_ctx = ctx
            if attempt >= orch.turns_limitation - 1:
                return [{
                    "kind": "finalize",
                    "phase": "failed",
                    "attempt": attempt,
                    "stage": "flash",
                    "error": err,
                }]
            return [{
                "kind": "codegen",
                "phase": "flash",
                "attempt": attempt,
                "stage": "repair",
                "meta": {"repair": {"kind": "compile", "error": err, "attempt": attempt}},
            }]

        outcome = self._synth_csim_only(new_code, log_prefix=f"[Step: {step_name}]", temp_suffix="flash")
        result = outcome["synth"]

        if not result["success"]:
            attempt_results.append({
                "attempt_index": attempt,
                "success": False,
                "stage": "synthesis",
                "report": result.get("report"),
                "error": result.get("error", ""),
            })
            step_turn_records.append({
                "turn": attempt, "phase": "B", "success": False,
                "error": result.get("error", ""),
            })
            ctx["flash_step_turn_records"] = step_turn_records
            ctx["flash_attempt_results"] = attempt_results
            orch._pipelined_ctx = ctx
            if attempt >= orch.turns_limitation - 1:
                return [{
                    "kind": "finalize",
                    "phase": "failed",
                    "attempt": attempt,
                    "stage": "flash",
                    "error": result.get("error", ""),
                }]
            next_attempt = attempt + 1
            return [{
                "kind": "codegen",
                "phase": "flash",
                "attempt": next_attempt,
                "stage": "repair",
                "meta": {
                    "repair": {
                        "kind": "synth",
                        "error": result.get("error", ""),
                        "report": result.get("report"),
                        "attempt": attempt,
                    },
                    "next_attempt": next_attempt,
                },
            }]

        csim_failed, csim_error = self._csim_failed(outcome)
        if csim_failed:
            attempt_results.append({
                "attempt_index": attempt,
                "success": False,
                "stage": "csim",
                "report": result.get("report"),
                "error": csim_error,
            })
            step_turn_records.append({
                "turn": attempt, "phase": "B", "success": False, "error": csim_error,
            })
            ctx["flash_step_turn_records"] = step_turn_records
            ctx["flash_attempt_results"] = attempt_results
            orch._pipelined_ctx = ctx
            if attempt >= orch.turns_limitation - 1:
                return [{
                    "kind": "finalize",
                    "phase": "failed",
                    "attempt": attempt,
                    "stage": "flash",
                    "error": csim_error,
                }]
            next_attempt = attempt + 1
            return [{
                "kind": "codegen",
                "phase": "flash",
                "attempt": next_attempt,
                "stage": "repair",
                "meta": {
                    "repair": {
                        "kind": "compile" if self._csim_link_error(csim_error) else "csim",
                        "error": csim_error,
                        "attempt": attempt,
                    },
                    "next_attempt": next_attempt,
                },
            }]

        orch.hls_code = new_code
        orch.synth_report = result["report"]
        ctx["flash_pending_code"] = new_code
        ctx["flash_step_result"] = {
            "success": True,
            "step_name": "flash",
            "report": result["report"],
            "code": new_code,
            "csim": outcome.get("csim"),
        }
        ctx["flash_done"] = True
        orch._pipelined_ctx = ctx
        return [{"phase": "finalize", "kind": "finalize", "attempt": attempt, "stage": "done"}]

    def _finalize_failure(self, error: str) -> None:
        results = {
            "benchmark": self.bench,
            "success": False,
            "phase": "reference" if not self.reference_validation.get("benchmark_ready") else "pipelined",
            "error": error,
            "reference_validation": self.reference_validation,
            "turn_results": getattr(self.orchestrator, "turn_results", []) if self.orchestrator else [],
            "steps": [],
            "pipelined": True,
        }
        # Keep gold bookkeeping consistent with success path: ground_truth_report
        # must mirror reference_validation.report even when the opt flow fails.
        results = _sanitize_saved_result_record(results, self.reference_validation)
        result_json = self.cell_dir / f"{self.bench}_multistep_results.json"
        result_json.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


def execute_job(
    *,
    job: BatchParallelJob,
    queue: BatchParallelQueue,
    bench_dir: Path,
    cell_dir: Path,
    variant_key: str,
    model_id: str,
    turns: int,
) -> None:
    session = TierABatchParallelBenchSession(
        variant_key=variant_key,
        bench=job.bench,
        bench_dir=bench_dir,
        cell_dir=cell_dir,
        model_id=model_id,
        turns=turns,
    )
    try:
        session.handle_job(job, queue)
    except Exception as exc:
        logging.exception("tier_a batch_parallel bench %s job %s failed", job.bench, job.id)
        queue.complete(job.id, error=str(exc))
        raise
