"""Flash pipelined bench driver with split synth / cosim jobs for batch_parallel."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from batch_parallel_env import configure_cosim_env, configure_synth_env
from batch_parallel_queue import BatchParallelJob, BatchParallelQueue
from flash_pipelined_bench import FlashPipelinedBenchSession


class BatchParallelBenchSession(FlashPipelinedBenchSession):
  """Extends flash pipelined session: synth jobs are synth-only; cosim is a separate kind."""

  def handle_job(self, job: BatchParallelJob, queue: BatchParallelQueue) -> None:
    if job.kind == "codegen":
      configure_synth_env(cosim_timeout_s=int(os.getenv("C2HLS_COSIM_TIMEOUT", "43200")))
      super().handle_job(job, queue)  # type: ignore[arg-type]
      return

    if job.kind == "synth":
      configure_synth_env(cosim_timeout_s=int(os.getenv("C2HLS_COSIM_TIMEOUT", "43200")))
      orch = self._ensure_orchestrator()
      followups = self._run_synth(job)  # type: ignore[arg-type]
      self._save_state(orch)
      self._apply_followups(followups, queue)
      return

    if job.kind == "cosim":
      configure_cosim_env(cosim_timeout_s=int(os.getenv("C2HLS_COSIM_TIMEOUT", "43200")))
      self._ensure_orchestrator()
      followups = self._run_cosim(job)
      self._save_state(self.orchestrator)
      self._apply_followups(followups, queue)
      return

    raise ValueError(f"unknown job kind {job.kind}")

  def _max_repair_attempt(self) -> int:
    raw = os.getenv("C2HLS_MAX_REPAIR_ATTEMPT", "").strip()
    if raw:
      return int(raw)
    return max(int(getattr(self, "turns", 4) or 4), 7)

  def _apply_followups(self, followups: list[dict[str, Any]], queue: BatchParallelQueue) -> None:
    max_attempt = self._max_repair_attempt()
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
      attempt = int(spec.get("attempt") or 0)
      if attempt > max_attempt:
        err = (
          spec.get("error")
          or f"repair attempt {attempt} exceeds max_repair_attempt={max_attempt}"
        )
        logging.warning(
          "bench %s refusing enqueue kind=%s attempt=%s (%s)",
          self.bench,
          spec.get("kind"),
          attempt,
          err,
        )
        self._finalize_failure(err)
        queue.set_bench_status(self.variant_key, self.bench, "failed")
        continue
      queue.enqueue(
        variant=self.variant_key,
        bench=self.bench,
        kind=spec["kind"],
        phase=spec["phase"],
        attempt=attempt,
        stage=spec.get("stage") or "",
        meta=dict(spec.get("meta") or {}),
      )

  def _synth_only(self, code: str, *, log_prefix: str, temp_suffix: str) -> dict:
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
      run_csim_check=False,
      run_cosim_check=False,
      cosim_depths=orch.cosim_depths,
      log_prefix=log_prefix,
      temp_tag=tag,
    )

  def _cosim_followup(self, job: BatchParallelJob) -> dict[str, Any]:
    return {
      "kind": "cosim",
      "phase": job.phase,
      "attempt": job.attempt,
      "stage": "cosim",
    }

  def _run_synth(self, job: BatchParallelJob) -> list[dict[str, Any]]:  # type: ignore[override]
    from c2hls import _classify_synth_error, compile_check_cpp

    orch = self._ensure_orchestrator()

    if job.phase == "phase_b":
      ctx = getattr(orch, "_pipelined_ctx", {})
      if ctx.get("phase_b_best_state") is None:
        ctx["phase_b_best_state"] = None
      if ctx.get("phase_b_error_history") is None:
        ctx["phase_b_error_history"] = []
      best_state = ctx.get("phase_b_best_state")
      error_class_history = ctx["phase_b_error_history"]
      threshold = orch.synthesis.revert_threshold
      attempt = job.attempt

      logging.info("[Phase B] Synthesis attempt %d (batch_parallel synth-only)", attempt)
      orch.hls_code = orch._preflight_generated_hls_code(
        orch.hls_code, f"Phase B attempt {attempt}",
      )

      ok, err = compile_check_cpp(
        orch.hls_code, orch.header_code, orch.header_name,
        extra_files=orch.extra_files,
      )
      if not ok:
        error_class_history.append(_classify_synth_error(err))
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

      outcome = self._synth_only(orch.hls_code, log_prefix="[Phase B]", temp_suffix="phase_b")
      result = outcome["synth"]
      orch.turn_results.append({
        "turn": attempt,
        "phase": "B",
        "success": result["success"],
        "report": result.get("report", {}),
        "error": result.get("error", ""),
      })

      if result["success"]:
        orch.synth_report = result["report"]
        ctx["phase_b_best_state"] = orch.synthesis._record_best(orch.hls_code, result, outcome)
        orch._pipelined_ctx = ctx
        return [self._cosim_followup(job)]

      error_class_history.append(_classify_synth_error(result.get("error", "")))
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

    if job.phase == "flash":
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

      logging.info("[Step: %s] Synthesis attempt %d (batch_parallel synth-only)", step_name, attempt)
      new_code = orch._preflight_generated_hls_code(
        new_code, f"Step {step_name} attempt {attempt}",
      )
      ctx["flash_pending_code"] = new_code

      ok, err = compile_check_cpp(
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

      outcome = self._synth_only(new_code, log_prefix=f"[Step: {step_name}]", temp_suffix="flash")
      result = outcome["synth"]

      if result["success"]:
        orch.hls_code = new_code
        orch.synth_report = result["report"]
        ctx["flash_pending_code"] = new_code
        orch._pipelined_ctx = ctx
        return [self._cosim_followup(job)]

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

    raise ValueError(f"unknown synth phase {job.phase}")

  def _run_cosim(self, job: BatchParallelJob) -> list[dict[str, Any]]:
    from c2hls import _run_synth_csim_cosim, join_temp_tag

    orch = self._ensure_orchestrator()
    tag = join_temp_tag(self.bench, job.phase if job.phase != "flash" else "flash")
    code = orch.hls_code
    if job.phase == "flash":
      code = getattr(orch, "_pipelined_ctx", {}).get("flash_pending_code") or orch.hls_code
    outcome = _run_synth_csim_cosim(
      code,
      header_code=orch.header_code,
      header_name=orch.header_name,
      top_function=orch.translated_hls_top,
      part=orch.part,
      clock_ns=orch.clock_ns,
      extra_files=orch.extra_files,
      testbench_code=orch.testbench_code,
      run_csim_check=bool(orch.testbench_code),
      run_cosim_check=bool(orch.testbench_code and orch.supports_cosim),
      cosim_depths=orch.cosim_depths,
      log_prefix=f"[{job.phase} cosim]",
      temp_tag=tag,
    )
    cosim = outcome.get("cosim") or {}
    synth = outcome.get("synth") or {}
    if not synth.get("success"):
      return [{
        "kind": "codegen",
        "phase": job.phase,
        "attempt": job.attempt,
        "stage": "repair",
        "meta": {"repair": {"kind": "synth", "error": synth.get("error", ""), "attempt": job.attempt}},
      }]

    if cosim.get("passed"):
      if job.phase == "phase_b":
        from c2hls import record_flow_enabled

        ctx = getattr(orch, "_pipelined_ctx", {})
        ctx["phase_b_done"] = True
        ctx["phase_b_success"] = True
        orch._pipelined_ctx = ctx
        orch.generated_cosim = cosim
        if record_flow_enabled():
          orch._flow_phase_b_code = orch.hls_code
          orch._flow_phase_b_report = dict(orch.synth_report or {})
        orch._baseline_report = dict(orch.synth_report or {})
        return [{
          "kind": "codegen",
          "phase": "flash",
          "attempt": 0,
          "stage": "optimize",
        }]
      ctx = getattr(orch, "_pipelined_ctx", {})
      ctx["flash_step_result"] = {
        "success": True,
        "step_name": "flash",
        "report": orch.synth_report,
        "code": code,
        "cosim": cosim,
      }
      ctx["flash_done"] = True
      orch._pipelined_ctx = ctx
      return [{"phase": "finalize", "kind": "finalize", "attempt": job.attempt, "stage": "done"}]

    next_attempt = int(job.attempt) + 1
    cosim_err = cosim.get("error") or "cosim failed"
    if next_attempt > self._max_repair_attempt():
      return [{
        "phase": "failed",
        "kind": "finalize",
        "attempt": job.attempt,
        "stage": "done",
        "error": (
          f"cosim repair exhausted at attempt {job.attempt}: {cosim_err}"
        ),
      }]
    return [{
      "kind": "codegen",
      "phase": job.phase,
      "attempt": next_attempt,
      "stage": "repair",
      "meta": {
        "repair": {
          "kind": "cosim",
          "error": cosim_err,
          "attempt": job.attempt,
        }
      },
    }]


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
  session = BatchParallelBenchSession(
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
    logging.exception("batch_parallel bench %s job %s failed", job.bench, job.id)
    queue.complete(job.id, error=str(exc))
    raise
