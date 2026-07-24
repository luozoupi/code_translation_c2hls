"""Multistep pipelined bench driver for batch_parallel (synth/cosim split + lat-opt)."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from batch_parallel_env import configure_cosim_env, configure_synth_env
from batch_parallel_multistep_lib import opt_steps_from_env
from batch_parallel_queue import BatchParallelJob, BatchParallelQueue
from multistep_pipelined_bench import MultistepPipelinedBenchSession


class MultistepBatchParallelBenchSession(MultistepPipelinedBenchSession):
    """Multistep session with synth-only intermediates, per-step lat-opt, final cosim."""

    def __init__(self, **kwargs: Any) -> None:
        if kwargs.get("opt_steps") is None:
            kwargs["opt_steps"] = opt_steps_from_env()
        super().__init__(**kwargs)

    def _ensure_orchestrator(self):
        orch = super()._ensure_orchestrator()
        clock_env = (os.getenv("C2HLS_CLOCK_NS") or "").strip()
        if clock_env:
            try:
                orch.clock_ns = float(clock_env)
            except ValueError:
                pass
        part_env = (os.getenv("C2HLS_PART") or "").strip()
        if part_env:
            orch.part = part_env
        return orch

    def handle_job(self, job: BatchParallelJob, queue: BatchParallelQueue) -> None:
        # Parent MultistepPipelinedBenchSession always ensures the orchestrator
        # before codegen/synth; do the same here (codegen uses self.orchestrator).
        orch = self._ensure_orchestrator()
        if job.kind == "codegen":
            configure_synth_env(cosim_timeout_s=int(os.getenv("C2HLS_COSIM_TIMEOUT", "604800")))
            followups = self._run_codegen(job)  # type: ignore[arg-type]
            self._save_state(orch)
            self._apply_followups(followups, queue)
            return

        if job.kind == "synth":
            configure_synth_env(cosim_timeout_s=int(os.getenv("C2HLS_COSIM_TIMEOUT", "604800")))
            followups = self._run_synth(job)  # type: ignore[arg-type]
            self._save_state(orch)
            self._apply_followups(followups, queue)
            return

        if job.kind == "cosim":
            configure_cosim_env(cosim_timeout_s=int(os.getenv("C2HLS_COSIM_TIMEOUT", "604800")))
            followups = self._run_cosim(job)
            self._save_state(orch)
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
            if attempt > max_attempt and spec.get("kind") != "cosim":
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

    def _write_multistep_seed(self, phase: str, code: str, report: dict | None) -> None:
        seed_cpp = self.cell_dir / f"{self.bench}_multistep_{phase}.cpp"
        seed_report = self.cell_dir / f"{self.bench}_multistep_{phase}_report.json"
        seed_cpp.write_text(code or "", encoding="utf-8")
        seed_report.write_text(
            json.dumps(report or {}, indent=2, default=str) + "\n", encoding="utf-8"
        )

    def _maybe_run_latency_opt(self, phase: str) -> None:
        from post_flash_latency_opt import maybe_chain_latency_opt

        orch = self._ensure_orchestrator()
        code = orch.hls_code or ""
        report = dict(orch.synth_report or {})
        self._write_multistep_seed(phase, code, report)
        source_role = f"multistep_{phase}"
        try:
            outcome = maybe_chain_latency_opt(
                bench=self.bench,
                bench_dir=self.bench_dir,
                cell_dir=self.cell_dir,
                orchestrator=orch,
                source_role=source_role,
                skip_existing=True,
            )
        except Exception as exc:
            logging.warning(
                "[latency_opt] multistep %s %s skipped: %s", self.bench, phase, exc
            )
            return
        if outcome is None or not outcome.success:
            return
        result = outcome.result or {}
        paths_kernel = self.cell_dir / f"{self.bench}_multistep_{phase}_latency_opt.cpp"
        if paths_kernel.is_file():
            improved = paths_kernel.read_text(encoding="utf-8")
            if improved.strip():
                orch.hls_code = improved
        lat = result.get("latency_cycles")
        if lat is not None:
            report = dict(orch.synth_report or {})
            report["latency_cycles"] = lat
            orch.synth_report = report
        report_path = self.cell_dir / f"{self.bench}_multistep_{phase}_latency_opt_report.json"
        if report_path.is_file():
            try:
                loaded = json.loads(report_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    orch.synth_report = loaded
            except json.JSONDecodeError:
                pass
        ctx = getattr(orch, "_pipelined_ctx", {}) or {}
        if phase == "phase_b":
            orch._flow_phase_b_code = orch.hls_code
            orch._flow_phase_b_report = dict(orch.synth_report or {})
            orch._baseline_report = dict(orch.synth_report or {})
        else:
            result_key = orch._pipelined_step_result_key(phase)
            step_result = dict(ctx.get(result_key) or {})
            if step_result:
                step_result["code"] = orch.hls_code
                step_result["report"] = dict(orch.synth_report or {})
                step_result["latency_opt"] = True
                ctx[result_key] = step_result
                step_results = list(ctx.get("step_results") or [])
                for idx, item in enumerate(step_results):
                    if item.get("step_name") == phase:
                        step_results[idx] = step_result
                        break
                ctx["step_results"] = step_results
                orch._pipelined_ctx = ctx

    def _run_synth(self, job: BatchParallelJob) -> list[dict[str, Any]]:  # type: ignore[override]
        followups = super()._run_synth(job)  # type: ignore[arg-type]
        if not followups:
            return followups
        first = followups[0]
        # Successful phase completion → lat-opt then either next step or final cosim.
        if first.get("phase") == "failed":
            return followups
        # Successful advance (codegen to a *different* phase) or terminal finalize.
        if job.phase == "phase_b" and first.get("kind") == "codegen" and first.get("phase") != "phase_b":
            self._maybe_run_latency_opt("phase_b")
            return followups
        if (
            job.phase in self.opt_steps
            and first.get("kind") == "codegen"
            and first.get("phase") != job.phase
        ):
            self._maybe_run_latency_opt(job.phase)
            return followups
        if job.phase in self.opt_steps and first.get("phase") == "finalize":
            self._maybe_run_latency_opt(job.phase)
            self._prepare_selected_for_cosim()
            return [{
                "kind": "cosim",
                "phase": "selected",
                "attempt": 0,
                "stage": "cosim",
            }]
        if job.phase == "phase_b" and first.get("phase") == "finalize":
            # No opt steps configured: lat-opt phase_b then cosim.
            self._maybe_run_latency_opt("phase_b")
            self._prepare_selected_for_cosim()
            return [{
                "kind": "cosim",
                "phase": "selected",
                "attempt": 0,
                "stage": "cosim",
            }]
        return followups

    def _prepare_selected_for_cosim(self) -> None:
        orch = self._ensure_orchestrator()
        ctx = getattr(orch, "_pipelined_ctx", {}) or {}
        baseline_report = dict(
            orch._flow_phase_b_report or getattr(orch, "_baseline_report", None) or {}
        )
        step_results = list(ctx.get("step_results") or [])
        self._promote_pipelined_best_so_far(orch, baseline_report, step_results)
        selected_cpp = self.cell_dir / f"{self.bench}_selected.cpp"
        selected_report = self.cell_dir / f"{self.bench}_selected_report.json"
        selected_cpp.write_text(orch.hls_code or "", encoding="utf-8")
        selected_report.write_text(
            json.dumps(orch.synth_report or {}, indent=2, default=str) + "\n",
            encoding="utf-8",
        )

    def _run_cosim(self, job: BatchParallelJob) -> list[dict[str, Any]]:
        from c2hls import _run_synth_csim_cosim, join_temp_tag

        orch = self._ensure_orchestrator()
        selected_cpp = self.cell_dir / f"{self.bench}_selected.cpp"
        if selected_cpp.is_file():
            code = selected_cpp.read_text(encoding="utf-8")
            if code.strip():
                orch.hls_code = code
        tag = join_temp_tag(self.bench, "selected")
        outcome = _run_synth_csim_cosim(
            orch.hls_code,
            header_code=orch.header_code,
            header_name=orch.header_name,
            top_function=orch.translated_hls_top,
            part=orch.part,
            clock_ns=orch.clock_ns,
            extra_files=orch.extra_files,
            testbench_code=orch.testbench_code,
            run_csim_check=False,
            run_cosim_check=bool(orch.testbench_code and orch.supports_cosim),
            cosim_depths=orch.cosim_depths,
            log_prefix="[selected cosim]",
            temp_tag=tag,
        )
        cosim = outcome.get("cosim") or {}
        synth = outcome.get("synth") or {}
        if synth.get("success") and orch.synth_report is None and synth.get("report"):
            orch.synth_report = synth.get("report")
        orch.generated_cosim = cosim
        cosim_pass = bool(cosim.get("passed"))
        cosim_required = os.getenv("C2HLS_COSIM_REQUIRED", "0").strip().lower() in {
            "1", "true", "yes", "on",
        }
        if cosim_pass or not cosim_required:
            # Soft-fail: still finalize with selected kernel even if cosim fails.
            if not cosim_pass:
                logging.warning(
                    "bench %s selected cosim failed (soft): %s",
                    self.bench,
                    (cosim.get("error") or "")[:300],
                )
            return [{"phase": "finalize", "kind": "finalize", "attempt": job.attempt, "stage": "done"}]
        next_attempt = int(job.attempt) + 1
        if next_attempt > self._max_repair_attempt():
            return [{
                "phase": "failed",
                "kind": "finalize",
                "attempt": job.attempt,
                "stage": "cosim",
                "error": cosim.get("error") or "cosim failed",
            }]
        return [{
            "kind": "codegen",
            "phase": self.opt_steps[-1] if self.opt_steps else "phase_b",
            "attempt": next_attempt,
            "stage": "repair",
            "meta": {
                "repair": {
                    "kind": "cosim",
                    "error": cosim.get("error") or "cosim failed",
                    "attempt": job.attempt,
                },
                "next_attempt": next_attempt,
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
    session = MultistepBatchParallelBenchSession(
        variant_key=variant_key,
        bench=job.bench,
        bench_dir=bench_dir,
        cell_dir=cell_dir,
        model_id=model_id,
        turns=turns,
    )
    session.handle_job(job, queue)
