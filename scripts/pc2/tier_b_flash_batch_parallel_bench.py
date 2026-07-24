"""Tier B MachSuite flash batch_parallel session: csim+csynth+cosim with repair rounds."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from batch_parallel_queue import BatchParallelJob, BatchParallelQueue
from tier_a_batch_parallel_bench import TierABatchParallelBenchSession


class TierBFlashBatchParallelBenchSession(TierABatchParallelBenchSession):
    """Tier-A style synth+csim repairs, then enqueue cosim (with cosim repairs)."""

    def _run_synth_phase_b(self, job: BatchParallelJob) -> list[dict[str, Any]]:
        followups = super()._run_synth_phase_b(job)
        return self._rewrite_success_to_cosim(followups, job)

    def _run_synth_flash(self, job: BatchParallelJob) -> list[dict[str, Any]]:
        followups = super()._run_synth_flash(job)
        return self._rewrite_success_to_cosim(followups, job)

    def _rewrite_success_to_cosim(
        self,
        followups: list[dict[str, Any]],
        job: BatchParallelJob,
    ) -> list[dict[str, Any]]:
        """After synth+csim success, run RTL cosim before advancing.

        Honors ``C2HLS_RUN_COSIM=0`` so campaigns can stay on csim+csynth only.
        """
        import os

        raw = os.getenv("C2HLS_RUN_COSIM", "1").strip().lower()
        if raw in ("0", "false", "no", "off"):
            return followups

        out: list[dict[str, Any]] = []
        for spec in followups:
            kind = spec.get("kind")
            phase = spec.get("phase")
            stage = spec.get("stage")
            # phase_b success previously jumped straight to flash codegen
            if (
                job.phase == "phase_b"
                and kind == "codegen"
                and phase == "flash"
                and stage == "optimize"
            ):
                out.append(self._cosim_followup(job))
                continue
            # flash success previously finalized immediately
            if (
                job.phase == "flash"
                and phase == "finalize"
                and kind == "finalize"
                and stage == "done"
            ):
                out.append(self._cosim_followup(job))
                continue
            out.append(spec)
        return out


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
    session = TierBFlashBatchParallelBenchSession(
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
        logging.exception(
            "tier_b flash batch_parallel bench %s job %s failed", job.bench, job.id
        )
        queue.complete(job.id, error=str(exc))
        raise
