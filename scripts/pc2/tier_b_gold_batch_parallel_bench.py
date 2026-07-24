"""Tier B gold-gate batch_parallel bench session (reference csynth+csim only, no LLM)."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from batch_parallel_env import configure_synth_env
from batch_parallel_queue import BatchParallelJob, BatchParallelQueue
from flash_pipelined_bench import _load_benchmark_inputs
from tier_a_batch_parallel_bench import TierABatchParallelBenchSession
from tier_b_gold_lib import apply_bench_synth_timeout_from_meta


class TierBGoldBatchParallelBenchSession(TierABatchParallelBenchSession):
    """Run validate_gold_reference once per bench, then finalize (no codegen)."""

    RESULT_NAME = "gold_gate_results.json"

    def handle_job(self, job: BatchParallelJob, queue: BatchParallelQueue) -> None:
        self._apply_bench_synth_timeout()
        if job.kind == "synth" and job.phase == "reference":
            configure_synth_env(cosim_timeout_s=int(__import__("os").getenv("C2HLS_COSIM_TIMEOUT", "43200")))
            followups = self._run_reference_synth(job)
            self._apply_followups(followups, queue)
            return
        raise ValueError(f"tier_b_gold does not support job kind={job.kind} phase={job.phase}")

    def _run_reference_synth(self, job: BatchParallelJob) -> list[dict[str, Any]]:
        logging.info("[Reference] Gold gate for %s (tier_B)", self.bench)
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
            "phase": "finalize",
            "kind": "finalize",
            "attempt": job.attempt,
            "stage": "done",
        }]

    def _finalize_success(self) -> None:
        ref = self.reference_validation or self._load_reference_validation()
        synth = (ref.get("synthesis") or {}) if isinstance(ref.get("synthesis"), dict) else {}
        csim = (ref.get("csim") or {}) if isinstance(ref.get("csim"), dict) else {}
        results = {
            "benchmark": self.bench,
            "success": True,
            "phase": "gold_gate",
            "gold_pass": True,
            "top_function": ref.get("top_function") or self.inputs["meta"].get("hls_top", ""),
            "synthesis": synth.get("status"),
            "csim": csim.get("status"),
            "invalid_reason": "",
            "reference_validation": ref,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        self.cell_dir.mkdir(parents=True, exist_ok=True)
        (self.cell_dir / self.RESULT_NAME).write_text(
            json.dumps(results, indent=2) + "\n",
            encoding="utf-8",
        )
        legacy = self.cell_dir / f"{self.bench}_multistep_results.json"
        legacy.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    def _finalize_failure(self, error: str) -> None:
        ref = self.reference_validation or self._load_reference_validation()
        synth = (ref.get("synthesis") or {}) if isinstance(ref.get("synthesis"), dict) else {}
        csim = (ref.get("csim") or {}) if isinstance(ref.get("csim"), dict) else {}
        results = {
            "benchmark": self.bench,
            "success": False,
            "phase": "gold_gate",
            "gold_pass": False,
            "top_function": ref.get("top_function") or self.inputs["meta"].get("hls_top", ""),
            "synthesis": synth.get("status"),
            "csim": csim.get("status"),
            "invalid_reason": error,
            "reference_validation": ref,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        self.cell_dir.mkdir(parents=True, exist_ok=True)
        (self.cell_dir / self.RESULT_NAME).write_text(
            json.dumps(results, indent=2) + "\n",
            encoding="utf-8",
        )
        legacy = self.cell_dir / f"{self.bench}_multistep_results.json"
        legacy.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


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
    _ = turns
    session = TierBGoldBatchParallelBenchSession(
        variant_key=variant_key,
        bench=job.bench,
        bench_dir=bench_dir,
        cell_dir=cell_dir,
        model_id=model_id,
        turns=0,
    )
    try:
        session.handle_job(job, queue)
    except Exception as exc:
        logging.exception("tier_b gold batch_parallel bench %s job %s failed", job.bench, job.id)
        queue.complete(job.id, error=str(exc))
        raise
