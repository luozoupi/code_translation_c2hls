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

# Final cosim tries at most this many ranked candidates (best → next1 → next2).
COSIM_FALLBACK_MAX_ATTEMPTS = 3


def skip_final_cosim() -> bool:
    return os.getenv("C2HLS_MULTISTEP_SKIP_FINAL_COSIM", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _latency_cycles_from_report(report: dict | None) -> float | None:
    """Ranking key: worst-case csynth latency (min of max)."""
    if not isinstance(report, dict):
        return None
    lat = report.get("latency_cycles_worst")
    if lat is None:
        lat = report.get("latency_cycles")
    try:
        return float(lat) if lat is not None else None
    except (TypeError, ValueError):
        return None


def _load_json_dict(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def collect_cosim_candidates(
    *,
    cell_dir: Path,
    bench: str,
    phases: list[str],
) -> list[dict[str, Any]]:
    """Collect successful pre/post lat-opt kernels with csynth latency for ranking."""
    import re

    candidates: list[dict[str, Any]] = []
    round_re = re.compile(r"^.+_r(\d+)\.cpp$")
    for phase_idx, phase in enumerate(phases):
        pre_cpp = cell_dir / f"{bench}_multistep_{phase}.cpp"
        pre_report_path = cell_dir / f"{bench}_multistep_{phase}_report.json"
        pre_report = _load_json_dict(pre_report_path)
        pre_lat = _latency_cycles_from_report(pre_report)
        if pre_cpp.is_file() and pre_lat is not None and pre_cpp.read_text(encoding="utf-8").strip():
            candidates.append(
                {
                    "id": f"{phase}:pre_lat_opt",
                    "phase": phase,
                    "variant": "pre_lat_opt",
                    "phase_idx": phase_idx,
                    "latency_cycles": pre_lat,
                    "code_path": str(pre_cpp),
                    "report_path": str(pre_report_path),
                    "report": dict(pre_report or {}),
                }
            )

        stem = f"{bench}_multistep_{phase}_latency_opt"
        round_bodies: set[str] = set()
        for round_cpp in sorted(cell_dir.glob(f"{stem}_r*.cpp")):
            match = round_re.match(round_cpp.name)
            if not match:
                continue
            round_idx = match.group(1)
            body = round_cpp.read_text(encoding="utf-8")
            if not body.strip():
                continue
            round_report_path = cell_dir / f"{stem}_r{round_idx}_report.json"
            round_report = _load_json_dict(round_report_path)
            round_lat = _latency_cycles_from_report(round_report)
            if round_lat is None or not lat_opt_improved_vs_seed(pre_lat, round_lat):
                continue
            round_bodies.add(body)
            candidates.append(
                {
                    "id": f"{phase}:post_lat_opt_r{round_idx}",
                    "phase": phase,
                    "variant": "post_lat_opt",
                    "phase_idx": phase_idx,
                    "latency_cycles": round_lat,
                    "code_path": str(round_cpp),
                    "report_path": str(round_report_path) if round_report_path.is_file() else "",
                    "report": dict(round_report or {"latency_cycles": round_lat}),
                }
            )

        post_cpp = cell_dir / f"{stem}.cpp"
        post_result = _load_json_dict(cell_dir / f"{stem}_result.json")
        post_report_path = cell_dir / f"{stem}_report.json"
        post_report = _load_json_dict(post_report_path)
        post_ok = bool(post_result and post_result.get("success"))
        post_lat = _latency_cycles_from_report(post_report)
        if post_lat is None and post_result is not None:
            try:
                post_lat = (
                    float(post_result["latency_cycles"])
                    if post_result.get("latency_cycles") is not None
                    else None
                )
            except (TypeError, ValueError):
                post_lat = None
        # Only rank post-lat-opt when it strictly beat the seed; otherwise seed stays
        # the chain/selection candidate for this phase.
        if (
            post_ok
            and post_cpp.is_file()
            and post_lat is not None
            and lat_opt_improved_vs_seed(pre_lat, post_lat)
            and post_cpp.read_text(encoding="utf-8").strip()
        ):
            post_body = post_cpp.read_text(encoding="utf-8")
            if post_body in round_bodies:
                continue
            candidates.append(
                {
                    "id": f"{phase}:post_lat_opt",
                    "phase": phase,
                    "variant": "post_lat_opt",
                    "phase_idx": phase_idx,
                    "latency_cycles": post_lat,
                    "code_path": str(post_cpp),
                    "report_path": str(post_report_path) if post_report_path.is_file() else "",
                    "report": dict(post_report or {"latency_cycles": post_lat}),
                }
            )
    return candidates


def rank_cosim_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sort by latency asc; ties: later phase, then post_lat_opt over pre."""
    return sorted(
        candidates,
        key=lambda c: (
            float(c.get("latency_cycles") if c.get("latency_cycles") is not None else float("inf")),
            -int(c.get("phase_idx") or 0),
            0 if c.get("variant") == "post_lat_opt" else 1,
            str(c.get("id") or ""),
        ),
    )


def lat_opt_improved_vs_seed(
    pre_lat: float | None,
    post_lat: float | None,
) -> bool:
    """True only when lat-opt is strictly lower latency than the step seed."""
    if pre_lat is None or post_lat is None:
        return False
    return float(post_lat) < float(pre_lat)


def build_latency_table(
    *,
    cell_dir: Path,
    bench: str,
    phases: list[str],
) -> list[dict[str, Any]]:
    """Clear per-phase pre/post csynth latency record for results JSON."""
    rows: list[dict[str, Any]] = []
    for phase in phases:
        pre_report = _load_json_dict(cell_dir / f"{bench}_multistep_{phase}_report.json")
        pre_lat = _latency_cycles_from_report(pre_report)
        post_result = _load_json_dict(
            cell_dir / f"{bench}_multistep_{phase}_latency_opt_result.json"
        )
        post_report = _load_json_dict(
            cell_dir / f"{bench}_multistep_{phase}_latency_opt_report.json"
        )
        post_lat = _latency_cycles_from_report(post_report)
        if post_lat is None and isinstance(post_result, dict):
            try:
                post_lat = (
                    float(post_result["latency_cycles"])
                    if post_result.get("latency_cycles") is not None
                    else None
                )
            except (TypeError, ValueError):
                post_lat = None
        lat_opt_ran = bool(post_result)
        lat_opt_success = bool(post_result and post_result.get("success"))
        improved = lat_opt_improved_vs_seed(pre_lat, post_lat) if lat_opt_success else False
        rows.append(
            {
                "phase": phase,
                "pre_lat_opt_csynth_latency": pre_lat,
                "post_lat_opt_csynth_latency": post_lat if lat_opt_success else None,
                "lat_opt_ran": lat_opt_ran,
                "lat_opt_success": lat_opt_success,
                "lat_opt_improved": improved,
            }
        )
    return rows


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

    def _phase_list(self) -> list[str]:
        return ["phase_b", *list(self.opt_steps or [])]

    def _maybe_run_latency_opt(self, phase: str) -> None:
        from post_flash_latency_opt import maybe_chain_latency_opt

        orch = self._ensure_orchestrator()
        code = orch.hls_code or ""
        report = dict(orch.synth_report or {})
        pre_lat = self._latency_cycles(report)
        self._write_multistep_seed(phase, code, report)
        source_role = f"multistep_{phase}"
        latency_record = {
            "phase": phase,
            "pre_lat_opt_csynth_latency": pre_lat,
            "post_lat_opt_csynth_latency": None,
            "lat_opt_ran": False,
            "lat_opt_success": False,
            "lat_opt_improved": False,
        }
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
            self._store_latency_record(phase, latency_record)
            return
        if outcome is None or not outcome.success:
            self._store_latency_record(phase, latency_record)
            return
        result = outcome.result or {}
        latency_record["lat_opt_ran"] = True
        latency_record["lat_opt_success"] = True
        paths_kernel = self.cell_dir / f"{self.bench}_multistep_{phase}_latency_opt.cpp"
        report_path = self.cell_dir / f"{self.bench}_multistep_{phase}_latency_opt_report.json"
        post_report: dict[str, Any] = {}
        if report_path.is_file():
            try:
                loaded = json.loads(report_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    post_report = loaded
            except json.JSONDecodeError:
                pass
        lat = result.get("latency_cycles")
        post_lat = self._latency_cycles(post_report) if post_report else None
        if post_lat is None and lat is not None:
            try:
                post_lat = float(lat)
            except (TypeError, ValueError):
                post_lat = None
        latency_record["post_lat_opt_csynth_latency"] = post_lat
        improved = lat_opt_improved_vs_seed(pre_lat, post_lat)
        latency_record["lat_opt_improved"] = improved
        if not improved:
            logging.info(
                "[latency_opt] multistep %s %s: keeping seed (pre=%s post=%s)",
                self.bench,
                phase,
                pre_lat,
                post_lat,
            )
            self._store_latency_record(phase, latency_record)
            return

        # Strictly better than seed → adopt into the live chain.
        if paths_kernel.is_file():
            improved_code = paths_kernel.read_text(encoding="utf-8")
            if improved_code.strip():
                orch.hls_code = improved_code
        if post_report:
            orch.synth_report = dict(post_report)
        elif lat is not None:
            report = dict(orch.synth_report or {})
            report["latency_cycles"] = lat
            orch.synth_report = report
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
                step_result["latency_record"] = dict(latency_record)
                ctx[result_key] = step_result
                step_results = list(ctx.get("step_results") or [])
                for idx, item in enumerate(step_results):
                    if item.get("step_name") == phase:
                        step_results[idx] = step_result
                        break
                ctx["step_results"] = step_results
                orch._pipelined_ctx = ctx
        self._store_latency_record(phase, latency_record)

    def _store_latency_record(self, phase: str, record: dict[str, Any]) -> None:
        orch = self._ensure_orchestrator()
        ctx = getattr(orch, "_pipelined_ctx", {}) or {}
        table = dict(ctx.get("latency_records") or {})
        table[phase] = dict(record)
        ctx["latency_records"] = table
        orch._pipelined_ctx = ctx
        if phase == "phase_b":
            orch._flow_phase_b_latency_record = dict(record)
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
            if skip_final_cosim():
                return [{
                    "kind": "finalize",
                    "phase": "finalize",
                    "attempt": 0,
                    "stage": "done",
                }]
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
            if skip_final_cosim():
                return [{
                    "kind": "finalize",
                    "phase": "finalize",
                    "attempt": 0,
                    "stage": "done",
                }]
            return [{
                "kind": "cosim",
                "phase": "selected",
                "attempt": 0,
                "stage": "cosim",
            }]
        return followups

    def _ranked_cosim_candidates(self) -> list[dict[str, Any]]:
        return rank_cosim_candidates(
            collect_cosim_candidates(
                cell_dir=self.cell_dir,
                bench=self.bench,
                phases=self._phase_list(),
            )
        )

    def _write_selected_kernel(self, code: str, report: dict | None) -> None:
        selected_cpp = self.cell_dir / f"{self.bench}_selected.cpp"
        selected_report = self.cell_dir / f"{self.bench}_selected_report.json"
        selected_cpp.write_text(code or "", encoding="utf-8")
        selected_report.write_text(
            json.dumps(report or {}, indent=2, default=str) + "\n",
            encoding="utf-8",
        )

    def _prepare_selected_for_cosim(self) -> None:
        orch = self._ensure_orchestrator()
        ranked = self._ranked_cosim_candidates()
        ctx = getattr(orch, "_pipelined_ctx", {}) or {}
        ctx["cosim_ranked_candidates"] = [
            {
                "id": c.get("id"),
                "phase": c.get("phase"),
                "variant": c.get("variant"),
                "latency_cycles": c.get("latency_cycles"),
                "code_path": c.get("code_path"),
            }
            for c in ranked
        ]
        ctx["cosim_fallback_attempts"] = []
        orch._pipelined_ctx = ctx
        if ranked:
            best = ranked[0]
            code = Path(best["code_path"]).read_text(encoding="utf-8")
            orch.hls_code = code
            orch.synth_report = dict(best.get("report") or {})
            self._write_selected_kernel(code, orch.synth_report)
            return
        # Fallback: legacy promote among in-memory step results.
        baseline_report = dict(
            orch._flow_phase_b_report or getattr(orch, "_baseline_report", None) or {}
        )
        step_results = list(ctx.get("step_results") or [])
        self._promote_pipelined_best_so_far(orch, baseline_report, step_results)
        self._write_selected_kernel(orch.hls_code or "", orch.synth_report)

    def _promote_pipelined_best_so_far(
        self,
        orch,
        baseline_report: dict,
        step_results: list[dict],
    ) -> dict | None:
        """Prefer lowest csynth among on-disk pre/post candidates (seed kept if lat-opt regressed)."""
        ranked = self._ranked_cosim_candidates()
        if ranked:
            best = ranked[0]
            code_path = Path(str(best.get("code_path") or ""))
            code = code_path.read_text(encoding="utf-8") if code_path.is_file() else ""
            cur_lat = self._latency_cycles(orch.synth_report)
            best_lat = best.get("latency_cycles")
            try:
                best_lat_f = float(best_lat) if best_lat is not None else None
            except (TypeError, ValueError):
                best_lat_f = None
            if code.strip() and best_lat_f is not None:
                if cur_lat is None or best_lat_f < cur_lat:
                    orch.hls_code = code
                    orch.synth_report = dict(best.get("report") or {"latency_cycles": best_lat_f})
                    return {
                        "promoted": True,
                        "from_step_name": best.get("phase"),
                        "from_variant": best.get("variant"),
                        "from_candidate_id": best.get("id"),
                        "from_latency_cycles": best_lat_f,
                        "previous_latency_cycles": cur_lat,
                    }
                return None
        return super()._promote_pipelined_best_so_far(orch, baseline_report, step_results)

    def _run_cosim(self, job: BatchParallelJob) -> list[dict[str, Any]]:
        from c2hls import _run_synth_csim_cosim, join_temp_tag

        orch = self._ensure_orchestrator()
        ranked = self._ranked_cosim_candidates()
        meta = dict(job.meta or {})
        cand_idx = int(meta.get("cosim_candidate_index") or 0)
        if not ranked:
            # No ranked artifacts: try current selected / orch code once.
            selected_cpp = self.cell_dir / f"{self.bench}_selected.cpp"
            if selected_cpp.is_file():
                code = selected_cpp.read_text(encoding="utf-8")
                if code.strip():
                    orch.hls_code = code
            ranked = [
                {
                    "id": "selected:current",
                    "phase": "selected",
                    "variant": "current",
                    "phase_idx": 0,
                    "latency_cycles": self._latency_cycles(orch.synth_report),
                    "code_path": str(selected_cpp),
                    "report": dict(orch.synth_report or {}),
                }
            ]

        if cand_idx >= len(ranked) or cand_idx >= COSIM_FALLBACK_MAX_ATTEMPTS:
            return self._cosim_exhausted_followups(job, orch)

        candidate = ranked[cand_idx]
        code_path = Path(str(candidate.get("code_path") or ""))
        code = code_path.read_text(encoding="utf-8") if code_path.is_file() else (orch.hls_code or "")
        orch.hls_code = code
        if candidate.get("report"):
            orch.synth_report = dict(candidate["report"])
        self._write_selected_kernel(code, orch.synth_report)

        tag = join_temp_tag(self.bench, f"cosim_{candidate.get('id', cand_idx)}".replace(":", "_"))
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
            log_prefix=f"[cosim {candidate.get('id')}]",
            temp_tag=tag,
        )
        cosim_raw = outcome.get("cosim")
        cosim = cosim_raw if isinstance(cosim_raw, dict) else {}
        synth = outcome.get("synth") or {}
        if synth.get("success") and synth.get("report"):
            orch.synth_report = synth.get("report")
        orch.generated_cosim = cosim_raw
        # Skipped cosim (None) is not a failure — do not burn fallback slots.
        cosim_ran = cosim_raw is not None
        cosim_pass = True if not cosim_ran else bool(cosim.get("passed"))

        ctx = getattr(orch, "_pipelined_ctx", {}) or {}
        attempts = list(ctx.get("cosim_fallback_attempts") or [])
        attempts.append(
            {
                "index": cand_idx,
                "id": candidate.get("id"),
                "phase": candidate.get("phase"),
                "variant": candidate.get("variant"),
                "latency_cycles": candidate.get("latency_cycles"),
                "cosim_ran": cosim_ran,
                "passed": cosim_pass,
                "error": (cosim.get("error") or "")[:500] if cosim_ran and not cosim_pass else "",
            }
        )
        ctx["cosim_fallback_attempts"] = attempts
        ctx["cosim_ranked_candidates"] = [
            {
                "id": c.get("id"),
                "phase": c.get("phase"),
                "variant": c.get("variant"),
                "latency_cycles": c.get("latency_cycles"),
                "code_path": c.get("code_path"),
            }
            for c in ranked
        ]
        orch._pipelined_ctx = ctx

        if cosim_pass:
            ctx["cosim_selected_id"] = candidate.get("id")
            orch._pipelined_ctx = ctx
            return [
                {
                    "phase": "finalize",
                    "kind": "finalize",
                    "attempt": job.attempt,
                    "stage": "done",
                }
            ]

        next_idx = cand_idx + 1
        if next_idx < len(ranked) and next_idx < COSIM_FALLBACK_MAX_ATTEMPTS:
            logging.warning(
                "bench %s cosim failed for %s; trying next candidate index=%s",
                self.bench,
                candidate.get("id"),
                next_idx,
            )
            return [
                {
                    "kind": "cosim",
                    "phase": "selected",
                    "attempt": int(job.attempt) + 1,
                    "stage": "cosim",
                    "meta": {"cosim_candidate_index": next_idx},
                }
            ]

        return self._cosim_exhausted_followups(job, orch)

    def _cosim_exhausted_followups(
        self, job: BatchParallelJob, orch
    ) -> list[dict[str, Any]]:
        cosim = getattr(orch, "generated_cosim", None) or {}
        cosim_required = os.getenv("C2HLS_COSIM_REQUIRED", "0").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        err = cosim.get("error") or "cosim failed for top ranked candidates"
        logging.warning(
            "bench %s cosim exhausted top-%s candidates: %s",
            self.bench,
            COSIM_FALLBACK_MAX_ATTEMPTS,
            str(err)[:300],
        )
        ctx = getattr(orch, "_pipelined_ctx", {}) or {}
        ctx["cosim_exhausted"] = True
        orch._pipelined_ctx = ctx
        if not cosim_required:
            # Soft-fail: keep best (first) selected kernel and finalize.
            ranked = self._ranked_cosim_candidates()
            if ranked:
                best = ranked[0]
                code = Path(best["code_path"]).read_text(encoding="utf-8")
                orch.hls_code = code
                orch.synth_report = dict(best.get("report") or {})
                self._write_selected_kernel(code, orch.synth_report)
            return [
                {
                    "phase": "finalize",
                    "kind": "finalize",
                    "attempt": job.attempt,
                    "stage": "done",
                }
            ]
        return [
            {
                "phase": "failed",
                "kind": "finalize",
                "attempt": job.attempt,
                "stage": "cosim",
                "error": err,
            }
        ]

    def _finalize_success(self) -> None:
        orch = self._ensure_orchestrator()
        ctx = getattr(orch, "_pipelined_ctx", {}) or {}
        latency_table = build_latency_table(
            cell_dir=self.cell_dir,
            bench=self.bench,
            phases=self._phase_list(),
        )
        # Prefer in-memory records when side files are incomplete.
        mem = dict(ctx.get("latency_records") or {})
        for row in latency_table:
            phase = row["phase"]
            if phase in mem:
                merged = dict(row)
                merged.update({k: v for k, v in mem[phase].items() if v is not None})
                row.clear()
                row.update(merged)
        cosim_fallback = {
            "max_attempts": COSIM_FALLBACK_MAX_ATTEMPTS,
            "ranked_candidates": list(ctx.get("cosim_ranked_candidates") or []),
            "attempts": list(ctx.get("cosim_fallback_attempts") or []),
            "selected_id": ctx.get("cosim_selected_id"),
            "exhausted": bool(ctx.get("cosim_exhausted")),
        }
        # Call parent finalize then enrich the written JSON.
        super()._finalize_success()
        result_json = self.cell_dir / f"{self.bench}_multistep_results.json"
        try:
            results = json.loads(result_json.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        if not isinstance(results, dict):
            return
        results["latency_table"] = latency_table
        results["cosim_fallback"] = cosim_fallback
        # Attach per-step latency_record when present.
        for step in list(results.get("steps") or []):
            if not isinstance(step, dict):
                continue
            phase = step.get("step_name")
            if phase and phase in mem:
                step["latency_record"] = dict(mem[phase])
        if "phase_b" in mem:
            results["phase_b_latency_record"] = dict(mem["phase_b"])
        result_json.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
        orch.save_multistep_results(str(self.cell_dir), self.bench, results)

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
