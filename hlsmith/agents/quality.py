"""QualityRepairAgent: phase-specific agent control flow."""
import logging
import math
import os
from typing import Any, Optional

from hlsmith import pipeline as core
from .base import _AgentBase


class QualityRepairAgent(_AgentBase):
    """LLM repair plus an optional deterministic QoR design-space search."""
    AGENT_NAME = "quality_repair"
    MODEL_ENV = core.QUALITY_REPAIR_MODEL_ENV

    @staticmethod
    def design_sweep_enabled() -> bool:
        return core._qor_design_sweep_enabled()

    def review_optimization_candidate(self, *, code, report, csim,
                                      skill_meta, synth_success):
        """Inspect candidate evidence without gold metrics or another model call."""
        orch = self.orch
        review = core.review_qor_candidate(
            parent_code=orch.hls_code or "", candidate_code=code,
            parent_report=orch.synth_report or {}, candidate_report=report or {},
            skill_meta=skill_meta, library=orch.skill_library,
            feasibility=core._paper_candidate_feasibility(
                report or {}, csim=csim,
                correctness_required=bool(orch.testbench_code),
                part=orch.part, clock_ns=orch.clock_ns,
            ),
            capacity=core._resource_capacity_for(orch.part), synth_success=synth_success,
        )
        if not hasattr(orch, "qor_profile_reviews"):
            orch.qor_profile_reviews = []
        orch.qor_profile_reviews.append(review)
        return review

    def _run_llm_repair(
        self,
        ground_truth_report: dict,
        initial_comparison: Optional[dict],
    ) -> dict:
        orch = self.orch
        summary = {"attempted": False, "applied": False, "attempts": []}
        current_comparison = initial_comparison or core.compare_reports(
            orch.synth_report, ground_truth_report,
        )
        quality_guidance = orch.feedback.render(
            "quality_gap",
            bench_name=orch.benchmark_name,
            report=orch.synth_report,
            ground_truth_report=ground_truth_report,
            comparison=current_comparison,
        )
        if not quality_guidance:
            summary["reason"] = "No timing/resource issues triggered quality repair"
            return summary

        summary["attempted"] = True
        summary["initial_score"] = core._quality_score(
            orch.benchmark_name, orch.synth_report, current_comparison,
        )
        summary["initial_comparison"] = current_comparison
        current_score = summary["initial_score"]

        for turn in range(orch.quality_repair_turns):
            current_focus = core._quality_focus(
                orch.benchmark_name, orch.synth_report, current_comparison,
            )
            prompt = core.hls_quality_repair.format(
                hls_code=orch.hls_code,
                current_report=core.format_report_summary(orch.synth_report),
                quality_context=core._build_quality_context(
                    orch.synth_report,
                    current_comparison,
                    part=orch.part,
                    clock_ns=orch.clock_ns,
                ),
                benchmark_context=orch.benchmark_context,
                quality_guidance=quality_guidance,
            )
            proposed_code = self._request_code_revision(prompt)
            attempt = {
                "turn": turn,
                "focus": current_focus,
                "score_before": current_score,
            }
            if not proposed_code:
                attempt["status"] = "no_code"
                summary["attempts"].append(attempt)
                continue

            candidate = orch._evaluate_candidate_with_repairs(
                proposed_code, "[Quality Repair]",
            )
            if not candidate.get("success"):
                attempt["status"] = "failed"
                attempt["error"] = candidate.get("error", "")
                summary["attempts"].append(attempt)
                continue

            candidate_comparison = core.compare_reports(
                candidate["report"], ground_truth_report,
            )
            candidate_score = core._quality_score(
                orch.benchmark_name, candidate["report"], candidate_comparison,
            )
            tests_preserved = (
                core._preserves_passed_test(orch.generated_csim, candidate.get("csim"))
                and core._preserves_passed_test(orch.generated_cosim, candidate.get("cosim"))
            )
            attempt.update({
                "candidate_score": candidate_score,
                "comparison": candidate_comparison,
                "tests_preserved": tests_preserved,
            })

            focus_improved = core._quality_focus_improved(
                orch.benchmark_name, current_focus,
                orch.synth_report, current_comparison,
                candidate["report"], candidate_comparison,
            )
            attempt["focus_improved"] = focus_improved

            if (tests_preserved and focus_improved
                    and candidate_score + core.QUALITY_SCORE_EPSILON < current_score):
                orch.hls_code = candidate["code"]
                orch.synth_report = candidate["report"]
                orch.generated_csim = candidate.get("csim")
                orch.generated_cosim = candidate.get("cosim")
                current_comparison = candidate_comparison
                current_score = candidate_score
                summary["applied"] = True
                attempt["status"] = "accepted"
                summary["attempts"].append(attempt)
                logging.info("[Quality Repair] Accepted improved candidate with score %.3f",
                             candidate_score)

                quality_guidance = orch.feedback.render(
                    "quality_gap",
                    bench_name=orch.benchmark_name,
                    report=orch.synth_report,
                    ground_truth_report=ground_truth_report,
                    comparison=current_comparison,
                )
                if not quality_guidance:
                    break
                continue

            attempt["status"] = "rejected"
            if not tests_preserved:
                attempt["rejection_reason"] = "Functional checks regressed"
            elif not focus_improved:
                attempt["rejection_reason"] = (
                    f"Candidate did not improve the active {current_focus} focus"
                )
            else:
                attempt["rejection_reason"] = (
                    f"Quality score did not improve enough "
                    f"({candidate_score:.3f} vs {current_score:.3f})"
                )
            summary["attempts"].append(attempt)

        summary["final_score"] = current_score
        summary["final_comparison"] = current_comparison
        return summary

    @staticmethod
    def _qor_candidate_record(
        payload: dict,
        evaluation: dict,
        *,
        fixed_knob_values: dict[str, Any],
    ) -> dict:
        report = evaluation.get("report") or {}
        feasibility = evaluation.get("feasibility") or {}
        event = evaluation.get("event") or {}
        selection_score = core.C2HLSOrchestrator._best_so_far_score(report)
        changed_knobs = list(payload.get("changed_knobs") or [])
        changed_ids = {
            item.get("knob_id") for item in changed_knobs if item.get("knob_id")
        }
        effective_values = dict(fixed_knob_values)
        for item in changed_knobs:
            if item.get("knob_id"):
                effective_values[item["knob_id"]] = item.get("to")
        return {
            "candidate_id": payload["candidate_id"],
            "stage": payload.get("stage"),
            "code_sha256": payload.get("code_sha256"),
            "source_diff_sha256": payload.get("source_diff_sha256"),
            "changed_knobs": changed_knobs,
            "fixed_knob_values": {
                knob_id: value
                for knob_id, value in fixed_knob_values.items()
                if knob_id not in changed_ids
            },
            "effective_knob_values": effective_values,
            "status": (
                "feasible"
                if feasibility.get("feasible") is True
                else event.get("failure_class") or "failed"
            ),
            "success": evaluation.get("success") is True,
            "feasible": feasibility.get("feasible") is True,
            "feasibility": dict(feasibility),
            "csim": core._sanitize_test_summary(evaluation.get("csim")),
            "metrics": core.extract_qor_metrics(report),
            "compiler_realization": core.compiler_knob_realization(
                report, changed_knobs
            ),
            "report_sha256": core.C2HLSOrchestrator._candidate_report_sha256(report),
            "selection_score": (
                selection_score if math.isfinite(selection_score) else None
            ),
            "error": str(evaluation.get("error") or ""),
            "qor_evaluation_index": event.get("qor_evaluation_index"),
        }

    def run_design_sweep(self) -> dict:
        """Run a frozen-parent, reference-blind local design experiment."""

        orch = self.orch
        summary: dict[str, Any] = {
            "schema_version": "c2hls.qor-design-sweep.v1",
            "method_version": core.QOR_METHOD_VERSION,
            "enabled": self.design_sweep_enabled(),
            "attempted": False,
            "applied": False,
            "reference_blind": True,
            "objective": "minimum feasible latency_cycles_worst",
            "cosim_policy": "not_run_during_design_sweep",
            "candidates": [],
        }
        if not summary["enabled"]:
            summary["reason"] = f"{core.QOR_DESIGN_SWEEP_ENV}=0"
            return summary
        if not orch.hls_code or not orch.synth_report:
            summary["reason"] = "missing parent code or CSynth report"
            return summary
        if not orch.testbench_code:
            summary["reason"] = "missing CSim testbench; refusing QoR mutation"
            return summary
        if not (
            isinstance(orch.generated_csim, dict)
            and orch.generated_csim.get("passed") is True
        ):
            summary["reason"] = "frozen parent lacks passing CSim evidence"
            return summary

        max_knobs = core._positive_env_int(core.QOR_SWEEP_MAX_KNOBS_ENV, 4)
        max_candidates = core._positive_env_int(core.QOR_SWEEP_MAX_CANDIDATES_ENV, 8)
        factor_values = core._positive_env_int_list(
            core.QOR_SWEEP_VALUES_ENV, "1,2,4,8,16"
        )
        ii_values = core._positive_env_int_list(
            core.QOR_SWEEP_II_VALUES_ENV, "1,2,4,8"
        )
        tile_values = core._positive_env_int_list(
            core.QOR_SWEEP_TILE_VALUES_ENV, "4,8,16,32,64"
        )
        widen_values = core._positive_env_int_list(
            core.QOR_SWEEP_WIDEN_VALUES_ENV, "64,128,256,512,1024"
        )
        exact_knob_name = os.getenv(core.QOR_SWEEP_KNOB_NAME_ENV, "").strip()
        interactions_enabled = core._env_flag(core.QOR_SWEEP_INTERACTIONS_ENV)
        max_interactions = (
            min(
                core._positive_env_int(core.QOR_SWEEP_MAX_INTERACTIONS_ENV, 2),
                max(0, max_candidates - 2),
            )
            if interactions_enabled
            else 0
        )
        ofat_budget = max_candidates - max_interactions
        parent_code = orch.hls_code
        parent_hash = core.qor_code_sha256(parent_code)
        parent_origin = dict(getattr(orch, "qor_parent_origin", {}) or {})
        origin_step = str(parent_origin.get("step_name") or "").strip().lower()
        origin_step = origin_step.replace("double_buffer", "doublebuffer")
        preferred_kinds = core.STEP_PREFERRED_KINDS.get(origin_step, ())
        knobs = core.discover_qor_knobs(
            parent_code,
            factor_values=factor_values,
            ii_values=ii_values,
            tile_values=tile_values,
            stream_depth_values=factor_values,
            widen_values=widen_values,
            max_knobs=(None if exact_knob_name else max_knobs),
            preferred_kinds=preferred_kinds,
        )
        if exact_knob_name:
            knobs = [knob for knob in knobs if knob.name == exact_knob_name][
                :max_knobs
            ]
        fixed_values = {
            knob.knob_id: (
                knob.current_value
                if knob.current_value is not None
                else knob.current_label
            )
            for knob in knobs
        }
        parent_feasibility = core._paper_candidate_feasibility(
            orch.synth_report,
            csim=orch.generated_csim,
            correctness_required=bool(orch.testbench_code),
            part=orch.part,
            clock_ns=orch.clock_ns,
        )
        parent_score = orch._best_so_far_score(orch.synth_report)
        parent_record = {
            "candidate_id": "frozen_parent",
            "stage": "parent",
            "code_sha256": parent_hash,
            "source_diff_sha256": None,
            "changed_knobs": [],
            "fixed_knob_values": fixed_values,
            "effective_knob_values": fixed_values,
            "status": "feasible" if parent_feasibility.get("feasible") else "infeasible",
            "success": True,
            "feasible": parent_feasibility.get("feasible") is True,
            "feasibility": parent_feasibility,
            "csim": core._sanitize_test_summary(orch.generated_csim),
            "metrics": core.extract_qor_metrics(orch.synth_report),
            "report_sha256": orch._candidate_report_sha256(orch.synth_report),
            "selection_score": parent_score if math.isfinite(parent_score) else None,
            "error": "",
            "qor_evaluation_index": None,
        }
        summary.update({
            "attempted": True,
            "parent_code_sha256": parent_hash,
            "parent_report_sha256": parent_record["report_sha256"],
            "parent_origin": parent_origin,
            "toolchain": {
                "vitis_version": getattr(orch, "vitis_version", ""),
                "part": orch.part,
                "clock_ns": orch.clock_ns,
            },
            "configuration": {
                "max_knobs": max_knobs,
                "max_candidates": max_candidates,
                "factor_values": list(factor_values),
                "ii_values": list(ii_values),
                "tile_values": list(tile_values),
                "widen_values": list(widen_values),
                "exact_knob_name": exact_knob_name or None,
                "interactions": interactions_enabled,
                "max_interactions": max_interactions,
                "preferred_knob_kinds": list(preferred_kinds),
            },
            "discovered_knobs": [knob.public() for knob in knobs],
            "parent": parent_record,
        })
        if not knobs:
            parent_record["pareto_frontier"] = parent_record["feasible"]
            summary["reason"] = "no safely parameterized QoR knobs discovered"
            summary.update({
                "candidate_count": 0,
                "feasible_candidate_count": 0,
                "budget_exhausted": False,
                "parent_integrity_checks": 0,
                "parent_integrity_passed": True,
                "pareto_candidate_ids": (
                    ["frozen_parent"] if parent_record["feasible"] else []
                ),
                "knob_trends": [],
                "winner_candidate_id": "frozen_parent",
                "winner_code_sha256": parent_hash,
                "winner_metrics": parent_record["metrics"],
                "winner_explanation": core.winner_explanation(
                    parent_record, parent_record,
                ),
                "fixed_parent_code_sha256_after_evaluation": parent_hash,
            })
            return summary

        candidate_payloads = core.build_ofat_candidates(
            parent_code,
            knobs,
            max_candidates=ofat_budget,
        )
        records: list[dict[str, Any]] = []
        skipped_payloads: list[dict[str, Any]] = []
        evaluated: dict[str, dict[str, Any]] = {}
        codes = {"frozen_parent": parent_code}
        parent_integrity_checks = 0
        budget_exhausted = False
        compiler_inactive_knobs: dict[str, dict[str, Any]] = {}

        def skip_if_compiler_inactive(payload: dict) -> bool:
            inactive = [
                compiler_inactive_knobs[item.get("knob_id")]
                for item in payload.get("changed_knobs") or []
                if item.get("knob_id") in compiler_inactive_knobs
            ]
            if not inactive:
                return False
            compact = {key: value for key, value in payload.items() if key != "code"}
            compact.update({
                "status": "skipped_compiler_inactive_knob",
                "reason": (
                    "Vitis previously reported that this frozen-parent knob is "
                    "overridden; additional values cannot control the synthesized "
                    "design."
                ),
                "compiler_realization": inactive,
            })
            skipped_payloads.append(compact)
            return True

        def evaluate_payload(payload: dict) -> Optional[dict]:
            nonlocal parent_integrity_checks, budget_exhausted
            parent_integrity_checks += 1
            if core.qor_code_sha256(orch.hls_code) != parent_hash:
                raise RuntimeError(
                    "QoR frozen-parent hash changed during candidate evaluation"
                )
            compact_metadata = {
                key: value for key, value in payload.items() if key != "code"
            }
            evaluation = orch._evaluate_qor_candidate(
                payload["code"],
                f"[QoR DOE: {payload['candidate_id']}]",
                compact_metadata,
            )
            record = self._qor_candidate_record(
                payload,
                evaluation,
                fixed_knob_values=fixed_values,
            )
            records.append(record)
            evaluated[payload["candidate_id"]] = evaluation
            codes[payload["candidate_id"]] = payload["code"]
            realization = record.get("compiler_realization") or {}
            for item in realization.get("knobs") or []:
                if item.get("overridden") is True and item.get("knob_id"):
                    compiler_inactive_knobs[str(item["knob_id"])] = dict(item)
            if evaluation.get("budget_exhausted"):
                budget_exhausted = True
            return record

        for payload in candidate_payloads:
            if skip_if_compiler_inactive(payload):
                continue
            evaluate_payload(payload)
            if budget_exhausted:
                break

        preferred_values: dict[str, int] = {}
        parent_rank = core.qor_selection_rank(parent_record)
        for knob in knobs:
            options = [
                record
                for record in records
                if record.get("feasible") is True
                and len(record.get("changed_knobs") or []) == 1
                and record["changed_knobs"][0].get("knob_id") == knob.knob_id
            ]
            if not options:
                continue
            best = min(options, key=core.qor_selection_rank)
            if core.qor_selection_rank(best) < parent_rank:
                preferred_values[knob.knob_id] = int(best["changed_knobs"][0]["to"])

        remaining = max(0, max_candidates - len(records))
        if interactions_enabled and not budget_exhausted and remaining > 0:
            interaction_payloads = core.build_interaction_candidates(
                parent_code,
                knobs,
                preferred_values,
                max_candidates=min(max_interactions, remaining),
            )
            for payload in interaction_payloads:
                if skip_if_compiler_inactive(payload):
                    continue
                evaluate_payload(payload)
                if budget_exhausted:
                    break

        if core.qor_code_sha256(orch.hls_code) != parent_hash:
            raise RuntimeError("QoR frozen-parent hash changed before selection")
        all_records = [parent_record, *records]
        frontier_ids = core.pareto_candidate_ids(all_records)
        for record in all_records:
            record["pareto_frontier"] = record["candidate_id"] in frontier_ids
        eligible = [
            record for record in all_records
            if record.get("feasible") is True
            and record.get("pareto_frontier") is True
        ]
        winner = min(
            eligible,
            key=core.qor_selection_rank,
            default=parent_record,
        )
        should_promote = (
            winner["candidate_id"] != "frozen_parent"
            and (
                parent_record.get("feasible") is not True
                or core.qor_selection_rank(winner) < parent_rank
            )
        )
        if should_promote:
            selected = evaluated[winner["candidate_id"]]
            orch.hls_code = codes[winner["candidate_id"]]
            orch.synth_report = selected["report"]
            orch.generated_csim = selected.get("csim")
            orch.generated_cosim = None
            summary["applied"] = True
        else:
            winner = parent_record

        knob_trends = core.summarize_knob_trends(
            records,
            knobs,
            parent=parent_record,
        )
        compiler_observed = sum(
            (record.get("compiler_realization") or {}).get("all_observed") is True
            for record in records
        )
        compiler_overridden = sum(
            (record.get("compiler_realization") or {}).get("any_overridden") is True
            for record in records
        )
        explanation = core.winner_explanation(parent_record, winner)
        if compiler_overridden:
            explanation += (
                f" Vitis reported that the requested knob was compiler-overridden "
                f"in {compiler_overridden}/{len(records)} evaluated candidates."
            )
        elif knob_trends and all(
            trend.get("observable_synthesis_response") is False
            for trend in knob_trends
        ):
            explanation += (
                " Source values changed, but the observed schedule, timing, latency, "
                "and resource metrics did not; this knob is inactive for the frozen "
                "parent under the tested toolchain."
            )

        summary.update({
            "candidates": records,
            "skipped_candidates": skipped_payloads,
            "candidate_count": len(records),
            "skipped_candidate_count": len(skipped_payloads),
            "feasible_candidate_count": sum(
                1 for record in records if record.get("feasible") is True
            ),
            "budget_exhausted": budget_exhausted,
            "parent_integrity_checks": parent_integrity_checks,
            "parent_integrity_passed": True,
            "pareto_candidate_ids": frontier_ids,
            "knob_trends": knob_trends,
            "compiler_realization_observed_candidate_count": compiler_observed,
            "compiler_override_candidate_count": compiler_overridden,
            "winner_candidate_id": winner["candidate_id"],
            "winner_code_sha256": winner["code_sha256"],
            "winner_metrics": winner["metrics"],
            "winner_explanation": explanation,
            "fixed_parent_code_sha256_after_evaluation": core.qor_code_sha256(
                parent_code
            ),
        })
        return summary

    def run(
        self,
        ground_truth_report: dict,
        initial_comparison: Optional[dict] = None,
    ) -> dict:
        orch = self.orch
        summary: dict[str, Any] = {
            "schema_version": "c2hls.quality-repair.v2",
            "attempted": False,
            "applied": False,
            "attempts": [],
            "design_sweep": {
                "enabled": self.design_sweep_enabled(),
                "attempted": False,
            },
        }
        orch.quality_repair_result = summary
        if not orch.synth_report or not orch.hls_code:
            summary["reason"] = "Quality repair missing code or synthesis report"
            return summary

        if orch.quality_repair_turns > 0 and ground_truth_report:
            llm_summary = self._run_llm_repair(
                ground_truth_report,
                initial_comparison,
            )
        else:
            llm_summary = {
                "attempted": False,
                "applied": False,
                "attempts": [],
                "reason": (
                    "LLM repair requires controller-visible comparison evidence"
                    if not ground_truth_report
                    else "LLM quality repair disabled"
                ),
            }
        summary["llm_repair"] = llm_summary
        summary["attempts"] = list(llm_summary.get("attempts") or [])
        summary["attempted"] = llm_summary.get("attempted") is True
        summary["applied"] = llm_summary.get("applied") is True
        for key in (
            "initial_score",
            "initial_comparison",
            "final_score",
            "final_comparison",
        ):
            if key in llm_summary:
                summary[key] = llm_summary[key]

        if self.design_sweep_enabled():
            design_summary = self.run_design_sweep()
            summary["design_sweep"] = design_summary
            summary["attempted"] = (
                summary["attempted"] or design_summary.get("attempted") is True
            )
            summary["applied"] = (
                summary["applied"] or design_summary.get("applied") is True
            )
        if not summary["attempted"]:
            summary["reason"] = (
                llm_summary.get("reason")
                or summary["design_sweep"].get("reason")
                or "Quality repair disabled"
            )
        summary["final_score"] = core._quality_score(
            orch.benchmark_name,
            orch.synth_report,
            (
                core.compare_reports(orch.synth_report, ground_truth_report)
                if ground_truth_report
                else {}
            ),
        )
        orch.quality_repair_result = summary
        return summary
