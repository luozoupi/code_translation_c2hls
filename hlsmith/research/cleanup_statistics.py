"""Paired, kernel-clustered uncertainty for the prospective cleanup study."""
from __future__ import annotations

from collections import Counter
import math
import random


_TERMINAL = {
    "passed", "generation_nonpass", "blocked_input", "failed_native",
    "failed_csim", "failed_rtl", "csim_timeout", "rtl_timeout",
    "generation_timeout", "validation_timeout",
    "failed_feasibility", "unqualified_rtl_synthesis",
}
_CONDITIONAL_QOR = (
    "Latency ratios condition on both selected designs passing validation and "
    "having positive measured RTL cycles. This selected subset can differ by "
    "cohort and excludes nonpasses; its QoR comparison does not estimate an "
    "unconditional improvement or establish equivalence."
)


def _interval(values):
    if not values:
        return None
    ordered = sorted(values)
    def quantile(probability):
        index = (len(ordered) - 1) * probability
        left = int(index)
        fraction = index - left
        return ordered[left] * (1 - fraction) + ordered[min(left + 1, len(ordered) - 1)] * fraction
    return [quantile(.025), quantile(.975)]


def _observed(row):
    if row is None or row.get("status") not in _TERMINAL:
        return False
    passed = row.get("rtl_pass")
    return isinstance(passed, bool) and passed == (row["status"] == "passed")


def _eligible(row):
    if not _observed(row) or not row["rtl_pass"]:
        return False
    cycles = row.get("rtl_cycles")
    return isinstance(cycles, (int, float)) and not isinstance(cycles, bool) and math.isfinite(cycles) and cycles > 0


def summarize_cleanup_statistics(rows, *, kernels, arms, repeats, cohorts,
                                 paired_comparisons, bootstrap_seed, draws=1000):
    """Summarize a registered matrix without coding unfinished cells as failures.

    Rows use ``benchmark``, ``arm``, ``repeat``, ``cohort``, ``status``,
    ``rtl_pass`` and ``rtl_cycles``. Eligibility means a verified validation pass
    with positive finite measured cycles, not an additional device-fit claim.
    Outcome differences are target-minus-baseline validation-pass indicators;
    log cycle ratios are baseline/target. Every bootstrap draw samples the full
    registered set of kernel clusters, retaining all their arms and repeats.
    """
    kernels, arms, repeats, cohorts = map(tuple, (kernels, arms, repeats, cohorts))
    if any(not axis or len(axis) != len(set(axis)) for axis in (kernels, arms, repeats, cohorts)):
        raise ValueError("matrix axes must be nonempty and unique")
    if isinstance(draws, bool) or not isinstance(draws, int) or not 100 <= draws <= 10000:
        raise ValueError("bootstrap draws must be an integer from 100 to 10000")
    if isinstance(bootstrap_seed, bool) or not isinstance(bootstrap_seed, int):
        raise ValueError("an explicit integer seed is required for offline resampling")
    lookup = {}
    for row in rows:
        key = (row["benchmark"], row["arm"], row["repeat"], row["cohort"])
        if any(value not in axis for value, axis in zip(key, (kernels, arms, repeats, cohorts))):
            raise ValueError("row is outside the registered matrix")
        if key in lookup:
            raise ValueError("duplicate registered cell")
        lookup[key] = row
    per_cohort = len(kernels) * len(arms) * len(repeats)
    summaries = {}
    all_complete = True
    for cohort in cohorts:
        registered = [lookup.get((kernel, arm, repeat, cohort))
                      for kernel in kernels for arm in arms for repeat in repeats]
        observed = [row for row in registered if _observed(row)]
        passed = sum(row["rtl_pass"] for row in observed)
        complete = len(observed) == per_cohort
        all_complete &= complete
        summaries[cohort] = {
            "registered_cells": per_cohort, "observed_outcomes": len(observed),
            "unobserved_cells": per_cohort - len(observed),
            "missing_rows": sum(row is None for row in registered),
            "validation_passes": passed, "latency_eligible_cells": sum(_eligible(row) for row in observed),
            "status_counts": dict(Counter(row.get("status", "unknown") if row else "missing" for row in registered)),
            "registered_validation_pass_fraction": passed / per_cohort if complete else None,
            "complete": complete,
        }
    comparisons = []
    generator = random.Random(bootstrap_seed)
    for baseline, target in paired_comparisons:
        if baseline not in cohorts or target not in cohorts or baseline == target:
            raise ValueError("paired comparisons require distinct registered cohorts")
        cluster_summaries = []
        for kernel in kernels:
            differences, log_ratios = [], []
            for arm in arms:
                for repeat in repeats:
                    left = lookup.get((kernel, arm, repeat, baseline))
                    right = lookup.get((kernel, arm, repeat, target))
                    if _observed(left) and _observed(right):
                        differences.append(int(right["rtl_pass"]) - int(left["rtl_pass"]))
                    if _eligible(left) and _eligible(right):
                        log_ratios.append(math.log(left["rtl_cycles"]) - math.log(right["rtl_cycles"]))
            cluster_summaries.append((sum(differences), len(differences), sum(log_ratios), len(log_ratios)))
        observed_pairs = sum(row[1] for row in cluster_summaries)
        passing_pairs = sum(row[3] for row in cluster_summaries)
        outcome_clusters = sum(row[1] > 0 for row in cluster_summaries)
        latency_clusters = sum(row[3] > 0 for row in cluster_summaries)
        difference = sum(row[0] for row in cluster_summaries) / observed_pairs if observed_pairs else None
        log_ratio = sum(row[2] for row in cluster_summaries) / passing_pairs if passing_pairs else None
        outcome_draws, latency_draws = [], []
        for _ in range(draws):
            sample = [cluster_summaries[generator.randrange(len(kernels))] for _ in kernels]
            count = sum(row[1] for row in sample)
            if count and outcome_clusters >= 2:
                outcome_draws.append(sum(row[0] for row in sample) / count)
            count = sum(row[3] for row in sample)
            if count and latency_clusters >= 2:
                latency_draws.append(sum(row[2] for row in sample) / count)
        log_interval = _interval(latency_draws)
        comparisons.append({
            "baseline": baseline, "target": target, "registered_pairs": per_cohort,
            "observed_outcome_pairs": observed_pairs, "unobserved_pairs": per_cohort - observed_pairs,
            "validation_pass_difference_target_minus_baseline": difference,
            "validation_pass_difference_ci95": _interval(outcome_draws),
            "observed_outcome_kernel_clusters": outcome_clusters,
            "jointly_passing_latency_pairs": passing_pairs, "latency_kernel_clusters": latency_clusters,
            "mean_log_rtl_cycle_ratio_baseline_over_target": log_ratio,
            "mean_log_rtl_cycle_ratio_ci95": log_interval,
            "geomean_rtl_cycle_ratio_baseline_over_target": math.exp(log_ratio) if log_ratio is not None else None,
            "geomean_rtl_cycle_ratio_ci95": [math.exp(value) for value in log_interval] if log_interval else None,
            "valid_outcome_bootstrap_draws": len(outcome_draws), "valid_latency_bootstrap_draws": len(latency_draws),
            "conclusion": "descriptive_no_equivalence_claim" if all_complete else "inconclusive_incomplete_study",
            "outcome_scope": "observed paired validation-pass indicators; pending/missing cells are not failures",
            "latency_scope": _CONDITIONAL_QOR,
        })
    return {
        "schema_version": "hlsmith.cleanup-statistics.v1", "cohorts": summaries,
        "comparisons": comparisons, "complete": all_complete,
        "conclusion": "descriptive_no_equivalence_claim" if all_complete else "inconclusive_incomplete_study",
        "bootstrap": {"method": "paired percentile cluster bootstrap by kernel, preserving all arms/repeats within each sampled kernel",
                      "registered_kernel_clusters": len(kernels), "draws": draws, "confidence": .95,
                      "seed": bootstrap_seed, "seed_scope": "offline resampling only; hosted model seeds are unsupported",
                      "minimum_observed_clusters_for_interval": 2},
        "eligibility_definition": "verified validation pass and positive finite measured RTL cycles; no additional device-fit claim",
        "equivalence_claim": False, "automatic_promotion": False,
        "qualification": _CONDITIONAL_QOR,
    }
