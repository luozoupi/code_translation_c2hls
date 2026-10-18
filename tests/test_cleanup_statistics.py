import copy

import pytest

from hlsmith.research.cleanup_statistics import summarize_cleanup_statistics


KERNELS = tuple(f'kernel{i}' for i in range(26))
ARMS = tuple(f'arm{i}' for i in range(7))
COHORTS = ('frozen-paper', 'clean-paper', 'clean-researcher')


def rows_for(outcome=None, cycles=None):
    rows = []
    for kernel in KERNELS:
        for arm in ARMS:
            for repeat in (1, 2, 3):
                for cohort in COHORTS:
                    passed = True if outcome is None else outcome(kernel, arm, repeat, cohort)
                    rows.append({'benchmark': kernel, 'arm': arm, 'repeat': repeat, 'cohort': cohort,
                                 'status': 'passed' if passed else 'failed_csim', 'rtl_pass': passed,
                                 'rtl_cycles': (100 if cycles is None else cycles(kernel, arm, repeat, cohort)) if passed else None})
    return rows


def summarize(rows, **kwargs):
    return summarize_cleanup_statistics(rows, kernels=KERNELS, arms=ARMS, repeats=(1, 2, 3), cohorts=COHORTS,
                   paired_comparisons=(COHORTS[:2], COHORTS[1:]), bootstrap_seed=20261018, **kwargs)


def test_complete_balanced_cohorts_preserve_denominators_and_ratio_direction():
    rows = rows_for(cycles=lambda k, a, r, c: {'frozen-paper': 100, 'clean-paper': 50, 'clean-researcher': 25}[c])
    report = summarize(rows)
    assert report['complete']
    assert not report['equivalence_claim'] and not report['automatic_promotion']
    for cohort in report['cohorts'].values():
        assert cohort['registered_cells'] == cohort['observed_outcomes'] == cohort['validation_passes'] == 546
    for comparison in report['comparisons']:
        assert comparison['registered_pairs'] == comparison['jointly_passing_latency_pairs'] == 546
        assert comparison['validation_pass_difference_ci95'] == [0, 0]
        assert comparison['geomean_rtl_cycle_ratio_baseline_over_target'] == pytest.approx(2)
        assert comparison['geomean_rtl_cycle_ratio_ci95'] == pytest.approx([2, 2])


def test_missing_or_pending_cell_is_never_a_failure_or_an_equivalence_result():
    rows = rows_for()
    rows.pop()
    rows[0].update(status='pending', rtl_pass=False, rtl_cycles=None)
    report = summarize(rows, draws=200)
    assert report['conclusion'] == 'inconclusive_incomplete_study'
    assert report['cohorts']['frozen-paper']['unobserved_cells'] == 1
    assert report['cohorts']['clean-researcher']['missing_rows'] == 1
    assert report['cohorts']['frozen-paper']['registered_validation_pass_fraction'] is None
    assert all(c['conclusion'] == 'inconclusive_incomplete_study' for c in report['comparisons'])
    assert report['comparisons'][0]['observed_outcome_pairs'] == 545
    assert report['comparisons'][0]['validation_pass_difference_target_minus_baseline'] == 0


def test_measured_resource_and_timing_failures_remain_in_observed_denominator():
    rows = rows_for()
    rows[0].update(status='failed_feasibility', rtl_pass=False, rtl_cycles=None)
    rows[1].update(status='unqualified_rtl_synthesis', rtl_pass=False, rtl_cycles=None)
    report = summarize(rows, draws=100)
    assert report['complete']
    assert report['cohorts']['frozen-paper']['observed_outcomes'] == 546
    assert report['cohorts']['frozen-paper']['validation_passes'] == 545
    assert report['cohorts']['clean-paper']['validation_passes'] == 545


def test_bootstrap_resamples_kernel_clusters_not_546_individual_cells():
    # All21 cells within a kernel move together. Sampling cells independently
    # would produce a much narrower interval than this kernel-level effect.
    rows = rows_for(outcome=lambda k, a, r, c: c == 'frozen-paper' or int(k[6:]) < 13)
    report = summarize(rows)
    comparison = report['comparisons'][0]
    assert comparison['validation_pass_difference_target_minus_baseline'] == -.5
    lower, upper = comparison['validation_pass_difference_ci95']
    assert lower < -.60 and upper > -.40
    assert comparison['jointly_passing_latency_pairs'] == 13 * 21
    assert comparison['latency_kernel_clusters'] == 13
    assert 'condition' in comparison['latency_scope'].lower()


def test_bootstrap_seed_is_reproducible_and_has_only_offline_scope():
    rows = rows_for(cycles=lambda k, a, r, c: 100 + int(k[6:]) * (1 if c == 'frozen-paper' else 2))
    assert summarize(rows, draws=200) == summarize(rows, draws=200)
    assert 'hosted model seeds are unsupported' in summarize(rows, draws=100)['bootstrap']['seed_scope']


def test_latency_requires_positive_finite_measurements_and_sufficient_clusters():
    rows = rows_for()
    for row in rows:
        if row['benchmark'] != KERNELS[0]:
            row['rtl_cycles'] = float('nan')
    report = summarize(rows, draws=100)
    assert report['cohorts']['frozen-paper']['validation_passes'] == 546
    assert report['cohorts']['frozen-paper']['latency_eligible_cells'] == 21
    assert report['comparisons'][0]['latency_kernel_clusters'] == 1
    assert report['comparisons'][0]['geomean_rtl_cycle_ratio_ci95'] is None


def test_duplicate_cells_and_invalid_resampling_configuration_fail_explicitly():
    rows = rows_for()
    with pytest.raises(ValueError, match='duplicate'):
        summarize(rows + [copy.deepcopy(rows[0])], draws=100)
    with pytest.raises(ValueError, match='bootstrap draws'):
        summarize(rows, draws=0)
