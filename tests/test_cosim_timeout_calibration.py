"""Timeout calibration must preserve censoring and measurement scope."""
import json
from pathlib import Path

import pytest

from scripts.calibrate_sigmetrics_cosim_timeout import (
    build_report, diagnostic_policy, parse_phase_log, pass_only_statistics,
    simulation_clock_rate, timing_observation,
)


def observation(status, seconds, cycles=100):
    return timing_observation({'id': 'v1', 'benchmark': 'bench', 'status': status},
                              {'cosim': {'status': status, 'runtime_seconds': seconds,
                                         'kernel_runtime_cycles': cycles}}, 10800)


def test_timeout_is_censored_not_exact_completion_time():
    value = observation('timeout', 10805, None)
    assert value['completion_time_seconds'] is None
    assert value['completion_time_right_censored']
    assert value['cosim_flow_timeout_lower_bound_seconds'] == 10800
    assert value['observed_validation_wall_seconds'] == 10805


def test_kernel_cycles_cannot_be_used_as_simulator_clock_rate():
    value = observation('pass', 7000, 32152)
    assert value['measured_kernel_latency_cycles'] == 32152
    assert value['simulator_clock_rate_per_wall_second'] is None
    with pytest.raises(ValueError, match='total_simulated_clocks'):
        simulation_clock_rate({'kernel_runtime_cycles': 32152, 'runtime_seconds': 7000})
    assert simulation_clock_rate({'total_simulated_clocks': 1000, 'simulation_wall_seconds': 5}) == 200


def test_pass_only_tail_is_explicitly_conditional():
    result = pass_only_statistics([observation('pass', 100), observation('pass', 200),
                                   observation('timeout', 10805)])
    assert result['count'] == 2
    assert result['median_seconds'] == 150
    assert result['p95_nearest_rank_seconds'] == 200
    assert result['suitable_as_universal_timeout'] is False


def test_phase_parser_retains_nested_times_and_transaction_evidence():
    result = parse_phase_log(iter([
        'INFO: Finished Command csynth_design CPU user time: 8 seconds. Elapsed time: 16.69 seconds;\n',
        'INFO: Finished Command cosim_design CPU user time: 50 seconds. Elapsed time: 6930.56 seconds;\n',
        'run: Time (s): cpu = 00:00:14 ; elapsed = 01:54:01 . Memory (MB): peak = 1617\n',
        '// RTL Simulation : 0 / 1 [0.00%] @ "119000"\n',
        'Warning: OPMODE Input Warning : invalid\n',
        'Time: 188618150 ps Iteration: 11\n',
        '// RTL Simulation : 1 / 1 [100.00%] @ "107513000"\n',
    ]))
    assert result['hls_command_wall_seconds']['csynth_design'] == [16.69]
    assert result['xsim_run_wall_seconds'] == [6841]
    assert result['phase_times_additive'] is False
    assert result['completed_transactions_observed'] == 1
    assert result['dsp_opmode_warning_count'] == 1


def test_clock_advance_does_not_imply_transaction_progress():
    result = parse_phase_log(['Time: 100 ps\n', 'Time: 200 ns\n'])
    assert result['last_observed_simulation_timestamp_ps'] == 200000
    assert result['completed_transactions_observed'] is None
    assert result['clock_advance_proves_useful_progress'] is False


def test_six_hour_rule_is_diagnostic_and_preserves_originals():
    result = diagnostic_policy()
    assert result['per_case_cosim_flow_cap_seconds'] == 21600
    assert result['same_cap_for_every_selected_case']
    assert result['preserve_original_measurements']
    assert 'not a calibrated' in result['status']
    assert 'separately' in result['denominator']


def test_report_accounts_for_missing_measurements_without_mutating_input(tmp_path):
    replay = tmp_path / 'replay'
    measurements = replay / 'historical_timeout_replay_20261001/raw/measurements'
    measurements.mkdir(parents=True)
    (replay / 'manifest.json').write_text(json.dumps({'timeout_seconds': 10800}))
    tasks = [{'id': str(i), 'benchmark': 'bench', 'status': status, 'kind': 'historical'}
             for i, status in enumerate(['pass', 'timeout', 'running', 'pending'])]
    (replay / 'state.json').write_text(json.dumps({'tasks': tasks}))
    for i, status in enumerate(['pass', 'timeout']):
        (measurements / f'{i}.json').write_text(json.dumps({'validation_id': str(i),
            'cosim': {'status': status, 'runtime_seconds': 100 if i == 0 else 10805}}))
    before = {p: p.read_bytes() for p in replay.rglob('*') if p.is_file()}
    report = build_report(replay, phase_ids=[])
    assert report['replay_summary']['saved_outcome_counts'] == {'pass': 1, 'timeout': 1}
    assert report['replay_summary']['without_saved_measurement'] == 2
    assert report['replay_summary']['state_liveness_checked'] is False
    assert report['calibration']['status'] == 'insufficient_evidence'
    assert report['eligible_diagnostic_validation_ids'] == ['1']
    assert {p: p.read_bytes() for p in replay.rglob('*') if p.is_file()} == before
