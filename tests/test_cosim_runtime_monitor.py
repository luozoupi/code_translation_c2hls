"""Observations must not turn log silence or simulator time into correctness."""
import json
import shutil

import pytest

import cosim_runtime_monitor as runtime
from cosim_runtime_monitor import CosimRuntimeMonitor, parse_log_line


class Clock:
    value = 100.0

    def __call__(self):
        return self.value

    def advance(self, seconds):
        self.value += seconds


@pytest.fixture
def setup(tmp_path):
    scratch = tmp_path / 'scratch'
    scratch.mkdir()
    clock = Clock()
    monitor = CosimRuntimeMonitor([scratch], tmp_path / 'observations.jsonl',
                                  clock=clock, stale_after_seconds=10)
    return scratch, clock, monitor


@pytest.mark.parametrize('unit,expected', [('ps', 12), ('ns', 12000), ('us', 12000000)])
def test_explicit_time_units(unit, expected):
    assert parse_log_line(f'Time: 12 {unit} Iteration: 10')['simulated_time_ps'] == expected
    assert 'simulated_time_ps' not in parse_log_line('Elapsed time: 12 seconds')


def test_transactions_distinguish_intra_percentage_and_unitless_time():
    item = parse_log_line('// RTL Simulation : 0 / 1 [51.13%] @ "113456"')
    assert item['transaction']['completed'] == 0
    assert item['transaction']['intra_percent'] == 51.13
    assert item['transaction']['reported_simulation_time_unit'] is None
    assert 'simulated_time_ps' not in item
    assert parse_log_line('WARNING: OPMODE Input Warning : [DSP48E1]')['dsp_opmode_warning_records'] == 1
    assert not parse_log_line('Time: 1 ps Process: prcs_opmode_drc').get('warning_records')


def test_incremental_reads_do_not_recount_and_preserve_partial_lines(setup):
    scratch, clock, monitor = setup
    log = scratch / 'xsim.log'
    log.write_bytes(b'WARNING: OPMODE Input Warning\nTime: 12 n')
    first = monitor.sample()
    assert first['logs'][0]['warning_records'] == 1
    assert first['logs'][0]['simulated_time_ps'] is None
    clock.advance(1)
    unchanged = monitor.sample()
    assert unchanged['bytes_read'] == 0
    assert unchanged['logs'][0]['warning_records'] == 1
    with log.open('ab') as stream:
        stream.write(b's\nTime: 13 ns\n')
    clock.advance(1)
    second = monitor.sample()
    assert second['bytes_read'] == len(b's\nTime: 13 ns\n')
    assert second['logs'][0]['simulated_time_ps'] == 13000
    assert second['progress'] == 'clock_advancing_transaction_progress_unknown'


def test_time_advancing_intra_percentage_and_cpu_never_claim_transaction(setup):
    scratch, clock, monitor = setup
    log = scratch / 'xsim.log'
    log.write_text('Time: 1 us\n// RTL Simulation : 0 / 2 [1.00%] @ "1000"\n')
    monitor.sample()
    clock.advance(2)
    with log.open('a') as stream:
        stream.write('Time: 2 us\n// RTL Simulation : 0 / 2 [9.00%] @ "2000"\n')
    sample = monitor.sample()
    assert sample['progress'] == 'clock_advancing_transaction_progress_unknown'
    clock.advance(20)
    stalled = monitor.sample()
    assert stalled['progress'] == 'no_recent_clock_advance_observed'
    assert stalled['deadlock_inferred'] is False
    with log.open('a') as stream:
        stream.write('// RTL Simulation : 1 / 2 [0.00%] @ "3000"\n')
    assert monitor.sample()['progress'] == 'completed_transaction_observed'


def test_large_logs_are_bounded_and_warning_counts_mark_skips(tmp_path):
    scratch = tmp_path / 'scratch'
    scratch.mkdir()
    log = scratch / 'xsim.log'
    log.write_text('Running: csynth_design\n' + 'WARNING: OPMODE Input Warning\n' * 10000 + 'Time: 9 ns\n')
    monitor = CosimRuntimeMonitor([scratch], tmp_path / 'samples.jsonl',
                                  max_bytes_per_sample=1024, max_bytes_per_file=1024)
    sample = monitor.sample()
    row = sample['logs'][0]
    assert sample['bytes_read'] <= 1024
    assert row['bytes_skipped'] > 200000
    assert 0 < row['dsp_opmode_warning_records'] < 10000
    assert row['simulated_time_ps'] == 9000
    assert monitor.sample()['bytes_read'] == 0
    summary = monitor.finish()
    assert 'incomplete' in summary['warning_count_semantics']


def test_rotation_and_truncation_do_not_compare_old_clock_or_completion(setup):
    scratch, clock, monitor = setup
    log = scratch / 'xsim.log'
    log.write_text('Time: 1 ns\nTime: 2 ns\n// RTL Simulation : 1 / 1 [100%]\n')
    monitor.sample()
    log.write_text('Time: 3 ns\n')  # truncate same inode
    clock.advance(1)
    sample = monitor.sample()
    assert sample['logs'][0]['generation'] == 1
    assert sample['logs'][0]['transaction'] is None
    assert sample['progress'] == 'no_recent_clock_advance_observed'
    log.rename(scratch / 'old.txt')
    log.write_text('Time: 4 ns\n')
    sample = monitor.sample()
    assert sample['logs'][0]['generation'] == 2
    assert sample['logs'][0]['last_clock_advance'] is None


def test_archival_disappearance_preserves_evidence_and_finish_is_idempotent(setup):
    scratch, clock, monitor = setup
    log = scratch / 'xsim.log'
    log.write_text('Running: csynth_design\nStarting XSIM ...\nTime: 10 ns\n')
    monitor.sample()
    shutil.rmtree(scratch)
    clock.advance(3)
    last = monitor.sample()
    assert last['scratch_dirs'][0]['present'] is False
    assert last['logs'][0]['present'] is False
    assert last['logs'][0]['simulated_time_ps'] == 10000
    summary = monitor.finish(outcome={'status': 'archived'})
    assert summary['outcome'] == {'status': 'archived'}
    assert len(summary['logs'][str(log)]['phase_events']) == 3
    assert monitor.finish() is summary
    with pytest.raises(RuntimeError):
        monitor.sample()
    records = [json.loads(line) for line in monitor.path.read_text().splitlines()]
    assert records[-1]['event'] == 'finished'
    assert json.loads(monitor.path.with_suffix('.summary.json').read_text()) == summary


def test_tool_completion_and_elapsed_are_evidence_not_independent_output_check(setup):
    scratch, clock, monitor = setup
    log = scratch / 'vitis.log'
    log.write_text('INFO: [HLS 200-111] Finished Command csynth_design CPU user time: 1.0 seconds. Elapsed time: 8.58 seconds;\n'
                   'INFO: [COSIM 212-1000] *** C/RTL co-simulation finished: PASS ***\n')
    monitor.sample()
    summary = monitor.finish()
    state = summary['logs'][str(log)]
    assert state['completion_marker'] == 'tool_reported_pass'
    assert state['command_elapsed'] == [{'command': 'csynth_design', 'seconds': 8.58}]
    assert summary['outcome'] is None


def test_budgets_rotate_so_noisy_logs_cannot_starve_other_logs(tmp_path):
    scratch = tmp_path / 'scratch'
    scratch.mkdir()
    paths = [scratch / f'{i}.log' for i in range(3)]
    for path in paths:
        path.write_text('Time: 1 ns\n' * 100)
    monitor = CosimRuntimeMonitor([scratch], tmp_path / 'samples.jsonl',
                                  max_bytes_per_file=64, max_bytes_per_sample=64)
    seen = set()
    for _ in range(3):
        sample = monitor.sample()
        assert sample['bytes_read'] <= 64
        seen.update(row['path'] for row in sample['logs'] if row['bytes_read_this_sample'])
        for path in paths:
            with path.open('a') as stream:
                stream.write('Time: 2 ns\n' * 100)
    assert seen == {str(p) for p in paths}


def test_live_root_identity_prevents_pid_reuse_and_collects_descendants(tmp_path, monkeypatch):
    def proc(pid, parent, start):
        return {'pid': pid, 'ppid': parent, 'start_ticks': start, 'cpu_ticks': 3,
                'name': 'xsim', 'state': 'R'}
    table = {123456: proc(123456, 1, 5), 123457: proc(123457, 123456, 6)}
    monkeypatch.setattr(runtime, '_process_table', lambda: table)
    monitor = CosimRuntimeMonitor([tmp_path / 'absent'], tmp_path / 'samples.jsonl', root_pids=[123456])
    assert {p['pid'] for p in monitor.sample()['live_processes']} == {123456, 123457}
    table.clear()
    table[123456] = proc(123456, 1, 99)
    assert monitor.sample()['live_processes'] == []


def test_never_append_to_existing_attempt_or_follow_symlink_log(tmp_path):
    scratch = tmp_path / 'scratch'
    scratch.mkdir()
    outside = tmp_path / 'secret.log'
    outside.write_text('Time: 99 ns\n')
    (scratch / 'alias.log').symlink_to(outside)
    target = tmp_path / 'samples.jsonl'
    monitor = CosimRuntimeMonitor([scratch], target)
    assert monitor.sample()['logs'] == []
    with pytest.raises(FileExistsError):
        CosimRuntimeMonitor([scratch], target)


def test_explicit_negative_deadlock_message_is_not_diagnostic():
    assert not parse_log_line('No deadlock detected').get('explicit_deadlock_diagnostic')
    assert parse_log_line('ERROR: Deadlock detected in dataflow')['explicit_deadlock_diagnostic']


def test_phase_windows_are_observations_not_exact_durations(setup):
    scratch, clock, monitor = setup
    log = scratch / 'vitis.log'
    log.write_text('Running: csynth_design\n')
    monitor.sample()
    clock.advance(4)
    monitor.sample()
    clock.advance(6)
    with log.open('a') as stream:
        stream.write('Starting XSIM ...\n')
    monitor.sample()
    summary = monitor.finish()
    first, second = summary['logs'][str(log)]['phase_intervals']
    assert first['start_observation_window_seconds'] == [None, 0]
    assert first['end_observation_window_seconds'] == [4, 10]
    assert first['duration_observation_bounds_seconds'] == [4, None]
    assert second['end_observation_window_seconds'] is None


def test_missing_scratch_and_late_creation_are_safe(setup):
    scratch, clock, monitor = setup
    scratch.rmdir()
    assert monitor.sample()['progress'] == 'unknown_no_simulation_time_evidence'
    scratch.mkdir()
    log = scratch / 'xsim.log'
    content = 'Time: 1 ns\n'
    log.write_text(content)
    monitor.sample()
    monitor.finish(outcome={'status': 'timeout'})
    assert log.read_text() == content
