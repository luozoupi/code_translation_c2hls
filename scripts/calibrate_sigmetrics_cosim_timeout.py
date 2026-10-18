#!/usr/bin/env python3
"""Audit frozen co-simulation timing without rerunning or modifying experiments.

This is an evidence report, not a fitted timeout predictor. Only selected archived
logs are streamed to recover phase timings; archives are never unpacked to disk.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import re
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPLAY = ROOT / 'results_sweeps/sigmetrics_timeout_replay_20261001'
DEFAULT_HISTORY = ROOT / 'results_sweeps/historical_registry_recovery_20260928'
TERMINAL = {'pass', 'fail', 'timeout', 'not_run'}
PHASE_PATTERN = re.compile(r'Finished Command (\w+).*?Elapsed time:\s*([\d.]+) seconds')
XSIM_WALL_PATTERN = re.compile(r'^run: Time \(s\):.*?elapsed\s*=\s*(\d+):(\d+):(\d+)')
PROGRESS_PATTERN = re.compile(r'RTL Simulation\s*:\s*(\d+)\s*/\s*(\d+)')
SIM_TIME_PATTERN = re.compile(r'\bTime:\s*([\d.]+)\s*(fs|ps|ns|us|ms|s)\b')
UNITS_PS = {'fs': .001, 'ps': 1, 'ns': 1000, 'us': 1e6, 'ms': 1e9, 's': 1e12}


def read_json(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def positive(value):
    if isinstance(value, bool):
        return None
    try:
        value = float(value)
    except (ValueError, TypeError):
        return None
    return value if math.isfinite(value) and value > 0 else None


def timing_observation(task, measurement, timeout_seconds):
    """Kernel latency is descriptive; it never becomes simulator throughput."""
    cosim = measurement.get('cosim') or {}
    status = cosim.get('status') or ('pass' if cosim.get('passed') is True else task['status'])
    observed = positive(cosim.get('runtime_seconds'))
    return {
        'validation_id': task['id'], 'prior_validation_id': task.get('prior_id'),
        'benchmark': task['benchmark'], 'status': status,
        'measurement_path': task.get('measurement_path'),
        'synthesis_estimated_kernel_cycles': positive(task.get('estimate_cycles')),
        'measured_kernel_latency_cycles': positive(cosim.get('kernel_runtime_cycles')),
        'observed_validation_wall_seconds': observed,
        'completion_time_seconds': observed if status == 'pass' else None,
        'completion_time_right_censored': status == 'timeout',
        'cosim_flow_timeout_lower_bound_seconds': timeout_seconds if status == 'timeout' else None,
        'censoring_scope': ('co-sim tool flow including synthesis, setup, and RTL simulation'
                            if status == 'timeout' else None),
        'observed_wall_scope': 'native golden + candidate C simulation + complete co-sim call; excludes archive creation',
        'total_simulated_clocks': None,
        'simulator_clock_rate_per_wall_second': None,
        'archive': cosim.get('work_dir'),
    }


def simulation_clock_rate(sample):
    """Only complete-clock observations with separately timed simulation qualify.

    Intentionally does not accept kernel_runtime_cycles/runtime_seconds as a
    substitute: those quantities describe different measurement scopes.
    """
    clocks = positive(sample.get('total_simulated_clocks'))
    wall = positive(sample.get('simulation_wall_seconds'))
    if clocks is None or wall is None:
        raise ValueError('measured total_simulated_clocks and simulation_wall_seconds are required')
    return clocks / wall


def pass_only_statistics(observations):
    values = sorted(x['completion_time_seconds'] for x in observations
                    if x['status'] == 'pass' and x['completion_time_seconds'] is not None)
    return {
        'population': 'observed passes only, conditional on completion within the existing cap',
        'count': len(values),
        'min_seconds': min(values) if values else None,
        'median_seconds': statistics.median(values) if values else None,
        'p95_nearest_rank_seconds': values[math.ceil(.95 * len(values)) - 1] if values else None,
        'max_seconds': max(values) if values else None,
        'suitable_as_universal_timeout': False,
        'reason': 'Unfinished and timed-out designs are excluded; the pass-only tail cannot estimate their completion times.',
    }


def parse_phase_log(lines):
    """Parse streaming lines with constant memory; overlapping phases are not summed."""
    phases = collections.defaultdict(list)
    wall = []
    warnings = 0
    last_time_ps = None
    progress = []
    deadlock = []
    line_count = 0
    for line in lines:
        line_count += 1
        match = PHASE_PATTERN.search(line)
        if match:
            phases[match[1]].append(float(match[2]))
        match = XSIM_WALL_PATTERN.search(line)
        if match:
            hours, minutes, seconds = map(int, match.groups())
            wall.append(hours * 3600 + minutes * 60 + seconds)
        if 'OPMODE Input Warning' in line:
            warnings += 1
        match = SIM_TIME_PATTERN.search(line)
        if match:
            stamp = float(match[1]) * UNITS_PS[match[2]]
            last_time_ps = stamp if last_time_ps is None else max(last_time_ps, stamp)
        match = PROGRESS_PATTERN.search(line)
        if match:
            pair = [int(match[1]), int(match[2])]
            if not progress or pair != progress[-1]:
                progress.append(pair)
                progress = progress[-16:]
        if re.search(r'(?:deadlock\s+(?:detected|found)|detected\s+(?:a\s+)?deadlock)', line, re.I):
            deadlock.append(line.strip()[:500])
            deadlock = deadlock[-4:]
    return {
        'line_count': line_count, 'hls_command_wall_seconds': dict(phases),
        'xsim_run_wall_seconds': wall, 'dsp_opmode_warning_count': warnings,
        'last_observed_simulation_timestamp_ps': last_time_ps,
        'transaction_progress': progress,
        'completed_transactions_observed': max((x[0] for x in progress), default=None),
        'explicit_deadlock_diagnostics': deadlock,
        'clock_advance_proves_useful_progress': False,
        'phase_times_additive': False,
        'note': 'Command and simulator times overlap. Warning counts and clock advance do not establish forward transaction progress or rule out deadlock.',
    }


def archived_phase_evidence(observation):
    archive = Path(observation['archive'] or '')
    if not archive.is_file() or not archive.name.endswith('.tar.zst'):
        return {'validation_id': observation['validation_id'], 'available': False}
    validation_id = observation['validation_id']
    evidence = {'validation_id': validation_id, 'benchmark': observation['benchmark'],
                'archive': str(archive), 'available': True, 'logs': {}}
    receipt = archive.with_suffix('').with_suffix('.tar.receipt.json')
    if receipt.is_file():
        evidence['archive_receipt_path'] = str(receipt)
        evidence['archive_receipt_sha256'] = sha(receipt)
        evidence['archived_sha256_from_receipt'] = read_json(receipt).get('sha256')
    for suffix in ('cosim.log', 'cosim/hls_proj/sol1/sim/verilog/xsim.log'):
        member = f'{validation_id}/{suffix}'
        # tar writes just the requested member to stdout, never filesystem paths.
        with subprocess.Popen(['tar', '--zstd', '-xOf', str(archive), member],
                              stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                              text=True, errors='replace') as process:
            parsed = parse_phase_log(process.stdout)
            parsed['archive_member'] = member
            parsed['read_returncode'] = process.wait()
            evidence['logs'][suffix] = parsed
    return evidence


def diagnostic_policy(prior_cap=10800):
    return {
        'name': 'separate fixed-budget timeout sensitivity',
        'status': 'predeclared diagnostic escalation, not a calibrated runtime prediction',
        'prior_cap_seconds': prior_cap,
        'per_case_cosim_flow_cap_seconds': 2 * prior_cap,
        'cap_selection_reason': 'Double the prior three-hour cap for a six-hour diagnostic attempt on right-censored cases; existing evidence does not identify a correct timeout.',
        'cohort': 'Validation identities with saved terminal three-hour timeout measurements at policy freeze; freeze the selected identity list before launch.',
        'same_cap_for_every_selected_case': True,
        'required_matches': ['candidate source hash', 'testbench/input hashes', 'golden output and tolerances',
                             'frozen validation source/harness', 'Vitis version', 'FPGA part', 'clock', 'trace/logging options'],
        'only_experimental_change': 'wall-clock co-sim cap; collect observational phase/progress telemetry without changing generated design or testbench',
        'preserve_original_measurements': True,
        'separate_result_directory_and_validation_ids': True,
        'denominator': 'All identities declared in this diagnostic cohort, reported separately from the frozen main-study and replay denominators.',
        'interrupted_runs': 'Record interruption separately; incomplete attempts are not saved timeout results.',
        'progress_policy': 'Record phase, completed/total transactions, timestamps, diagnostics, and host load. Clock/log activity alone does not prove useful progress. Do not extend or shorten the fixed cap selectively based on this telemetry.',
        'concurrency': 'Record worker count, competing jobs and host load; calibrate any later rule at the intended concurrency.',
    }


def build_report(replay, history=None, phase_ids=None):
    replay = Path(replay)
    manifest_path, state_path = replay / 'manifest.json', replay / 'state.json'
    manifest, state = read_json(manifest_path), read_json(state_path)
    prior_cap = manifest['timeout_seconds']
    tasks = [t for t in state['tasks'] if t['kind'] == 'historical']
    observations, issues = [], []
    input_hashes = {str(manifest_path): sha(manifest_path), str(state_path): sha(state_path)}
    for task in tasks:
        path = Path(task.get('measurement_path') or replay / 'historical_timeout_replay_20261001' / 'raw' / 'measurements' / (task['id'] + '.json'))
        if not path.is_file():
            if task['status'] in TERMINAL:
                issues.append(f'Terminal state without saved measurement: {task["id"]}')
            continue
        measurement = read_json(path)
        digest = sha(path)
        if measurement.get('validation_id') != task['id']:
            raise ValueError('measurement identity mismatch: ' + str(path))
        if task.get('measurement_sha256') and task['measurement_sha256'] != digest:
            raise ValueError('measurement hash mismatch: ' + str(path))
        item = timing_observation({**task, 'measurement_path': str(path)}, measurement, prior_cap)
        observations.append(item)
        input_hashes[str(path)] = digest
    counts = dict(collections.Counter(x['status'] for x in observations))
    eligible = [x for x in observations if x['status'] == 'timeout']
    if phase_ids is None:
        phase_ids = [next((x['validation_id'] for x in observations if x['status'] == status), None)
                     for status in ('pass', 'timeout')]
    index = {x['validation_id']: x for x in observations}
    phases = [archived_phase_evidence(index[i]) for i in phase_ids if i is not None]
    pilot = next((x for x in phases if index[x['validation_id']]['status'] == 'timeout' and
                  x.get('logs', {}).get('cosim/hls_proj/sol1/sim/verilog/xsim.log', {}).get('line_count')), None)
    historical = None
    if history is not None:
        original_path = Path(history) / 'manifest.json'
        original = read_json(original_path)
        input_hashes[str(original_path)] = sha(original_path)
        ids = {r['validation_id'] for r in original['rows'] if r.get('inventory_status') == 'queued'}
        original_counts = collections.Counter()
        for identity in sorted(ids):
            path = Path(original['raw_root']) / 'measurements' / (identity + '.json')
            if path.is_file():
                original_counts[(read_json(path).get('cosim') or {}).get('status', 'unknown')] += 1
                input_hashes[str(path)] = sha(path)
            else:
                original_counts['missing_measurement'] += 1
        historical = {'unique_validation_denominator': len(ids), 'saved_outcome_counts': dict(original_counts)}
    by_benchmark = {}
    for benchmark in sorted({x['benchmark'] for x in observations}):
        subset = [x for x in observations if x['benchmark'] == benchmark]
        by_benchmark[benchmark] = {'counts': dict(collections.Counter(x['status'] for x in subset)),
                                   'conditional_pass_times': pass_only_statistics(subset)}
    return {
        'schema_version': 'c2hls.cosim-timeout-evidence.v1',
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'replay': str(replay), 'historical_original': historical,
        'replay_summary': {'denominator': len(tasks), 'saved_outcome_counts': counts,
                           'saved_terminal_measurements': len(observations),
                           'without_saved_measurement': len(tasks) - len(observations),
                           'recorded_state_counts': dict(collections.Counter(t['status'] for t in tasks)),
                           'state_liveness_checked': False,
                           'note': 'Recorded running states can be stale; this offline tool does not infer process liveness.'},
        'input_file_sha256': input_hashes, 'data_issues': issues,
        'cycle_wall_scatter': observations,
        'conditional_pass_times': pass_only_statistics(observations), 'by_benchmark': by_benchmark,
        'phase_evidence_samples': phases,
        'calibration': {
            'status': 'insufficient_evidence', 'recommended_calibrated_per_benchmark_timeout_seconds': None,
            'future_formula': 'setup_wall_seconds + margin * target_total_simulated_clocks / measured_simulator_clocks_per_wall_second',
            'required_observations': ['phase-separated host wall time', 'total simulated clocks including reset, interfaces and all testbench calls',
                                      'completed transactions and stall diagnostics', 'comparable design complexity, logging, host load and concurrency'],
            'hardware_time_formula': 'kernel_latency_cycles * clock_period_seconds',
            'hardware_cycles_are_total_simulated_clocks': False,
            'limitations': ['Historical runtime_seconds includes native golden, candidate C simulation, synthesis, setup, and RTL simulation.',
                            'Terminal timeouts are right-censored completion times; recorded wall seconds are exposure, not exact completion times.',
                            'Pass-only tails are conditional on completion and cannot set a universal cap.',
                            'Nested tool phase durations must not be added as if independent.'],
        },
        'diagnostic_policy': diagnostic_policy(prior_cap),
        'eligible_diagnostic_validation_ids': [x['validation_id'] for x in eligible],
        'pilot_recommendation': ({'validation_id': pilot['validation_id'], 'benchmark': pilot['benchmark'],
                                  'reason': 'Saved three-hour timeout with archived XSIM log evidence. Instrument a fixed six-hour sensitivity attempt to distinguish slow simulation from lack of transaction progress; clock advance is not proof of eventual completion.',
                                  'prior_measurement': index[pilot['validation_id']]['measurement_path'],
                                  'archive': pilot['archive']} if pilot else None),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replay', type=Path, default=DEFAULT_REPLAY)
    parser.add_argument('--history', type=Path, default=DEFAULT_HISTORY)
    parser.add_argument('--output', type=Path, required=True, help='New JSON report path; existing files are never overwritten.')
    parser.add_argument('--phase-validation-id', action='append', dest='phase_ids',
                        help='Stream logs for this saved historical validation (repeatable). Default: first pass and first timeout.')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = build_report(args.replay, args.history, args.phase_ids)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write('\n')
    print(json.dumps({'report': str(args.output), 'replay_summary': report['replay_summary'],
                      'calibration': report['calibration']['status'], 'pilot': report['pilot_recommendation']}))


if __name__ == '__main__':
    main()
