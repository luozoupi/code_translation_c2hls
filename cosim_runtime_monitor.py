"""Bounded, read-only observations of Vitis co-simulation jobs.

Use one instance per validation attempt (its scratch roots must be private)::

    monitor = CosimRuntimeMonitor([scratch], output / 'runtime.jsonl',
                                  root_pids=[worker.pid])
    while worker.poll() is None:
        monitor.sample()  # caller controls polling interval, e.g. 30 seconds
    summary = monitor.finish(outcome={'returncode': worker.returncode})

No simulator options, testbenches, timeouts, or process states are changed.
``finish`` makes one final bounded sample; it does not wait for missing logs.
The JSONL is newly created, never appended to another attempt. Its sibling
``.summary.json`` is written on finish. Phase transitions are observation-time
bounds, not exact execution times. Buffered output or skipped bytes can hide
phases. Time advancing (including XSIM's intra-transaction percentage) does not
prove useful computation; only completed transactions constitute that evidence.
Warning counts count observed log records, can duplicate across logs, and are
lower bounds when bytes are skipped. No absence of messages establishes deadlock.
"""
from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

_NUMBER = r'(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?'
_TIME = re.compile(r'\b(?:simulation\s+time|time)\s*[:=]\s*(' + _NUMBER + r')\s*(ps|ns|us|ms|s)\b', re.I)
_TRANSACTION = re.compile(r'RTL Simulation\s*:\s*(\d+)\s*/\s*(\d+)\s*\[\s*(' + _NUMBER + r')%\s*\](?:\s*@\s*"([^"\r\n]+)")?', re.I)
_ELAPSED = re.compile(r'Finished Command (csynth_design|cosim_design).*?Elapsed time:\s*(' + _NUMBER + r')\s*seconds', re.I)
_FACTORS_PS = {'ps': 1, 'ns': 1e3, 'us': 1e6, 'ms': 1e9, 's': 1e12}
_PHASE_PATTERNS = (
    ('synthesis', re.compile(r'Running:\s*csynth_design\b|Starting synthesis', re.I)),
    ('cosim_setup', re.compile(r'Running:\s*cosim_design\b|Instrumenting C test bench|Starting C TB testing|Generating (?:C post check|RTL) test bench', re.I)),
    ('simulator_setup', re.compile(r'Starting (?:verilog simulation|XSIM|static elaboration)|Running:\s*xelab\b', re.I)),
    ('rtl_simulation', re.compile(r'RTL Simulation\s*:|\bTime\s*:\s*' + _NUMBER + r'\s*(?:ps|ns|us)\b|\$finish called at time', re.I)),
    ('c_postcheck', re.compile(r'Starting C post checking', re.I)),
    ('cosim_complete', re.compile(r'C/RTL co-simulation finished:\s*(?:PASS|FAIL)|cosim done with 0 errors|cosim_design finished successfully', re.I)),
)


def parse_log_line(line: str) -> dict:
    """Parse explicit log evidence; never infer cycle units or correctness."""
    result = {}
    for phase, pattern in _PHASE_PATTERNS:
        if pattern.search(line):
            result['phase'] = phase
            break
    match = _TIME.search(line)
    if match:
        result['simulated_time_ps'] = float(match[1]) * _FACTORS_PS[match[2].lower()]
    match = _TRANSACTION.search(line)
    if match:
        result['transaction'] = {'completed': int(match[1]), 'total': int(match[2]),
                                 'intra_percent': float(match[3]),
                                 'reported_simulation_time_raw': match[4],
                                 'reported_simulation_time_unit': None}
    if re.search(r'\bwarning\b', line, re.I):
        result['warning_records'] = 1
        if 'opmode' in line.lower():
            result['dsp_opmode_warning_records'] = 1
    match = _ELAPSED.search(line)
    if match:
        result['command_elapsed'] = {'command': match[1].lower(), 'seconds': float(match[2])}
    if re.search(r'C/RTL co-simulation finished:\s*PASS|cosim done with 0 errors|cosim_design finished successfully', line, re.I):
        result['completion_marker'] = 'tool_reported_pass'
    elif re.search(r'C/RTL co-simulation finished:\s*FAIL', line, re.I):
        result['completion_marker'] = 'tool_reported_fail'
    if re.search(r'\$finish called at time', line, re.I):
        result['simulation_finish_marker'] = True
    if (re.search(r'\bdeadlock\b.*\bdetected\b|\bdetected\b.*\bdeadlock\b', line, re.I)
            and not re.search(r'\bno\s+deadlock\b|\b0\s+deadlocks?\b', line, re.I)):
        result['explicit_deadlock_diagnostic'] = True
    return result


def _process_table(proc_root=Path('/proc')):
    """Read process identity and CPU only; command arguments may contain secrets."""
    table = {}
    try:
        entries = proc_root.iterdir()
        for entry in entries:
            if not entry.name.isdigit():
                continue
            try:
                raw = (entry / 'stat').read_text()
                after = raw.rsplit(')', 1)[1].split()
                table[int(entry.name)] = {
                    'pid': int(entry.name), 'ppid': int(after[1]),
                    'name': raw.split('(', 1)[1].rsplit(')', 1)[0],
                    'state': after[0], 'start_ticks': int(after[19]),
                    'cpu_ticks': int(after[11]) + int(after[12]),
                }
            except (OSError, ValueError, IndexError):
                continue
    except OSError:
        pass
    return table


class CosimRuntimeMonitor:
    """Poll private scratch roots and root processes without controlling them.

    Each sample reads at most ``max_bytes_per_sample`` total log bytes, and
    ``max_bytes_per_file`` from each log. If a backlog exceeds the per-file
    budget, the newest tail is read (plus a bounded prefix on first discovery);
    skipped bytes are counted permanently. Discovery and long partial lines
    are capped too. Missing/archived scratch roots retain their last evidence.
    ``stale_after_seconds`` labels absent *observations*, never a deadlock.
    """
    def __init__(self, scratch_dirs, jsonl_path, root_pids=(), *,
                 max_bytes_per_sample=262144, max_bytes_per_file=65536,
                 stale_after_seconds=300, max_log_files=128,
                 max_scan_entries=20000, max_line_bytes=8192,
                 clock=time.monotonic):
        if min(max_bytes_per_sample, max_bytes_per_file, max_log_files,
               max_scan_entries, max_line_bytes) <= 0 or stale_after_seconds < 0:
            raise ValueError('monitor bounds must be positive, staleness nonnegative')
        self.scratch_dirs = [Path(p).absolute() for p in scratch_dirs]
        self.path = Path(jsonl_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation prevents accidental reuse of another attempt's data.
        with self.path.open('x'):
            pass
        self.clock = clock
        self.started = clock()
        self.previous = None
        self.max_bytes_per_sample = max_bytes_per_sample
        self.max_bytes_per_file = max_bytes_per_file
        self.stale_after_seconds = stale_after_seconds
        self.max_log_files = max_log_files
        self.max_scan_entries = max_scan_entries
        self.max_line_bytes = max_line_bytes
        self.files = {}
        self.identities = {}
        self.root_pids = set(int(p) for p in root_pids)
        table = _process_table()
        self.identities.update({p: table[p]['start_ticks'] for p in self.root_pids if p in table})
        self.missing_initial_root_pids = sorted(self.root_pids - self.identities.keys())
        self.previous_cpu = {}
        self.samples = 0
        self.closed = False
        self.final_summary = None
        self.scan_limited_ever = False
        self._rotation = 0

    def _discover(self):
        found = set()
        count = 0
        limited = False
        for root in self.scratch_dirs:
            for directory, dirs, names in os.walk(root, followlinks=False):
                dirs[:] = sorted(d for d in dirs if not Path(directory, d).is_symlink())
                count += len(dirs) + len(names)
                if count > self.max_scan_entries:
                    limited = True
                    break
                for name in sorted(names):
                    path = Path(directory, name)
                    if path.suffix.lower() != '.log' or path.is_symlink() or path == self.path:
                        continue
                    if path not in self.files and len(self.files) >= self.max_log_files:
                        limited = True
                        continue
                    found.add(path)
                    if path not in self.files:
                        self.files[path] = self._new_file()
            if count > self.max_scan_entries:
                break
        self.scan_limited_ever |= limited
        return found, limited

    @staticmethod
    def _new_file():
        return {'identity': None, 'offset': 0, 'pending': b'', 'generation': 0,
                'bytes_read': 0, 'bytes_skipped': 0, 'partial_bytes_discarded': 0,
                'phase': 'unknown', 'phase_events': [], 'phase_events_dropped': 0,
                'command_elapsed': [], 'warning_records': 0, 'dsp_opmode_warning_records': 0,
                'simulated_time_ps': None, 'transaction': None,
                'last_clock_advance': None, 'last_transaction_advance': None,
                'last_output': None, 'last_read_elapsed': None, 'completion_marker': None,
                'explicit_deadlock_diagnostic': False, 'simulation_finish_marker': False}

    def _consume(self, state, data, elapsed, bounds, discard_first=False):
        if discard_first:
            newline = data.find(b'\n')
            if newline < 0:
                state['partial_bytes_discarded'] += len(data)
                return
            state['partial_bytes_discarded'] += newline + 1
            data = data[newline + 1:]
        data = state['pending'] + data
        lines = data.split(b'\n')
        state['pending'] = lines.pop()
        if len(state['pending']) > self.max_line_bytes:
            state['partial_bytes_discarded'] += len(state['pending'])
            state['pending'] = b''
        for raw in lines:
            if len(raw) > self.max_line_bytes:
                state['partial_bytes_discarded'] += len(raw)
                continue
            item = parse_log_line(raw.decode('utf-8', errors='replace'))
            phase = item.get('phase')
            if phase and phase != state['phase']:
                state['phase'] = phase
                state['phase_events'].append({'phase': phase, 'observed_elapsed_seconds': elapsed,
                                              'observation_window_seconds': bounds,
                                              'generation': state['generation']})
                if len(state['phase_events']) > 512:
                    state['phase_events'].pop(0)
                    state['phase_events_dropped'] += 1
            for key in ('warning_records', 'dsp_opmode_warning_records'):
                state[key] += item.get(key, 0)
            if 'simulated_time_ps' in item:
                value = item['simulated_time_ps']
                before = state['simulated_time_ps']
                if before is not None and value > before:
                    state['last_clock_advance'] = elapsed
                state['simulated_time_ps'] = value
            if 'transaction' in item:
                before = state['transaction']
                value = item['transaction']
                if value['completed'] > (before['completed'] if before else 0):
                    state['last_transaction_advance'] = elapsed
                state['transaction'] = value
            for key in ('completion_marker', 'explicit_deadlock_diagnostic', 'simulation_finish_marker'):
                if key in item:
                    state[key] = item[key]
            if 'command_elapsed' in item:
                state['command_elapsed'].append(item['command_elapsed'])
                state['command_elapsed'] = state['command_elapsed'][-64:]

    def _read(self, path, state, budget, elapsed):
        try:
            with path.open('rb') as stream:
                stat = os.fstat(stream.fileno())
                identity = (stat.st_dev, stat.st_ino)
                new = state['identity'] is None
                reset = not new and (identity != state['identity'] or stat.st_size < state['offset'])
                if reset:
                    state['generation'] += 1
                    state['partial_bytes_discarded'] += len(state['pending'])
                    state.update(offset=0, pending=b'', phase='unknown', simulated_time_ps=None,
                                 transaction=None, last_clock_advance=None, last_transaction_advance=None,
                                 completion_marker=None, simulation_finish_marker=False)
                state['identity'] = identity
                remaining = max(0, stat.st_size - state['offset'])
                allowed = min(budget, self.max_bytes_per_file, remaining)
                if not allowed:
                    if not remaining:
                        state['last_read_elapsed'] = elapsed
                    return 0, remaining
                state['last_output'] = elapsed
                bounds = [None if new or reset else state['last_read_elapsed'], elapsed]
                chunks = []
                if remaining > allowed:
                    # Preserve early phase markers when first seeing a large file.
                    prefix = allowed // 2 if state['offset'] == 0 else 0
                    if prefix:
                        stream.seek(0)
                        chunks.append((stream.read(prefix), False))
                    tail = allowed - prefix
                    skip = remaining - allowed
                    state['bytes_skipped'] += skip
                    stream.seek(stat.st_size - tail)
                    chunks.append((stream.read(tail), True))
                    bounds[0] = None
                else:
                    stream.seek(state['offset'])
                    chunks.append((stream.read(allowed), False))
                consumed = 0
                for data, gap in chunks:
                    if gap:
                        state['partial_bytes_discarded'] += len(state['pending'])
                        state['pending'] = b''
                    self._consume(state, data, elapsed, bounds, discard_first=gap)
                    consumed += len(data)
                state['offset'] = stream.tell()
                state['bytes_read'] += consumed
                state['last_read_elapsed'] = elapsed
                return consumed, max(0, stat.st_size - state['offset'])
        except OSError:
            return 0, None

    def _processes(self):
        table = _process_table()
        selected = {p for p, start in self.identities.items()
                    if p in table and table[p]['start_ticks'] == start}
        # Private scratch CWD also identifies descendants missed between polls.
        for pid, item in table.items():
            try:
                cwd = Path(os.readlink(f'/proc/{pid}/cwd'))
                if any(cwd == root or root in cwd.parents for root in self.scratch_dirs):
                    selected.add(pid)
            except OSError:
                pass
        while True:
            children = {p for p, item in table.items() if item['ppid'] in selected}
            if children <= selected:
                break
            selected.update(children)
        rows = []
        for pid in sorted(selected):
            item = dict(table[pid])
            identity = (pid, item['start_ticks'])
            item['cpu_ticks_delta'] = None if identity not in self.previous_cpu else max(0, item['cpu_ticks'] - self.previous_cpu[identity])
            self.previous_cpu[identity] = item['cpu_ticks']
            self.identities[pid] = item['start_ticks']
            rows.append(item)
        return rows

    def sample(self):
        """Append and return one observation; never blocks on tool completion."""
        if self.closed:
            raise RuntimeError('monitor already finished')
        elapsed = self.clock() - self.started
        found, limited = self._discover()
        budget = self.max_bytes_per_sample
        log_rows = []
        paths = sorted(self.files)
        if paths:
            start = self._rotation % len(paths)
            paths = paths[start:] + paths[:start]
            self._rotation += 1  # prevent a noisy first log starving all others
        for path in paths:
            state = self.files[path]
            used, unread = self._read(path, state, budget, elapsed) if path in found else (0, None)
            budget -= used
            log_rows.append({'path': str(path), 'present': path in found if not limited else (True if path in found else None), 'bytes_read_this_sample': used,
                             'unread_bytes': unread, **{k: v for k, v in state.items()
                               if k not in ('identity', 'pending', 'phase_events', 'command_elapsed')},
                             'pending_partial_line_bytes': len(state['pending'])})
        processes = self._processes()
        recent_transaction = any(s['last_transaction_advance'] is not None and
                                 elapsed - s['last_transaction_advance'] <= self.stale_after_seconds
                                 for s in self.files.values())
        recent_clock = any(s['last_clock_advance'] is not None and
                           elapsed - s['last_clock_advance'] <= self.stale_after_seconds
                           for s in self.files.values())
        explicit_deadlock = any(s['explicit_deadlock_diagnostic'] for s in self.files.values())
        if recent_transaction:
            progress = 'completed_transaction_observed'
        elif recent_clock:
            progress = 'clock_advancing_transaction_progress_unknown'
        elif any(s['simulated_time_ps'] is not None for s in self.files.values()):
            progress = 'no_recent_clock_advance_observed'
        else:
            progress = 'unknown_no_simulation_time_evidence'
        record = {'schema_version': 'c2hls.cosim-runtime-observation.v1', 'event': 'sample',
                  'sample': self.samples, 'at_utc': datetime.now(timezone.utc).isoformat(),
                  'elapsed_seconds': elapsed, 'sample_interval_seconds': None if self.previous is None else elapsed - self.previous,
                  'scratch_dirs': [{'path': str(p), 'present': p.is_dir()} for p in self.scratch_dirs],
                  'logs': log_rows, 'discovery_limited': limited,
                  'bytes_read': self.max_bytes_per_sample - budget,
                  'live_processes': processes, 'cpu_ticks_per_second': os.sysconf('SC_CLK_TCK'),
                  'progress': progress, 'explicit_deadlock_diagnostic_observed': explicit_deadlock,
                  'deadlock_inferred': False,
                  'missing_initial_root_pids': self.missing_initial_root_pids}
        self._emit(record)
        self.previous = elapsed
        self.samples += 1
        return record

    def _emit(self, record):
        with self.path.open('a') as stream:
            stream.write(json.dumps(record, sort_keys=True) + '\n')

    @staticmethod
    def _phase_intervals(events):
        intervals = []
        for index, event in enumerate(events):
            start = event['observation_window_seconds']
            following = events[index + 1] if index + 1 < len(events) else None
            end = (following['observation_window_seconds']
                   if following and following['generation'] == event['generation'] else None)
            lower = max(0, end[0] - start[1]) if end and end[0] is not None else None
            upper = max(0, end[1] - start[0]) if end and start[0] is not None else None
            intervals.append({'phase': event['phase'], 'generation': event['generation'],
                              'start_observation_window_seconds': start,
                              'end_observation_window_seconds': end,
                              'duration_observation_bounds_seconds': [lower, upper]})
        return intervals

    def finish(self, outcome=None):
        """Return/persist observations; caller outcome is metadata, not inferred."""
        if self.closed:
            return self.final_summary
        last = self.sample()
        summary = {'schema_version': 'c2hls.cosim-runtime-summary.v1', 'event': 'finished',
                   'elapsed_seconds': last['elapsed_seconds'], 'samples': self.samples,
                   'outcome': outcome, 'last_progress': last['progress'],
                   'discovery_limited_ever': self.scan_limited_ever,
                   'phase_timing_method': 'log observation windows; buffering and missing bytes prevent exact boundaries',
                   'warning_count_semantics': 'observed records, not unique events; incomplete if bytes skipped and may duplicate across logs',
                   'deadlock_inferred': False,
                   'logs': {str(p): {k: v for k, v in s.items() if k not in ('identity', 'pending')}
                            for p, s in self.files.items()}}
        for state in summary['logs'].values():
            state['phase_intervals'] = self._phase_intervals(state['phase_events'])
        self._emit(summary)
        summary_path = self.path.with_suffix('.summary.json')
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
        self.closed = True
        self.final_summary = summary
        return summary
