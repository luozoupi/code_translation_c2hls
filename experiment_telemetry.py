"""Opt-in tool-flow accounting and origin-aware candidate evidence."""
from __future__ import annotations

import contextvars
import hashlib
import time
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
from functools import wraps

_ACTIVE = contextvars.ContextVar('c2hls_tool_ledger', default=None)
_SCOPE = contextvars.ContextVar('c2hls_tool_scope', default='unspecified')


@contextmanager
def capture_tool_calls():
    events = []
    token = _ACTIVE.set(events)
    try:
        yield events
    finally:
        _ACTIVE.reset(token)


@contextmanager
def tool_scope(name):
    token = _SCOPE.set(name)
    try:
        yield
    finally:
        _SCOPE.reset(token)


def record_tool(kind):
    """Count attempted flow invocations, including exceptions, without code/log text."""
    def decorate(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            events = _ACTIVE.get()
            if events is None:
                return function(*args, **kwargs)
            code = args[0] if args else kwargs.get('hls_code', kwargs.get('c_code', ''))
            event = {'index': len(events), 'kind': kind, 'scope': _SCOPE.get(),
                     'started_at': datetime.now(timezone.utc).isoformat(),
                     'code_sha256': hashlib.sha256(code.encode()).hexdigest() if isinstance(code, str) else None,
                     'success': False, 'exception_type': None}
            events.append(event)
            start = time.monotonic()
            try:
                result = function(*args, **kwargs)
                if isinstance(result, dict):
                    event['success'] = bool(result.get('success', result.get('passed', False)))
                    event['explicitly_skipped'] = bool(result.get('skipped') or result.get('ran') is False)
                elif isinstance(result, tuple):
                    event['success'] = bool(result[0])
                return result
            except Exception as exc:
                event['exception_type'] = type(exc).__name__
                raise
            finally:
                event['elapsed_seconds'] = time.monotonic() - start
                event['finished_at'] = datetime.now(timezone.utc).isoformat()
        return wrapped
    return decorate


def tool_summary(events):
    counts = Counter(event['kind'] for event in events)
    return {'schema_version': 'c2hls.tool-flows.v1', 'complete_for_capture_scope': True,
            'unit': 'function flow invocation; not OS subprocess count',
            'counts': dict(counts), 'total_flow_invocations': len(events),
            'vitis_synthesis_flow_invocations': counts['csynth'] + counts['rtl_cosim'],
            'note': 'Each RTL COSIM flow includes resynthesis; compiler/native/CSim flows are counted separately.',
            'events': events}


def unified_candidate_ledger(result):
    """Join LLM candidates, deterministic search, and retained frozen parents.

    Keep legacy event streams intact. An origin without model calls has a null
    LLM link; a retained parent is not a newly generated successful candidate.
    """
    summary = result.get('synthesis_evaluations') or {}
    usage = result.get('llm_usage') or {}
    llm_events = usage.get('events') or []
    rows, issues = [], []
    for index, event in enumerate(summary.get('events') or []):
        candidate_index = event.get('candidate_evaluation_index', index)
        links = [i for i, value in enumerate(llm_events) if value.get('candidate_evaluation_index') == candidate_index]
        if not links:
            issues.append('missing_llm_event:' + str(candidate_index))
        rows.append({**{k: v for k, v in event.items() if not k.startswith('_')},
                     'candidate_id': 'llm:' + str(candidate_index), 'origin': 'llm',
                     'llm_event_indices': links, 'cost_scope': 'current_search'})
    for index, event in enumerate(summary.get('qor_design_events') or []):
        rows.append({**event, 'candidate_id': 'deterministic:' + str(event.get('qor_evaluation_index', index)),
                     'origin': 'deterministic_search', 'llm_event_indices': None,
                     'cost_scope': 'current_search'})
    parent = result.get('frozen_phase_b') or {}
    if parent.get('code_sha256'):
        rows.append({'candidate_id': 'frozen_parent:' + parent['code_sha256'],
                     'origin': 'frozen_parent', 'code_sha256': parent['code_sha256'],
                     'llm_event_indices': None, 'synthesis_ran': False,
                     'cost_scope': 'previous_parent_construction', 'source_provenance': parent})
    selected_hash = result.get('selected_code_sha256')
    for row in rows:
        row['selected_source_match'] = bool(selected_hash and row.get('code_sha256') == selected_hash)
    if selected_hash and not any(row['selected_source_match'] for row in rows):
        issues.append('selected_source_missing_from_candidates')
    executed_synthesis = sum(row.get('synthesis_ran') is True for row in rows)
    if summary.get('count') is not None and executed_synthesis != summary['count']:
        issues.append('synthesis_count_disagrees_with_events')
    if usage.get('calls') is not None and usage['calls'] != len(llm_events):
        issues.append('llm_count_disagrees_with_events')
    if usage.get('candidate_requests') is not None and usage['candidate_requests'] != len(llm_events):
        issues.append('provider_attempts_missing_usage_events')
    if usage.get('usage_missing_calls'):
        issues.append('provider_token_usage_missing')
    if llm_events and sum(int(x.get('total_tokens') or 0) for x in llm_events) != int(usage.get('total_tokens') or 0):
        issues.append('token_total_disagrees_with_events')
    return {'schema_version': 'c2hls.candidate-ledger.v1', 'complete': not issues,
            'issues': issues, 'candidates': rows,
            'origin_counts': dict(Counter(row['origin'] for row in rows))}


def current_tool_summary():
    events = _ACTIVE.get()
    return tool_summary(events) if events is not None else None
