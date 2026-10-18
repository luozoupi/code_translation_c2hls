"""Main-study request allowance; checked before provider accounting starts."""
from __future__ import annotations
import json


class SearchAllowanceExhausted(RuntimeError):
    """A prespecified local token allowance ended search without an API request."""


def request_input_bound(messages):
    payload = json.dumps(messages, ensure_ascii=True, sort_keys=True, separators=(',', ':')).encode('utf-8')
    return len(payload) + 1024 * (len(messages) + 1)


def check_request_allowance(owner, messages, *, request_cap=250000, aggregate_cap=500000):
    bound = request_input_bound(messages)
    usage = getattr(owner, 'llm_usage_events', [])
    used = sum(int(event.get('input_tokens') or 0) for event in usage)
    if bound > request_cap:
        raise SearchAllowanceExhausted('next_request_conservative_input_bound_exceeds_' + str(request_cap))
    if used + bound > aggregate_cap:
        raise SearchAllowanceExhausted('aggregate_input_reservation_exceeds_' + str(aggregate_cap))
    return {'next_input_bound': bound, 'previous_actual_input_tokens': used}
