"""Locked campaign-wide API reservations and explicit per-run input limits.

Rates are a frozen accounting assumption, not a provider billing receipt.
Failed or interrupted requests retain their full reservation until inspected.
No credentials, prompt text, or response content are written to this ledger.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path


class BudgetExhausted(RuntimeError):
    pass


@contextmanager
def locked_ledger(path):
    path = Path(path)
    with path.with_suffix(path.suffix + '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        data = json.loads(path.read_text())
        yield data
        temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
        temp.write_text(json.dumps(data, indent=2, sort_keys=True) + '\n')
        temp.replace(path)


def initialize(path, *, limit_usd, prices):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {'schema_version': 'c2hls.api-budget.v1', 'limit_usd': limit_usd,
            'prices_per_million': prices, 'reservations': [],
            'created_at': datetime.now(timezone.utc).isoformat(),
            'billing_scope': 'conservative estimates from frozen rates, not invoice totals'}
    with path.open('x') as stream:
        json.dump(data, stream, indent=2)
        stream.write('\n')


def cost(price, input_tokens, output_tokens, cached_tokens=0):
    cached = min(input_tokens, max(0, cached_tokens))
    return ((input_tokens-cached)*price['input'] + cached*price.get('cached_input',price['input'])
            + output_tokens*price['output']) / 1_000_000


def reserve(path, *, model, input_bound, output_bound, run_id, prompt_sha256):
    with locked_ledger(path) as ledger:
        price = ledger['prices_per_million'].get(model)
        if price is None:
            raise BudgetExhausted('model lacks a frozen price entry')
        if any(row.get('overrun') for row in ledger['reservations']):
            raise BudgetExhausted('a prior request exceeded its reserved bound; reconcile before continuing')
        if any(row['run_id'] == run_id and row['status'] == 'reserved' for row in ledger['reservations']):
            raise BudgetExhausted('this run has a request with unknown usage; do not retry it silently')
        amount = cost(price, input_bound, output_bound)
        used = sum(row['charged_or_reserved_usd'] for row in ledger['reservations'])
        if used + amount > ledger['limit_usd'] + 1e-12:
            raise BudgetExhausted('campaign API budget exhausted')
        identity = uuid.uuid4().hex
        ledger['reservations'].append({'id': identity, 'model': model, 'run_id': run_id,
            'prompt_sha256': prompt_sha256, 'input_token_bound': input_bound,
            'output_token_bound': output_bound, 'reserved_usd': amount,
            'charged_or_reserved_usd': amount, 'status': 'reserved',
            'created_at': datetime.now(timezone.utc).isoformat()})
    return identity


def settle(path, identity, *, input_tokens, output_tokens, cached_tokens=0, response_model=None, request_id=None):
    with locked_ledger(path) as ledger:
        row = next(row for row in ledger['reservations'] if row['id'] == identity)
        if row['status'] != 'reserved':
            raise ValueError('reservation already settled')
        amount = cost(ledger['prices_per_million'][row['model']], input_tokens, output_tokens, cached_tokens)
        row.update(status='settled', input_tokens=input_tokens, output_tokens=output_tokens,
                   cached_tokens=cached_tokens, response_model=response_model, request_id=request_id,
                   charged_or_reserved_usd=amount,
                   overrun=(input_tokens > row['input_token_bound'] or output_tokens > row['output_token_bound']))


def _value(obj, name, default=0):
    return (obj.get(name, default) if isinstance(obj, dict) else getattr(obj, name, default)) or default


def send_budgeted_request(owner, create, *, model, messages, max_tokens, kwargs):
    path = os.getenv('C2HLS_API_BUDGET_LEDGER', '').strip()
    if not path:
        return create(**kwargs)
    input_limit = int(os.environ['C2HLS_AGGREGATE_INPUT_TOKEN_BUDGET'])
    events = getattr(owner, 'llm_usage_events', [])
    if any(not event.get('usage_available') for event in events):
        raise BudgetExhausted('prior request has no token usage')
    used = sum(int(event.get('input_tokens') or 0) for event in events)
    payload = json.dumps(messages, ensure_ascii=True, sort_keys=True, separators=(',', ':')).encode('utf-8')
    # Conservative text-byte reservation, with message framing allowance.
    # Provider usage is audited afterwards; a bound violation blocks later calls.
    input_bound = len(payload) + 1024 * (len(messages) + 1)
    if used + input_bound > input_limit:
        raise BudgetExhausted('aggregate input token reservation exceeds per-run cap')
    identity = reserve(path, model=model, input_bound=input_bound, output_bound=max_tokens,
                       run_id=os.getenv('C2HLS_SAMPLE_ID', 'unspecified'),
                       prompt_sha256=hashlib.sha256(payload).hexdigest())
    start = time.monotonic()
    try:
        response = create(**kwargs)
        usage = getattr(response, 'usage', None)
        if usage is not None:
            input_tokens = int(_value(usage,'prompt_tokens', _value(usage,'input_tokens')))
            output_tokens = int(_value(usage,'completion_tokens', _value(usage,'output_tokens')))
            details = _value(usage,'prompt_tokens_details', {})
            cached = int(_value(details,'cached_tokens', _value(usage,'prompt_cache_hit_tokens')))
            settle(path, identity, input_tokens=input_tokens, output_tokens=output_tokens,
                   cached_tokens=cached, response_model=getattr(response,'model',None),
                   request_id=getattr(response,'_request_id',None))
            if input_tokens > input_bound or output_tokens > max_tokens:
                raise BudgetExhausted('provider token usage exceeded conservative reservation')
        return response
    finally:
        with locked_ledger(path) as ledger:
            row = next(row for row in ledger['reservations'] if row['id'] == identity)
            row['elapsed_seconds'] = time.monotonic() - start
