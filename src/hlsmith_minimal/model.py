"""One DeepSeek/OpenAI-compatible HTTP adapter and an offline response reader."""
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import urllib.error
import urllib.request
from urllib.parse import urlsplit


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class ModelError(RuntimeError):
    pass


class Model:
    """No automatic retries; an uncertain request stops this run.

    Before a live request reserve the full declared provider input context plus
    the output limit. This is intentionally conservative and avoids claiming a
    tokenizer estimate is an enforceable billing bound. Amounts are peak-rate
    estimates, not an invoice. Missing usage retains the reservation.
    """
    def __init__(self, out, *, responses=None, config=None, max_calls=3, max_usd=None):
        self.out = Path(out)
        if self.out.exists() and any(self.out.iterdir()):
            raise ValueError('model output must be new or empty; accounting is never reset')
        self.out.mkdir(parents=True, exist_ok=True)
        self.config = dict(config or {})
        self.model = self.config.get('model', 'deepseek-flash')
        self.endpoint = self.config.get('endpoint', 'https://api.deepseek.com').rstrip('/')
        address = urlsplit(self.endpoint)
        if address.scheme not in {'http', 'https'} or not address.netloc or address.username or address.password or address.query or address.fragment:
            raise ValueError('endpoint must be an HTTP(S) base URL without credentials or query parameters')
        self.max_calls, self.max_usd = max_calls, max_usd
        self.calls = []
        self.responses = None
        self.origin = None
        if type(max_calls) is not int or max_calls < 1:
            raise ValueError('max_calls must be a positive integer')
        if responses is not None:
            path = Path(responses).resolve()
            data = json.loads(path.read_text())
            self.responses = data['responses']
            if not isinstance(self.responses, list) or not self.responses:
                raise ValueError('responses.json needs a nonempty responses array')
            self.origin = {'path': str(path), 'sha256': sha(path), 'declared_origin': data.get('origin', 'unspecified')}
        elif type(max_usd) not in (int, float) or not math.isfinite(max_usd) or max_usd <= 0:
            raise ValueError('live API use requires a positive --max-api-usd')
        if self.model != 'deepseek-flash' and not all(k in self.config for k in ('prices', 'context_input_bound', 'declared_release')):
            raise ValueError('other models require explicit prices, context_input_bound and declared_release')
        self.prices = dict(self.config.get('prices', {'input': .30, 'cached_input': .006, 'output': 1.20}))
        if (set(self.prices) != {'input', 'cached_input', 'output'}
                or any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in self.prices.values())
                or self.prices['cached_input'] > self.prices['input']):
            raise ValueError('prices must contain finite nonnegative input/cached_input/output rates')
        self.context_bound = self.config.get('context_input_bound', 1048576)
        self.max_tokens = self.config.get('max_output_tokens', 8192)
        for value in (self.context_bound, self.max_tokens):
            if type(value) is not int or value < 1:
                raise ValueError('context and output limits must be positive integers')
        timeout = self.config.get('timeout_seconds', 120)
        if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError('provider timeout_seconds must be positive and finite')
        self._save()

    def _save(self):
        write_json(self.out / 'calls.json', {'mode': 'saved_responses' if self.responses is not None else 'api',
            'max_calls': self.max_calls, 'max_api_usd': self.max_usd, 'calls': self.calls,
            'charged_or_reserved_usd': sum(c['charged_or_reserved_usd'] for c in self.calls),
            'rates_per_million': self.prices, 'billing_scope': 'conservative peak-rate estimate; not provider invoice',
            'rates_verified_on': self.config.get('rates_verified_on', '2026-10-26' if self.model == 'deepseek-flash' and 'prices' not in self.config else None),
            'rates_source': self.config.get('rates_source', 'https://api-docs.deepseek.com/quick_start/pricing/'
                                             if self.model == 'deepseek-flash' and 'prices' not in self.config else None),
            'replay_origin': self.origin})

    def complete(self, messages):
        if len(self.calls) >= self.max_calls:
            raise ModelError('request limit reached')
        if any(c['status'] == 'uncertain' for c in self.calls):
            raise ModelError('a previous request has uncertain accounting')
        index = len(self.calls)
        folder = self.out / f'{index:03d}'
        if self.responses is not None and index >= len(self.responses):
            raise ModelError('saved responses exhausted; no API fallback')
        payload = {'model': self.model, 'messages': messages, 'max_tokens': self.max_tokens,
                   'temperature': 0, 'stream': False}
        if self.model == 'deepseek-flash':
            payload['thinking'] = {'type': 'disabled'}
        raw = json.dumps(payload, ensure_ascii=False, separators=(',', ':')).encode()
        reservation = 0
        key = None
        if self.responses is None:
            key = os.environ.get(self.config.get('key_env', 'DEEPSEEK_API_KEY'), '').strip()
            if not key:
                raise ModelError('provider key environment variable is empty')
            # Byte admission is just a size guard; billing reserves the entire
            # configured provider context, not this approximate token count.
            if len(raw) + 8192 > self.context_bound:
                raise ModelError('request exceeds conservative input size guard')
            reservation = (self.context_bound * self.prices['input'] + self.max_tokens * self.prices['output']) / 1e6
            if sum(c['charged_or_reserved_usd'] for c in self.calls) + reservation > self.max_usd:
                raise ModelError('insufficient API allowance for the full request reservation')
        folder.mkdir(exist_ok=False)
        (folder / 'request.json').write_bytes(raw)
        record = {'index': index, 'started_at': now(), 'status': 'uncertain',
                  'requested_model': self.model, 'endpoint': self.endpoint,
                  'declared_release': self.config.get('declared_release', 'DeepSeek-V4.1-Flash-1017'),
                  'immutable_weights_claim': False, 'seed_supported': False,
                  'request_sha256': hashlib.sha256(raw).hexdigest(),
                  'effective_decoding': {'thinking': 'disabled' if self.model == 'deepseek-flash' else 'provider_default',
                                         'temperature': 0, 'max_output_tokens': self.max_tokens},
                  'charged_or_reserved_usd': reservation, 'reservation_usd': reservation}
        self.calls.append(record)
        self._save()
        try:
            if self.responses is not None:
                response = self.responses[index]
                if isinstance(response, str):
                    response = {'content': response}
                text = response['content']
                record.update(status='replayed', returned_model=response.get('model'),
                              usage=None, provider_calls=0, replay_origin=self.origin)
                write_json(folder / 'response.json', response)
            else:
                request = urllib.request.Request(self.endpoint + '/chat/completions', data=raw,
                    headers={'Content-Type': 'application/json', 'Authorization': 'Bearer ' + key}, method='POST')
                with urllib.request.urlopen(request, timeout=self.config.get('timeout_seconds', 120)) as stream:
                    body = stream.read(16 * 1024 * 1024 + 1)
                    record['request_id'] = stream.headers.get('x-request-id') or stream.headers.get('request-id')
                if len(body) > 16 * 1024 * 1024:
                    raise ModelError('provider response exceeds read limit')
                (folder / 'response.json').write_bytes(body)
                response = json.loads(body)
                usage = response.get('usage') or {}
                details = usage.get('prompt_tokens_details') or {}
                inputs, outputs = usage.get('prompt_tokens'), usage.get('completion_tokens')
                cached = usage.get('prompt_cache_hit_tokens', details.get('cached_tokens', 0))
                if any(type(v) is not int or v < 0 for v in (inputs, outputs, cached)) or cached > inputs:
                    raise ModelError('provider usage missing or invalid; reservation retained')
                if inputs > self.context_bound or outputs > self.max_tokens:
                    raise ModelError('provider token usage exceeds reserved bounds')
                charge = ((inputs - cached) * self.prices['input'] + cached * self.prices['cached_input']
                          + outputs * self.prices['output']) / 1e6
                record.update(status='settled', charged_or_reserved_usd=charge, returned_model=response.get('model'),
                              response_id=response.get('id'), provider_created=response.get('created'), usage=usage,
                              cache_accounting='reported' if ('prompt_cache_hit_tokens' in usage or 'cached_tokens' in details)
                              else 'cache benefit unknown; billed estimate assumes all input uncached')
                text = response['choices'][0]['message']['content']
                record['finish_reason'] = response['choices'][0].get('finish_reason')
            if not isinstance(text, str) or not text.strip():
                raise ModelError('response contains no source text')
            return text
        except (OSError, ValueError, KeyError, IndexError, TypeError, AttributeError, ModelError) as error:
            record['error'] = type(error).__name__
            if isinstance(error, urllib.error.HTTPError):
                record['http_status'] = error.code
            # Never copy HTTP error bodies, headers or credential-bearing URLs
            # to a user-facing exception. The reservation stays on the ledger.
            raise ModelError('request failed; inspect calls.json and preserved response') from None
        finally:
            record['finished_at'] = now()
            self._save()
