"""Provider boundary tests; every HTTP request is mocked."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from hlsmith_minimal.model import Model, ModelError


class Response:
    headers = {'x-request-id': 'request-123'}

    def __init__(self, value):
        self.raw = json.dumps(value).encode()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def read(self, maximum):
        return self.raw[:maximum]


class ModelTests(TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.messages = [{'role': 'user', 'content': 'Translate this public kernel.'}]

    def test_replay_cannot_call_network_or_reset_previous_accounting(self):
        path = self.root / 'responses.json'
        path.write_text(json.dumps({'origin': 'illustrative_fixture', 'responses': [{'content': 'int kernel() { return 1; }'}]}))
        with patch('urllib.request.urlopen', side_effect=AssertionError('network forbidden')):
            model = Model(self.root / 'model', responses=path)
            self.assertIn('int kernel', model.complete(self.messages))
            with self.assertRaisesRegex(ModelError, 'exhausted'):
                model.complete(self.messages)
        ledger = json.loads((self.root / 'model/calls.json').read_text())
        self.assertEqual(ledger['calls'][0]['provider_calls'], 0)
        self.assertIsNone(ledger['calls'][0]['usage'])
        self.assertEqual(ledger['charged_or_reserved_usd'], 0)
        with self.assertRaisesRegex(ValueError, 'never reset'):
            Model(self.root / 'model', responses=path)

    def test_full_request_reservation_stops_before_network_when_allowance_small(self):
        with patch.dict('os.environ', {'DEEPSEEK_API_KEY': 'dummy-unit-test-only'}), \
                patch('urllib.request.urlopen', side_effect=AssertionError('network forbidden')):
            model = Model(self.root / 'model', max_usd=.01)
            with self.assertRaisesRegex(ModelError, 'reservation'):
                model.complete(self.messages)
        self.assertEqual(model.calls, [])

    def test_missing_usage_retains_reservation_and_prevents_another_call(self):
        response = {'model': 'returned-alias', 'choices': [{'message': {'content': 'code'}}]}
        with patch.dict('os.environ', {'DEEPSEEK_API_KEY': 'dummy-unit-test-only'}), \
                patch('urllib.request.urlopen', return_value=Response(response)) as network:
            model = Model(self.root / 'model', max_usd=1)
            with self.assertRaises(ModelError):
                model.complete(self.messages)
            reservation = model.calls[0]['charged_or_reserved_usd']
            self.assertGreater(reservation, .3)
            self.assertEqual(model.calls[0]['status'], 'uncertain')
            with self.assertRaisesRegex(ModelError, 'uncertain'):
                model.complete(self.messages)
            self.assertEqual(network.call_count, 1)
        self.assertTrue((self.root / 'model/000/response.json').exists())

    def test_decoding_usage_request_identity_and_secret_exclusion(self):
        response = {'model': 'returned-deployment-identity', 'id': 'response-123', 'created': 12345,
            'choices': [{'message': {'content': 'int kernel() { return 1; }'}, 'finish_reason': 'stop'}],
            'usage': {'prompt_tokens': 100, 'completion_tokens': 20, 'prompt_cache_hit_tokens': 50}}
        with patch.dict('os.environ', {'DEEPSEEK_API_KEY': 'dummy-unit-test-only'}), \
                patch('urllib.request.urlopen', return_value=Response(response)) as network:
            model = Model(self.root / 'model', max_usd=1)
            self.assertIn('int kernel', model.complete(self.messages))
            payload = json.loads(network.call_args.args[0].data)
        self.assertEqual(payload['thinking'], {'type': 'disabled'})
        self.assertEqual(payload['temperature'], 0)
        self.assertNotIn('seed', payload)
        call = model.calls[0]
        self.assertEqual(call['status'], 'settled')
        self.assertEqual(call['returned_model'], 'returned-deployment-identity')
        self.assertEqual(call['request_id'], 'request-123')
        self.assertAlmostEqual(call['charged_or_reserved_usd'], (50 * .3 + 50 * .006 + 20 * 1.2) / 1e6)
        for path in (self.root / 'model').rglob('*.json'):
            self.assertNotIn('dummy-unit-test-only', path.read_text())

    def test_other_provider_does_not_inherit_deepseek_pricing_verification(self):
        path = self.root / 'responses.json'
        path.write_text(json.dumps({'responses': ['source']}))
        model = Model(self.root / 'model', responses=path, config={'model': 'another-model', 'declared_release': 'declared',
            'context_input_bound': 4096, 'prices': {'input': 1, 'cached_input': 1, 'output': 2}})
        ledger = json.loads((model.out / 'calls.json').read_text())
        self.assertIsNone(ledger['rates_source'])
        self.assertIsNone(ledger['rates_verified_on'])

    def test_nullable_cache_details_are_estimated_as_uncached_without_losing_usage(self):
        response = {'model': 'returned-alias', 'choices': [{'message': {'content': 'source'}}],
                    'usage': {'prompt_tokens': 100, 'completion_tokens': 20, 'prompt_tokens_details': None}}
        with patch.dict('os.environ', {'DEEPSEEK_API_KEY': 'dummy-unit-test-only'}), \
                patch('urllib.request.urlopen', return_value=Response(response)):
            model = Model(self.root / 'model', max_usd=1)
            self.assertEqual(model.complete(self.messages), 'source')
        self.assertEqual(model.calls[0]['status'], 'settled')
        self.assertIn('unknown', model.calls[0]['cache_accounting'])
        self.assertAlmostEqual(model.calls[0]['charged_or_reserved_usd'], (100 * .3 + 20 * 1.2) / 1e6)
