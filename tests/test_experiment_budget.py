import json
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import pytest
from experiment_budget import BudgetExhausted, initialize, reserve, settle, send_budgeted_request


@pytest.fixture
def ledger(tmp_path):
    path=tmp_path/'budget.json'
    initialize(path, limit_usd=0.002, prices={'model': {'input':1, 'output':1, 'cached_input':0.1}})
    return path


def request(path, run_id):
    return reserve(path, model='model', input_bound=500, output_bound=500, run_id=run_id, prompt_sha256='hash')


def test_shared_reservations_prevent_parallel_overspend(ledger):
    def attempt(i):
        try:return request(ledger,str(i))
        except BudgetExhausted:return None
    with ThreadPoolExecutor(max_workers=8) as pool:ids=list(pool.map(attempt,range(8)))
    assert sum(identity is not None for identity in ids)==2
    assert sum(row['charged_or_reserved_usd'] for row in json.loads(ledger.read_text())['reservations'])<=0.002


def test_settlement_releases_only_known_unused_reservation(ledger):
    identity=request(ledger,'a')
    settle(ledger,identity,input_tokens=10,output_tokens=10)
    request(ledger,'b')
    with pytest.raises(BudgetExhausted):request(ledger,'c')
    assert json.loads(ledger.read_text())['reservations'][0]['charged_or_reserved_usd']==0.00002


def test_unknown_usage_cannot_be_silently_retried(ledger):
    request(ledger,'a')
    with pytest.raises(BudgetExhausted,match='unknown usage'):request(ledger,'a')


def test_reservation_overrun_blocks_further_calls(ledger):
    identity=request(ledger,'a')
    settle(ledger,identity,input_tokens=501,output_tokens=0)
    with pytest.raises(BudgetExhausted,match='prior request'):request(ledger,'b')


def test_input_cap_prevents_provider_call(ledger,monkeypatch):
    monkeypatch.setenv('C2HLS_API_BUDGET_LEDGER',str(ledger))
    monkeypatch.setenv('C2HLS_AGGREGATE_INPUT_TOKEN_BUDGET','5')
    calls=[]
    with pytest.raises(BudgetExhausted,match='per-run cap'):
        send_budgeted_request(SimpleNamespace(llm_usage_events=[]),lambda **kw:calls.append(kw),
            model='model',messages=[{'role':'user','content':'test'}],max_tokens=1,kwargs={})
    assert not calls and not json.loads(ledger.read_text())['reservations']


def test_deepseek_cache_usage_and_response_identity_are_preserved(tmp_path, monkeypatch):
    path = tmp_path / 'deepseek.json'
    initialize(path, limit_usd=1, prices={'deepseek-flash': {'input': .3, 'cached_input': .006, 'output': 1.2}})
    monkeypatch.setenv('C2HLS_API_BUDGET_LEDGER', str(path))
    monkeypatch.setenv('C2HLS_AGGREGATE_INPUT_TOKEN_BUDGET', '100000')
    owner = SimpleNamespace(llm_usage_events=[])
    response = SimpleNamespace(model='deepseek-flash', id='response', _request_id='request',
                              usage=SimpleNamespace(prompt_tokens=1000, completion_tokens=100,
                                                    prompt_cache_hit_tokens=800))
    send_budgeted_request(owner, lambda **kwargs: response, model='deepseek-flash',
                          messages=[{'role': 'user', 'content': 'test'}], max_tokens=100, kwargs={})
    row = json.loads(path.read_text())['reservations'][0]
    assert row['cached_tokens'] == 800
    assert row['charged_or_reserved_usd'] == pytest.approx((200 * .3 + 800 * .006 + 100 * 1.2) / 1e6)
    assert row['response_model'] == 'deepseek-flash' and row['request_id'] == 'request'
    assert owner.last_response_metadata == {'response_model': 'deepseek-flash', 'response_id': 'response', 'request_id': 'request'}
