"""Matched heldout bank reads keep the training neighborhood and causal boundary."""
from contextlib import nullcontext
from pathlib import Path
import runpy
from types import SimpleNamespace

import torch


def test_supplied_mixed_plan_uses_prefix_query_and_deduplicates(monkeypatch):
    module = runpy.run_path(str(Path(__file__).parents[1] /
                                'scripts/evaluate_published_bank.py'))
    planner = module['supplied_mixed_plans']
    monkeypatch.setitem(planner.__globals__, 'autocast_context', lambda _: nullcontext())
    calls = []

    class Agent:
        config = object()
        query_maps = [lambda query: query, lambda query: query]

        def prompt_ids(self, query):
            assert query == 'earlier question'
            return torch.tensor([[1]])

        def plan_loop_memory(self, prompt, provider, *, include_routing_query):
            assert include_routing_query and prompt.tolist() == [[1]]
            provider(1, torch.tensor([[2.0]]), torch.tensor([[3.0]]))

    class Searcher:
        def search(self, address, **kwargs):
            assert address.item() == 3.0
            calls.append(kwargs)
            return SimpleNamespace(selections=[SimpleNamespace(record_id=rid)
                for rid in ('positive', 'a', 'b', 'c')])

    episode = SimpleNamespace(episode_id='q', query='earlier question',
        answer='future answer', query_time=7, required_ids=('positive',),
        support_annotation='verified', provenance={'domain': 'test'})
    result = planner(Agent(), Searcher(), [episode], namespace='bank',
                     generation='g', limits=(3, 2))
    assert [[item.record_id for item in plan.selections] for plan in result['q'][0]] == [
        ['positive', 'a', 'b'], ['positive', 'a']]
    assert [call['space'] for call in calls] == ['s0', 's1']
    assert all(call['query_time'] == 7 and call['domain'] == 'test' for call in calls)
