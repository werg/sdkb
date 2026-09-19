from dataclasses import asdict
import importlib.util
from pathlib import Path
import json

import pytest
import yaml

from sdkb.data import make_multiuse_world, counterfactual_multiuse, load_episodes
from sdkb.launch import launch


@pytest.mark.parametrize('kind', ['restoration', 'permission'])
def test_binding_counterfactual_preserves_world_identity_and_shared_sources(kind):
    episodes = make_multiuse_world(13, bindings=3)
    variants = [counterfactual_multiuse(e, kind) for e in episodes]
    sources = {}
    changed = unchanged = 0
    for original, variant in zip(episodes, variants, strict=True):
        assert counterfactual_multiuse(variant, kind) == original
        assert (variant.episode_id, variant.query, variant.required_ids, variant.query_time) == (
            original.episode_id, original.query, original.required_ids, original.query_time)
        for before, after in zip(original.supports, variant.supports, strict=True):
            assert (before.record_id, before.created_at, before.kind) == (after.record_id, after.created_at, after.kind)
            if before.kind != kind:
                assert before == after
            assert sources.setdefault(after.record_id, after) == after
        if original.task_family == 'multiuse/action':
            expected_change = kind == 'permission' or original.capability == original.allowed_capability
            assert (variant.answer != original.answer) == expected_change
            changed += expected_change
            unchanged += not expected_change
        if original.task_family == 'multiuse/identifier':
            assert variant.answer == original.answer
    assert changed and (unchanged or kind == 'permission')


def test_binding_curriculum_uses_separate_worlds_and_evaluates_counterfactuals(tmp_path, tiny_config, monkeypatch):
    import sdkb.evaluation as evaluation
    calls, original = [], evaluation.stored_transfer_evaluation
    def observe(*args, **kwargs):
        result = original(*args, **kwargs)
        calls.append((kwargs.get('full_evidence_only', False), {r['condition'] for r in result['rows']}))
        return result
    monkeypatch.setattr(evaluation, 'stored_transfer_evaluation', observe)
    tiny_config.train.max_prompt_tokens = 1500
    config = tmp_path / 'base.yaml'
    config.write_text(yaml.safe_dump(asdict(tiny_config)))
    recipe = tmp_path / 'recipe.yaml'
    recipe.write_text(yaml.safe_dump(dict(base_config=str(config), protocol='binding', seed=17,
        causal_train_worlds=2, bindings=2, sources=[],
        stages=[dict(name='memory', arm='memory', steps=1)],
        evaluation=dict(max_episodes=2, causal_worlds=1))))
    out = tmp_path / 'run'
    assert launch(recipe, out)['status'] == 'complete'
    train = load_episodes(out / 'data/train.jsonl')
    validation = load_episodes(out / 'data/validation.jsonl')
    assert {e.environment for e in train}.isdisjoint(e.environment for e in validation)
    assert {e.task_family for e in train} == {'multiuse/action', 'multiuse/identifier',
                                             'multiuse/restoration', 'multiuse/permission'}
    report = json.loads((out / 'multiuse-evaluation.json').read_text())
    assert set(report['counterfactuals']) == {'cf_restoration', 'cf_permission'}
    assert 'drop_0' in report['summary']
    assert [flag for flag, _ in calls] == [False, True, True, False, True, True]
    assert all(conditions == {'all'} for flag, conditions in calls if flag)
    assert launch(recipe, out, resume=True)['status'] == 'complete'
    script = Path(__file__).parents[1] / 'scripts/evaluate_binding_context.py'
    spec = importlib.util.spec_from_file_location('binding_context', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.evaluate(out / 'memory', out / 'fresh-multiuse.jsonl', out / 'world-context')
    result = json.loads((out / 'world-context/results.json').read_text())
    action_rows = [r for r in result['rows'] if r['task_family'] == 'multiuse/action']
    assert all(r['selected_record_count'] == 4 for r in action_rows if r['condition'] == 'all')
    assert all(r['selected_record_count'] == 2 for r in action_rows if r['condition'] == 'selected_pair')
    # Two distractors remain visible after removing a true required support, but
    # they cannot satisfy the missing contribution on an allowed retry branch.
    dropped = [r for r in action_rows if r['condition'].startswith('drop_') and r['answer'] != 'STOP']
    assert dropped and all(r['selected_record_count'] == 3 and not r['complete_support'] for r in dropped)
    assert result['writes']['all']['writer_calls'] == 4


def test_full_evidence_fast_path_matches_reference_rows(tmp_path, tiny_config):
    from sdkb.agent import SDKBAgent
    from sdkb.evaluation import build_shared_bank, stored_transfer_evaluation
    from sdkb.store import DiskStore
    episodes = make_multiuse_world(2, bindings=2)
    agent = SDKBAgent(tiny_config).eval()
    store = DiskStore(tmp_path / 'bank.sqlite')
    build_shared_bank(agent, store, episodes)
    reference = stored_transfer_evaluation(agent, store, episodes)
    fast = stored_transfer_evaluation(agent, store, episodes, full_evidence_only=True)
    assert fast['rows'] == [r for r in reference['rows'] if r['condition'] == 'all']
