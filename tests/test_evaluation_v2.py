import torch

from sdkb.agent import SDKBAgent
from sdkb.data import make_episode, make_multiuse_world, save_episodes, load_episodes
from sdkb.evaluation import build_shared_bank, stored_transfer_evaluation, score_answers
from sdkb.sessions import read_session
from sdkb.store import DiskStore, StoredRecord
from sdkb.training import build_evaluation_store, stored_evaluation
from sdkb.metrics import paired_world_bootstrap


def test_multiuse_roundtrip_and_unique_sources(tmp_path):
    episodes = make_multiuse_world(0, bindings=3)
    assert len(episodes) == 15
    assert len({s.record_id for e in episodes for s in e.supports}) == 6
    assert {e.task_family for e in episodes} == {'multiuse/action', 'multiuse/restoration',
                                                  'multiuse/permission', 'multiuse/identifier'}
    path = tmp_path / 'data.jsonl'
    save_episodes(path, episodes)
    assert load_episodes(path) == episodes
    for e in episodes:
        if e.task_family == 'multiuse/action':
            assert len(e.required_ids) == 2  # No target-dependent cardinality leakage.


def test_write_once_stored_evaluation_and_shared_compute(tmp_path, tiny_config, monkeypatch):
    agent = SDKBAgent(tiny_config).eval()
    episodes = make_multiuse_world(0, bindings=1) + make_multiuse_world(1, bindings=1)
    store = DiskStore(tmp_path / 'bank.sqlite')
    writes = build_shared_bank(agent, store, episodes)
    assert writes['writer_calls'] == 4 and writes['queries'] == 10
    def forbidden(*args, **kwargs):
        raise AssertionError('Producer called during stored evaluation')
    monkeypatch.setattr(agent, 'produce', forbidden)
    result = stored_transfer_evaluation(agent, DiskStore(store.path), episodes, drop_supports=True)
    assert result['summary']['all']['complete_support_recall'] == 1
    assert result['paired_memory_benefit']['n_worlds'] == 2
    assert result['paired_memory_benefit']['n_queries'] == 10
    # Control should run without touching any stored source payload.
    tiny_config.train.arm = 'shared_compute'
    monkeypatch.setattr(store, 'fetch', forbidden)
    control = stored_transfer_evaluation(agent, store, episodes[:1])
    assert control['rows'][0]['read_count'] == 0


def test_full_sequence_scoring_is_not_mean_scoring(tiny_config, monkeypatch):
    agent = SDKBAgent(tiny_config)
    monkeypatch.setattr(agent, 'target_ids', lambda answer: torch.zeros(1, 1 if answer == 'short' else 5, dtype=torch.long))
    def loss(prompt, target, memory, *, reduction):
        assert reduction == 'sum'
        return torch.tensor(2. if target.numel() == 1 else 3.)
    monkeypatch.setattr(agent, 'conditioned_nll', loss)
    row = score_answers(agent, None, None, 'short', ('short', 'long'))
    assert row['predicted_action'] == 'short' and row['mean_score_prediction'] == 'long'


def test_branch_specific_support_and_counterfactuals(tmp_path, tiny_config):
    agent = SDKBAgent(tiny_config).eval()
    episode = make_episode(0, restore=True, allowed_capability=1, capability=0)
    store = DiskStore(tmp_path / 'bank.sqlite')
    build_evaluation_store(agent, store, [episode], 'test')
    result = stored_evaluation(agent, store, [episode], 'test')
    assert result['summary']['B']['complete_support_recall'] == 1
    assert result['summary']['A']['complete_support_recall'] == 0
    assert result['counterfactuals']['cf_restoration']['unchanged_pairs'] == 1
    assert result['counterfactuals']['cf_permission']['change_pairs'] == 1


def test_multiread_training_and_inference_agree(tmp_path, tiny_config):
    tiny_config.memory.read_steps = 3
    tiny_config.memory.read_top_k = 1
    agent = SDKBAgent(tiny_config).eval()
    episode = make_episode(0, distractors=0)
    store = DiskStore(tmp_path / 'bank.sqlite')
    build_shared_bank(agent, store, [episode])
    result = read_session(agent, store, agent.prompt_ids(episode.query), namespace='global',
                          generation='frozen-v1', query_time=episode.query_time,
                          oracle_ids=episode.required_ids)
    assert len(result.plans) == 2
    assert len(set(result.selected_ids[0])) == 2
    assert result.memory.shape[1] == tiny_config.memory.read_slots
    assert not torch.equal(result.query_keys[0], result.query_keys[1])


def test_exact_search_exclusions_still_fill_topk(tmp_path):
    store = DiskStore(tmp_path / 'store.sqlite')
    for i in range(6):
        store.put(StoredRecord(str(i), torch.tensor([1., float(i)]), torch.ones(3)))
    plan = store.search(torch.tensor([1., 0.]), top_k=2, exclude_ids=frozenset({'0', '1'}), key_chunk_size=1)
    assert [s.record_id for s in plan.selections] == ['2', '3']


def test_bootstrap_clusters_queries_by_world():
    rows = [{'episode': str(i), 'environment': str(i // 3), 'condition': c, 'choice_correct': c == 'all'}
            for i in range(6) for c in ['all', 'none']]
    metric = paired_world_bootstrap(rows, draws=30)
    assert metric['n_worlds'] == 2 and metric['n_queries'] == 6
    assert metric['ci95'] == [1., 1.]
