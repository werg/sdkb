import torch
import random

from sdkb.agent import SDKBAgent
from sdkb.checkpoints import restore_checkpoint, save_checkpoint
from sdkb.key_index import PublishedKeyIndex
from sdkb.optimizers import make_optimizer
from sdkb.store import DiskStore, ReadPlan, Selection, StoredRecord
from sdkb.training_bank import TrainingBank


def test_training_overlay_updates_search_keys_and_stored_payloads(tmp_path):
    base = DiskStore(tmp_path / 'base.sqlite')
    base.put(StoredRecord('a', torch.tensor([1., 0.]), torch.tensor([1., 2.]),
        namespace='corpus', space='s0', generation='g', created_at=1))
    cache = DiskStore(tmp_path / 'cache.sqlite')
    index = PublishedKeyIndex(base, namespace='corpus', generation='g', spaces=('s0',),
                              expected_sources=1)
    bank = TrainingBank(base, cache, index)
    bank.update([StoredRecord('a', torch.tensor([0., 2.]), torch.tensor([7., 8.]),
                              space='s0')])
    plan = ReadPlan('corpus', 's0', 'g', 'research', 2, (Selection('a', 0.),))
    torch.testing.assert_close(bank.fetch_many((plan,))[0][0], torch.tensor([7., 8.]))
    torch.testing.assert_close(index.keys_for_ids('s0', ('a',), domain='research',
                                                  query_time=2)[0], torch.tensor([0., 1.]))
    reopened_index = PublishedKeyIndex(base, namespace='corpus', generation='g',
                                       spaces=('s0',), expected_sources=1)
    reopened = TrainingBank(base, cache, reopened_index)
    torch.testing.assert_close(reopened.fetch_many((plan,))[0][0], torch.tensor([7., 8.]))
    torch.testing.assert_close(reopened_index.keys_for_ids(
        's0', ('a',), domain='research', query_time=2)[0], torch.tensor([0., 1.]))


def test_training_overlay_batches_multiple_plans_and_spaces(tmp_path):
    base = DiskStore(tmp_path / 'base.sqlite')
    for space in ('s0', 's1'):
        for offset, record_id in enumerate(('a', 'b')):
            base.put(StoredRecord(record_id, torch.tensor([1., float(offset)]),
                torch.tensor([float(offset)]), namespace='corpus', space=space,
                generation='g', created_at=1))
    cache = DiskStore(tmp_path / 'cache.sqlite')
    index = PublishedKeyIndex(base, namespace='corpus', generation='g',
                              spaces=('s0', 's1'), expected_sources=2)
    bank = TrainingBank(base, cache, index)
    bank.update([
        StoredRecord('a', torch.tensor([1., 0.]), torch.tensor([7.]), space='s0'),
        StoredRecord('b', torch.tensor([1., 1.]), torch.tensor([8.]), space='s1'),
    ])
    plans = tuple(ReadPlan('corpus', space, 'g', 'research', 2,
                           tuple(Selection(record_id, 0.) for record_id in ('a', 'b')))
                  for space in ('s0', 's1'))
    rows = bank.fetch_many(plans)
    torch.testing.assert_close(rows[0][0], torch.tensor([7.]))
    torch.testing.assert_close(rows[0][1], torch.tensor([1.]))
    torch.testing.assert_close(rows[1][0], torch.tensor([0.]))
    torch.testing.assert_close(rows[1][1], torch.tensor([8.]))


def test_training_overlay_is_part_of_exact_checkpoint_resume(tiny_config, tmp_path):
    agent = SDKBAgent(tiny_config)
    optimizer = make_optimizer(agent)
    run = tmp_path / 'run'
    run.mkdir()
    base = DiskStore(tmp_path / 'base.sqlite')
    base.put(StoredRecord('a', torch.tensor([1.] * tiny_config.memory.key_dim),
        torch.ones(tiny_config.memory.payload_dims[0]), namespace='corpus', space='s0',
        generation='g', created_at=1))
    cache = DiskStore(run / 'training_cache.sqlite')
    index = PublishedKeyIndex(base, namespace='corpus', generation='g', spaces=('s0',),
                              expected_sources=1)
    bank = TrainingBank(base, cache, index)
    first = torch.full((tiny_config.memory.payload_dims[0],), 3.)
    bank.update([StoredRecord('a', torch.arange(tiny_config.memory.key_dim).float() + 1,
                              first, space='s0')])
    save_checkpoint(agent, optimizer, run, 1, random.Random(7), cache, 'data', keep=2)
    bank.update([StoredRecord('a', torch.arange(tiny_config.memory.key_dim).float() + 2,
                              torch.full_like(first, 9.), space='s0')])
    assert restore_checkpoint(agent, optimizer, run, random.Random(9), 'data') == 1
    restored_index = PublishedKeyIndex(base, namespace='corpus', generation='g',
                                       spaces=('s0',), expected_sources=1)
    restored = TrainingBank(base, cache, restored_index)
    plan = ReadPlan('corpus', 's0', 'g', 'research', 2, (Selection('a', 0.),))
    torch.testing.assert_close(restored.fetch_many((plan,))[0][0], first)
