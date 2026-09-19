"""Fit a query-independent two-to-one compactor against a frozen useful reader.

This post-hoc diagnostic uses oracle full clusters. It does not train the backbone,
route evidence, or establish net disk savings while raw subset fallback is kept.
"""
import argparse
import copy
from dataclasses import asdict
import json
from pathlib import Path
import signal

from safetensors.torch import load_file, save_file
import torch

from sdkb.archiving import ensure_free
from sdkb.checkpoints import resolve_checkpoint, _fsync, _fsync_dir
from sdkb.cluster_store import ClusterBank, state_fingerprint
from sdkb.compaction import SyntheticCompactor, contribution_loss, mean_and_mass
from sdkb.data import load_episodes, make_multiuse_world, save_episodes
from sdkb.evaluation import build_shared_bank, stored_transfer_evaluation
from sdkb.evaluation_adapter import load_frozen_agent
from sdkb.operations import atomic_json, control_dir, run_lock, stop_requested
from sdkb.optimizers import MuonAdamW
from sdkb.runtime import configure_memory, available_host_memory, memory_metrics
from sdkb.store import DiskStore, ReadPlan, Selection
from sdkb.tracking import Tracking
from sdkb.training import config_from_run, autocast_context
from sdkb.trajectories import file_sha256


def plan_for(episode):
    return ReadPlan('global', 's0', 'frozen-v1', 'research', episode.query_time,
                    tuple(Selection(rid, 0.) for rid in sorted(episode.required_ids)))


def serialized_codes(compactor, raw):
    """Differentiable cast reproduces the persisted BF16-value/FP32-mass forward."""
    values, weights = compactor(raw, raw.new_ones(raw.shape[:2]))
    return values.bfloat16().float(), weights.float()


@torch.no_grad()
def features(agent, episodes, store, output):
    path = output / 'features.safetensors'
    manifest = output / 'features.json'
    if path.exists() and manifest.exists():
        if json.loads(manifest.read_text())['sha256'] != file_sha256(path):
            raise ValueError('Frozen feature bytes changed')
        return load_file(str(path))
    raw, queries = [], []
    with autocast_context(agent.config):
        for index, episode in enumerate(episodes):
            if len(episode.required_ids) != 2:
                continue
            values = torch.stack(store.fetch(plan_for(episode)))
            captured = []
            def provider(completed, query):
                if completed == 1:
                    captured.append(query.detach().cpu()[0])
                return None
            agent.plan_loop_memory(agent.prompt_ids(episode.query), provider)
            if len(captured) != 1:
                raise ValueError('Expected exactly one causal first-boundary query')
            raw.append(values)
            queries.append(captured[0])
            if index % 128 == 0:
                print(json.dumps({'extracted_queries': index}), flush=True)
            if stop_requested(output.parent):
                raise RuntimeError('Stopped during reproducible feature extraction')
    result = {'raw': torch.stack(raw), 'query': torch.stack(queries)}
    temporary = path.with_suffix('.tmp')
    save_file(result, str(temporary))
    _fsync(temporary)
    temporary.replace(path)
    _fsync_dir(path.parent)
    atomic_json(manifest, {'sha256': file_sha256(path), 'queries': len(raw)})
    return result


@torch.no_grad()
def persist_codes(agent, compactor, store, episodes, view):
    bank = ClusterBank(store, view=view, reader_hash=state_fingerprint(agent.reader))
    covered = set()
    with autocast_context(agent.config):
        for episode in episodes:
            ids = frozenset(episode.required_ids)
            if len(ids) != 2 or ids <= covered:
                continue
            if ids & covered:
                raise ValueError('Overlapping clusters require an explicit separate view')
            plan = plan_for(episode)
            raw = torch.stack(store.fetch(plan)).float().to(agent.device)[None]
            if compactor is None:
                values, weights = mean_and_mass(raw, raw.new_ones(raw.shape[:2]))
            else:
                values, weights = serialized_codes(compactor, raw)
            existing = bank.fetch(plan)
            if existing.used_clusters:
                torch.testing.assert_close(existing.values[0], values[0].bfloat16().cpu(), rtol=0, atol=0)
                torch.testing.assert_close(existing.weights[0], weights[0].float().cpu(), rtol=0, atol=0)
            else:
                bank.put(plan, values[0].bfloat16(), weights[0].float())
            covered.update(ids)
    return bank


def run(source, train_file, output, steps=400, batch_size=32, seed=59):
    if steps < 1 or batch_size < 1:
        raise ValueError('Positive optimizer steps and batch size required')
    output.mkdir(parents=True, exist_ok=True)
    with run_lock(output, clear_stop=False):
        checkpoint = resolve_checkpoint(source, verify=True)
        config = copy.deepcopy(config_from_run(checkpoint))
        if (config.memory.read_timing != 'loop_boundary' or config.memory.read_steps != 1
                or len(config.memory.payload_dims) != 1 or config.memory.storage_dtype != 'bfloat16'):
            raise ValueError('Probe requires a one-read, single-space BF16 recurrent checkpoint')
        config.train.optimizer = 'muon'
        config.train.steps, config.train.seed = steps, seed
        config.train.learning_rate = 1e-4
        config.train.gradient_accumulation = 1
        config.train.loop_counts = []
        config.train.retrieval = 'oracle'
        config.train.evidence_scope = 'required'
        config.train.episodes_file = str(train_file)
        config.train.wandb_group = 'frozen-reader-posthoc-compaction'
        configure_memory(config.train)
        torch.set_num_threads(config.train.threads)
        torch.manual_seed(seed)
        agent, _ = load_frozen_agent(config, checkpoint)
        agent.requires_grad_(False)
        episodes = {'train': load_episodes(train_file), 'heldout': [
            e for i in range(32) for e in make_multiuse_world(i, split='posthoc-compaction-20260919', bindings=2)]}
        config.train.train_worlds = len({e.environment for e in episodes['train']})
        identity = {'checkpoint_manifest_sha256': file_sha256(checkpoint / 'manifest.json'),
                    'train_episodes_sha256': file_sha256(train_file), 'config': asdict(config),
                    'batch_size': batch_size, 'steps': steps, 'seed': seed,
                    'compactor': {'width': config.memory.reader_width, 'records': 1},
                    'script_sha256': file_sha256(__file__), 'reader_hash': state_fingerprint(agent.reader),
                    'objective': 'conditional numerator/mass and free reader rollout on serialized BF16 codes'}
        if (output / 'inputs.json').exists() and json.loads((output / 'inputs.json').read_text()) != identity:
            raise ValueError('Compaction run identity changed')
        atomic_json(output / 'inputs.json', identity)
        data, stores = {}, {}
        for split, group in episodes.items():
            directory = output / split
            directory.mkdir(exist_ok=True)
            save_episodes(directory / 'episodes.jsonl', group)
            stores[split] = DiskStore(directory / 'bank.sqlite')
            if not (directory / 'features.json').exists():
                build_shared_bank(agent, stores[split], group)
            data[split] = {k: v.float().to(agent.device) for k, v in
                           features(agent, group, stores[split], directory).items()}
        def forbidden(*_args, **_kwargs):
            raise AssertionError('Writer invoked after offline feature/bank creation')
        agent.produce = forbidden
        compactor = SyntheticCompactor(config.memory.payload_dims[0], config.memory.reader_width, 1).to(agent.device)
        names = [[n for n, p in compactor.named_parameters() if (p.ndim == 2) == matrix]
                 for matrix in (True, False)]
        named = dict(compactor.named_parameters())
        groups = [{'params': [named[n] for n in group], 'lr': config.train.learning_rate} for group in names]
        optimizer = MuonAdamW(groups[:1], groups[1:], config.train)
        sampler = torch.Generator().manual_seed(seed)
        completed, path = 0, output / 'resume.pt'
        if path.exists():
            state = torch.load(path, weights_only=True, map_location=agent.device)
            if state['identity'] != identity or state['parameter_names'] != names:
                raise ValueError('Compactor resume identity/ownership changed')
            compactor.load_state_dict(state['compactor'])
            optimizer.load_state_dict(state['optimizer'])
            sampler.set_state(state['sampling_rng'].cpu())
            torch.set_rng_state(state['torch_rng'].cpu())
            if state['cuda_rng']:
                torch.cuda.set_rng_state_all([v.cpu() for v in state['cuda_rng']])
            completed = state['step']
        def save():
            ensure_free(output, sum(p.numel() * p.element_size() for p in compactor.parameters()) * 4,
                        config.train.min_free_disk_bytes)
            temporary = path.with_suffix('.tmp')
            torch.save({'identity': identity, 'compactor': compactor.state_dict(),
                        'optimizer': optimizer.state_dict(), 'parameter_names': names, 'step': completed,
                        'sampling_rng': sampler.get_state(), 'torch_rng': torch.get_rng_state(),
                        'cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}, temporary)
            _fsync(temporary)
            temporary.replace(path)
            _fsync_dir(path.parent)
            print(json.dumps({'checkpoint_saved': completed}), flush=True)
        if not path.exists():
            save()
        with Tracking(config, output) as tracking:
            if tracking.run is not None:
                tracking.run.config.update({'compaction_probe': identity, 'frozen_base_model': True})
            while completed < steps:
                available = available_host_memory()
                if stop_requested(output) or (available is not None and available < config.train.min_system_available_bytes):
                    save()
                    raise RuntimeError('Stopped at complete optimizer boundary with full resume state')
                indices = torch.randint(len(data['train']['raw']), (batch_size,), generator=sampler).to(agent.device)
                raw, query = data['train']['raw'][indices], data['train']['query'][indices]
                optimizer.zero_grad(set_to_none=True)
                with autocast_context(config):
                    values, weights = serialized_codes(compactor, raw)
                    loss = contribution_loss(agent.reader, raw, raw.new_ones(raw.shape[:2]), values, weights, query)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(compactor.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
                completed += 1
                if completed % 20 == 0:
                    row = {'step': completed, 'loss': loss.item(), **memory_metrics(config.train.device)}
                    with (output / 'metrics.jsonl').open('a') as handle:
                        handle.write(json.dumps(row) + '\n')
                    tracking.log(row)
                    print(json.dumps(row), flush=True)
            save()
        if state_fingerprint(agent.reader) != identity['reader_hash']:
            raise AssertionError('Frozen reader changed')
        result = {'identity': identity, 'feature_losses': {}}
        compactor.eval()
        with torch.no_grad(), autocast_context(config):
            for split, values in data.items():
                result['feature_losses'][split] = {}
                for method in ('mean', 'trained'):
                    total = 0.
                    for start in range(0, len(values['raw']), batch_size):
                        raw = values['raw'][start:start + batch_size]
                        query = values['query'][start:start + batch_size]
                        code, mass = (serialized_codes(compactor, raw) if method == 'trained' else
                                      mean_and_mass(raw, raw.new_ones(raw.shape[:2])))
                        code = code.bfloat16().float()
                        total += len(raw) * contribution_loss(agent.reader, raw, raw.new_ones(raw.shape[:2]),
                                                              code, mass.float(), query).item()
                    result['feature_losses'][split][method] = total / len(values['raw'])
        atomic_json(output / 'fit-results.json', result)
        print(json.dumps(result), flush=True)
        # Persist codes before disabling compaction. Inference may only fetch them.
        banks = {method: persist_codes(agent, compactor if method == 'trained' else None,
                                       stores['heldout'], episodes['heldout'], method)
                 for method in ('mean', 'trained')}
        compactor.forward = forbidden
        for method, bank in banks.items():
            if (output / f'{method}-stored.json').exists():
                continue
            report = stored_transfer_evaluation(agent, stores['heldout'], episodes['heldout'],
                                                cluster_bank=bank, drop_supports=True)
            report['code_storage'] = bank.sizes()
            atomic_json(output / f'{method}-stored.json', report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--train', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=400)
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()
    def stop(_signal, _frame):
        (control_dir(args.output) / 'STOP').touch()
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    run(args.source, args.train, args.output, args.steps, args.batch_size)
