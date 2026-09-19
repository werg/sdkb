"""Supervised character readout from frozen payloads; not a decoder capability test."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random
import re
import signal

import torch
from torch import nn
from torch.nn import functional as F

from sdkb.checkpoints import resolve_checkpoint
from sdkb.data import load_episodes
from sdkb.operations import atomic_json, control_dir, run_lock, stop_requested
from sdkb.probe_state import restore_probe_state, save_probe_state
from sdkb.runtime import configure_memory, available_host_memory, compute_watchdog, memory_metrics
from sdkb.store import DiskStore, lookup_record
from sdkb.tracking import Tracking
from sdkb.training import config_from_run, autocast_context
from sdkb.trajectories import file_sha256


def split_identifiers(episodes, heldout_worlds):
    worlds = list(dict.fromkeys(e.environment for e in episodes))
    if not 1 <= heldout_worlds < len(worlds):
        raise ValueError('Need nonempty disjoint train and held-out worlds')
    heldout = set(worlds[:heldout_worlds])
    groups = {'train': [], 'heldout': []}
    seen = set()
    for e in episodes:
        if e.task_family != 'multiuse/identifier':
            continue
        if len(e.required_ids) != 1 or re.fullmatch(r'api_[0-9a-f]{6}', e.answer) is None:
            raise ValueError('Expected one source and generated six-hex endpoint')
        rid = e.required_ids[0]
        if rid in seen:
            raise ValueError('Identifier source repeated across examples')
        seen.add(rid)
        source = next(s for s in e.supports if s.record_id == rid)
        if source.created_at >= e.query_time or f'Its endpoint is {e.answer}.' not in source.text:
            raise ValueError('Endpoint label/source/causal boundary differs')
        groups['heldout' if e.environment in heldout else 'train'].append(e)
    if any(not v for v in groups.values()):
        raise ValueError('Both splits need identifier examples')
    return groups


class Readout(nn.Module):
    def __init__(self, training, hidden):
        super().__init__()
        self.register_buffer('center', training.mean(0))
        self.register_buffer('scale', training.std(0, unbiased=False).clamp_min(1e-6))
        width = training.shape[1]
        self.net = (nn.Sequential(nn.Linear(width, hidden, bias=False), nn.SiLU(), nn.Linear(hidden, 96, bias=False))
                    if hidden else nn.Linear(width, 96, bias=False))

    def forward(self, values):
        return self.net((values-self.center)/self.scale).reshape(-1, 6, 16)


@torch.no_grad()
def assess(model, values, labels, config):
    with autocast_context(config):
        prediction = model(values).argmax(-1)
        shuffled = model(values.roll(1, dims=0)).argmax(-1)
    return {name: {'n': len(labels), 'exact_six': int((p == labels).all(-1).sum()),
                   'characters_correct': int((p == labels).sum()), 'characters': labels.numel()}
            for name, p in [('payload', prediction), ('shifted_payload', shuffled)]}


def run(source, episodes_path, bank_path, output, steps=1600, heldout_worlds=32):
    if steps < 1:
        raise ValueError('Positive optimizer budget required')
    output.mkdir(parents=True, exist_ok=True)
    with run_lock(output, clear_stop=False):
        checkpoint = resolve_checkpoint(source, verify=True)
        config = config_from_run(checkpoint)
        config.train.optimizer = 'muon'
        config.train.steps, config.train.seed, config.train.learning_rate = steps, 83, .001
        config.train.gradient_accumulation = 1
        config.train.loop_counts = []
        config.train.wandb_group = 'frozen-payload-identifier-readout'
        config.train.episodes_file = str(episodes_path)
        config.model.freeze_backbone = True
        configure_memory(config.train)
        torch.set_num_threads(config.train.threads)
        reference = json.loads((bank_path.parent/'bank-manifest.json').read_text())
        manifest_hash, data_hash = file_sha256(checkpoint/'manifest.json'), file_sha256(episodes_path)
        if reference['checkpoint_manifest_sha256'] != manifest_hash or reference['episodes_sha256'] != data_hash:
            raise ValueError('Stored bank writer/source corpus differs')
        groups = split_identifiers(load_episodes(episodes_path), heldout_worlds)
        config.train.train_worlds = len({e.environment for e in groups['train']})
        store, values, labels = DiskStore(bank_path), {}, {}
        for split, episodes in groups.items():
            payloads = []
            for e in episodes:
                record = lookup_record(store, e.required_ids[0], namespace='global', space='s0',
                                       generation='frozen-v1', query_time=e.query_time)
                source_record = next(s for s in e.supports if s.record_id == e.required_ids[0])
                if (record.source_id != e.required_ids[0] or record.created_at != source_record.created_at
                        or record.payload.numel() != config.memory.payload_dims[0]):
                    raise ValueError('Stored source provenance differs')
                payloads.append(record.payload.float().flatten())
            values[split] = torch.stack(payloads).to(config.train.device)
            labels[split] = torch.tensor([[int(c, 16) for c in e.answer[4:]] for e in episodes], device=config.train.device)
        identity = {'source_manifest_sha256': manifest_hash, 'episodes_sha256': data_hash,
                    'bank_sha256': file_sha256(bank_path), 'bank_identity': reference, 'config': asdict(config),
                    'steps': steps, 'heldout_worlds': heldout_worlds, 'batch': 128,
                    'script_sha256': file_sha256(__file__), 'split_episodes': {k: [e.episode_id for e in v] for k, v in groups.items()},
                    'notice': 'Six supervised hexadecimal classifiers from payload only. Negative readout results do not prove information absent.'}
        if (output/'inputs.json').exists() and json.loads((output/'inputs.json').read_text()) != identity:
            raise ValueError('Readout experiment identity changed')
        atomic_json(output/'inputs.json', identity)
        for name, hidden in [('linear', 0), ('mlp256', 256)]:
            root = output/name
            root.mkdir(exist_ok=True)
            random.seed(83)
            torch.manual_seed(83)
            model = Readout(values['train'], hidden).to(config.train.device)
            optimizer = torch.optim.Muon(model.parameters(), lr=.001, momentum=config.train.muon_momentum,
                ns_steps=config.train.muon_ns_steps, weight_decay=config.train.weight_decay, adjust_lr_fn='match_rms_adamw')
            sampler = torch.Generator().manual_seed(83)
            arm_identity = identity | {'arm': name, 'hidden': hidden, 'parameters': sum(p.numel() for p in model.parameters())}
            path = root/'resume.pt'
            start = restore_probe_state(path, model, optimizer, sampler, arm_identity)
            def save(step):
                save_probe_state(path, model, optimizer, sampler, arm_identity, step, reserve_bytes=config.train.min_free_disk_bytes)
            if not path.exists():
                save(0)
            with Tracking(config, root) as tracking:
                if tracking.run is not None:
                    tracking.run.config.update({'readout': {'arm': name, 'hidden': hidden, 'parameters': arm_identity['parameters']}})
                for step in range(start, steps):
                    available = available_host_memory()
                    if stop_requested(output) or (available is not None and available < config.train.min_system_available_bytes):
                        save(step)
                        raise RuntimeError('Readout stopped at complete optimizer boundary')
                    index = torch.randint(len(values['train']), (128,), generator=sampler).to(config.train.device)
                    with compute_watchdog(config.train.stall_timeout_seconds):
                        optimizer.zero_grad(set_to_none=True)
                        with autocast_context(config):
                            loss = F.cross_entropy(model(values['train'][index]).flatten(0, 1), labels['train'][index].flatten())
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.clip_grad_norm, error_if_nonfinite=True)
                        optimizer.step()
                    if (step+1) % 20 == 0:
                        row = {'step': step+1, 'loss': loss.item(), 'resume_attempt': tracking.attempt, **memory_metrics(config.train.device)}
                        with (root/'metrics.jsonl').open('a') as handle:
                            handle.write(json.dumps(row)+'\n')
                        tracking.log(row)
                        print(json.dumps(row), flush=True)
                save(steps)
            report = {split: assess(model, values[split], labels[split], config) for split in groups}
            atomic_json(root/'results.json', {'identity': arm_identity, 'results': report})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'episodes', 'bank', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--steps', type=int, default=1600)
    parser.add_argument('--heldout-worlds', type=int, default=32)
    args = parser.parse_args()
    def stop(_signal, _frame):
        (control_dir(args.output)/'STOP').touch()
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    run(args.source, args.episodes, args.bank, args.output, args.steps, args.heldout_worlds)
