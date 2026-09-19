"""Frozen-feature address learnability probe; never a stored-memory capability score."""
import argparse
import copy
import json
import signal
from pathlib import Path

from safetensors.torch import load_file, load_model, save_file
import torch
from torch import nn
from torch.nn import functional as F

from sdkb.agent import SDKBAgent
from sdkb.checkpoints import resolve_checkpoint
from sdkb.data import load_episodes, make_multiuse_world, save_episodes
from sdkb.operations import atomic_json, run_lock, stop_requested
from sdkb.training import autocast_context, config_from_run
from sdkb.trajectories import file_sha256


def pair_loss(scores, required, lengths):
    """Batched PL objective for one required item or an unordered required pair."""
    first, second = required.unbind(-1)
    logp = scores.log_softmax(-1)
    a, b = logp.gather(1, first[:, None])[:, 0], logp.gather(1, second[:, None])[:, 0]
    mask_a = F.one_hot(first, scores.shape[-1]).bool()
    mask_b = F.one_hot(second, scores.shape[-1]).bool()
    ab = a + scores.gather(1, second[:, None])[:, 0] - scores.masked_fill(mask_a, -torch.inf).logsumexp(-1)
    ba = b + scores.gather(1, first[:, None])[:, 0] - scores.masked_fill(mask_b, -torch.inf).logsumexp(-1)
    return -torch.where(lengths == 1, a, torch.logaddexp(ab, ba)).mean()


class AddressProbe(nn.Module):
    def __init__(self, agent, raw_query):
        super().__init__()
        self.raw_query = raw_query
        self.key = copy.deepcopy(agent.key_head)
        self.address = copy.deepcopy(agent.address_maps[0])
        self.query_map = copy.deepcopy(agent.query_maps[0])
        self.query_head = copy.deepcopy(agent.query_head) if raw_query else None

    def forward(self, features):
        key = F.normalize(self.key(features['key']), dim=-1)
        key = F.normalize(self.address(key), dim=-1)
        query = (F.normalize(self.query_head(features['raw_query']), dim=-1)
                 if self.raw_query else features['query'])
        query = self.query_map(query)
        return (F.normalize(query.float(), dim=-1)[:, None] *
                F.normalize(key.float(), dim=-1)).sum(-1) / .1


@torch.no_grad()
def extract(agent, episodes, config, root):
    result = {'key': [], 'raw_query': [], 'query': [], 'required': [], 'lengths': []}
    sources, captured = {}, []
    def capture(_module, args):
        captured.append(args[0].detach().clone())
    hook = agent.key_head.register_forward_pre_hook(capture)
    try:
        for e in episodes:
            for source in e.supports:
                if source.created_at >= e.query_time:
                    raise ValueError('Future source')
                if source.record_id in sources:
                    if sources[source.record_id][0] != source.text:
                        raise ValueError('Source identity changed')
                    continue
                captured.clear()
                with autocast_context(config):
                    agent.produce(agent.text_ids(source.text, source=True))
                if len(captured) != 1:
                    raise ValueError('Expected one writer key')
                sources[source.record_id] = (source.text, captured[0][0].cpu())
            if stop_requested(root):
                raise RuntimeError('Probe stopped during reproducible feature extraction')
    finally:
        hook.remove()
    hook = agent.query_head.register_forward_pre_hook(capture)
    try:
        for i, e in enumerate(episodes):
            captured.clear()
            queries = []
            def provider(completed, query):
                if completed == 1:
                    queries.append(query.detach().cpu()[0])
                return None
            with autocast_context(config):
                agent.plan_loop_memory(agent.prompt_ids(e.query), provider)
            if not queries or not captured:
                raise ValueError('Missing causal first query')
            ids = [s.record_id for s in e.supports]
            required = [ids.index(rid) for rid in e.required_ids]
            if len(ids) != 4 or len(required) not in {1, 2}:
                raise ValueError('Probe requires four candidates and one/two required records')
            result['key'].append(torch.stack([sources[rid][1] for rid in ids]))
            result['raw_query'].append(captured[0][0].cpu())
            result['query'].append(queries[0])
            result['required'].append(torch.tensor((required * 2)[:2]))
            result['lengths'].append(torch.tensor(len(required)))
            if i % 128 == 0:
                print(json.dumps({'extracted_queries': i, 'total': len(episodes)}), flush=True)
            if stop_requested(root):
                raise RuntimeError('Probe stopped during reproducible feature extraction')
    finally:
        hook.remove()
    return {name: torch.stack(values) for name, values in result.items()}


@torch.no_grad()
def score(model, features, episodes, config):
    with autocast_context(config):
        scores = model(features)
        loss = pair_loss(scores, features['required'], features['lengths']).item()
    ranks = scores.argsort(descending=True, stable=True).cpu().tolist()
    families = {}
    for rank, e in zip(ranks, episodes, strict=True):
        ids = [s.record_id for s in e.supports]
        selected = {ids[i] for i in rank[:2]}
        group = families.setdefault(e.task_family, {'count': 0, 'full_required': 0, 'sufficient': 0, 'top_one_required': 0})
        group['count'] += 1
        group['full_required'] += set(e.required_ids) <= selected
        group['sufficient'] += any(set(g) <= selected for g in e.sufficient_groups or (e.required_ids,))
        group['top_one_required'] += ids[rank[0]] in e.required_ids
    return {'loss': loss, 'by_family': families}


def run(checkpoint, root, steps):
    root.mkdir(parents=True, exist_ok=True)
    with run_lock(root, clear_stop=False):
        checkpoint = resolve_checkpoint(checkpoint, verify=True)
        config = config_from_run(checkpoint)
        torch.set_num_threads(config.train.threads)
        torch.manual_seed(43)
        agent = SDKBAgent(config).to(config.train.device).eval()
        load_model(agent, str(checkpoint / 'model.safetensors'), device=config.train.device)
        if config.memory.read_steps != 1 or len(config.memory.payload_dims) != 1:
            raise ValueError('Probe is limited to first-read single-space models')
        train = load_episodes(config.train.episodes_file)
        heldout = [e for i in range(32) for e in make_multiuse_world(i, split='routing-feature-probe-20260919', bindings=2)]
        save_episodes(root / 'heldout.jsonl', heldout)
        identity = {'checkpoint_manifest_sha256': file_sha256(checkpoint / 'manifest.json'),
                    'train_sha256': file_sha256(config.train.episodes_file),
                    'heldout_sha256': file_sha256(root / 'heldout.jsonl'), 'steps': steps,
                    'train_config': vars(config.train), 'seed': 43, 'learning_rate': config.train.learning_rate,
                    'optimizer': 'native Muon', 'batch': 'all training queries per update',
                    'notice': 'Frozen-feature learnability probe, unequal parameter counts; no downstream capability claim.'}
        identity_path = root / 'inputs.json'
        if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
            raise ValueError('Probe identity changed')
        atomic_json(identity_path, identity)
        features = {}
        for split, episodes in [('train', train), ('heldout', heldout)]:
            path = root / f'{split}-features.safetensors'
            if not path.exists():
                data = extract(agent, episodes, config, root)
                temporary = path.with_suffix('.tmp')
                save_file(data, str(temporary))
                temporary.replace(path)
            features[split] = {k: v.to(config.train.device) for k, v in load_file(str(path)).items()}
        models = {name: AddressProbe(agent, raw).to(config.train.device)
                  for name, raw in [('compressed_query', False), ('full_state_query', True)]}
        with torch.no_grad(), autocast_context(config):
            a, b = [model(features['train']) for model in models.values()]
        if not torch.equal(a, b):
            raise ValueError(f'Initial scores differ: {(a-b).abs().max().item()}')
        del agent
        report = {'identity': identity, 'initial_scores_exact': True, 'arms': {}}
        for name, model in models.items():
            optimizer = torch.optim.Muon(model.parameters(), lr=config.train.learning_rate,
                momentum=config.train.muon_momentum, ns_steps=config.train.muon_ns_steps,
                weight_decay=config.train.weight_decay, adjust_lr_fn='match_rms_adamw')
            state_path = root / f'{name}-resume.pt'
            start = 0
            if state_path.exists():
                state = torch.load(state_path, weights_only=True, map_location=config.train.device)
                if state['identity'] != identity:
                    raise ValueError('Resume identity changed')
                model.load_state_dict(state['model'])
                optimizer.load_state_dict(state['optimizer'])
                start = state['step']
            def save(step):
                temporary = state_path.with_suffix('.tmp')
                torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                            'step': step, 'identity': identity}, temporary)
                temporary.replace(state_path)
            initial = score(model, features['train'], train, config) if start == 0 else None
            for step in range(start, steps):
                if stop_requested(root):
                    save(step)
                    raise RuntimeError('Probe stopped; address optimizer state saved')
                optimizer.zero_grad(set_to_none=True)
                with autocast_context(config):
                    loss = pair_loss(model(features['train']), features['train']['required'], features['train']['lengths'])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.clip_grad_norm)
                optimizer.step()
                if (step + 1) % 100 == 0:
                    print(json.dumps({'arm': name, 'step': step + 1, 'loss': loss.item()}), flush=True)
            save(steps)
            report['arms'][name] = {'parameters': sum(p.numel() for p in model.parameters()),
                'initial_train': initial, 'train': score(model, features['train'], train, config),
                'heldout': score(model, features['heldout'], heldout, config)}
            atomic_json(root / 'results.json', report)
        print(json.dumps(report), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=800)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error('--steps must be positive')
    from sdkb.operations import control_dir
    def request_stop(_signal, _frame):
        (control_dir(args.output) / 'STOP').touch()
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    run(args.checkpoint, args.output, args.steps)
