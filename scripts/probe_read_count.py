"""Learn one-versus-two record requests from a frozen causal address query."""
import argparse
import copy
from dataclasses import asdict
import json
from pathlib import Path
import signal

from safetensors.torch import load_file
import torch
from torch import nn
from torch.nn import functional as F

from probe_routing_stop import StopProbe
from sdkb.checkpoints import resolve_checkpoint
from sdkb.operations import atomic_json, control_dir, run_lock, stop_requested
from sdkb.optimizers import MuonAdamW
from sdkb.runtime import configure_memory, available_host_memory, memory_metrics
from sdkb.tracking import Tracking
from sdkb.training import config_from_run, autocast_context
from sdkb.trajectories import file_sha256


def run(source, features_root, output, steps=200):
    output.mkdir(parents=True, exist_ok=True)
    with run_lock(output, clear_stop=False):
        checkpoint = resolve_checkpoint(source, verify=True)
        config = copy.deepcopy(config_from_run(checkpoint))
        config.model.freeze_backbone = True
        config.memory.independent_routing_query = True
        config.train.seed, config.train.steps = 53, steps
        config.train.gradient_accumulation = 1
        config.train.learning_rate = .01
        config.train.loop_counts = []
        config.train.oracle_anchor_weight = 0.
        config.train.wandb_group = 'read-count-feature-probe'
        inputs = json.loads((features_root / 'inputs.json').read_text())
        endpoint = features_root / 'full_state_query-resume.pt'
        state = torch.load(endpoint, weights_only=True, map_location='cpu')
        if (state['identity'] != inputs or state['step'] != inputs['steps']
                or inputs['checkpoint_manifest_sha256'] != file_sha256(checkpoint / 'manifest.json')):
            raise ValueError('Frozen address source differs')
        configure_memory(config.train)
        torch.set_num_threads(config.train.threads)
        torch.manual_seed(53)
        router = StopProbe(state['model'], False).to(config.train.device).eval().requires_grad_(False)
        identity = {'source_endpoint_sha256': file_sha256(endpoint), 'source_identity': inputs,
                    'choices': [1, 2], 'steps': steps, 'batch_size': 1280, 'seed': 53,
                    'learning_rate': .01, 'optimizer': 'Muon matrix + AdamW bias',
                    'parameters': 130, 'config': asdict(config), 'script_sha256': file_sha256(__file__),
                    'notice': 'Frozen-address query classifier; support cardinality labels are training-only.'}
        values, labels, scores = {}, {}, {}
        identity['features_sha256'] = {}
        for split in ('train', 'heldout'):
            path = features_root / f'{split}-features.safetensors'
            identity['features_sha256'][split] = file_sha256(path)
            features = {k: v.to(config.train.device) for k, v in load_file(str(path)).items()}
            with torch.no_grad(), autocast_context(config):
                query = F.normalize(router.query_head(features['raw_query']), dim=-1)
                values[split] = F.normalize(router.query_map(query).float(), dim=-1)
                scores[split], _ = router(features)
            labels[split] = features['lengths'] - 1
            if not torch.all((labels[split] == 0) | (labels[split] == 1)):
                raise ValueError('Only one/two required-record groups are supported')
        if (output / 'inputs.json').exists() and json.loads((output / 'inputs.json').read_text()) != identity:
            raise ValueError('Count probe identity changed')
        atomic_json(output / 'inputs.json', identity)
        head = nn.Linear(values['train'].shape[-1], 2).to(config.train.device)
        nn.init.zeros_(head.weight)
        nn.init.zeros_(head.bias)
        optimizer = MuonAdamW([{'params': [head.weight], 'lr': .01}],
                             [{'params': [head.bias], 'lr': .01}], config.train)
        generator = torch.Generator().manual_seed(53)
        start, path = 0, output / 'resume.pt'
        if path.exists():
            saved = torch.load(path, weights_only=True, map_location=config.train.device)
            if saved['identity'] != identity:
                raise ValueError('Read-count resume identity differs')
            head.load_state_dict(saved['head'])
            optimizer.load_state_dict(saved['optimizer'])
            generator.set_state(saved['sampling_rng'].cpu())
            start = saved['step']
        def save(step):
            temporary = path.with_suffix('.tmp')
            torch.save({'identity': identity, 'head': head.state_dict(), 'step': step,
                        'optimizer': optimizer.state_dict(), 'sampling_rng': generator.get_state()}, temporary)
            temporary.replace(path)
        with Tracking(config, output) as tracking:
            if tracking.run is not None:
                tracking.run.config.update({'read_count_probe': identity, 'frozen_address_model': True})
            for step in range(start, steps):
                available = available_host_memory()
                if stop_requested(output) or (available is not None and available < config.train.min_system_available_bytes):
                    save(step)
                    raise RuntimeError('Read-count probe stopped with complete optimizer/sampling state')
                indices = torch.randint(len(labels['train']), (1280,), generator=generator).to(config.train.device)
                optimizer.zero_grad(set_to_none=True)
                with autocast_context(config):
                    loss = F.cross_entropy(head(values['train'][indices]).float(), labels['train'][indices])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(head.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
                if (step + 1) % 20 == 0:
                    row = {'step': step + 1, 'loss': loss.item(), **memory_metrics(config.train.device)}
                    tracking.log(row)
                    print(json.dumps(row), flush=True)
            save(steps)
            result = {'identity': identity, 'by_split': {}}
            for split in ('train', 'heldout'):
                with torch.no_grad(), autocast_context(config):
                    predicted = head(values[split]).argmax(-1)
                result['by_split'][split] = {'queries': len(predicted),
                    'correct_count': int((predicted == labels[split]).sum()),
                    'confusion': {f'{actual+1}->{chosen+1}': int(((labels[split] == actual) & (predicted == chosen)).sum())
                                  for actual in (0, 1) for chosen in (0, 1)}}
                atomic_json(output / f'{split}-counts.json', {'predicted': (predicted + 1).cpu().tolist()})
            atomic_json(output / 'results.json', result)
            print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--features', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    def stop(_signal, _frame):
        (control_dir(args.output) / 'STOP').touch()
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    run(args.source, args.features, args.output)
