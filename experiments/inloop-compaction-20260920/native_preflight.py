"""Native shared-gradient and stored-code validation; no optimizer update or capability claim."""
import argparse
import copy
import json
from pathlib import Path

import torch

from sdkb.agent import SDKBAgent
from sdkb.checkpoints import resolve_checkpoint
from sdkb.cluster_store import state_fingerprint
from sdkb.data import make_multiuse_world
from sdkb.evaluation import build_shared_bank, build_persistent_codes
from safetensors.torch import load_model
from sdkb.operations import atomic_json
from sdkb.replay import ReplayTape
from sdkb.sessions import read_session
from sdkb.store import DiskStore
from sdkb.training import config_from_run, autocast_context, stored_channel, environment_report
from sdkb.trajectories import file_sha256


def main(source, output, *, raw=False, no_autocast_cache=False, objective='interleaved'):
    output.mkdir(parents=True, exist_ok=False)
    checkpoint = resolve_checkpoint(source, verify=True)
    config = config_from_run(checkpoint)
    config.memory.compaction, config.memory.compact_records = 'synthetic', 1
    config.memory.compaction_objective = objective
    config.memory.behavior_kl_weight = .2 if objective == 'paired' else 0.
    config.memory.compaction_warmup = 0
    config.memory.compaction_probability = .5
    config.validate()
    torch.set_num_threads(config.train.threads)
    torch.manual_seed(97)
    torch.set_autocast_cache_enabled(not no_autocast_cache)
    a = SDKBAgent(config).to(config.train.device).train()
    missing, unexpected = load_model(a, str(checkpoint/'model.safetensors'), strict=False, device=config.train.device)
    assert missing and all(n.startswith('compactor.') for n in missing) and not unexpected
    a.backbone.loops = 3
    b = copy.deepcopy(a)
    e = next(e for e in make_multiuse_world(0, split='native-compaction-preflight', bindings=2) if len(e.required_ids) == 2)
    required = [i for i, s in enumerate(e.supports) if s.record_id in e.required_ids]
    sources = [a.text_ids(s.text, source=True) for s in e.supports]
    prompt, target = a.prompt_ids(e.query), a.target_ids(e.answer)
    cpu, cuda = torch.get_rng_state(), torch.cuda.get_rng_state_all()
    with autocast_context(config):
        records = [stored_channel(a, a.produce(s)) for s in sources]
        reference = a(prompt, target, records, required, compact=not raw)
    reference.loss.backward()
    expected_cpu, expected_cuda = torch.get_rng_state(), torch.cuda.get_rng_state_all()
    torch.set_rng_state(cpu)
    torch.cuda.set_rng_state_all(cuda)
    tape = ReplayTape(verify_outputs=True)
    with autocast_context(config):
        records = [tape.capture(b, lambda s=s, owner=b: stored_channel(owner, owner.produce(s))) for s in sources]
        replayed = b(prompt, target, records, required, compact=not raw)
    replayed.loss.backward()
    tape.backward()
    errors = {}
    for (name, x), (_, y) in zip(a.named_parameters(), b.named_parameters(), strict=True):
        if x.grad is None:
            assert y.grad is None, name
        else:
            delta = x.grad-y.grad
            errors[name] = {'max_abs': float(delta.abs().max()),
                            'reference_max_abs': float(x.grad.abs().max()),
                            'relative_l2': float(delta.norm()/x.grad.norm().clamp_min(1e-20))}
    atomic_json(output/'gradient-diagnostics.json', errors)
    relative_tolerance = 1e-5 if no_autocast_cache else .005
    assert max(e['relative_l2'] for e in errors.values()) < relative_tolerance
    assert a.write_slots.grad.abs().sum() > 0
    if not raw:
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in a.compactor.parameters())
    assert torch.equal(torch.get_rng_state(), expected_cpu)
    assert all(torch.equal(x, y) for x, y in zip(torch.cuda.get_rng_state_all(), expected_cuda, strict=True))
    del b, records, replayed, tape
    a.zero_grad(set_to_none=True)
    a.eval().requires_grad_(False)
    with torch.no_grad(), autocast_context(config):
        records = [stored_channel(a, a.produce(s)) for s in sources]
        integrated = a(prompt, target, records, required, compact=True)
        store = DiskStore(output/'bank.sqlite')
        build_shared_bank(a, store, [e], writer_identity=file_sha256(checkpoint/'manifest.json'))
        codes, sizes = build_persistent_codes(a, store, [e])
        def forbidden(*_args, **_kwargs):
            raise AssertionError('Stored inference regenerated a payload or code')
        a.produce = forbidden
        a.compactor.forward = forbidden
        a._compact_values = forbidden
        session = read_session(a, store, prompt, namespace='global', generation='frozen-v1',
                               query_time=e.query_time, oracle_ids=e.required_ids, cluster_bank=codes)
        stored = a.conditioned_nll(prompt, target, session.memory)
    integrated_nll = integrated.compact_nll
    torch.testing.assert_close(integrated_nll, stored, atol=.03, rtol=0)
    atomic_json(output/'results.json', {'source_manifest_sha256': file_sha256(checkpoint/'manifest.json'),
        'script_sha256': file_sha256(__file__), 'environment': environment_report(),
        'objective': objective, 'gradient_compact': not raw, 'autocast_cache_enabled': not no_autocast_cache,
        'gradient_relative_l2_tolerance': relative_tolerance,
        'gradient_errors_by_parameter': errors, 'gradient_notice': 'BF16 shared-parameter accumulation order is diagnosed explicitly.',
        'loss': float(reference.loss.detach()), 'rng_exact': True,
        'integrated_nll': float(integrated_nll), 'stored_nll': float(stored), 'stored_nll_atol': .03,
        'reader_hash': state_fingerprint(a.reader), 'code_storage': sizes,
        'notice': 'Native BF16 execution check only. Random new compactor; no training or capability result.'})
    print(json.dumps({'max_gradient_error': max(e['max_abs'] for e in errors.values()), 'stored_nll_error': float((integrated_nll-stored).abs())}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--raw', action='store_true')
    parser.add_argument('--no-autocast-cache', action='store_true')
    parser.add_argument('--objective', choices=['interleaved', 'paired'], default='interleaved')
    args = parser.parse_args()
    main(args.source, args.output, raw=args.raw, no_autocast_cache=args.no_autocast_cache, objective=args.objective)
