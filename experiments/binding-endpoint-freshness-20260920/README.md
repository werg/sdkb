# Endpoint freshness with retained rule tasks

## Protocol committed before launch

The focused 64-target diagnostic fitted all original training answers but failed
all changed and fresh endpoints, and degraded rule/action behavior. Character
questions failed even with selected source text (0/96), so they are not a viable
training interface yet. This study keeps the full-endpoint question that the
pretrained decoder can answer from text and retains the original rule/action tasks.

Warm-start both arms from the useful original recurrent-core MLP checkpoint in
`inputs.json`, with fresh native Muon state. Compare 256 versus 8,192 source worlds.
Each world supplies 14 full-endpoint questions, four action questions, two permission
questions and two restoration questions. Slot-major ordering and power-of-two world
counts reproduce the same task, source-count and recurrent-depth RNG schedule.
Run 4,000 updates, accumulation four, seed 127, core LR 5e-6 and memory LR 1e-4.
Both arms consume 10,097 identifier microbatches and 5,903 rule/action microbatches.
The smaller arm exposes 512 distinct targets 8–36 times each; the larger samples
7,516 distinct targets, mostly once. Its full corpus has 16,371 distinct endpoint
strings among 16,384 endpoint records; collisions are retained and disclosed.
World content and endpoint freshness both differ. Equal updates and task schedules
are not equal token/FLOP budgets or a pure endpoint-only intervention.

Before model evaluation, seal 32 fresh worlds/320 questions. Take the first worlds
whose source IDs, original endpoints and inverted endpoints are disjoint from source
finetuning and both nested training corpora; seed 27 is rejected by that policy.
`inputs.json` pins all input hashes, accepted seeds and the deterministic sampler.
Use fixed full-endpoint questions and original rule/action questions. Evaluate source
and both endpoints with the same stored-only oracle confirmation: exact generation,
no/zero-memory controls, endpoint/permission/restoration interventions, and teacher
NLL reported separately. Query text and oracle-selected prior records contain no
teacher answer. Required-record selection is diagnostic supervision, not global
routing. Correct counterfactual copying and rule retention determine progress;
format, training fit and decreasing NLL do not establish capacity substitution.

## Storage and run control

Artifacts go directly to `/archive/runs/binding-endpoint-freshness-20260920`, which
is `/mnt/external/sdkb-archive/runs/binding-endpoint-freshness-20260920` on this host.
Paths are configuration, not a required deployment platform. Initial/final/emergency
checkpoints only (cadence 10,000, keep two), complete optimizer/hyperparameter/RNG
and partial-gradient resume, offline W&B, 10 GiB disk and 8 GiB host reserve, 0.35
CUDA fraction, and a 300-second compute watchdog are configured. Never add GPU
allocation to CPU usage on this unified-memory machine. No NVIDIA stack changes.

For each SIZE in 256 and 8192, from the frozen study checkout:

```bash
sdkb train --config /archive/runs/binding-endpoint-freshness-20260920/worlds-SIZE.yaml \
  --output /archive/runs/binding-endpoint-freshness-20260920/worlds-SIZE \
  --init-from /archive/runs/binding-muon-continuation-20260919/recurrent_core/checkpoints/step-000001600-ae2efaf16b1c
python experiments/binding-endpoint-freshness-20260920/run_confirmation.py \
  --root /archive/runs/binding-endpoint-freshness-20260920
```

The controller evaluates the source, waits for both training ownership locks to
release with committed step-4000 checkpoints, then runs at most two evaluators.
Stop the study root to stop the queue and flush active evaluator progress. Also
request `sdkb runs stop --output` for each training stage to commit emergency state.
Wait for locks to release. Resume each stage using the same config/output and
`--resume` in place of `--init-from`. To restart the deliberately stopped queue,
clear only root/queue controls under their locks after the old queue exits:

```python
from pathlib import Path
from sdkb.operations import run_lock
root = Path('/archive/runs/binding-endpoint-freshness-20260920')
with run_lock(root / 'confirmation-queue'), run_lock(root):
    pass
```

Then rerun the controller; validated completed evaluations are reused. A hard kill
can recover only committed state. Training checkpoints are immutable; resume the
mutable stage root. Deduplicate identical initial weights only after training locks
release, retaining complete recovery state. The pre-launch full suite passed 428
tests with four existing warnings; Ruff passed.

## Launch record

Both training arms and the bounded confirmation controller launched from frozen
commit `c58028c`. Both external initial checkpoints committed successfully and
recorded all four named Muon/AdamW groups and offline W&B identity. Source held-out
confirmation runs alongside training. Endpoint results remain pending.

## Source controls on the sealed split

The source gets 128/128 actions and 64/64 in each direct rule family, including all
tested changed-rule answers and invariant action branches. Identifiers remain 0/64
in every stored-memory condition. No/zero-memory actions are 64/128 and 61/128.
The selected-text identifier control gets 64/64, with all 64 repeated no-memory
predictions exactly reproducing the frozen latent reference (0/64 correct). This
confirms the question interface is viable for this split. Source text is a separate
control with unequal token budgets, never an inference fallback. The controls ran
from frozen `c58028c`; `source-confirmation.json` and `source-text.json` pin outputs.
