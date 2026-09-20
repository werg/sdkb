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


## Supplemental diagnostics declared during training

Before either endpoint is available, `prepare_training_diagnostic.py` seals the
same 64 endpoint questions from the first 32 training worlds for both arms. It
keeps the original whole-answer episode and drops six replica questions, verifies
that each complete episode is identical in both corpora, and records the planned
sampler exposures. No model outcomes choose this subset. It supplements the
pre-launch held-out protocol; it does not replace it or constitute held-out evidence.

All 64 targets are sampled 11–36 times by the small arm. In the large arm, 42 have
zero endpoint-target exposures, 19 have one, two have two, and one has three.
These are target-supervision counts, not absence of the endpoint from source text
in other tasks. Evaluate both final models with the existing frozen six-condition
oracle evaluator on this common corpus, and stratify the results by the recorded
exposures. This distinguishes training fit from fresh recall without changing the
models, training budget or primary confirmation split. In addition, run the existing
selected-text/no-memory control for both held-out endpoints to check that their
decoders retain the source-text task. The repeated no-memory predictions must
match each latent reference exactly. All outputs remain external; the committed
`training-diagnostic-inputs.json` pins selection and exposure counts.


`run_supplemental.py` waits for the primary confirmation queue to release its lock,
requires both complete step-4,000 references, then runs at most two fit evaluations
followed by at most two selected-text controls. It uses the same study-root and
`confirmation-queue` stop controls, terminates and joins active children on exit,
and rechecks the sealed corpora after waiting. Launch it with the original
`c58028c` model checkout's `PYTHONPATH` and Git environment; pass that checkout as
`--frozen-model-code`. The controller may come from its separately frozen newer
checkout; it refuses an inconsistent model import path. Resume by clearing only
released queue/root controls as above and rerunning; child progress is reusable.
The shared queue's selected-entry-point regression and full suite pass (449 tests).


## Training endpoints committed and audited

Both native training processes exited successfully at update 4,000. The audit
verifies the sealed corpus hashes, all 4,000 sampled recurrent depths, the final
16,000-microbatch sampler state, complete Muon state without pending accumulation,
and clean frozen `c58028c` provenance. `endpoint-audit.json` pins both final model
and manifest hashes. Each arm retains exactly its initial and final checkpoints.
Peak logged CUDA allocations were about 2.52 GiB; minimum logged host available
memory was about 51.2 GiB. Training took about 106 minutes per arm under shared
machine contention; these times are not an isolated throughput comparison.

After training locks released, SHA256-verified hard-link deduplication removed
2,034,699,352 redundant initial-model bytes while preserving every complete
checkpoint and its optimizer/config/RNG/cache state. The record is
`../operations-20260920/endpoint-freshness-initial-dedup.json`. Primary held-out
confirmation is running; capability results remain pending.


The completed training curves are reproducible with `plot_training.py --root RUN_ROOT`;
PNG/SVG figures stay in the external run's `analysis` directory. The final
100-update mean target NLL is 0.173 for 256 worlds and 1.050 for 8,192 worlds,
versus 1.826 and 1.866 in the first 100 updates. `training-nll.json` pins metric
file hashes and the calculation. These are teacher-forced training losses with
one seed and different data streams, not exact-recall or generalization evidence.


## Completed confirmation

Both primary and supplemental controllers exited successfully. The identity-checked
`comparison.json` retains all generation/counterfactual summaries, teacher NLL,
text controls, exposure strata and raw-result hashes.

| Result | Source | 256 worlds | 8,192 worlds |
|---|---:|---:|---:|
| Fresh original endpoints /64 | 0 | 0 | 0 |
| Changed endpoints /64 | 0 | 0 | 1 |
| Both original and changed endpoint correct /64 | 0 | 0 | 0 |
| Original actions /128 | 128 | 128 | 128 |
| Permission-change action pairs both correct /128 | 128 | 128 | 128 |
| Restoration-change action pairs both correct /64 | 64 | 64 | 63 |
| Selected-text endpoints /64 | 64 | 64 | 64 |
| Fresh identifier mean question target NLL | 5.1163 | 3.3872 | 1.7493 |
| Common training-corpus endpoints /64 | Not run | 45 | 0 |

The small arm preserves every tested direct-rule answer and invariant action branch.
The large arm misses one changed-restoration action, one changed-permission direct
answer, and two direct-permission answers after an irrelevant endpoint change.
No original action flips under endpoint changes in either arm. No/zero-memory
actions are 64/46 for the small arm and 34/46 for the large arm, each out of 128.
All 64 repeated no-memory identifier strings in each text control exactly match
the corresponding latent reference; none is correct.

Training fit is not general copying. The small arm's 45/64 common-corpus answers
fall to 0/64 after endpoint replacement and 26/64 after an irrelevant permission
change; no/zero-memory get zero. The large arm gets zero in every condition and
every recorded exposure stratum (42 zero, 19 single, three multiple target exposures).
Neither endpoint is promoted as a solution to exact-detail transfer.

### Post-hoc error description

`analyze_identifier_errors.py` and `identifier-errors.json` record a descriptive
analysis of completed generations, without new model queries. Both continuations
produce correctly formatted original answers in 64/64 cases; the source produces
none. Correct hex positions are 102/384 for the small arm and 125/384 for the large
arm, versus 29/384 and 35/384 with zero payloads. First-position correctness is
44/64 and 47/64, with much weaker later positions. This is partial content transfer,
not exact recall. Malformed outputs receive zero position credit.

Of the 64 original predictions, 53 small-arm predictions belong to its 512-target
training vocabulary; only one large-arm prediction belongs to its sampled
7,516-target vocabulary. The larger corpus reduces this particular old-answer
reuse pattern and improves teacher NLL, while leaving reliable exact copying
unsolved. Wrong predictions change under endpoint interventions in all 64 cases
for both arms, but correct original/changed pairs remain zero. Single-seed,
post-hoc position statistics do not establish an information-theoretic limit or
identify one responsible module. The next diagnostic will compare recoverability
from stored payloads and reader outputs under identical queries.
