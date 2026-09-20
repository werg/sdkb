# Deliberate exact-copy training-fit diagnostic

The fixed-question continuation has 0/64 exact latent identifier generations even
on its most frequently sampled training questions, despite 64/64 selected-text
copying on held-out questions. Those training targets were seen only three to five
times. Before attributing failure to generalization or the interface, test whether
repeated training can fit a small supplied-memory communication task.

Warm-start from the completed fixed-question endpoint with fresh Muon optimizer
state. Train only the exact 64 identifier episodes selected by the preceding
training-fit audit, preserving their source/target/query identities. They cover 61
worlds. Use 1,600 updates, accumulation four, seed 113, unchanged core/memory learning
rates (5e-6/1e-4), interface, precision, replay, recurrent-depth distribution and
text-anchor weight. The declared sampler gives each identifier target 80–135
additional exposures (mean 100). The only task family is identifier copying.
This intentionally changes the task distribution and is not a matched-budget
capability comparison with the mixed-task curriculum.

Primary diagnosis: exact stored-memory generation on those same 64 training
queries. Report no/zero-memory controls and endpoint/rule interventions separately;
fitting original memories does not establish correct responses to changed endpoints.
Reusing fixed questions relies on oracle-selected records, not global retrieval.

Before launch, generate 32 fresh worlds/320 questions in the fixed-query form.
Evaluate the source and candidate on the identical held-out file, including all
rule/action families and the same counterfactual controls. The held-out source IDs,
original endpoint targets and inverted counterfactual endpoints are disjoint from
source finetuning, all broad continuation data and the preceding confirmation.
`inputs.json` pins the checkpoint, corpus, configuration, split and overlap checks.
Training fit, held-out copying and rule retention remain distinct outcomes. Do not
promote the endpoint based on training fit alone.

All artifacts go directly to `/archive/runs/binding-copy-fit-20260920`. Use initial,
final and emergency checkpoints only (cadence 10,000, keep two), complete main-trainer
resume state, offline W&B, a 10 GiB disk reserve, 8 GiB host reserve, 0.35 allocator
fraction and 300-second compute watchdog. No model downloads, teacher calls or
NVIDIA stack changes. The pre-run full suite passes 412 tests and native preflight
passes. Training has not yet launched when this protocol is committed.

Training and its bounded confirmation controller launched from frozen commit
`a75b4e8`. The external initial checkpoint committed successfully; offline W&B and
all four named Muon/AdamW parameter groups are recorded. The source held-out
evaluation runs alongside training; final training/held-out checks are queued.

## Run control and reproduction

Use the frozen `a75b4e8` checkout with the existing native environment. On this
machine the host archive is `/mnt/external/sdkb-archive`, mounted as `/archive`
in the container. The following paths are container paths; on another machine,
prepare a fresh run with its actual artifact paths and preserve the recorded
scientific settings and input hashes.

```bash
sdkb train --config /archive/runs/binding-copy-fit-20260920/copy.yaml \
  --output /archive/runs/binding-copy-fit-20260920/copy \
  --init-from /archive/runs/binding-fixed-query-detail-20260920/fixed/checkpoints/step-000001600-d8faf965ff8c
python experiments/binding-copy-fit-20260920/run_confirmation.py \
  --root /archive/runs/binding-copy-fit-20260920
```

To stop both training and queued/in-progress confirmation, request both controls:

```bash
sdkb runs stop --output /archive/runs/binding-copy-fit-20260920
sdkb runs stop --output /archive/runs/binding-copy-fit-20260920/copy
```

Training finishes its current microbatch/replay and commits complete emergency
state, including partial accumulated gradients. The queue stops launching jobs and
asks active evaluators to save completed progress. Wait for ownership locks to
release before restarting. Resume training with the same frozen checkout/config:

```bash
sdkb train --config /archive/runs/binding-copy-fit-20260920/copy.yaml \
  --output /archive/runs/binding-copy-fit-20260920/copy --resume
```

A deliberate queue restart also requires clearing its root stop request through
the operations helper; the controller deliberately uses `clear_stop=False` and
will not silently discard an operator stop. Completed confirmation files are
validated/reused. A hard kill can only return training to a committed checkpoint;
it cannot promise the newest partial microbatch state.

After the stopped training has been resumed and the old queue has exited, clear
only the study/queue stop controls while holding their ownership locks, then rerun
the confirmation command above:

```python
from pathlib import Path
from sdkb.operations import run_lock

root = Path('/archive/runs/binding-copy-fit-20260920')
with run_lock(root / 'confirmation-queue'), run_lock(root):
    pass
```

The initial source training-fit reference is already recorded at
`/archive/runs/binding-fixed-query-detail-20260920/train-fit/fixed/results.json`.
Its checkpoint and episode hashes match this protocol's source and training file;
it need not be regenerated for the comparison.

The source checkpoint confirmation on the newly declared held-out split is complete:
128/128 original and counterfactual actions, 61/64 direct permissions, 64/64 direct
restoration answers and 0/64 identifiers. `source-confirmation.json` records this
control before the candidate endpoint is evaluated.

## Completed training and training-set confirmation

All 1,600 Muon updates completed. Saved sampler state and every recurrent depth
match the declared plan: 6,400 identifier microbatches, 64 distinct targets,
80–135 exposures each. Initial weights exactly match the source. `endpoint.json`
pins the final checkpoint and resource observations. Only initial/final sets were
written; after ownership released, verified deduplication reclaimed 1,017,349,676
bytes while preserving both complete recovery sets and all paths.

| Training-set condition | Exact identifiers |
|---|---:|
| Original stored memories | **64/64** |
| No memory | 0/64 |
| Zeroed values | 1/64 |
| Endpoint replacement | **0/64** |
| Irrelevant permission change | **17/64** |
| Irrelevant restoration change | 64/64 |

Original target NLL is 0.00061444. Every endpoint replacement changes the model's
answer, but every changed prediction belongs to the original 64-target training
vocabulary; none of those counterfactual targets belongs to that vocabulary.
Permission changes falsely alter 47/64 identifiers, with 60/64 predictions still
in the training vocabulary. This descriptive post-hoc vocabulary audit is consistent
with memorized associations, not evidence of a learned general string-copying rule
or a proof of the internal mechanism. Stored memory is necessary for the training
fit, but correct intervention behavior is missing. `training-confirmation.json`
records all aggregates and result hashes. Fresh held-out confirmation is running.
