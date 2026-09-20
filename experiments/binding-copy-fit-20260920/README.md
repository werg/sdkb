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
