# MLP world-scoped learned selection (complete)

The longer selected-support MLP now composes both rule facts correctly. Test whether
verified group supervision can teach its existing key/query path to select records
for the requested entity among the four records in a world. This is candidate-local
retrieval, not global retrieval or learned authorization.

Both arms warm-start the identical completed 1,600-update recurrent-core MLP
continuation, reset Muon state, and run 800 updates with seed 37. Keep the same
training worlds, core-only adaptation, reader, BF16 payload, learning rates,
accumulation, sampled depths and text anchor. Both use learned top-two selection
with 100 oracle-warmup updates. The sole between-arm difference is routing loss
weight: zero versus one. Thus this tests explicit group supervision over task-loss
training of the same discrete-selection system. Both receive identical training
examples and support labels; the control does not optimize the routing objective.
The text anchor receives required supports in both arms. Fact questions need one
record, but learned reads have a fixed two-record budget; report this distinction.

Use the existing sequential runner in `binding-continuation-20260919/run.py`.
Direct external output, cadence 1,000 plus initial/final/emergency, retention two,
offline W&B and inherited host/device guards apply. No periodic checkpoint is
expected inside this 800-update budget. Exact resumes preserve optimizer state;
the initial fork explicitly resets it in both arms.

At the fixed final endpoint, create 32 new shared worlds after both arms complete.
Evaluate both using `evaluate_binding_context.py --learned-world --read-budget 2`:
actual exact key ranking within supplied world eligibility, complete-support recall,
action/fact/identifier choice, support removal with reranking, zero payload and
counterfactuals with original plans held fixed. The named selected-pair control is
an oracle upper bound, not learned selection. Compare world-paired outcomes and
retain episode/checkpoint hashes. Evaluate the common source on the same worlds.
Do not choose an endpoint using held-out performance. One seed and synthetic worlds
cannot establish agent success, arbitrary entity binding or parameter substitution.

Preparation does not launch training. The full-graph/replay test covers the new
combination of in-loop learned top-two selection, competing records, serialized
BF16 payloads, live key/query gradients, and checkpointing on/off.


Launched sequentially from frozen source `304fb17`, control first. The queued
comparison uses that same frozen source and waits for both arms to finish before
creating its new evaluation worlds. `launch.json` records the external console and
process IDs; `inputs.json` records immutable configurations and source/data hashes.
All 244 core tests and Ruff passed before launch. The source's all-world diagnosis
subsequently confirmed identical fact answers for both entities in every opposed-rule
world, making learned selection a distinct, unresolved mechanism to test.


After choice evaluation commits, `generate.py --study EXTERNAL_STUDY_ROOT` validates
its checkpoint/episode identities and runs candidate-free generation from the same
banks: first eight worlds, 24 greedy tokens, learned top-two selection, zero-payload
and no-memory controls. This follow-up adds no training or checkpoint selection.
Generation code is frozen separately from the running training/evaluation checkout.


The control completed all 800 updates and its final checkpoint committed directly
externally in 24.2 seconds (1.393 GB); its only checkpoint sets are initial and final.
The supervised arm then started. `initialization-check.json` verifies both initial
weight-file hashes equal the shared source. Their first task NLL and raw routing
loss are identical (0.00328392605 and 1.13250297308); the weighted objectives and
gradients differ as intended. Final routing capability results follow below.


## Fixed-endpoint results

Both arms completed 800 updates without a resource stop. New common-world choice
results (32 worlds) are:

| Metric | Source | No routing loss | Group routing loss |
|---|---:|---:|---:|
| Action accuracy | 63.28125% | 60.9375% | 52.34375% |
| Action sufficient-support recall | 46.875% | 48.4375% | 3.90625% |
| Identifier accuracy | 29.6875% | 31.25% | 29.6875% |
| Permission accuracy | 57.8125% | 79.6875% | 79.6875% |
| Restoration accuracy | 51.5625% | 43.75% | 70.3125% |

Supervised minus control action accuracy is −8.59375 percentage points,
world-bootstrap 95% interval [−14.0625, −3.90625]. The supervised model's oracle
selected-pair action score also falls to 69.53125%, showing interference with its
previously successful composition. Action permission counterfactuals have only
1/128 jointly correct pairs; restoration changes have 1/64, with 3/64 false changes
on unchanged branches.

The apparently perfect direct-fact support recall does not establish entity
selection: all permission/identifier queries select both entities' permission
records; all restoration queries select both restoration records. Action queries
select two restoration records in 117/128 cases, two permission records in 6/128,
and mixed types in 5/128. None select the full required pair. Sufficient-support
recall can count permission alone on STOP branches; it is distinct from this
full-pair count. `selection-patterns.json` preserves both definitions.

Candidate-free generation on the first eight worlds scores 17/32 actions for both
continued arms (source 19/32); all three generate 0/16 exact identifiers. These
results reject success of this particular 800-update joint routing intervention.
They do not prove routing or the MLP architecture impossible. Training fit versus
held-out selection, and interference from updating the shared writer/reader, remain
separate diagnostic questions. See `confirmation-summaries.json`, `paired-results.json`,
`generation-summaries.json` and `completed-training.json` for controls and identities.
Training and choice evaluation used frozen `304fb17`; generation used `baba9ef`.


### Training-fit follow-up

A frozen `304fb17` diagnostic evaluates the first eight existing training worlds,
all 80 questions, using the same stored-only learned-selection protocol. Supervised
action accuracy is 65.625% and sufficient-support recall only 12.5%, versus control
68.75% and 53.125%. Thus failure is visible on this training subset too, not only
on held-out worlds. Direct-fact recall is again 100%; identifier choice reaches
93.75% in both continued arms. This is explicitly training fit, not generalization.
`training-fit.json` records the subset and result hashes.


The first-ranked direct-fact record is also weak on held-out worlds: supervised
permission 32/64, restoration 36/64, identifier 34/64. For permission, both entity
queries choose the same first record in all 32 worlds, with no world ranking both
entities correctly. This confirms that top-two fact recall was not concealing
reliable entity ranking. `top-one-entity-ranking.json` records all arms and hashes.
