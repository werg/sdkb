# Stored-payload training-fit diagnostic

The uniform-data, no-KL, R=2 trajectory control beats zeroed stored payloads by
only 0.00969 token-weighted NLL on 119 held-out episodes. The output-KL,
source-utility curriculum and raised-gate forks did not increase average
payload-specific benefit. This diagnostic checks whether the same frozen control
shows a substantially larger real-versus-zero effect on its own 530 training
episodes. No weights or retrieval policy change.

Use the completed uniform control checkpoint at
`/archive/runs/trajectory-output-distillation-20260920/control` and the exact
530-episode training JSONL whose digest is recorded in its checkpoint manifest.
Build one newly serialized R=1 writer bank on the external disk, reopen it, and
score all 530 episodes at R=2 with the original oracle read plans. Score real,
zeroed, wrong and absent values. The inference scorer must not call the writer.
Compare with its already completed held-out report using episode and trajectory
aggregates. Also report source counts, dataset/checkpoint/evaluator digests and
training versus validation target lengths. The full set is deliberate: do not
select high-performing training episodes after seeing their scores.

This is a **training-fit** diagnostic, so a larger training effect would indicate
fit/generalization tension; it would not be held-out transfer evidence. A similarly
small training effect would locate the issue earlier in the objective or read path,
but would not alone identify the cause. Teacher-forced NLL remains distinct from
generation, agent success and parameter substitution. The read path uses the
stored bank only, and all raw data and evaluation progress stay on the external
disk. The evaluation is resumable after a cooperative stop.

## Completed result

The frozen evaluator at `9bc499a` wrote 1,680 distinct training sources once,
reopened the bank and scored all 530 training episodes in four conditions. The
analyzer checked the checkpoint against its training-file digest, both evaluator
identities, complete condition grids, fixed source IDs under value interventions
and disjoint episode/trajectory IDs. The existing held-out bank required 375
writer calls for 119 episodes. Raw banks and rows are on the external disk;
`results.json` pins their report and input hashes.

| R=2 token-weighted teacher NLL | Training, 530 | Heldout, 119 |
|---|---:|---:|
| Real stored values | 0.647832 | 0.971024 |
| Zeroed values | 0.658005 | 0.980714 |
| Wrong values | 0.661011 | 0.982071 |
| No memory | 0.807649 | 1.089557 |

The real-versus-zero token-weighted gain is **0.01017 on training** and **0.00969
on heldout**. Mean episode gains are 0.01915 and 0.01722 respectively; their
descriptive trajectory-bootstrap intervals are 0.01726–0.02125 over 187 training
trajectories and 0.00595–0.02639 over 42 heldout trajectories. Thus the much lower
training NLL does not come with a large training-only source-value effect. The
no-memory difference is much larger, and the real-versus-none mean episode gain
is 0.34077 on training versus 0.29579 heldout; that contrast includes generic
recurrent computation and slot effects.

Training targets average 99.50 tokens versus 108.05 heldout, and the splits have
different repository groups. These are descriptive, unequal-distribution
comparisons, not a controlled causal estimate of generalization. The result
weakens a simple story that the model memorized strong payload use on training
trajectories and merely failed to transfer it. It does not identify whether the
limiting factor is the writer, reader, bridge, target objective or available
source-dependent supervision. The next experiment needs an explicit
source-dependent task and counterfactual target, with a fresh heldout split;
another NLL gain alone would not settle the question.
