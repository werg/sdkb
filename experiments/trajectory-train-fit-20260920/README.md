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
