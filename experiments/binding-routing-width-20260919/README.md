# Address-width diagnostic

Compare 64, 128 and 256 address dimensions using the same frozen source/query
features, source router, global negatives, seed 67, batch 128 and 3,200 Muon
updates as the corrected-precision experiment. Preserve the original blocks;
initialize added query rows to zero, added key rows with noise at 0.001 times
the source key weight standard deviation, and added map blocks to identity.
This avoids a dead all-zero extension. Measure initial score/ranking drift;
the initialization does not promise an identical function at wider dimensions.

Training uses 1,024 worlds (4,096 sources); held-out features cover 32 worlds
(128 sources). Report full required-pair retrieval for both candidate scopes,
parameter counts and serialized FP32 key bytes. Wider arms have larger budgets.
This is a feature diagnostic, not stored-bank or downstream capability evidence.
There is currently no production adapter for wider routing dimensions.

The source is the completed broad full-state router at
`/archive/probes/routing-breadth-20260919/full_state_query-resume.pt`.
Outputs go to `/archive/probes/routing-width-20260919`, including immutable
source/config hashes, offline W&B and JSONL metrics, initial/final/emergency
optimizer-boundary checkpoints with complete optimizer ownership and RNG state.
Resume requires unchanged scientific settings. SIGTERM/SIGINT or the run STOP
file request a checkpoint after the current complete optimizer update.

Use `scripts/probe_routing_width.py`; any promising feature result must be
validated through actual stored banks before claiming a model improvement.

## Completed result

| Width | Address parameters | Key bytes/record | Train full pairs /4096 | Held-out full pairs /128 |
|---|---:|---:|---:|---:|
| 64 | 139,264 | 256 | 3,742 | 87 |
| 128 | 294,912 | 512 | 4,003 | 80 |
| 256 | 655,360 | 1,024 | 4,013 | 79 |

Wider matrices fit the training pairs better without improving this held-out
feature diagnostic. Keep the 64-dimensional production interface; these results
do not justify a wider adapter. They suggest a generalization problem under this
initialization and budget, rather than establish a universal optimal width.
The unchanged-width control reproduces the corrected global run's final model,
optimizer and sampling RNG exactly. Initial maximum score changes were 0,
0.00718 and 0.01180 respectively (temperature-scaled scores).

All arms completed using native Muon from immutable commit `5cb9cfb`. Checkpoints
contain only these small routing matrices and optimizer state; no backbone copies
were created. Tests: 329 passed, four existing warnings; Ruff clean.
