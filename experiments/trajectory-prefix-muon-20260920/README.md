# Public-trajectory Muon curriculum: recurrent bridge

The pretrained one-pass text baseline benefits from prior context on 119 held-out
SWE-smith episodes: token-weighted NLL 1.490 with selected prior text versus 1.937
without it. This establishes a context-utility signal for a bounded curriculum,
not source sufficiency, latent-memory utility or agent success.

The first declared stage uses the same 530 training episodes, pretrained LFM
revision `40cb2ad3b3044d5a41eee083a6103c8b523afa45`, two recurrent passes, and a
**frozen pretrained backbone**. Only the native recurrent bridge receives text-path
gradients. The task NLL is supplemented by weight .1 KL to the one-pass parent;
because the backbone is frozen and one pass bypasses the bridge, this teacher
remains the original pretrained model.

200 native Muon updates, accumulation four, seed 197, memory/bridge LR .0001,
BF16. No source encoder or memory reader is trained in this stage. The declared
source text and targets retain their original causal preparation and provenance.
Repository-held-out validation is unchanged. Validate the resulting text path
before declaring or launching latent warmup; no synthetic-study weights or failed
alignment objective are imported.

All artifacts and runtime caches are external. Save initial/final/cooperative
emergency states only within this budget (cadence 10000), keep two complete sets,
reserve 10 GiB disk and 8 GiB host memory, CUDA fraction .35, watchdog 300 seconds,
offline W&B. Resume with the frozen launch checkout and unchanged configuration.
No live teacher, shell command from a trajectory, or benchmark environment runs.

Native preflight passed on the pinned LFM checkpoint. All 200 updates completed;
the initial and final complete checkpoints are on the external drive. The final
checkpoint is step 200, and the held-out scoring file is
`/archive/runs/trajectory-prefix-muon-20260920/heldout-text/results.json`.
The frozen training code is commit `48cabfa`; the evaluator records the checkpoint
model digest, input digests, code digest and each completed scoring row for resume.

| Same checkpoint and 119 repository-held-out episodes | R=1 token-weighted NLL | R=2 token-weighted NLL |
|---|---:|---:|
| Selected prior source text | 1.489802 | 1.447101 |
| No prior source text | 1.937263 | 1.902615 |

The R=1 scores exactly match the untouched pretrained baseline. R=2 improves
teacher-forced NLL in both text conditions; these 200 updates establish a usable
recurrent bridge for the next latent stage, not latent-memory transfer, source
sufficiency, composition, or agent success. The selected-text benefit is a context
control with a different token and compute budget. No model weights or raw
trajectory text are committed to Git.

## Latent warmup

`latent_warmup.yaml` declares the next 400-update, two-pass memory stage. It
warm-starts all compatible model weights from the verified bridge step 200,
resets optimizer state as a new scientific stage, keeps the pretrained backbone
frozen, and trains the writer, codecs, reader, query path, workspace and bridge
through a real in-loop stored-value read. The writer remains at depth one. Oracle
selection uses only the prepared source supports visible at each query time; the
target continuation never enters a query. `selected_producers_only` records the
fully live oracle compute policy, and native Muon remains the matrix optimizer.
Complete checkpoints, caches, data and offline W&B files stay on the external
drive. This is a teacher-likelihood and memory-intervention experiment.

`latent-inputs.json` pins the config, prepared train and validation files, base
revision, and exact bridge checkpoint model and manifest. `run_latent.py` verifies
these before the native preflight and before any training state is created. A
graceful stop writes complete optimizer, RNG and partial-gradient state; `--resume`
uses the same locked inputs and committed checkpoint.

The native stage completed all 400 updates in about 240 seconds, with one actual
in-loop read per logged example. Its initial and final complete checkpoints are
on the external drive. The frozen training code is `c421408`; the frozen
evaluation code is `5e34e49`. The evaluation was stopped after 177 scored rows
and resumed to 476/476 without re-encoding a completed frozen bank scope.

On all 119 repository-held-out episodes, the frozen stage-400 checkpoint gave:

| Condition | Token-weighted teacher NLL |
|---|---:|
| Real stored values | 1.324823 |
| Zeroed values, same read selection | 1.333195 |
| Wrong values, same read selection | 1.335170 |
| No memory | 1.829600 |
| Selected source text at R=2 | 1.397014 |

`latent_text_control.yaml` changes only the inference arm and its incompatible
training-only selected-producer flag before loading the identical latent checkpoint
for the selected-text control; it does not warm-start or retrain a model.

`latent-results.json` pins the external result digests and reports paired
trajectory-bootstrap diagnostics. Real values improved mean episode NLL over
zero values by 0.01173 (42-trajectory descriptive interval 0.01000–0.01350),
and over wrong values by 0.01359 (0.01011–0.01685). The original selected IDs
were identical in all 119 intervention pairs. This is a consistent but small
payload-specific effect. The much larger real-memory versus no-memory gap
includes the effect of the memory path, working slots and extra computation;
it cannot be credited to stored content. The selected-text condition changes
token and compute budgets. These are teacher-forced likelihood measurements,
not agent success, composition or a capacity-substitution result.

### Same-bank depth sweep before joint training

The frozen latent checkpoint was also scored at R=1, 2, 3 and 4 using one stored
bank and all 119 validation episodes. The evaluator checkout was `9835b06`;
`latent-depth-results.json` pins each external report and both text-control
files. R=3 and R=4 exceed this checkpoint's trained depth of two.

| Depth | Real values | Zero values | No memory | Selected text |
|---:|---:|---:|---:|---:|
| 1 | 1.937263 | 1.937263 | 1.937263 | 1.489802 |
| 2 | 1.324823 | 1.333195 | 1.829600 | 1.397014 |
| 3 | 1.270974 | 1.284375 | 1.743415 | 1.324983 |
| 4 | 1.239385 | 1.256570 | 1.675303 | 1.270449 |

These are token-weighted teacher NLLs. R=1 makes no read. More core passes help
all conditions, including zero values and no memory. The real-over-zero paired
mean episode NLL benefit grows from 0.01173 at R=2 to 0.01708 at R=3 and
0.02105 at R=4, but remains small. Fixed original IDs held in every intervention
pair. This same-checkpoint sweep isolates extra recurrence from a changed writer;
it does not establish that more depth improves actual agent tasks or capacity
substitution. Evaluation timing includes uncertain page-cache state and must not
be presented as cold-NVMe latency.

## Recurrent joint stage

`recurrent_joint.yaml` declares a new 400-update Muon warm-start from the exact
latent step-400 checkpoint. It samples consumer depths two and three per complete
optimizer update, keeps writer depth one, trains the shared native core and SDKB
modules, and uses a 0.1 selected-text one-pass NLL anchor. Prelude, coda and
embeddings follow the `recurrent_core` freeze policy. The same prepared sources,
validation split, oracle access and external checkpoint policy remain pinned.
`joint-inputs.json` and `run_joint.py` reject changes to the config, data, base
revision or latent initialization before native model preflight and training.
`joint_text_control.yaml` is the corresponding read-only text-scoring config for
the same eventual checkpoint; it switches the inference arm and disables the
training-only selected-producer and anchor settings.
The joint stage tests whether the small latent-warmup payload effect survives
training the shared recurrent core.

The native joint stage passed its actual LFM preflight and completed all 400
updates in about 1,017 seconds. Depths two and three were both sampled, with one
actual read per logged example. The final full optimizer/RNG checkpoint is on
the external drive. `joint-depth-results.json` pins the complete fixed-bank
held-out reports and text controls, scored by frozen evaluator `8f703a9`.

| Joint checkpoint depth | Real values | Zero values | Wrong values | No memory | Selected text |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.107569 | 1.107569 | 1.107569 | 1.107569 | 0.866650 |
| 2 | 0.978818 | 0.984149 | 0.985777 | 1.092322 | 0.865496 |
| 3 | 0.980331 | 0.988139 | 0.994084 | 1.086108 | 0.867777 |
| 4 | 0.985889 | 0.995611 | 1.006509 | 1.084728 | 0.872847 |

These are token-weighted teacher NLLs on the same 119 episodes and one bank.
The joint checkpoint improves absolute teacher NLL over latent warmup at every
condition, including R=1/no-memory. Its lowest real-value NLL is at R=2;
additional passes do not improve that score. Real values beat zero and wrong
payloads at R=2–4, with unchanged original selections, but selected source text
remains substantially better than the latent path. R=4 exceeds the joint
training depths. Training changed both the core and writer, so cross-stage NLL
improvements cannot be attributed to depth or memory alone. This is neither
agent success nor parameter substitution.
The paired mean episode real-over-zero benefit is 0.01186 at R=2, 0.01739 at
R=3 and 0.02077 at R=4; the 42-trajectory descriptive intervals and positive
episode counts are in the pinned report.

After all evaluators released the checkpoint files, verified hard-link
deduplication reclaimed 2,034,699,352 duplicate external weight bytes from
bridge-to-latent and latent-to-joint warm starts. All six complete checkpoint
sets still pass manifest and SHA256 verification. The exact replacements are in
`experiments/operations-20260920/trajectory-stage-dedup.json`.
