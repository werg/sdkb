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

## Recurrent joint stage

`recurrent_joint.yaml` declares a new 400-update Muon warm-start from the exact
latent step-400 checkpoint. It samples consumer depths two and three per complete
optimizer update, keeps writer depth one, trains the shared native core and SDKB
modules, and uses a 0.1 selected-text one-pass NLL anchor. Prelude, coda and
embeddings follow the `recurrent_core` freeze policy. The same prepared sources,
validation split, oracle access and external checkpoint policy remain pinned.
`joint-inputs.json` and `run_joint.py` reject changes to the config, data, base
revision or latent initialization before native model preflight and training.
The joint outcome is pending; a tiny positive payload effect in latent warmup
does not establish that extra recurrence will improve transfer.
