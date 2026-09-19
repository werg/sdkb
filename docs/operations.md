# Portable training operations

The Python runner supports CPU and CUDA PyTorch environments without Docker,
Spark, W&B or an external disk. Spark's ARM64 image and driver checks remain in
`scripts/spark.sh`. Do not install a replacement Torch into a working vendor
runtime merely to use the optional logging extra.

## Start, stop, inspect and resume

Foreground execution works with a terminal, a scheduler, or a container:

```bash
sdkb launch --recipe recipes/looped_causal.yaml --output /fast/sdkb-runs/causal
# Resume the identical recipe and output:
sdkb launch --recipe recipes/looped_causal.yaml --output /fast/sdkb-runs/causal --resume
```

For a local detached process, use the same Python environment that will train:

```bash
sdkb runs start --recipe recipes/looped_causal.yaml --output /fast/sdkb-runs/causal
sdkb runs status --output /fast/sdkb-runs/causal
sdkb runs stop --output /fast/sdkb-runs/causal
sdkb runs start --recipe recipes/looped_causal.yaml --output /fast/sdkb-runs/causal --resume
```

Status gives a persistent console-log path, resolved checkpoint locations and free space.
Optional archives expose active/pending copies, last completion and errors in
`archive-status.json`. A stop request is a control file,
not a signal sent to a potentially recycled PID. Training finishes the current microbatch and its complete producer replay, then
commits model, optimizer, RNG and cache together. If accumulation is incomplete,
the checkpoint also includes accumulated gradients, microbatch position, sampled
depth and running loss totals; resume finishes that same optimizer update. Preparation,
preflight and evaluation stop at the next stage boundary; they are not interrupted
mid-write. SIGINT/SIGTERM also request a graceful stop. A hard kill resumes the
last committed checkpoint. Detached runs do not automatically restart failures.

OS file locks reject concurrent writers to a launch or training directory and
release on process death. Control files live in `.sdkb-control` beside the run;
their presence alone does not imply a live process. A new explicit invocation
acknowledges an earlier stop request. Use one canonical run path through shared
mounts. These controls have been exercised on Linux ARM64; other platforms need
their own runtime validation.

Inside Docker, run `sdkb launch` in the foreground and let Docker own process
lifetime; do not detach a child inside an ephemeral `docker run --rm` container.
On Spark:

```bash
export SDKB_RUNS_DIR=/mnt/external/sdkb-runs       # existing mounted external storage
export SDKB_ARCHIVE_DIR=/mnt/external/sdkb-archive # optional separate archive directory
export SDKB_CONTAINER=sdkb-causal
./scripts/spark.sh build
./scripts/spark.sh start --recipe recipes/looped_causal.yaml --output /runs/causal
./scripts/spark.sh status
./scripts/spark.sh logs
./scripts/spark.sh stop
```

The detached container is retained for inspection. After it exits, remove that
specific stopped container with `docker rm "$SDKB_CONTAINER"`, then repeat `start`
with `--resume`. Logs can be detached without stopping training. Docker stop allows
600 seconds by default (`SDKB_STOP_TIMEOUT` overrides this); an expired grace
period becomes a hard kill. Existing unrelated GPU services are never stopped.

## Checkpoint frequency and storage placement

Choose checkpoint placement explicitly with `--output`. On this machine retained
checkpoints belong on the external disk. New runs can live entirely there, with
`archive_dir: null` to avoid a redundant second copy. Existing stopped runs can
move their checkpoint directories without changing dataset or result paths:

```bash
mkdir -p /mnt/external/sdkb-checkpoints/my-stage
sdkb relocate-checkpoints --run /fast/run/my-stage \
  --destination /mnt/external/sdkb-checkpoints/my-stage
sdkb runs configure --output /fast/run --checkpoint-every 1000 --no-archive
```

Relocation verifies every retained checkpoint, publishes the external directory,
replaces the old checkpoint directory with a link, then removes redundant local
files. A corrupt destination or a running stage fails closed. Optional
`--archive-hint PATH/TO/checkpoints` reuses verified existing archive files with
hardlinks when possible. Source/destination paths must be visible in the same
location inside and outside containers; the Spark wrapper exposes the archive at
both `/archive` and its host path. Run parents must also be stopped during a
manual stage relocation. New direct-external runs do not need symlinks.

Periodic saves default to **1,000 optimizer updates**, plus initial, final and
emergency saves. `runs configure` changes only checkpoint operations and records
`checkpoint-policy.json`; it does not rewrite immutable scientific inputs. Short
200–400-update stages therefore normally write initial/final checkpoints only.
Stop and final checkpoints remain mandatory regardless of periodic cadence.

A fast local staging directory and asynchronous archive remain optional when
external-write latency is unacceptable. They require deliberate retirement of
completed runs: keeping two local checkpoints **per stage forever** is not a
space-management policy. The external-first setup avoids that accumulation.

Copy a base config/recipe for the intended run and set these fields under `train`:

```yaml
archive_dir: /archive             # Docker mount; use the actual path outside Docker
checkpoint_every: 1000           # emergency and final saves are independent
keep_checkpoints: 2               # recent complete sets on the output filesystem
archive_keep_checkpoints: 3       # recent archived sets per unique run identity
min_free_disk_bytes: 10737418240  # optional 10 GiB reserve; default is 1 GiB
```

`archive_dir` must already exist on the intended mounted disk. Leaving it null
disables archival. Each stage has a persistent UUID and a separate archive
subdirectory. Archives cannot be mixed across run identities. W&B shares that
identity, and it is included in checkpoint recovery data.

Copies run in a background thread with 8 MiB buffers and fsync every 64 MiB,
bounding dirty-page pressure. Every immutable checkpoint file is checked against
its recorded SHA256 before atomic publication of the archive's `CURRENT` pointer.
An incomplete copy is never a recovery checkpoint. Where supported, archive pages
are advised out of the OS cache after flush; this is not a cold-cache benchmark.

There is at most one active and one pending copy. If storage falls behind, the
pending copy advances to the newest checkpoint: **the archive is a bounded recovery
history, not a promise to retain every training step**. Local pruning protects
active/pending copies in addition to the configured recent sets, so transient local
retention can exceed `keep_checkpoints` by up to two. Completed runs drain the
archive; graceful stops prioritize the local checkpoint and do not wait for the
external disk. Resume queues the current committed checkpoint again.

Before saves, free-space checks budget for weights, Adam state, the SQLite snapshot,
and the reserve. Archive errors surface as errors while leaving the local recovery
checkpoint intact. Reserve checks reduce disk-exhaustion risk; they cannot reserve
space against other processes writing to the same disk.

```bash
sdkb storage --path /fast/sdkb-runs/causal
sdkb storage --path /mnt/external/sdkb-archive
# Manual copy of a stopped stage into an existing dedicated archive directory:
sdkb archive --run /fast/sdkb-runs/causal/recurrent_joint --destination /archive/manual-stage
# Restore into a NEW directory, then resume using its committed config:
sdkb restore --archive /archive/RUN_UUID --output /fast/recovered-stage
sdkb train --output /fast/recovered-stage --resume
```

Archives hold checkpoint sets, including the stale training cache and run identity.
They are not full backups of downloaded datasets, prepared launch inputs, console
logs or W&B history. Preserve the prepared launch directory separately; exact
resume still requires the unchanged dataset at the path recorded in the config.
The restore command never overwrites an existing directory. For a new experiment
from an old checkpoint, use warm-start (`--init-from`) and a new run identity.

Storage inspection reports bytes and abandoned `.pending-*` directories; it does
not delete shared Hugging Face caches, other runs, Docker images, or archives.
Inspect interrupted copies before manually removing them. No whole-disk cleanup
command runs automatically.

## Optional W&B

Install `sdkb[tracking]` into the project environment; the Spark installer includes
it while constraining the NVIDIA runtime. Logging is disabled by default. Configure:

```yaml
wandb_mode: offline # disabled / offline / online
wandb_project: sdkb
wandb_entity: null
wandb_group: causal-mlp-seed17
```

Online mode uses credentials from the environment/login, never YAML. The Spark
wrapper forwards `WANDB_API_KEY` by name and optionally `WANDB_BASE_URL`. No upload
was required to validate the offline path. Source text, model weights and latent
payloads are not logged; automatic console and Git capture are disabled. Config
and scalar optimizer metrics are logged, including the actual optimizer step.

Online resumes use the same explicit ID with `resume="allow"`, following the
[W&B resume contract](https://docs.wandb.ai/models/runs/resuming). Offline restarts
produce separate local segments with the same identity; the SDK does not merge
offline history automatically. After a crash, W&B can contain uncheckpointed
metrics, so each attempt is tagged and local checkpoint-consistent JSONL remains
authoritative. W&B is telemetry, never a source for optimizer recovery.

## Lessons applied from bgkit

Bgkit's runbook, trainer and checkpoint archiver document failures from writing
large checkpoints synchronously to slow external storage, treating an archive's
existence as proof of completeness, mixing retention across runs, and losing run
identity during resume. SDKB adopts bounded flushing, verified atomic publication,
independent retention, durable identities and graceful stopping. It keeps the
implementation independent of bgkit's model code, Hydra configuration, absolute
mount paths and Spark-only optimizations.

## Optimizers and exact resume

`train.optimizer: muon` uses native `torch.optim.Muon` on eligible 2D matrix
transforms, with AdamW on embedding/output tables, learned slots, vectors,
non-matrix tensors and LoRA factors. It requires a runtime with native Muon;
older Torch runtimes fail clearly without installing a replacement. The saved
configuration controls momentum, Newton–Schulz steps, weight decay, Adam betas
and epsilon. Muon uses `adjust_lr_fn: match_rms_adamw`; learning rates still need
experimental validation and are not assumed equivalent to AdamW.

Every trainable parameter must have exactly one optimizer owner. Checkpoints save
both optimizers, actual per-group learning rates and settings, momentum/moments,
parameter names/order and all accumulation state. `optimizer.json` reports the
settings actually loaded after resume. Changed training hyperparameters are
rejected on exact resume; an optimizer switch is an explicit warm-start into a
new run. `sdkb train --output RUN --resume` loads the saved config automatically.
There is no LR scheduler or early-stopping controller in SDKB yet; no scheduler
state is implied. W&B identity and random episode-sampling position also survive.

## Runtime rails and profiling

Optional `cuda_memory_fraction` limits this process's CUDA allocator;
`min_system_available_bytes` checks host `MemAvailable` before allocation and
between updates, requesting a checkpointed stop under pressure. Neither adds host
RAM and CUDA totals together. Unsupported host memory reporting is recorded as
unknown. These controls are portable opt-ins rather than Spark assumptions.

`stall_timeout_seconds` dumps all-thread stacks during stalled compute. It is
disarmed for checkpoint writes and never hard-kills a process with unsaved work.
A hung native kernel must return before cooperative checkpointing is possible;
SIGKILL or power loss can only recover the last committed state.

Benchmark representative shapes before changing activation checkpointing or
batch size. GPU memory headroom alone is not throughput evidence. Keep cold-disk,
warm-cache, exact-scan and ANN claims separate. Model-specific bgkit DeltaNet
kernels do not apply to this LFM2 convolution/attention backbone.

The detailed adoption inventory is in [the bgkit audit](bgkit-audit.md).
