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

Status gives a persistent console-log path. A stop request is a control file,
not a signal sent to a potentially recycled PID. Training finishes the current
optimizer step and commits model, optimizer, RNG and cache together. Preparation,
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
export SDKB_RUNS_DIR="$HOME/sdkb-runs"             # fast local storage
export SDKB_ARCHIVE_DIR=/mnt/external/sdkb-archive # existing project directory
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

## Fast checkpoints and external archives

Choose the active filesystem with `--output`. Keep active optimizer checkpoints,
the training cache and frequently accessed data on fast storage. An external HDD
is appropriate for archive copies; its latency should not block every optimizer
checkpoint. This distinction also applies to other machines and network storage.

Copy a base config/recipe for the intended run and set these fields under `train`:

```yaml
archive_dir: /archive             # Docker mount; use the actual path outside Docker
keep_checkpoints: 2               # recent local complete sets
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
sdkb train --config /fast/recovered-stage/checkpoints/STEP_DIRECTORY/config.json \
  --output /fast/recovered-stage --resume
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
