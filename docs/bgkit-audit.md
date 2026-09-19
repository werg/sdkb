# bgkit operational lessons: SDKB adoption audit

Audited against local `~/bgkit/docs/runbook.md`, `docs/dgx_spark_perf_playbook.md`,
`src/bgkit/training/{base_trainer,checkpoint_archiver,checkpointing}.py`,
`scripts/restart-train.sh`, and the step watchdog on 2026-09-19.

The initial integration was incomplete: archives were populated, but roughly
49 GiB of completed-run checkpoints remained internally. Per-stage retention
did not retire completed runs. This was the same failure class bgkit documents.
The owner also requested less frequent saves; the former 50-update cadence was
not appropriate for these short pilots.

| Lesson | SDKB implementation / validation |
|---|---|
| External storage must actually relieve NVMe pressure | Verified relocation, stable run paths, direct external output for new runs; 25 stage/diagnostic directories relocated, internal run tree reduced to about 52 MiB. |
| A directory's existence is not archive completeness | SHA256 verification of every checkpoint file before publication or local removal; corrupt destination regression. |
| Retention must not mix runs | Persistent UUIDs, separate archive destinations, identity checks and owner locks. |
| Completed-run residue survives ordinary retention | Relocation removes local payloads, including final retained sets; future runs use external output. No cleanup of unrelated bgkit/cache data. |
| Storage placement must survive future launches | Ignored machine-local path settings default the Spark wrapper to the chosen external run root. Environment overrides remain portable; unavailable configured directories and relative training output paths fail before Docker launch. |
| Slow-disk dirty pages compete with GPU memory | Optional bounded archive buffers and periodic fsync/cache advice retained; fewer periodic writes. Current runs use direct external output to avoid local residue. Observed saves can take minutes under contention; fast local staging remains an explicit alternative with a local-retention obligation. |
| Archive lag/failure must be visible | Persistent active/pending/completed/error status and CLI checkpoint paths/free space. |
| Stop must preserve training rather than restart from old periodic state | Signal/control-file stop at a microbatch/replay boundary, including partial accumulated gradients and totals. CPU tests cover AdamW and Muon; actual LFM2.5 BF16 CUDA Muon resume also reproduced weights, optimizer and RNG exactly. |
| Preserve complete hyperparameters and sampling state | Config, both optimizer components/groups, names/order, LR/momentum settings, Python/Torch/CUDA RNGs, sampled depth, microbatch cursor, cache snapshot and data/revision checksums. |
| Never silently attach optimizer momentum to different parameters | Named ownership check on restore; optimizer/topology changes fail. |
| Group learning rates must remain independent and observable | Factory ownership audit and post-resume `optimizer.json`; no schedule that flattens group rates. |
| Muon requires an explicit parameter split | Native Torch Muon for eligible matrix transforms; embedding/output tables, slots, LoRA factors and other tensors use AdamW. |
| W&B is telemetry, not recovery authority | Stable run ID, scalar/config logging, explicit offline segments, checkpoint-consistent JSONL. Online wiring exists; current pilots remain offline. |
| Stalls need diagnostic evidence | Compute-only stack watchdog, disarmed for slow saves; no automatic destructive restart loop. |
| Spark RAM and VRAM share physical memory | Optional allocator fraction plus host `MemAvailable` reserve and checkpointed pressure stop; no summing capacities. |
| Slow memory growth needs a recorded trend | Logged updates include host available bytes, current CUDA allocation/reservation and attempt peak allocation in JSONL and W&B. Allocator readings do not synchronize the device. Earlier frozen runs retain their original telemetry. |
| Source/config drift invalidates run interpretation | Immutable launch/data checksums, isolated checkouts for live experiments, per-attempt environment history. Operational checkpoint policy is recorded separately. |
| Profile before changing checkpointing/batching | Two 100-update native Muon profiles: checkpointing off cut median update time 18%, raised peak CUDA allocation 2.14→2.53 GiB, and preserved every training metric and final weight-file hash. Applied only to this short binding distribution. |
| Runtime compatibility must be verified on the actual machine | Native ARM64 NVIDIA Torch preserved; actual pretrained preflight and GPU training exercised. No x86 emulation or Torch replacement. |
| Model-specific acceleration is not universal | bgkit's DeltaNet/FLA patches and LoRA training topology are not imported into LFM2. No matching operations exist here. |
| Capability metrics must remain separate | Teacher NLL, choice transfer/counterfactuals and actual agent success are distinct; this project has not established agent success or parameter substitution. |

SDKB currently has fixed learning rates, no early stopping and no live mutation of
scientific hyperparameters. Therefore it has no hidden schedule/early-stop/live
override state to restore. Adding those features must extend the atomic checkpoint
contract. Explicit warm-start forks are used for optimizer or objective changes.

GPU emergency validation is recorded in
`experiments/operations-20260919/muon-emergency-resume.json`.
The performance comparison is recorded in
`experiments/operations-20260919/checkpointing-profile.json`; compressed operator
traces and training artifacts remain on the external disk.


A follow-up inventory found 13.29 GiB of byte-identical warm-start weights in
separate external checkpoint sets. Verified hard-link deduplication reclaimed those
redundant copies without dropping any checkpoint or optimizer/RNG/cache state.
The command requires inactive stage/parent locks, matching metadata and SHA256
verification before atomic replacement. See the operations guide and
`experiments/operations-20260919/checkpoint-weight-dedup.json`. Internal run data
remains about 52 MiB; external free space was about 248 GiB after this maintenance.


Two additional duplicate final weight files in completed profiling and emergency
resume validation runs were then deduplicated (1.90 GiB), bringing this maintenance
pass to about 15.19 GiB of redundant weight copies removed. Their original
checkpoint paths and validation evidence remain intact. The current active run's
initial copy was explicitly excluded until training completes.


After the projection-only run finished, its byte-identical initial weights were
also deduplicated (another 0.95 GiB). Total verified redundant weight removal is
now about 16.1 GiB. This is additional to the earlier roughly 48 GiB removed from
the internal run tree by verified relocation.

The same recovery discipline now covers the standalone frozen-compactor probes:
raw bank creation and its source/writer/config identity plus raw-record digest
commit in one SQLite transaction. A restart verifies bytes and reuses the bank
without writer calls, even if feature extraction had not finished. Existing banks
without this manifest require an explicit offline rebuild at a new path; they are
not silently adopted. Historical frozen experiment checkouts retain their original
code. Compact-code markers, lineage and code/member tables also publish in one
transaction, so interruption cannot leave a marker blocking a retry. Probe metrics
are reconciled to the saved optimizer step before a new resume attempt appends rows.

Older routing-feature, STOP and count probes now use the shared small-probe
checkpoint contract: named optimizer groups, full optimizer hyperparameters,
Python/Torch/CUDA and sampler RNG, fsynced atomic publication with free-space
reserve, and JSONL reconciliation to the committed optimizer step. They write
initial/final/emergency state only, keep separate W&B attempts and group metadata,
and arm the stall watchdog only for compute. This closes gaps in their earlier
standalone save functions; historical checkouts still implement their original
formats. Current-source resume rejects the old format rather than guessing at
missing state. Use the original checkout to resume an old run, or an explicit
warm-start fork for a changed protocol. Main-trainer microbatch/replay recovery
remains a stronger, separate contract; these small probes stop between updates.

The post-hoc compactor probe now uses that same shared recovery helper. Its prior
standalone saver retained Torch/CUDA/sampler state but omitted Python RNG and did
not remove a failed temporary save. The replacement includes both and budgets
from actual state tensors. A download-free end-to-end regression stops after one
Muon update, reuses the committed bank/features, resumes, and exactly matches the
uninterrupted final model, optimizer, named groups and all RNG state. The shared
failed-save regression preserves the previous checkpoint and removes the partial
file. Existing compactor endpoints remain valid explicit warm-start sources;
resuming an older run requires its original checkout and format.
