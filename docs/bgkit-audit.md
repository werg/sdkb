# bgkit operational lessons: SDKB adoption audit

The later representation-distillation lesson is being tested explicitly:
bgkit2 matches student and frozen-teacher projection states on the same selected
positions. SDKB's optional `oracle_alignment_weight` instead matches corresponding
answer-prediction states of its latent and selected-text paths, using cosine loss.
Its detached text teacher shares current weights and retains the text NLL anchor;
it is not bgkit's separate frozen encoder, and no capability result is implied.
Matched evidence, causal positions, full replay and exact Muon emergency recovery
are regression requirements. The original stored-only inference path is preserved.

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


The 20 September follow-up rechecked the local reference at commit `2015354`
(clean worktree); exact operational source hashes are recorded in
`experiments/operations-20260920/bgkit-audit-source.json`. SDKB already arms stop
handling before model setup, disarms its compute watchdog around saves, and avoids
a blocking archive drain on emergency exit. The direct-external mode used here
still needs time to fsync the checkpoint itself; the Spark launcher gives a
600-second stop grace period. An abrupt kill cannot promise a new checkpoint.
The generic optional archive mode remains available for a faster staging disk.

The reference's newer data lesson is also applicable: targets must refer only to
source content actually encoded. SDKB source/prompt token limits raise explicitly;
they do not silently truncate decisive identifiers or rules. Repeated/query-view
experiments preserve immutable source versions and report their oracle scope.
No model-specific kernel patches, automatic destructive restart loop or live
hyperparameter mutation were imported. Those are not prerequisites for the current
fixed-schedule LFM experiments; the implemented recovery contract covers their
actual state rather than claiming nonexistent scheduler/controller support.


Long oracle confirmations now also preserve completed scoring variants/read plans
and incremental free-generation results. Cooperative stops flush the current
completed string; a normal restart reuses those results without rerunning writers,
scoring or finished generations. This is deterministic frozen-inference progress,
not optimizer state. Regression coverage interrupts both by control file and
SIGTERM, reproduces uninterrupted score/prediction outputs, and rejects changed
banks or mismatched generation prefixes. Existing running frozen evaluators retain
their original behavior.

A further allocation audit found that frozen diagnostics and standard evaluation/
preflight entry points did not consistently apply the configured startup host-memory
reserve and CUDA allocator fraction. Seven regression cases reproduced model
allocation before the reserve check. All seven entry points now call the shared
portable resource helper before allocating the model. The full suite passes 412
tests; a native pretrained model preflight also passes. This closes a startup gap,
not continuous inference-time pressure monitoring. Active frozen checkouts retain
their original code. See `experiments/operations-20260920/evaluation-reserves.json`.

The final resume-path audit closes two compatibility gaps: an immutable checkpoint
can no longer be used as the training output directory, and missing legacy RNG or
optimizer ownership/type fields no longer trigger approximate fallbacks. Direct
read-only state loading verifies dataset/base-model identity and leaves all checkpoint
files unchanged. Five new regression cases failed before the fix; current complete
AdamW/Muon and partial-accumulation resume tests still pass. The full suite is now
421 tests. Older incomplete states require their original checkout/mutable run or
an explicit warm-start; this does not change current checkpoint contents.


A final parameter-level Muon audit found that reader `null_tokens` fallback tables
were missed by the slot-table exclusion. Both MLP and attention regressions fail
under the old factory; the corrected factory assigns these tables to AdamW while
retaining Muon for reader transforms. A legacy-topology regression confirms refusal
before model mutation, and complete/partial-accumulation resume still matches exactly.
All 434 tests pass. Existing frozen studies retain their original policy and must
resume with that checkout; future warm-starts adopt corrected ownership. Evidence:
`experiments/operations-20260920/muon-fallback-ownership.json`.

The actual cached LFM topology was also constructed under native ARM64 vendor Torch
for a CPU ownership audit. Every Muon-owned tensor belongs to a recognized linear
transform module, and all 8,192 `reader.null_tokens` entries now belong to AdamW.
This audit writes no checkpoint and does not claim another CUDA gradient-parity run.


Runtime W&B errors are now isolated from optimizer recovery. Injected `log` and
`finish` failures previously aborted after an optimizer update; the corrected
tracking helper records error types, disables further logging for that attempt,
and preserves training and the original exception. A new attempt retries the same
run ID. A complete training comparison matches final weights, optimizer and RNG
exactly despite both injected failures. Setup errors remain fatal before training.
All 441 tests pass. This does not add a timeout to a hung SDK call. Evidence:
`experiments/operations-20260920/tracking-failure-isolation.json`.


Native BF16 emergency-resume validation was repeated after the fallback-table fix
from frozen `799b560`. With mixed live/cached sources and noise, a stop after one
of two accumulated microbatches again reproduces the uninterrupted final model,
both optimizer components and all RNG state exactly. Only the final complete sets
are retained; verified weight deduplication removes their duplicate model bytes.
`experiments/operations-20260920/muon-emergency-resume.json` pins the result.


A stop arriving during an initial/periodic checkpoint no longer triggers a second
identical save at the next loop boundary. Two failing-before tests reproduce the
redundant writes; resumed training preserves final model/optimizer/RNG results. A
stopped resume with an extended step budget still commits the new configuration.
Successful pytest checkpoint fixtures are now retired automatically, retaining one
failed session for debugging. Test temporary storage fell from about 258 MiB to
67 MiB after the passing suite. Experiment recovery sets are unaffected. All 446
tests pass; see `experiments/operations-20260920/stop-during-save.json`.


Host-side checkpoint controls now work without importing Torch or Safetensors.
Two subprocess regressions failed before moving model serialization imports into
actual save/load functions; they cover status/stop and archive/restore/relocation
with opaque checkpoint fixtures. The host, which has no Torch installed, also
inspected both live native-container training stages through the external mount.
All 448 tests and Ruff pass. This avoids installing another ML stack just to manage
storage. See `experiments/operations-20260920/lightweight-operations.json`.


The next frozen-readout workflow reuses atomic offline-bank manifests without
requiring a routing overlay. Its source-forward guard honors cooperative stops,
host/disk reserves and a compute-only watchdog; interrupted records roll back and
committed banks resume without writer calls. Its bounded queue has explicit resume
acknowledgement for signal-stopped probe children and refuses to clear active-child
controls. All 457 tests pass, with 18 focused checks after the final report-metadata
addition. All 18 native readout heads have since completed; results are recorded in
`experiments/binding-freshness-readout-20260920/summary.json`. Evidence:
`experiments/operations-20260920/frozen-readout-controls.json`.


A follow-up to the new offline writer moves CUDA completion inside its compute
watchdog, before serialization/fsync, and preserves the first manifest publisher
when reusing unchanged bank bytes. Two regressions failed before correction; all
459 tests pass. Completed frozen `f9625f7` banks/readouts retain their actual
implementation provenance. See `experiments/operations-20260920/offline-writer-guard.json`.


The local reference was rechecked at clean `011150a8281a98a8532322326171136841b2928b`.
Its six newer commits do not change the audited operational checkpoint/archive
helpers. They add cached tree browsing, whole-file windowed encoding, matched
opening text views, wrong-repository browsing controls, and agentic training that
does not substitute bare tree batches for tool-use trajectories. Applicable lessons
remain explicit here: never silently truncate decisive source content, never
re-encode cached memories during reads, match the actual information and tool
access of controls, and evaluate tool execution separately from authored answers.
SDKB already enforces the first two and its current experiments declare the latter
limits. A browsable repository-tree model is not part of this project's implemented
architecture; no such agent result or imported kernel/training topology is claimed.


Device-aware compute watchdogs now await asynchronous CUDA completion before an
active guard is disarmed. Main optimizer updates, small-probe updates and frozen
reader feature extraction use this path; saves remain outside it. CPU and disabled
guards do not add a CUDA wait. Three regressions failed before the API change;
468 tests pass, and a native CUDA event is complete before disarming in a bounded
smoke. This is not a GPU-hang injection or an automatic restart policy. Active
capacity runs retain frozen `b38564a`. Evidence:
`experiments/operations-20260920/compute-completion-watchdog.json`.


The long oracle transfer evaluator now also monitors the configured host reserve
after startup: before each offline source encoding, at periodic scoring progress,
and between generated answers. Pressure is latched for that attempt. An interrupted
bank transaction rolls back; completed generations publish before exit and resume
without repeated writers/scoring. Source forwards use the device-aware compute
guard, and writes check disk reserves. Two injected-pressure regressions failed
before correction; all 470 tests and eight focused runner cases pass. This does
not preempt a running kernel or continuously monitor every evaluation entry point.
Active frozen capacity training/confirmation retains `b38564a`. Evidence:
`experiments/operations-20260920/evaluation-pressure-resume.json`.


Following the profile-before-optimization lesson, an explicit fully-live/oracle
producer policy was tested against the full candidate reference. It avoids encoding
unread sources while preserving all Python sampler draws and actual scheduled
reads. Mixed caches and learned addressing are rejected. Full-graph/replay CPU
comparisons include noise, cumulative reads and exact partial-microbatch Muon
recovery. Two 50-update native BF16 Muon profiles reproduce every final weight,
optimizer/RNG value and training metric bitwise. Their sequential timings have
unequal contention and are not a causal throughput estimate. The option stays
recorded and opt-in; current capacity jobs are unchanged. One complete final set
per profile remains external, with another 1,017,349,676 duplicate weight bytes
removed. See `experiments/selected-producer-profile-20260920/README.md`.


Partial-update restoration now verifies the loss totals, microbatch cursor and
sampled depth fields before loading model weights. The new alignment total is
required only when that objective is enabled. Five missing-field regressions
failed before this guard; valid complete and partial Muon resumes remain exact.
The full suite passes 496 tests. This changes validation order, not valid snapshot
contents or the active frozen study. Evidence: `experiments/operations-20260920/accumulation-prevalidation.json`.


The first real-trajectory preparation exposed a remaining future-download gap:
run artifacts were external, but the wrapper still defaulted Hugging Face caches
to the internal disk. `SDKB_CACHE_DIR` now overrides an ignored `.sdkb/cache-dir`
setting, and explicitly configured missing cache paths fail before Docker launch.
This machine selects the external cache. The existing pinned model cache was copied
and every file/link verified (464,141,663 file bytes); after all frozen-container evaluators exited, the verified duplicate was retired,
reclaiming 464,141,663 bytes. Its legacy path resolves to external storage, and the
internal project cache is now about 620 KiB. New trajectory preparation
sets Hugging Face, Xet, dataset, XDG and temporary paths to external storage.
The full suite passes 499 tests. Evidence: `experiments/operations-20260920/external-cache-placement.json`.


A runtime-cache check found another 203,978,160 bytes of CUDA driver cache in the
project container's local filesystem. After all GPU jobs exited, the files were
copied and checksum-verified externally, the duplicate was removed, and the old
path became a symlink. The wrapper now places CUDA/Triton/Torch/XDG/W&B caches and
temporary files under its configured cache mount. The new argument regression
failed before the fix; the full suite passes 501 tests. No CUDA/Torch stack was
replaced. Evidence: `experiments/operations-20260920/cuda-cache-migration.json`.

The real-trajectory teacher evaluator now follows the same recovery discipline.
Each frozen-writer environment/intervention namespace commits with its input and
raw-byte digest; completed scopes are verified and reused without writer calls.
Scoring commits one condition and its captured original read plan at a time, so
STOP, signal, disk or host pressure does not repeat completed NLL work or reroute
value interventions. A stable output identity rejects changed checkpoint, data,
limits or evaluator code, and the launcher does not publish a stopped stage as
complete. The native LFM evaluation was stopped after 177 of 476 rows and resumed
to completion with zero writer calls on the resumed attempt. It kept all 119
original read selections fixed under both payload controls. The 509-test suite
and Ruff passed. This is frozen-inference progress, separate from training's
complete optimizer/RNG/partial-gradient checkpoint contract.

One subsequent guard audit found that this new teacher path did not yet arm the
device-aware stall watchdog around its actual writer and scorer computations.
The corrected path awaits CUDA completion before disarming each forward, while
leaving SQLite publication and progress-file fsync outside the guard. A focused
regression checks that writer and scoring forwards are guarded and progress I/O
is not; the full suite passes 511 tests. The already active frozen latent depth
sweep keeps its original evaluator checkout.

The local bgkit reference advanced to `b38afd4` after the earlier audit. Its
checkpoint/archive helpers remain unchanged. A new 2,000-file repository case
found an OOM from encoding a whole padded directory level, then bounded that
work in chunks. SDKB does not use bgkit's tree model, but the scaling lesson
applies to its teacher-bank intervention builder: retaining every encoded source
would grow with the evaluation corpus. The builder now keeps at most 64 recent
source encodings, records actual writer calls and peak cache size, and still
commits complete namespaces atomically. Evicted sources may be re-encoded only
while creating original offline scopes. Wrong-value scopes now copy the exact
serialized peer payload from completed original scopes without another writer
call, while retaining each original key, source ID and visibility metadata.
Completed banks and stored inference never invoke the writer. A 140-source
regression covers the bound and all 280 expected raw intervention records;
another compares wrong-value bank bytes to the original writer-based reference.
The new bgkit path-walk evaluation also
reinforces SDKB's existing distinction between teacher NLL and full agent
success; no path-walk capability is claimed here. Source identities are pinned
in `experiments/operations-20260920/bgkit-refresh-source.json`.
