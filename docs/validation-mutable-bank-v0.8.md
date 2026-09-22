# Mutable-bank v0.8 validation

Date: 22 September 2026.

## Executed checks

The complete core suite passed in the pinned Spark project container:

```text
626 passed, 4 third-party/runtime warnings in 54.70 seconds
```

`ruff check --no-cache src tests scripts` and Python bytecode compilation also
passed. No model download was needed for these checks, and the already running Phase
1 GPU process was not used as capability evidence.

Regression coverage establishes the following storage and recovery behavior:

* a logical-record update is rejected unless every configured space view is present;
* keys and serialized payloads are appended under one monotonic commit cursor;
* captured read plans continue to fetch their pinned payload revision after a later
  update;
* active journal heads patch the resident exact index, including newly authored IDs;
* checkpoints contain `bank-state.json` and do not contain a copied bank SQLite file;
* restoring a checkpoint verifies store identity and commit digest, truncates a later
  branch, restores the maintenance cursor, and rebuilds logical heads;
* an event's record revisions, lineage, metadata, dependencies, and visibility
  frontier commit or roll back together;
* changing a child invalidates transitive compact descendants, which remain
  unavailable until republished against declared current children;
* revision GC refuses to pass a retained non-base checkpoint pin; and
* the former overwrite-in-place overlay migrates into bounded batches and reopens
  with equivalent keys and payloads; and
* reopening the same legacy checkpoint after migration reuses the recorded migration
  cursor instead of copying the legacy overlay a second time.

## Live Phase 1 migration and recovery

The running four-space Phase 1 job was cooperatively stopped at optimizer step
3,182 and its legacy overlay was migrated into the journal. The migration published
232,876 active space views in 15 bounded commits. The legacy overlay table was then
removed.

After resuming, the trainer reached step 3,193 and cooperatively wrote a native
mutable-bank checkpoint at journal cursor 26. The checkpoint contains the model,
optimizer/training state, and a 249-byte `bank-state.json`; it does not copy the
1.27 GB working SQLite bank. The recorded store UUID and commit digest were verified
on reopen. Training then committed steps 3,194 and 3,195 at cursors 27 and 28 while
retaining 232,876 active views. This exercises migration, checkpoint publication,
exact reopening, and continued mutation with the actual Spark training process.

## Interpretation and limits

These checks establish local transaction, causality, migration, and exact-resume
semantics. They do not establish useful retrieval, compaction quality, network-store
behavior, or parameter substitution. SQLite remains the local correctness backend.
The network implementation must reproduce the same pinned-cursor and atomic
all-space publication contract before it can replace this reference.

The live Phase 1 run has completed the legacy migration and now uses the journal
checkpoint path. Other legacy spatial runs migrate independently when first opened
by the current trainer.
