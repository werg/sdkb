# Mutable-bank v0.8 validation

Date: 22 September 2026.

## Executed checks

The complete core suite passed in the pinned Spark project container:

```text
625 passed, 4 third-party/runtime warnings in 63.98 seconds
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
  with equivalent keys and payloads.

## Interpretation and limits

These checks establish local transaction, causality, migration, and exact-resume
semantics. They do not establish useful retrieval, compaction quality, network-store
behavior, or parameter substitution. SQLite remains the local correctness backend.
The network implementation must reproduce the same pinned-cursor and atomic
all-space publication contract before it can replace this reference.

The migration is activated when a legacy spatial run next starts. A Python process
that was already running before this implementation continues using the code and
tables it imported until it is gracefully checkpointed and resumed.
