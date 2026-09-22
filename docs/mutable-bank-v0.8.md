# Continuously learned mutable bank (SDKB 0.8)

This document supersedes the immutable-bank-generation target in
`trajectory-memory-v0.5.md`, the frozen-corpus-bank target language in
`training.md`, and any plan that treats refreshed keys, values, or compacted
representations as permanent generations. Historical validation records still
describe the experiments that actually used frozen bank snapshots.

## 1. What remains immutable

The durable evidence history remains append-only and auditable:

* source and agent trajectory bytes, causal event order, and availability time;
* opaque source identity, authorization scope, and external provenance;
* read/write tool events and the records visible when each event executed;
* verifier outcomes and corrections as new events rather than edited history;
* committed training checkpoints and captured per-step read plans; and
* the bank mutation journal needed to reproduce or roll back to a checkpoint.

These facts constrain what a memory is allowed to represent. They are not the
learned bank representation itself.

## 2. What changes continuously

A logical memory record is mutable derived state. Its active representation may
change whenever learning or maintenance improves it:

* canonical and per-space keys;
* positional payloads;
* query-relative utility and density statistics;
* index membership and neighborhood structure;
* compaction memberships, responsibilities, codes, and multiplicities;
* compatibility adapters and representation versions; and
* tombstone or supersession state.

An old trajectory does not change when its key or payload changes. The active
record continues to point to that trajectory and to the immutable evidence used
to produce it.

The system therefore has one logically evolving bank. A "generation" or "epoch"
is only a transactional revision boundary, recovery cursor, or reproducible
evaluation snapshot. It is not a permanently frozen parent bank that future
learning merely layers beside.

## 3. Transaction and read semantics

Every optimizer step reads a consistent bank revision:

1. Capture a bank epoch and causal visibility frontier.
2. Derive queries from causal trajectory prefixes.
3. Materialize and freeze the discrete read plans for the step.
4. Fetch the exact key and serialized-payload revisions named by those plans.
5. Complete consumer backward and selective producer replay.
6. Apply the optimizer step.
7. Regenerate touched keys and payloads and atomically publish every configured
   space view as the next logical-record revisions.
8. Update affected index entries and invalidate dependent compacted records in
   the same commit or before they can become visible.

A record must never change underneath an active forward/backward/replay graph.
This is snapshot isolation for a training step, not an immutable-bank policy.
Concurrent inference can similarly pin an epoch for the duration of a request.

Physical revisions may be append-only so readers and recovery retain stable
bytes. The logical record points to its newest admissible revision. Garbage
collection removes superseded physical revisions only after no checkpoint,
request, or compaction lineage can reference them.

## 4. Continuous key and payload learning

Query maps and address transforms learn on every applicable task step. Stored
keys and payloads receive utility gradients when their records enter the
materialized field or are supplied as verified positives. Selective replay sends
the accumulated key and payload cotangents through the writing trajectory, then
publishes regenerated serialized representations after the optimizer step.

Records outside the field naturally receive no direct gradient on that step.
This does not make them permanently frozen. A bounded maintenance scheduler must
sample low-traffic, stale, never-retrieved, recently invalidated, and randomly
selected records for regeneration and key-drift audits. Corpus size must affect
maintenance cadence and index work rather than force every record through every
optimizer step.

Supporting facts remain gentle positive anchors. They pre-seed useful geometry
and ensure early gradients reach relevant records, while task utility and
continuous gates remain the main learned signal.

## 5. Compaction is mutable derived computation

Compacted records are cached learned computations over child revisions. Their
keys, positional payloads, multiplicities, overlapping responsibilities, and
local field boundaries may all change.

Every compacted revision records the exact child revisions and responsibility
shares that produced it. Updating, deleting, or changing authorization for a
child invalidates all transitive compacted descendants. They are recomputed from
the surviving current children before publication. Old compacted revisions may
remain only while a pinned read or recovery point references them.

Recursive compaction is therefore recurrent computation through bank time, not a
sequence of immutable semantic layers. Computation flows forward through updated
codes even though gradients are local to the currently trained compaction step.

## 6. Prequential and contamination boundaries

The bank starts empty or from a declared small core and grows after each causal
event. Event `t` may read only logical records whose immutable source event was
admissible before `t`. Its target and writes become visible only after the event
finishes.

Later learning may refresh the representation of an earlier admissible source;
it may not make a later source visible to an earlier event. Thus causal
eligibility is determined by immutable event history while keys, payloads, and
compactions remain mutable. Replaying a historical evaluation pins both its
visibility frontier and the declared bank/checkpoint revision.

Recursive learning does not require frozen parent generations. A later
trajectory reads the current representations of earlier records, writes new
records with read lineage, and subsequent optimization may update both old and
new logical records.

## 7. Recovery without copying the bank into checkpoints

Model checkpoints record a bank journal cursor, active index revision, compaction
frontier, and maintenance-sampler state. Bank mutations are transactional and
step-labeled. Exact resume restores the model checkpoint and rolls the logical
bank view back or forward to its recorded cursor before another step begins.

Retain physical revisions since the oldest recoverable checkpoint. When a
checkpoint is retired, garbage collection may release revisions referenced only
by it. The large bank is never copied into every model checkpoint.

## 8. Current implementation gap

The v0.6 spatial trainer already updates touched keys and payloads in a
checkpointed mutable overlay and searches those updated keys on subsequent
steps. This is the correct learning behavior. Its published source bank remains
an immutable physical fallback, and refreshed full-bank builds are still used as
handoff boundaries. The prequential executor similarly uses a frozen parent plus
append-only authored records.

Those are transitional implementations. The remaining work is to make the
overlay the durable active logical bank:

* add explicit record revisions and a monotonic mutation journal;
* atomically update all space views and index deltas;
* checkpoint the journal cursor and maintenance scheduler;
* regenerate sampled stale and never-retrieved records;
* invalidate and rebuild transitive compacted descendants;
* add revision retention and garbage collection; and
* replace parent-plus-generation lookup with one visibility-filtered mutable
  catalog.

Until that executor exists, documentation and reports must say "frozen snapshot
plus mutable training overlay" when describing the current run. They must not
present immutable generations as the desired bank architecture.
