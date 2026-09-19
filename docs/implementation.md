# Implementation specification — 0.1

This file maps the research plan to executable behavior. `architecture.md` remains
the design document; the table in the root README is the implementation inventory.

## Shared writer, query and decoder

`MemoryAgent` uses a shared pretrained backbone (or the offline tiny model). A
source is tokenized using the student tokenizer, followed by `write_slots + 1`
learned input embeddings. The first resulting state produces the single canonical
key; the remaining states produce a fixed sequence of canonical value vectors.
Per-space address maps and codecs produce stored keys and payloads.

The writer receives only the source. The query head receives only the current
query prompt. Required-record annotations, target answers and opaque IDs are not
neural input features. Keys are used for retrieval and routing loss, not silently
concatenated into the reader payload. Future tool observations are not available.
Source and prompt limits fail explicitly instead of silently truncating rules.

The read result has fixed output slots, normalized and gated into embeddings after
the prompt. Teacher-forcing loss predicts the first answer token from the last
context state, then each next token from preceding context. Only answer positions
contribute to the supervised loss. Greedy reference decoding recomputes prefixes;
it is not an optimized serving engine.

The current end-to-end runner has one externally scheduled read per task. This is
a scope boundary, **not a memory-driven read cap in the replay algorithm**. The
reader can process variable neighborhoods, and ReplayTape can service multiple
producer captures and repeated consumer uses. Follow-up queries and asynchronous
model scheduling remain explicit backlog items.

## Reader equations

For record x_i, query q, residual slot r_j and round t, the MLP evaluates

```
h_ij = activation(U_t x_i + V_t r_j + Q_t q + e_j)
g_ij = sigmoid(G_t h_ij)     # or unit gate in the lower-level API
numerator_j = P_t sum_i weight_i * g_ij * h_ij
mass_j = sum_i weight_i * g_ij
```

`P_t` has no bias and is applied after pooling. A bias could instead be included
per record with its mass; adding one once after pooling would change the function.
The update reads normalized numerator, log(1 + mass) and residual state, performs a
nonlinear update and fixed-slot mixing, and repeats. Record ordering is invariant;
output-slot and within-record information are not treated as exchangeable.

The attention baseline has the same update interface with exponential scores and
value projections. Its statistics carry a detached numerical log shift to allow
stable merging. The mass is never discarded. Tests include derivative parity at
zero multiplicity, attention chunk merging, query/state/gate gradients and
permutation invariance. Double-precision tests establish algebraic parity; BF16
rounding can vary with chunk shape and is not claimed bit-identical.

Chunk checkpointing does not build an all-pairs neighborhood attention matrix. It
still retains the small statistic graph per chunk/round and the selected payload
inputs; it is not a constant-memory out-of-core GPU streamer. A production payload
streamer can reuse the additive boundary later. All chunks complete a round before
the shared residual advances. Processing each chunk through all rounds independently
would be a different computation.

The multi-space branch evaluates each space against one common residual state and
merges its proposed update. Different widths and neighbor counts are explicit.
The main test corpus has few records and oracle routing, so merely enabling four
spaces does not constitute an experiment in large-neighborhood multiscale retrieval.

## Exact first-order producer replay

`ReplayTape.capture` runs a producer without retaining its graph, returning detached
floating leaves. The consumer accumulates cotangents on these leaves, including
multiple uses of the same leaf. `backward` replays only records with a cotangent
and backpropagates their vector–Jacobian products.

The tape preserves PyTorch CPU/CUDA RNG, autocast mode/dtype, parameter versions,
module training flags and buffers. It rejects parameter updates, changed modes
and mutable buffers before replay. Buffer-mutating producers are unsupported rather
than approximately replayed. Inputs/closure contents must remain immutable until
replay; the training runner captures each source tensor explicitly.

All producer and consumer parameter gradients accumulate before the optimizer
step. Live outputs include storage codecs and precision casts in the captured
forward, so the replayed Jacobian belongs to the values actually consumed. Cached
entries are detached and absent from the tape. This is the exact gradient of the
chosen mixed cached/live computation, not an unbiased all-fresh objective.

Reader dropout/noise is replayed by checkpointing where applicable; producer RNG
belongs to the producer tape. The first implementation does not replay nested
historical read dependencies, support higher-order gradients or sharded distributed
optimizers, or differentiate through historical tool actions. No such guarantees
are implied by the first-order tests.

## Training cache and stored-only evaluation

The training cache stores the first payload produced for an immutable source ID.
A selected live read re-encodes the source with current parameters; an unselected
cached read truly uses the old serialized value. This intentionally exposes a
stale/live mixture. It is not a sophisticated generational refresh policy.

During frozen evaluation, an offline write phase encodes unseen support experiences
once into a new bank. That bank is reopened. The read phase consumes only the stored
payloads; a regression test makes writer calls raise during evaluation. Original
and counterfactual worlds have separate namespaces, preserving identical record
IDs and routing metadata without colliding in the store.

The configured precision (`bfloat16` by default) is used in the actual serialized
payloads. Compute may cast those values back to FP32 under autocast. JSON metadata
and safetensors payloads avoid arbitrary pickle loading. Only locally generated
optimizer checkpoints are loaded through `torch.load(weights_only=True)`.

## Routing and compaction

The exact CPU reference searches only records eligible under namespace, model
space/generation, authorization domain and time. Its bounded key chunks avoid
loading the whole key array into VRAM. It is a full scan, not ANN. The initial
learned router trains complete sufficient sets under a Plackett–Luce selection
objective, marginalized over valid orderings for small groups. This directly
rewards group starters even when isolated utility is zero. Single-utility ranking
is a separate utility that skips flat targets. Top-k itself is still discrete.

Temporary compaction replaces an entire selected neighborhood with its weighted
mean or a few amortized synthetic records. The synthetic writer preserves total
multiplicity and is shared across clusters. Contribution loss matches each round's
numerator/mass at raw-reader states and also a free compact rollout. Normal task
steps remain interleaved with compact task steps; a given compact step does not
also compute a second raw task loss. The stronger paired objective in the plan is
a follow-on ablation. The raw contribution teacher is detached.

`local_merge_loss`, random grouping, overlap responsibilities and storage-noise
functions are implemented/tested APIs. Persistent overlapping field storage and
an integrated regrouping-training schedule are **not** claimed. Responsibilities
split both contributions and mass; retrieving only some shares requires a separately
specified routing plan. Noise does not make independent facts compressible by itself.

`FullClusterCode` rejects partial selections and cross-domain reads. Persistent
arbitrary-subset decoding is not approximated by a full-cluster code. SQLite
lineage permits invalidation of compact derivatives but does not automatically run
a compactor or replace record indices. These are distinct pieces of infrastructure.

## Persistence and concurrency

SQLite records are immutable within namespace/ID/space/generation. A read plan
captures IDs and scores; values are rechecked for visibility when fetched. Tombstones
prevent resurrection under the same logical ID, and deletion invalidates transitive
compact parents. Regeneration needs a new logical record ID after deletion.

Authorization domains are caller-provided research boundaries, not user
authentication: a production service must bind caller identity to permitted domains.
Do not compact incompatible domains. Deletion is logical invalidation, not certified
physical erasure from WAL, backups or previous experiment artifacts.

AsyncRetriever uses CPU threads to overlap disk retrieval with caller work. The
model runner does not yet submit early-layer requests or manage delayed-result loops.
This narrow API is intentionally separate from claims about overlap performance.

## Run artifacts

A run records config, dependency/model revision, dataset content hash, metrics,
resource counters, a safetensors checkpoint and local optimizer state. Resume only
allows changing total step count; changing episode content is rejected before
reusing stale cache data. Checkpoints are atomically replaced at the file level,
not a fault-tolerant multi-file transaction. The prototype saves at run completion;
periodic checkpoints and interruption recovery remain development tasks.

A mutable `model.revision: main` is logged as the actual resolved commit but should
be pinned before training. Different runs can otherwise fetch different initial
weights/tokenizers even though their YAML strings look identical.
