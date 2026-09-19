> Historical evidence/development document. Current SDKB operations: [training](training.md), [Spark](spark.md), [validation](validation-v0.3.md).

# Development guide — v0.2

The governing research plan remains [architecture.md](architecture.md). This guide
covers the expanded executable boundary; [validation-v0.2.md](validation-v0.2.md)
records what was actually run.

## 1. Durable experiment state

A run's `CURRENT` file selects one immutable checkpoint directory. Publication is
atomic after model parameters, optimizer/RNG state, config, and the training-cache
SQLite snapshot are complete. Every component has a checksum. Root model/state
symlinks are conveniences, not recovery authorities.

`sdkb train --resume` verifies source-data identity and model revision, restores the
committed stale/live cache distribution, and truncates later or incomplete log
rows. SIGINT/SIGTERM request a checkpoint after the current complete optimizer
step. They do not interrupt a producer replay halfway through backward. SIGKILL
can lose steps since the last checkpoint; restarting uses the last committed one.

```bash
sdkb train --config configs/tiny_cpu.yaml --output runs/demo --steps 10 --stop-after 3
sdkb train --config configs/tiny_cpu.yaml --output runs/demo --steps 10 --resume
```

`--init-from RUN` instead copies compatible model weights with a **new optimizer,
new RNG schedule, and new cache**. Reader/storage dimensions must agree; newly
initialized compactor parameters are permitted. The source checkpoint is recorded.
This distinction prevents a curriculum/warm start from being mistaken for exact
continuation. Never use `--init-from` and `--resume` together.

Current checkpoints snapshot a small SQLite bank rather than implementing a large
incremental distributed checkpoint system. Budget this copying cost in larger runs.

## 2. Write once, then test multiple uses

`sdkb make-multiuse` creates independently named environments, multiple bindings,
and questions about normal action, permission, restoration, and exact identifiers.
`sdkb make-boolean` supplies the minimal A/B/XOR diagnostic. JSONL episodes support
`task_family`, candidate `choices`, and alternative `sufficient_groups`.

`sdkb evaluate-transfer` creates one heterogeneous bank across all supplied worlds.
Each source is written once; the writer is frozen and unavailable to the stored
read API. Different questions use the same serialized value. Learned retrieval
searches the complete namespace rather than an episode-private bank. Oracle
retrieval uses fixed annotated plans for controlled interpretation.

The primary multiple-choice score is **total answer-sequence NLL including EOS**.
Mean token NLL and mean-score predictions remain separate fields. v0.1 scores
used mean scoring; do not compare their unequal-length choice metrics directly.
Grouped bootstrap metrics resample entire worlds, not each use independently.

`--drop-supports` tests required-record removals. `--boolean-counterfactuals` builds
separate offline bit-flip banks with the same IDs, query, timestamps, and ordering;
no source encoder is called during either read. Report appropriate changes and
false changes separately. For retry tasks, restoration interventions matter only
on branches where retry is actually allowed.

The `shared_compute` arm executes source-independent query/reader/slot computations.
It is an extra-compute/position control, not an information-matched retrieval
alternative. Oracle text remains necessary to diagnose controller capacity.

## 3. Multiple causal reads and bounded inference staging

With `memory.read_steps > 1`, each next query conditions on the previously returned
soft working state. New selections exclude previously read records. The reader
recomposes cumulative evidence into fixed slots, and training supplies supervision
for remaining support-group members. Producer replay has gradient parity against
the complete graph for this path.

This is a **scheduled fixed-budget experiment**, not a learned invocation or halt
policy. `read_steps` is not a replay memory cap. Consumer gradients remain connected
through rounds; this version does not implement arbitrary nested historical reads.

`memory.stream_reads: true` activates single-space, stored-only payload streaming.
A captured plan is fetched in bounded chunks, with authorization/time/deletion
checks. All contributions are accumulated before advancing each reader round.
MLP and attention outputs match their materialized references in CPU tests.

Only one payload chunk and the fixed shared statistics are staged on the compute
device. This does not bound Python ID lists, the host database cache, or all model
memory. It rereads records at every reader round, uses synchronous transfers, and
does not claim asynchronous latency hiding. Compaction plus streaming is currently
rejected explicitly rather than silently materializing everything.

## 4. Compactability as an executable training option

Existing interleaved raw/compact training remains the default. Set
`compaction_objective: paired` to use both task branches on the same noisy inputs,
plus optional detached-teacher KL and conditional numerator/mass matching.
`compact_task_weight`, `behavior_kl_weight`, and `merge_loss_weight` control losses.
Raw-branch task gradients prevent an easier-to-compress but less useful target from
being the only training signal.

Grouping modes are `whole`, `random`, `local`, and `overlap`. Random partitions vary
across steps; local groups use detached greedy neighborhoods. Overlap assigns
nonnegative field responsibilities summing to one per original record. All field
shares are aggregated, preserving mass before approximation. Synthetic records
carry explicit multiplicities. Singleton/small groups stay raw rather than expanding.
These groupings concern the compacted value representation, not a learned routing
or authorization mechanism. No cross-domain compaction is allowed in persistence.

`train.optimization_scope: compactor` freezes every non-compactor parameter and
requires synthetic compaction on every step with no warmup. Use `--init-from` to
start from a useful writer/reader. This makes behavioral preservation a clean test
of amortized compaction rather than hidden retraining of the underlying model.

## 5. Persistent full-cluster codes

`ClusterBank` stores synthetic values and masses, reader-state fingerprints,
original-child membership, generation/domain/time metadata, and deletion lineage.
Keys remain attached to original records. A read uses a compact code **only when
all its children are selected**; partial selections fall back to raw stored values.
This preserves the original retrieval semantics without pretending that arbitrary
child masks can be decoded reliably from a shared approximate code.

Clusters within a view are disjoint. Overlapping-field training is supported, but
persistent overlapping views are not silently merged as duplicate evidence. Domain
mismatch, stale reader fingerprints, revoked children, and invalid timestamps fail
closed. Deleting a child invalidates its parent code. Code generation is offline;
the stored reader receives neither source trajectories nor a compactor callback.

```bash
sdkb evaluate-transfer --run runs/compact-trained --episodes runs/test.jsonl \
  --compact --persistent-compact --drop-supports
```

The helper builds profitable disjoint groups from required-ID annotations for this
oracle experiment. This is not a scalable learned cluster-discovery algorithm.
Original payloads remain for fallback, and the training agent still contains the
compactor module even though inference does not call it. Therefore active-payload
reduction is distinct from net storage savings and deployment parameter residency.
Stripping unused modules, exception handling, variable code budgets, and genuine
selection-conditioned compaction remain future tasks.

## 6. LFM/Spark preflight and common-data matrix

The student remains `LiquidAI/LFM2.5-230M`. No backbone switch was made based on the
CPU diagnostic. The conservative adapter still disables hybrid cache reuse;
gated shared-stack refinement remains experimental.

```bash
./scripts/spark.sh build
./scripts/spark.sh run sdkb doctor --require-spark
./scripts/spark.sh run python scripts/pin_model.py \
  --config configs/lfm25_230m_spark.yaml --output runs/lfm-pinned.yaml
./scripts/spark.sh run sdkb model-probe \
  --config runs/lfm-pinned.yaml --output runs/lfm-model-probe.json
./scripts/spark.sh run python scripts/prepare_matrix.py \
  --config runs/lfm-pinned.yaml --output runs/lfm-matrix \
  --steps 200 --worlds 64 --eval-worlds 32 --bindings 2 --seeds 17
./scripts/spark.sh run bash runs/lfm-matrix/run_all.sh
```

`model-probe` checks the actual public embedding path, base/one-loop equality,
zero-gated two-loop equality, causal-prefix isolation, and finite end-to-end soft
memory gradients. It reports the actual backend and resolved model revision.
Successful tiny tests do not substitute for this real-checkpoint preflight.

The matrix contains oracle text, no memory, source-independent shared compute,
MLP memory, attention memory, one reader round, and two scheduled reads. All arms
use identical source data and test queries per seed. They are **not asserted to
match parameters, token counts, bytes, and compute simultaneously**. Its generated
script resumes committed runs; it launches nothing during preparation.

Additional standalone Spark configs cover paired local compaction, overlapping
fields, streaming reads, oracle text, shared compute, and two causal reads. Pin
revisions for each experiment. A pretrained checkpoint download and a native GPU
run have not been performed in this environment.

## 7. Research priorities after this release

First reproduce the real-student numerical path and oracle-text baseline, then
compare cold-start and controlled warm-start latent transfer. The CPU diagnostic
supports this ordering but does not establish that pretraining needs the same
bootstrap. Improve counterfactual consistency before claiming robust composition.

Subsequent branches remain separable: learned global retrieval and complete-group
recall; overlap/local compaction ablations; partial-selection and exception codes;
key drift and incremental checkpointing; disk ANN; early asynchronous reads; safe
hybrid-state caches; real agentic verifier tasks; and information-matched resource
frontiers. None requires waiting for every other branch to be complete.
