# Adaptive routed writes, continuous bank learning, and document ingestion

**Target specification and implementation status — 22 September 2026**

This revision joins routing, writing, and compaction around one continuously
trained spatial interface. It also replaces the bootstrap convention “one source
string becomes one record” with explicit agentic ingestion trajectories.

## 1. Read weighting and key geometry

Search remains a hard, bounded storage operation. Within the materialized field,
record contribution is continuous. For query key (q), stored key (k_i), local
similarity boundary (b_q), and learned temperature (t_q), the external gate is

\[
a_i=\sigma((\operatorname{cos}(q,k_i)-b_q-\Delta_b(q))/t_q).
\]

The boundary begins at a detached local order statistic, normally the similarity
of the density reference candidate. A bounded query-conditioned adjustment changes
the radius. Temperature is query-conditioned and bounded between configured positive
limits. Each space has its own gate head. There is no global fixed distance scale.

The gate multiplies each record's per-reader contribution before aggregation. The
reader retains the pre-normalization numerator and total mass. It receives both the
conditional mean and `log(1 + mass)`, so adaptive radius does not hide whether a
result arose from weak sparse evidence or many strong records. Retrieval distance
does not change the stored payload itself.

Hard candidate discovery is still nondifferentiable and bounds I/O. Task gradients
train query and stored keys within the materialized field. A modest halo may be
materialized around the nominal result boundary. Records outside that field receive
no gradient from the query.

## 2. Supporting facts are a gentle anchor

When a dataset supplies supporting facts, they serve as positive anchors rather
than a complete declaration that all other records are wrong.

* Early curriculum rows may force annotated records into the materialized field.
* A configurable training-only floor prevents a supplied record from being
  numerically absent before geometry has formed.
* The support loss rewards each annotated record being reachable in at least one
  space. It does not require every space to retrieve every fact.
* Unlabelled records are judged primarily through downstream task utility.
* The support coefficient and forced-delivery rate should decrease as unaided
  retrieval becomes reliable.

The ordinary answer or agent loss remains the primary teacher. Continuous gates
route its gradient into both retrieval geometry and record contribution.

## 3. Selective writer replay and mutable stored records

The forward read uses the actual serialized representation. For a selected live
record, the writer executes its causal prefix through the `memory.write` call, the
key remains FP32, and the payload is cast to configured storage precision before it
becomes a detached replay leaf. The consumer reads those leaves. After consumer
backward, selective replay reconstructs the writer computation under the captured
RNG and autocast state and applies every accumulated key and payload cotangent.

After the optimizer step, touched records are regenerated and committed to a
mutable training overlay. The overlay contains both keys and payloads. Subsequent
searches and reads consult it before the immutable published generation. The overlay
lives in `training_cache.sqlite`, so model, optimizer, RNG, sampler position, and
mutable bank state share one checkpoint commit point. Inference reads stored base or
overlay tensors and never regenerates source trajectories.

Untouched records retain their prior stored version. This is expected bounded
staleness: the task supplied no gradient for them. Background generation refreshes
can reduce writer/version drift without making every optimizer step proportional to
the whole corpus.

## 4. Document ingestion trajectories

Document ingestion is an agent behavior, not a direct call to the latent writer.
Every training producer is a causal chat prefix ending at an explicit
`memory.write` tool call. The call contains auditable text, applicability conditions,
scope, evidence references, and parent read calls. Its tool result publishes the
opaque record ID and generation.

Whole trajectories reserve one key slot and the configured value slots immediately
after each visible write call. These slots traverse the recurrent model alongside
the transcript. Their final causal states produce the stored key and payload, so a
write after a search can depend on the injected latent result as well as visible
messages. The prequential publisher consumes these integrated states; it no longer
re-encodes only the write call's text argument.

The curriculum includes two complementary modes.

The current source-bank builder groups up to four passages from the same article.
A stable article-group hash assigns half the groups to holistic presentation and
half to streaming presentation. Every record keeps its original opaque identity;
the writer prefix for a later record includes the earlier visible write calls and
commit results from that group.

### Holistic mode

The prompt contains one complete document and asks the agent to inspect relevant
memory and retain reusable information. The agent decides how to divide the source
and emits several `memory.write` calls at distinct transcript positions. A
sublinear, initially logarithmic length rule supplies a baseline write count, while
structure, novelty, exact identifiers, and procedural density can request additional
writes. A purely logarithmic ceiling is not an information-capacity claim.

### Streaming mode

The document arrives as natural sections or deterministic overlapping chunks. Each
part is followed by a prompt to retain it and by a distinct `memory.write` call
before the next part arrives. Natural headings, paragraphs, code units, and table
boundaries take priority. The fallback splitter records exact character offsets,
overlap, ordinal, and policy.

Some trajectories then query the newly committed detailed records and emit later
synthesis writes. This creates within-trajectory read-before-write recursion while
retaining the underlying detailed records. Later bank generations repeat the same
pattern over earlier authored memories.

## 5. Compaction

Compaction uses the same local geometry. For compaction center (c), an input's
kernel weight is density-relative rather than governed by one fixed temperature.
Overlapping centers divide each raw record's responsibility so shares sum to one.
This conserves pre-normalization numerator and mass while allowing the information
to influence several local charts.

Compactors consume relative key geometry and local density. Absolute key position is
not an input in the initial design. A bounded learned scale adjustment may modify the
local density estimate. Query-time gates later decide how strongly each compacted
record contributes. Compaction-time and query-time gates therefore have distinct
roles.

Persistent multilevel multi-space compaction remains a subsequent implementation:
the present code provides adaptive field responsibilities and contribution/mass
losses, but the production spatial trainer still rejects integrated multi-space
compaction.

## 6. Implementation status

| Capability | Status on 22 September 2026 |
|---|---|
| Density-adaptive continuous MLP/attention record weights | Implemented in spatial training and stored materialized inference paths |
| Per-space bounded learned radius and temperature | Implemented |
| Union-of-spaces support anchor and supplied-record floor | Implemented |
| Serialized key and payload selective writer replay | Implemented for the pipelined spatial trainer |
| Checkpointed mutable key and payload overlay | Implemented |
| Holistic one-blob/model-chosen multi-write construction | Implemented as data primitive |
| Structured/manual streaming ingestion trajectories | Implemented as data primitive |
| Read-dependent integrated trajectory write slots | Implemented; regenerated spatial artifacts required |
| Adaptive overlapping compaction responsibilities | Implemented as a tested compaction primitive |
| Target Hotpot spatial artifacts with integrated write workspaces | Regenerated on Spark; broader document corpora pending |
| Persistent multi-space recursive compaction executor | Pending |
| Large Spark training validation of v0.6 | Integrated write-slot, gradient, serialized replay, and mutable-overlay smoke passed; sustained run pending |

Unit tests establish invariants and gradients. They do not establish retrieval
quality, compositional memory use, or parameter substitution.

## 7. Next interface generation: wider canonical writes

Eight canonical writer value slots are a compatibility setting for the active v0.6
run, not the intended capacity. The next interface generation starts at **32 writer
value slots per `memory.write` call**. Returned reader slots remain a separate
capacity decision; increasing writer slots does not silently change them.

This is an interface migration. The writer workspace, canonical value width, and
codec input matrices change shape, so an eight-slot checkpoint must not be resumed
as if it were exact. Copy compatible backbone, recurrent, query, reader, and key
parameters explicitly; initialize the additional writer slots and wider codec
inputs under a recorded conversion; then rebuild every stored bank with that writer.
Run 16/32/64-slot information-matched controls later, but do not delay the 32-slot
generation on that sweep.
