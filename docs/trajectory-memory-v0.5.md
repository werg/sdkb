# Trajectory memory program — v0.5 plan

**Status:** target design and implementation plan. The current HotpotQA curriculum
does not implement this behavior. It remains useful interface pretraining for the
writer, reader, recurrence, and global addressing.

## 1. Target behavior

SDKB should act as a small controller over a very large persistent knowledge base.
On long tasks it should repeatedly:

1. emit a visible `memory.search` tool call from the causal trajectory prefix;
2. receive a normal tool-result event whose attached latent payload was fetched
   from storage;
3. use that result in later reasoning, tool use, and follow-up searches;
4. emit one or more `memory.write` calls when it has acquired reusable information;
5. receive opaque committed record IDs and continue the task.

Memory use is therefore distributed through the trajectory. It is not one implicit
read before one short answer. A long trajectory may contain dozens of read and write
sites. The number of writes scales with useful input content rather than collapsing
an arbitrarily long input into one fixed record.

A **site** means one distinct tool call at one causal token/event position. Several
items submitted in one call count as one write site. Several native recurrent reads
used to compose one returned result count as one search site. Recursive bank
generations are a third, independent axis: the target requires repeated sites within
each trajectory *and* repeated generations whose new records were authored after
reading earlier generations.

Normal inference never re-encodes source trajectories during a read. A write is an
explicit new operation that encodes the content supplied to `memory.write`. Offline
encoding of an immutable corpus by a frozen writer remains a separate bank-building
operation.

## 2. Current fit and gaps

| Requirement | Current state | v0.5 change |
| --- | --- | --- |
| Frequent reads through a long trajectory | Native recurrence supports one workspace queried at depth boundaries, but the target run uses R=2 and one read per episode. | Add many transcript-level query positions and blank result workspaces; activate same-level sites together and scatter their results before the next pass. |
| Queries as tool calls | Query vectors are extracted automatically from hidden state. No memory call appears in the transcript. | Train and execute `memory.search` calls with textual arguments, then derive the latent search key from the causal call state. |
| Results as tool results | Latent tokens are inserted into one reserved workspace without a visible tool event. | Give every result event its own blank workspace span, then replace or update that span with aligned latent result tokens between recurrent passes. The visible envelope contains handles and status, not source text. |
| Prompted write behavior | Sources are passed directly to the writer or offline bank builder. | Put a memory-use policy in the system prompt and supervise `memory.write` calls whose arguments contain the learned reusable content. |
| Several writes for long input | Dataset preparation can chunk trajectories into several fixed producers, but the model does not decide or emit those writes. | Supervise distinct write calls at distinct causal positions, normally one logical record per call, with optional later synthesis calls. |
| Read-before-write recursive memory | The current two-generation bank refresh trains against a bank, but bank contents are source chunks, not memories authored by read-augmented trajectories. | Publish immutable generations containing outputs of trajectories that queried an earlier generation, retaining complete lineage. |
| Pretraining/posttraining corpus scale | The target bank contains 100,000 sources. | Build a sharded logical bank that can grow through 1M, 10M, and 100M records before scale-out corpus ingestion. |
| Capacity substitution evidence | No matched result yet. | Compare a small controller plus bank with larger resident models under equal information and declared byte, latency, and compute budgets. |

The visible tool protocol and native recurrent read are complementary. Tool-call
positions choose **where, when, and why** memory is accessed. At one recurrence
level, the model processes the complete teacher-forced trajectory in parallel,
extracts every active site's query from its causally masked position, and dispatches
those searches together. Their latent results are scattered into separate blank
site workspaces before the next recurrent pass. A later level can issue new queries
whose positions causally see results injected at earlier positions and levels.
Increasing the current `read_steps` alone does not create spatial sites.

## 3. Tool protocol

### 3.1 Search

The canonical assistant event is structurally equivalent to:

```json
{
  "name": "memory.search",
  "arguments": {
    "query": "transaction retry rules for capability-gated services",
    "scope": {"project": "opaque-project-id"},
    "purpose": "choose the next recovery action",
    "budget": "standard"
  }
}
```

The result event is structurally equivalent to:

```json
{
  "name": "memory.search",
  "call_id": "opaque-call-id",
  "status": "ok",
  "record_ids": ["opaque-record-id-1", "opaque-record-id-2"],
  "spaces": {"s0": 64, "s1": 32, "s2": 16, "s3": 8},
  "generation": "immutable-generation-id"
}
```

The tool result has latent payload tokens attached at that causal location. It does
not reveal source text on the normal latent-memory path. A selected-text arm may be
used as an information-matched teacher or control, and must be labeled separately.
Empty, denied, timed-out, and partial results remain distinct tool outcomes.

The query string is useful for supervision, debugging, sparse or hybrid retrieval,
and later policy editing. Retrieval keys still come from the causal model state so
the latent and textual query representations can be trained jointly. Scope is an
authorization filter. Learned ranking never grants access.

### 3.2 Write

The system prompt tells the agent to store information that is likely to help a
future task, including applicability conditions, evidence, outcome, and relevant
identifiers. A call is structurally equivalent to:

```json
{
  "name": "memory.write",
  "arguments": {
    "content": "After error E17, restore snapshot S before retrying when capability C is present.",
    "kind": "procedural_lesson",
    "applies_when": "service uses protocol version v3",
    "evidence_refs": ["opaque-observation-id"],
    "confidence": "observed"
  }
}
```

The result returns one committed logical record ID. It is encoded into all configured
spaces. The behavior curriculum uses one logical record per call so that several
records require several visible calls at separate causal positions. A future bulk
transport extension may accept several items, but that still counts as one site and
cannot satisfy a multi-site training target. Source commands remain inert text and
the write tool cannot execute them.

Writes retain raw/teacher/model-authored provenance, the producing model and writer
versions, trajectory ID, causal timestamp, evidence references, parent read IDs,
and verification status. Model interpretation is not silently promoted to fact.

## 4. Site and record schedules

The first supervised schedule is event- and length-based. These are starting
targets, not measured optima:

- Search after a new task is understood, after substantial external observations,
  after a failed attempt, before an irreversible action, and before a final synthesis.
- Insert at least one search opportunity per 128–256 newly visible trajectory
  tokens, subject to the event boundaries above.
- Short tasks should contain 1–2 search sites; medium tasks 3–8; long tool tasks
  8–32. The executor must not impose these as a hard architectural ceiling.
- Place write calls immediately after durable discoveries, resolved failures, or
  completed subgoals, with optional synthesis calls after later evidence. Short
  tasks should contain 0–2 write sites, medium tasks 2–8, and long tool tasks 4–32
  when the trace contains that many independently reusable lessons.
- Create one detail memory per roughly 96–192 useful source tokens or one coherent
  event, plus optional cross-item lessons after a successful or failed task.
- A pure knowledge-injection example first performs context/deduplication searches
  at distinct positions across the input, then makes approximately
  `ceil(useful_tokens / target_chunk)` distinct detail write calls and any justified
  synthesis calls. It must not batch all details into one terminal call.
- A long agent trajectory writes after meaningful discoveries and at task closure.
  Search and write calls may interleave as new evidence changes the task. Call count,
  call positions, retained source coverage, and evidence coverage are logged. A
  burst of calls at task closure cannot satisfy a distributed-site target.

Training examples include fewer and more sites than the nominal schedule so the
model learns placement rather than a mechanical interval. Every inserted site is
causal: its query and write content can use only preceding events.

## 5. Representation and retrieval budget

The present four stored views use 256, 512, 1,024, and 2,048 BF16 scalars, plus four
64-dimensional FP32 keys. That is 8,704 array bytes per logical record before
metadata and database overhead. For the current 65-token sources this is already a
large representation per token. The immediate bottleneck is more likely record
coverage, retrieval quality, selected bytes, and the eight-token returned workspace
than raw payload width alone.

The v0.5 starting architecture therefore uses:

| Axis | v0.5 starting point |
| --- | --- |
| Logical record granularity | 96–192 useful tokens or one coherent event |
| Stored views | Existing 256/512/1,024/2,048 BF16 payload widths |
| Search neighborhood | 64/32/16/8 records, 128 KiB of raw value arrays per site |
| Returned workspace | 16 decoder-width soft tokens |
| Search sites | 1–32 per trajectory according to length/events |
| Compaction | Disabled until uncompressed multi-site use works; then train contribution-and-mass preserving codes |

This is a fresh interface version because changing returned slots changes model
shape. Payload widths remain a measured capacity axis. Promote wider payloads only
when source reconstruction, exact-detail transfer, or future-utility curves show
that adding records and read budget has saturated. Evaluate 1×, 2×, and 4× stored
payload bytes at equal source coverage before selecting a corpus-scale encoding.

At 8,704 array bytes per record, lower-bound storage is:

| Logical records | Key and value arrays | Approximate source tokens at 128/item |
| ---: | ---: | ---: |
| 100 thousand | 0.87 GB | 12.8 million |
| 1 million | 8.7 GB | 128 million |
| 10 million | 87 GB | 1.28 billion |
| 100 million | 870 GB | 12.8 billion |
| 1 billion | 8.7 TB | 128 billion |

Two immutable generations, source text, indices, metadata, compaction workspaces,
and publication scratch space increase these totals. A trillion-token corpus at
128 tokens per record would require about 7.8 billion records and 68 TB of arrays
before overhead. At 256 tokens per record it would still require about 34 TB. The
current 3.6 TB external disk comfortably supports the 10M-record tier. One 100M
generation may fit after explicit accounting, but two raw generations already need
at least 1.74 TB of arrays; existing artifacts, sources, indices, metadata, and build
scratch make that an unsafe default on this machine. The 100M tier and the full
pretraining/posttraining objective require a platform-agnostic sharded store across
additional local storage or object storage.

"One knowledge base" means one logical, queryable catalog with common identity,
authorization, provenance, and generation rules. It need not be one SQLite file or
one physical device. Immutable shards and generations remain necessary.

### 5.1 Remote latency and depth-wave scheduling

Corpus-scale banks will commonly place search and payload shards behind a network.
Training does not stop separately at every spatial site. For recurrent level
$r$, it should:

1. Run the complete teacher-forced trajectory through the prelude and current core
   pass, with all sequence positions computed in the transformer's normal parallel
   causal execution. Learned Perceiver-style site workspaces contain no retrieved
   information on their first pass.
2. Gather queries from every `memory.search` position active at level $r$. Causal
   masking prevents a site from using later tokens even though positions execute in
   parallel.
3. Dispatch all site/space searches asynchronously, batching compatible requests by
   bank generation and authorization scope. Each request records model, time, site,
   and level identity.
4. Park that pass's differentiable full-trajectory state and run other trajectory
   batches while network requests are outstanding.
5. When the required response set is ready, compose each site's latent result and
   scatter it only into that site's workspace. Run the next shared core pass over
   the whole trajectory.

Thus lookup latency is paid by **recurrence depth waves**, not once per spatial site.
Eight or thirty-two same-level sites can overlap; the wave waits for its required tail
response. Later-level queries may depend on earlier-position results from prior
levels, so dependent levels cannot be collapsed into one retrieval wave. This depth
structure is the intended within-trajectory recursion.

The highest-throughput path retains the differentiable full-sequence state and graph
across the wait. Bound the number and bytes of parked batches so network stalls cannot
exhaust host or unified memory. When a graph does not fit, a fallback may checkpoint
the level boundary and reproduce RNG and autocast to recompute it after payloads
arrive. In either mode, backward accumulates every site and level's query, routing,
reader, recurrent, writer, gate, and shared-parameter paths before the optimizer
step. Checkpoint recomputation consumes captured responses and never issues a search.

The ready pool must contain enough independent full-trajectory passes to cover
`lookup latency / GPU time per recurrent pass`, plus margin for tail latency and
length variation. Apply bounded queues and backpressure. Timeout, denial,
cancellation, empty result, retry, and late response remain explicit per-site tool
outcomes with idempotent request IDs.

Remote payload bandwidth is not assumed free. At the initial 128 KiB raw value
budget, 8--32 search sites transfer roughly 1--4 MiB per long trajectory before
protocol and index overhead. Log request queue time, search time, payload-fetch time,
bytes, ready-queue depth, parked-state bytes, recompute cost, GPU idle time, and
end-to-end site latency. Compare synchronous and pipelined throughput using the same
read plans.

Autoregressive serving cannot process future, ungenerated positions in the same way
as teacher-forced training. Once it emits a search call and receives the result, a
fast path can populate that site's workspace immediately after the fixed prelude and
enter the recurrent core with memory already present. It need not reproduce the
training-only blank pass for a query it has just generated. A post-training or policy
optimization phase should expose both blank-first-pass and immediate-result schedules
and measure transfer between them; they are not silently assumed equivalent. Serving
can batch available calls and interleave independent users, but required retrieval
waves still contribute to request latency. Use measured hot caches, co-located
shards, request batching, and policy-visible latency budgets.

## 6. Recursive bank generations

Each generation has a frozen writer identity and explicit parent generations.

1. **G0 corpus bank:** encode raw pretraining-like documents, documentation,
   structured knowledge, and verified posttraining trajectories. Keep provenance
   classes separate inside one logical catalog.
2. **G1 read-augmented experience bank:** run supervised/teacher trajectories that
   query G0 repeatedly, then commit their `memory.write` items. Train later tasks
   against both G0 and G1.
3. **G2 utility-trained bank:** run longer tasks against G0+G1. Credit writes by
   their measured utility on future heldout tasks, not merely by similarity to a
   teacher summary. Publish accepted writes as G2.
4. **G3 and later refreshes:** repeat read → act → observe → write → freeze → publish
   for several generations. Retain raw evidence links and measure whether recursive
   memories add utility or amplify errors.
5. **Compacted generations:** after raw recursive banks work, train compact codes
   against fixed queries and reader states. Publish them as new immutable views;
   never relabel old raw records as if the writer or compactor had not changed.

Each training mixture must include direct raw records, teacher-authored memories,
model-authored read-free memories, and model-authored read-augmented memories. Log
their contribution separately. Recursive content must become a substantial fraction
of later training, while raw/verified anchors remain present to limit self-reinforcing
drift.

### 6.1 Prequential growth within a generation

Large-corpus work begins from an empty bank or a small immutable core and consumes a
time-ordered event stream. For each event, the system records the current visibility
frontier, performs all reads against records committed before that frontier, finishes
the trajectory and outcome, and atomically publishes its eligible write calls only
afterward. A target therefore cannot retrieve itself or a memory derived from its
future answer. Resume state includes the event cursor, bank generation/frontier,
pending atomic write set, sampler state, and model checkpoint reference.

Training samples multiple logged bank-size bands instead of observing only the final
bank. Later recursive slices run against a bank containing many earlier
read-augmented trajectories. The mixture deliberately includes fresh and stale
records, duplicates, conflicts, failures, corrections, and records with uncertain
future utility, matching a developing experiential store rather than a clean static
encyclopedia.

Use three clocks: original evidence availability, ingestion, and trajectory event
time. Authorization and causality are checked before learned selection. Heldout
targets and their derivatives remain unavailable until their evaluation event has
finished. Garbage collection creates a versioned successor view using tombstones,
deduplication, or valid compaction; logged historical reads continue to resolve
against their original immutable frontier. Preserve lineage through collection so a
recursive summary cannot outlive deletion of evidence it depends on.

## 7. Training curriculum

### Phase A — Tool syntax and placement

Edit recorded tool trajectories to include `memory.search` opportunities and
`memory.write` targets at separate causal positions. Train call/no-call decisions,
query text, call count, placement, and valid arguments. Use negative examples where
searching or writing would add no value. This phase may use selected text in tool
results as a teacher arm.

### Phase B — Latent result use

Replace teacher text with stored latent results while preserving the same tool
call/result positions and selected record IDs. Train downstream tokens and actions,
query addressing, reader composition, and recurrent integration. Include fixed-plan
zero, wrong, dropped, and source-counterfactual payload interventions.

### Phase C — Scaled write extraction

Give documents and trajectories of varied length. Require multiple `memory.write`
calls at different context positions, proportional to useful content. One call per
record is the default curriculum contract. Supervise factual fidelity, provenance
links, coverage, applicability conditions, deduplication, and separation of
observations from interpretations. Reopen the stored payloads and test
reconstruction, extraction, and future-task utility.

### Phase D — Read-then-write trajectories

Train complete traces where several search results influence actions and later
writes. A write records its parent reads. The main target is future utility: later
source-disjoint tasks should improve when the new record is available and regress
under a fixed-plan intervention.

### Phase E — Long agent tasks

Use tool-rich software, browsing, research, and multi-document tasks with 8–32
search sites and 4–32 write sites where supported by the trace. Both kinds of calls
must occupy multiple causal positions rather than one batched location. Start with
edited successful teacher trajectories, then add self-distillation and
bounded-environment execution. Teacher likelihood, tool-call validity, and actual
task success remain separate metrics.

### Phase F — Policy optimization

Once supervised behavior is stable, optimize search placement, query quality,
write selection, and item count using future-task reward and explicit read/write
costs. Reinforcement learning must operate through the same tool protocol and
immutable bank generations; it must not gain hidden source access.

Include the autoregressive fast path in this phase. After the model emits a search
call, retrieve its payload and inject the result after the fixed prelude on the
continuing prefix, without requiring a blank recurrent pass. Mix this schedule with
the full-trajectory blank-first-pass schedule during post-training and measure task
success, query placement, payload dependence, latency, and calibration separately.

Retain reconstruction and short multi-hop examples throughout as interface anchors,
but the majority of later updates should contain multiple sites and at least one
read-before-write dependency.

## 8. Implementation releases

### Release 1 — Transcript contract and data builder

- Define canonical `memory.search` and `memory.write` schemas and result envelopes.
- Preserve call IDs, opaque record IDs, causal timestamps, authorization scope,
  evidence references, writer identity, and parent-read lineage.
- Build an editor that inserts supervised sites without exposing future events.
- Add length/event-based site-count manifests and source coverage accounting.

### Release 2 — Stateful memory tool executor

- Represent every visible search call with a site-aligned blank latent workspace and
  an activation recurrence level.
- Gather all active site queries after a whole-trajectory core pass, execute their
  stored retrievals, and scatter results into their respective workspaces for the
  next pass.
- Support many calls per trajectory, cancellation, retries, and checkpointable
  session state.
- Add asynchronous request IDs, bounded pending/ready queues, response hashes,
  idempotent retries, and batch dispatch across sites and independent trajectories.
- Support the autoregressive immediate-result path that injects a completed call's
  workspace after the fixed prelude, while retaining the blank-first-pass training
  path as the reference computation.
- Keep the stored-only read invariant and make source encoding available only to
  explicit writes or offline builds.

### Release 3 — Multi-site training and replay

- Train over complete edited trajectories rather than one support/query answer.
- Capture every site/level discrete read/write plan before backward.
- Accumulate query, key, value, reader, recurrent, gate, and shared-parameter paths
  across all sites before the optimizer step.
- Retain bounded differentiable full-sequence states across remote waits; support
  level-boundary recomputation as a lower-memory fallback and never query storage
  during recomputation.
- Batch sites and producers without truncating counts; checkpoint the site cursor,
  request/response hashes, tool state, RNG, optimizer, and partially accumulated
  gradients.

### Release 4 — Corpus-scale growing bank

- Replace a single SQLite scan with sharded contiguous keys and measured exact or
  approximate search, while SQLite remains the correctness reference.
- Separate payload shards, source archive, metadata, and indices behind storage
  interfaces that support local NVMe, external disk, and object storage.
- Publish logical records only after every required space view and manifest hash
  is complete. Support incremental shard resume and generation-level rollback.
- Add an event-stream builder with atomic post-trajectory writes, immutable
  visibility frontiers, resumable cursors, and coverage by bank-size band. Keep
  heldout targets and their derivatives beyond the active frontier.
- Add versioned garbage collection with provenance/deletion propagation and evaluate
  stale, conflicting, redundant, corrected, and rare retained records.

### Release 5 — Recursive generations

- Produce G1 from read-augmented trajectories, then train against G0+G1.
- Repeat through at least three authored generations before making recursive-memory
  claims.
- Add trust/provenance sampling, duplicate/conflict handling, and error-amplification
  evaluations.

### Release 6 — Integrated compaction

- Extend contribution-and-mass preserving compaction to multi-space, multi-site
  trajectory reads.
- Train against the same reader states and selection opportunities as raw records.
- Account for exceptions, raw fallback, indices, rewrite work, and total retained
  bytes. Do not serve a subset from a full-cluster code.

## 9. Required measurements

Report all of the following by trajectory length and provenance class:

- search opportunities, distinct calls made, unique causal positions, calls per
  1,000 visible tokens, inter-call distance, terminal-call concentration, and trigger
  type;
- query-to-required-source recall, precision, rank, and cross-space overlap;
- selected records and bytes per site and per completed task;
- downstream task success and teacher NLL under real, zero, wrong, and dropped values;
- blank-first-pass versus immediate-result task quality, payload dependence,
  recurrent compute, and latency after post-training;
- write opportunities, distinct calls, unique causal positions, records written,
  useful-token/evidence coverage, duplication, contradiction, factual fidelity,
  terminal-call concentration, and future-task utility;
- fraction of writes whose lineage includes earlier reads, recursion depth, and
  marginal utility by generation;
- bank records, logical source tokens, raw and compact bytes, index bytes, source
  archive bytes, temporary publication bytes, and generation overlap;
- end-to-end latency, search latency, payload I/O, model compute, and energy where
  available; include queue time, parked-state bytes, ready depth, recompute overhead,
  GPU idle fraction, and synchronous-versus-pipelined throughput; distinguish warm
  cache, cold storage, exact search, ANN, local disk, and remote service;
- resident parameters and memory for the small controller plus services.

The capacity-substitution test compares small-controller-plus-bank systems against
larger resident models and text-retrieval controls with equal source access. It must
trace a quality–resident-memory–latency–storage frontier across bank sizes. More bank
bytes alone, lower teacher NLL, successful ingestion, or high GPU utilization does
not establish parameter substitution.

## 10. Immediate course decision

Retain the completed live-writer phase and its 100,000-source bank as Phase 0
interface and addressing pretraining. Replace the unstarted one-query bank-training
stages with whole-trajectory spatial stages containing several visible search calls.
This compatibility stage keeps eight read slots so it can initialize from the frozen
bank writer exactly. Start the later v0.5 capacity run as a fresh 16-read-slot
interface initialized from compatible backbone weights; do not disguise that shape
change as an ordinary resume.

The first v0.5 milestone is a stored-only long trajectory with at least eight visible
search calls and four write calls, each at distinct causal positions separated by
other trajectory events. At least two writes must causally depend on earlier latent
results, and a later heldout task must change correctly when one authored memory is
removed under a fixed read plan. Corpus ingestion for that milestone is prequential:
start empty or from a small core, retrieve against each prior frontier, and publish
writes only after the event completes. The first scale milestone is 1 million logical
records, followed by 10 million after search, ingestion, and recovery measurements
pass. Do not schedule the 100 million tier on the present external disk until source,
index, scratch, retained-generation, and free-space budgets prove that publication
and rollback both fit. Prefer the scale-out store before that tier.
