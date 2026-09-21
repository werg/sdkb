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

Normal inference never re-encodes source trajectories during a read. A write is an
explicit new operation that encodes the content supplied to `memory.write`. Offline
encoding of an immutable corpus by a frozen writer remains a separate bank-building
operation.

## 2. Current fit and gaps

| Requirement | Current state | v0.5 change |
| --- | --- | --- |
| Frequent reads through a long trajectory | Native recurrence supports scheduled state-dependent reads, but the target run uses R=2 and one read per episode. | Add transcript-level sites after observations and between actions; each site may use one or more native recurrent reads. |
| Queries as tool calls | Query vectors are extracted automatically from hidden state. No memory call appears in the transcript. | Train and execute `memory.search` calls with textual arguments, then derive the latent search key from the causal call state. |
| Results as tool results | Latent tokens are inserted into reserved workspace slots without a visible tool event. | Return an ordinary tool-result envelope plus aligned latent result tokens. The deployment result contains handles and status, not source text. |
| Prompted write behavior | Sources are passed directly to the writer or offline bank builder. | Put a memory-use policy in the system prompt and supervise `memory.write` calls whose arguments contain the learned reusable content. |
| Several writes for long input | Dataset preparation can chunk trajectories into several fixed producers, but the model does not decide or emit those writes. | Supervise a length- and event-scaled sequence of write calls, with one logical record per item and optional higher-level synthesis records. |
| Read-before-write recursive memory | The current two-generation bank refresh trains against a bank, but bank contents are source chunks, not memories authored by read-augmented trajectories. | Publish immutable generations containing outputs of trajectories that queried an earlier generation, retaining complete lineage. |
| Pretraining/posttraining corpus scale | The target bank contains 100,000 sources. | Build a sharded logical bank that can grow through 1M, 10M, and 100M records before scale-out corpus ingestion. |
| Capacity substitution evidence | No matched result yet. | Compare a small controller plus bank with larger resident models under equal information and declared byte, latency, and compute budgets. |

The visible tool protocol and native recurrent read are complementary. A tool call
chooses **when and why** to access memory in the agent trajectory. Native recurrence
controls how a single result is composed into latent working state. Increasing
`read_steps` alone does not teach the agent to place calls throughout a multi-turn
task.

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
    "items": [
      {
        "content": "After error E17, restore snapshot S before retrying when capability C is present.",
        "kind": "procedural_lesson",
        "applies_when": "service uses protocol version v3",
        "evidence_refs": ["opaque-observation-id"],
        "confidence": "observed"
      }
    ]
  }
}
```

The result returns one committed logical record ID per item. Each item is encoded
into all configured spaces. A multi-item call is transport batching, not one fused
record. Source commands remain inert text and the write tool cannot execute them.

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
- Create one detail memory per roughly 96–192 useful source tokens or one coherent
  event, plus optional cross-item lessons after a successful or failed task.
- A pure knowledge-injection example first performs context/deduplication searches
  distributed across the input, then writes `ceil(useful_tokens / target_chunk)`
  detail items and any justified synthesis items.
- A long agent trajectory writes after meaningful discoveries and at task closure.
  Write count, retained source coverage, and evidence coverage are logged. A single
  end-of-task summary cannot satisfy a many-item target.

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

## 7. Training curriculum

### Phase A — Tool syntax and placement

Edit recorded tool trajectories to include `memory.search` opportunities and
`memory.write` targets. Train call/no-call decisions, query text, item count, and
valid arguments. Use negative examples where searching or writing would add no
value. This phase may use selected text in tool results as a teacher arm.

### Phase B — Latent result use

Replace teacher text with stored latent results while preserving the same tool
call/result positions and selected record IDs. Train downstream tokens and actions,
query addressing, reader composition, and recurrent integration. Include fixed-plan
zero, wrong, dropped, and source-counterfactual payload interventions.

### Phase C — Scaled write extraction

Give documents and trajectories of varied length. Require multiple `memory.write`
items proportional to useful content. Supervise factual fidelity, provenance links,
coverage, applicability conditions, deduplication, and separation of observations
from interpretations. Reopen the stored payloads and test reconstruction, extraction,
and future-task utility.

### Phase D — Read-then-write trajectories

Train complete traces where several search results influence actions and later
writes. A write records its parent reads. The main target is future utility: later
source-disjoint tasks should improve when the new record is available and regress
under a fixed-plan intervention.

### Phase E — Long agent tasks

Use tool-rich software, browsing, research, and multi-document tasks with 8–32
search sites and several writes. Start with edited successful teacher trajectories,
then add self-distillation and bounded-environment execution. Teacher likelihood,
tool-call validity, and actual task success remain separate metrics.

### Phase F — Policy optimization

Once supervised behavior is stable, optimize search placement, query quality,
write selection, and item count using future-task reward and explicit read/write
costs. Reinforcement learning must operate through the same tool protocol and
immutable bank generations; it must not gain hidden source access.

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

- Pause generation on a memory tool call, execute stored retrieval/write, append
  the tool result, and continue generation.
- Attach latent result slots to the tool-result location.
- Support many calls per trajectory, cancellation, retries, and checkpointable
  session state.
- Keep the stored-only read invariant and make source encoding available only to
  explicit writes or offline builds.

### Release 3 — Multi-site training and replay

- Train over complete edited trajectories rather than one support/query answer.
- Capture every discrete read/write plan before backward.
- Accumulate query, key, value, reader, recurrent, gate, and shared-parameter paths
  across all sites before the optimizer step.
- Batch sites and producers without truncating counts; checkpoint the site cursor,
  tool state, RNG, optimizer, and partially accumulated gradients.

### Release 4 — Corpus-scale bank

- Replace a single SQLite scan with sharded contiguous keys and measured exact or
  approximate search, while SQLite remains the correctness reference.
- Separate payload shards, source archive, metadata, and indices behind storage
  interfaces that support local NVMe, external disk, and object storage.
- Publish logical records only after every required space view and manifest hash
  is complete. Support incremental shard resume and generation-level rollback.

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

- search opportunities, calls made, calls per 1,000 visible tokens, and trigger type;
- query-to-required-source recall, precision, rank, and cross-space overlap;
- selected records and bytes per site and per completed task;
- downstream task success and teacher NLL under real, zero, wrong, and dropped values;
- write opportunities, items written, useful-token/evidence coverage, duplication,
  contradiction, factual fidelity, and future-task utility;
- fraction of writes whose lineage includes earlier reads, recursion depth, and
  marginal utility by generation;
- bank records, logical source tokens, raw and compact bytes, index bytes, source
  archive bytes, temporary publication bytes, and generation overlap;
- end-to-end latency, search latency, payload I/O, model compute, and energy where
  available; distinguish warm cache, cold storage, exact search, and ANN;
- resident parameters and memory for the small controller plus services.

The capacity-substitution test compares small-controller-plus-bank systems against
larger resident models and text-retrieval controls with equal source access. It must
trace a quality–resident-memory–latency–storage frontier across bank sizes. More bank
bytes alone, lower teacher NLL, successful ingestion, or high GPU utilization does
not establish parameter substitution.

## 10. Immediate course decision

Let the current 100,000-source curriculum complete. Treat it as Phase 0 interface
and addressing pretraining, not the final trajectory program. In parallel, implement
Releases 1–3 and prepare a v0.5 dataset with multi-site tool traces. Start v0.5 as a
fresh 16-read-slot interface initialized from compatible backbone weights; do not
disguise the shape change as an ordinary resume.

The first v0.5 milestone is a stored-only long trajectory with at least eight visible
search calls, four write calls, two writes causally dependent on earlier latent
results, and a later heldout task that changes correctly when one authored memory is
removed under a fixed read plan. The first scale milestone is 1 million logical
records, followed by 10 million after search, ingestion, and recovery measurements
pass. Do not schedule the 100 million tier on the present external disk until source,
index, scratch, retained-generation, and free-space budgets prove that publication
and rollback both fit. Prefer the scale-out store before that tier.
