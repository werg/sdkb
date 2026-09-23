# Keyspace distillation contingency

**Status:** design, not implemented. Use this if the corrected four-space R3
stage still lacks sustained source-disjoint recall at its actual 16/8/4/4
neighborhood budgets around step 300. The immediate gate is a packed
`memory.search` validation set with causal source eligibility. Training loss
and top-256 candidate recall alone do not pass the gate.

## Interface and conversion

The current four payload widths are 256, 512, 1,024 and 2,048 scalars. The
retrieval key in every space is 64-dimensional. A shared 64-dimensional writer
key is transformed by four `address_maps`; a shared query key is transformed by
four `query_maps`. This makes it easy for the spaces to share a narrow geometry.

The target key interface has one direct writer-state-to-key head and one direct
causal-query-state-to-key head per space. Each starts at 64 output dimensions;
key width is an independent scaling choice from payload width. Initialize each
direct head by composing the current shared head and its space map. Because
both old and new outputs are normalized after the map, this conversion should
preserve scores and rankings at initialization. An actual-model test must check
that equivalence, stored-precision serialization, gradients through replay,
and the bank refresh required after training the new heads. Existing stored
keys are not silently compatible with later changed heads.

## Teacher examples

Cache frozen teacher embeddings for source *content*, not record IDs or future
task answers. For document ingestion, each `memory.write` record gets the
source chunk it stores plus a bounded local heading/context. For an authored
experience, use only the content available at that write position, including
its causal read results and observed outcome. Large-blob trajectories with
several writes need a declared source-span/content assignment for each write;
the same whole blob is not copied as every write's teacher target.

At each `memory.search` site, embed the visible call arguments and the relevant
causal task context. Different call positions produce separate queries.
Teacher-forced targets and observations after the call are excluded. A frozen
teacher may encode all sources offline, but its training candidate field must
still obey each site's authorization, availability time and scope.

## Training sequence

1. Benchmark candidate teachers on source-disjoint, causally eligible query to
   source retrieval. Keep the source and query embedding cache on the external
   disk. Do not distill a teacher that cannot retrieve the verified evidence.
2. Start with one or two small frozen dense encoders plus the existing soft
   lexical signal. Train separate small per-space projections of teacher
   vectors against verified supports, then freeze those projections while
   distilling the student. Distinct models, source views and candidate mixtures
   provide diversity; four random projections of one identical target do not.
3. On a field of verified supports, sampled eligible records and hard records,
   distill each space's query-to-source score distribution. Keep the verified
   support union loss, downstream task loss, stored-key objective, live writer
   replay objective and bounded key refresh. The teacher does not enter the
   stored payload or inference path.
4. Freeze a writer snapshot, refresh all stored keys and payloads from their
   immutable source trajectories into an external staged journal, and validate
   actual stored-key retrieval before continuing bank training.

Do not force semantic labels onto the four spaces. Measure per-space rank,
top-k overlap, unique verified-support coverage, downstream removal effects,
and source-disjoint transfer. Beneficial redundancy is allowed; a low score
correlation by itself is not an objective.
