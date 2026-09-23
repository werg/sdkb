# Keyspace distillation contingency

**Status:** design, not implemented. Use this if the corrected four-space R3
stage fails the gate below. The warmup changes the key interface and adds
teacher distillation in one stage. It trains only key heads over cached writer
and query states, then returns to the full trajectory curriculum with a
coherent refreshed bank.

## Gate

The gate set is a fixed, packed `memory.search` validation set of at least 512
sites. Sources are disjoint from training sources, and each site's candidates
obey causal source eligibility (authorization, availability time and scope).
The set is evaluated every 50 optimizer steps with exact stored-key search and
no writer calls.

The primary metric is selected-support recall: the fraction of sites where at
least one verified support lands in that space's selected neighborhood at the
actual 16/8/4/4 budgets. Random expectation is about 16/100,000 in s0. Also
log median support rank and top-256 recall per space.

Initial trigger: run the warmup if, at both the step-250 and step-300
evaluations, every space has selected-support recall below 5%. The union over
spaces must also be below 10%. These thresholds are defaults and should be
recorded with any change. Training loss, sampled-batch recall and top-256
candidate recall alone neither pass nor trigger the gate.

The same gate and thresholds judge the warmup's result. Evaluate them first on
the refreshed bank at the end of the warmup, then again at step 300 of the
continued trajectory stage.

## Interface and conversion

The current four payload widths are 256, 512, 1,024 and 2,048 scalars. Each
space's retrieval key is 64-dimensional. A shared writer `key_head` feeds four
bias-free `address_maps`, and a shared routing query head feeds four bias-free
`query_maps`. Every space's key therefore lies in the image of the same
64-dimensional writer projection.

The target interface has one direct writer-state-to-key head `W_s` and one
direct query-state-to-key head `Q_s` per space. Initialize `W_s = A_s · W_key`
and `Q_s = M_s · W_route`. `W_route` is `routing_query_head` when present;
otherwise it is a *copy* of `query_head`. The reader's own `query_head` stays
in place, because it still conditions the reader.

Today a stored key is `normalize(A_s · normalize(W_key h))`. The intermediate
normalization only rescales by a positive number, and `A_s` has no bias, so
the outer normalization cancels it. The key therefore equals
`normalize(W_s h)`. Query scores follow the same argument because
`cosine_similarities` normalizes both sides. In fp32 the scores are identical.
In BF16 the folded weights round differently, so the test tolerance should
allow small score differences and a small rank-change rate, not exact
rankings.

The warmup keeps every key width at 64 so the equivalence test holds. Per-space
key width becomes configurable, and a later width change requires:

- per-space key widths in the structural checkpoint check (`training.py`),
  `AdaptiveDistanceGate` construction, and the index width check in
  `training_bank.py`;
- updated parameter-prefix lists in `optimizers.py`, `training.py`,
  `offline_bank.py` and `probes.py`;
- an explicit checkpoint conversion path; missing head weights are never loaded
  silently.

An actual-model test must check score equivalence at initialization,
stored-precision serialization, gradients to `W_s` and `Q_s`, and the full bank
refresh after the warmup. Keys produced by earlier heads are not compatible
with trained direct heads.

## Cached states

The warmup freezes the backbone, `write_slots`, value head, codecs, reader and
distance gates. Only `W_s` and `Q_s` train. Writer keys are then exact
functions of cached states:

- for a document source, the writer key-slot state `tail[:, 0]` from the
  checkpoint's actual writer forward (`produce_batch`);
- for a trajectory write, the causal key-slot state that
  `produce_from_write_states` receives;
- for each `memory.search` site, the routing features at that site's packed
  causal prefix.

The cache stores states as the actual autocast forward produced them, at the
serialized precision, under that checkpoint's RNG. It records the checkpoint
and writer-state token. At BF16, the cache costs `2 × width` bytes per state,
roughly 0.2 GB per 100,000 states at width 1,024. It lives on the external
disk. The cache is invalid for any other checkpoint. This does not truncate
replay (invariant 3): with everything upstream frozen, `normalize(W_s h)` over
the cached `h` is the exact live key.

Because keys are one matmul away, every training step computes current keys for
the whole eligible field. Search and loss use the same fresh keys, so the
warmup has no stale-key drift and needs no bounded refresh. The downstream
task loss and live writer replay are not part of the warmup, because they
require the frozen components. They resume in the continuation.

## Teacher examples

Frozen teacher embeddings encode source *content*, never record IDs or future
task answers. For document ingestion, each `memory.write` record gets the source
chunk it stores plus a bounded local heading/context. For an authored
experience, the teacher sees only the content available at that write position,
including its causal read results and observed outcome. A large-blob trajectory
with several writes needs a declared source-span assignment for each write. The
same whole blob is never copied as every write's teacher target.

At each `memory.search` site, the teacher embeds the visible call arguments plus
a context window chosen by a fixed rule, not by relevance judged after seeing
the task. The default rule is the last 512 prefix tokens before the call. The
span is recorded per site. Different call positions produce separate queries.
Teacher-forced targets and observations after the call are excluded. A frozen
teacher may encode all sources offline, but each site's candidate field obeys
its authorization, availability time and scope.

## Diversity

Each space gets a different frozen local dense encoder, preferring different
model families, training data or tokenizers. Each space also gets a different
declared source view and query view, for example:

| Space | Source view | Query view |
| --- | --- | --- |
| s0 | chunk only | call arguments only |
| s1 | chunk + heading path | arguments + fixed prefix window |
| s2 | chunk + neighbouring chunk | arguments + window + prior tool results |
| s3 | heading path + chunk summary span | fixed prefix window only |

The space-to-teacher and view assignment is configuration, not a semantic label.
Random projections of one identical target are not a source of diversity. The
existing soft lexical signal remains a shared auxiliary loss, not one space's
teacher. Teachers run locally with frozen weights. No paid teacher endpoint is
used.

Each space's teacher projection (teacher dimension to 64) trains on the
**training split only**, contrastively against verified supports, and is then
frozen. A projection never sees a gate or validation source.

## Warmup sequence

1. **Benchmark teachers.** On a held-out portion of the training split,
   measure each candidate encoder and view's source-disjoint, causally eligible
   recall@16 and recall@256 against verified supports. The lexical signal on
   the same field is the baseline. A teacher must beat lexical at both cutoffs
   to be used. If fewer than four teacher/view pairs qualify, the leftover
   spaces reuse the strongest qualified encoder with a different view pair.
   Teacher and projection compute are reported. Source and query embedding
   caches live on the external disk.
2. **Cache and convert.** Build the state caches from the gate-failing
   checkpoint. Fold the shared heads into direct heads and run the equivalence
   test on the actual model.
3. **Distill.** For each site and space, the field is all causally eligible
   records. The loss combines:
   - KL(teacher ‖ student) over the eligible field. Student logits are cosine
     scaled by 10, matching the routing losses. Each teacher's temperature is
     tuned on the held-out training portion so its distribution ranks verified
     supports best, then fixed.
   - The verified-support union contrastive loss over the same field.
   - The soft lexical auxiliary.
4. **Control.** Rerun the same warmup with teacher weight zero and all else
   identical. The control uses the same heads, cache, field, supports and
   steps.
5. **Refresh and validate.** Compute every stored key with the trained heads
   from the cached states into a new external staged journal. Payloads are
   unchanged because their producers stayed frozen. Publish the journal only
   when all four views of all sources are present. Evaluate the gate on it,
   using the warmup and the control's refreshed banks.
6. **Continue.** Warm-start the full trajectory curriculum from the refreshed
   model and journal. It restores the downstream task loss, stored-key
   objective, live writer replay, key-stability term and bounded key refresh.
   Writer parameters train at the existing small nonzero learning rate.

## Measurements

Do not force semantic labels onto the four spaces. For warmup and control,
report:

- per-space rank and gate recall;
- top-k overlap between spaces;
- unique verified-support coverage per space;
- downstream effects of removing each space;
- source-disjoint transfer.

Report whether one-stage gains came from the interface or from the teacher only
through the teacher-weight-zero control. The interface change alone is
uncontrolled in this design. Beneficial redundancy is allowed. A low score
correlation by itself is not an objective.
