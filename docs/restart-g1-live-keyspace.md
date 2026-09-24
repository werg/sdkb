# Restart from g1-live: keyspace pretraining and Phase 2 merge

**Status:** proposed recipe, 23 September 2026. Nothing here is trained yet.
It supersedes the R3 continuation chain (`spatial-r2-g1` through
`spatial-r3-livekey-v11`) and the head-only
[keyspace distillation plan](keyspace-distillation-plan.md) as the next training
path. Those runs and their artifacts stay on the external disk unchanged.

## 1. What we learned

Each finding is measured, and the evidence is recorded in
`/archive/runs/four-space-target-curriculum-v2-20260921/keyspace-warmup-20260923`.

1. **The key slot carries almost no retrievable content in R3.** Writer key-slot states
   of the R3 checkpoint have mean cosine 0.9988 with their common direction.
   Linear 64-d, linear 256-d and 2-layer MLP heads trained over the full
   100k field all stall near 2% held-out recall@16 (about 12% at 256). The same
   linear head over mean-pooled source-token states and a pooled 64-token causal
   query window reaches 64% recall@16, 89% at 256, and median rank 5. The
   information exists in the backbone. The key slot never learned to collect it.
   On `g1-live` itself (22k-source sampled field, 1,213 held-out level-1 sites,
   the same linear probe), the key slot reaches 7.1% recall@16 (median rank
   519), and pooled token states reach 61.9% (median rank 6). The slot holds a
   little content, but far less than the backbone does. On the same field, the
   stopped R3 checkpoint's slot is worse (2.7% recall@16, median rank 1,862),
   while its pooled states are about the same (58.5%). The R3 chain degraded
   the key slot, which confirms `g1-live` as the rollback point.
2. **BF16 key heads destroy ranking.** Over those collinear states, a BF16
   query scores all 100,000 stored keys with about five distinct values. Every
   autocast recall metric in R3 and in early warmup evaluations was either
   tie-inflated or meaningless. Key heads must run in fp32, and evaluation must
   rank ties pessimistically.
3. **Routing never had to work.** `g1-live` and its parent used oracle-selected
   reads for their whole stages (`routing_warmup` equal to steps). Its key loss
   only separated each episode's few local candidates. Every spatial stage
   supplied the verified supports to the reader. Nothing made global addressing
   load-bearing.
4. **Stored keys drifted.** The mutable bank mixed revisions from many writer
   states (median cosine near 0 against base keys). Head changes without
   coherent refresh make the index compare incomparable vectors.
5. **Full-field objectives need their own scale and conditioning.** A cosine
   ×10 softmax over 100k records cannot concentrate probability. Minibatch Adam
   on raw collinear states scrambles rankings before the loss moves.
   Standardized fp32 heads with a learned logit scale do learn.
6. **Teacher signal is strong and diverse.** On held-out training sites, a
   lexical index already gets 98% any-support recall@16 but only 63%
   all-support. Local dense teachers reach up to 80% all-support. Best per
   teacher (source view : query view):
   - EmbeddingGemma-300m, `chunk_neighbor:content`: 0.80
   - e5-base-v2, `chunk_neighbor:arguments`: 0.75
   - granite-english-r2, `chunk:content`: 0.73
   - LFM2.5-Embedding, `chunk:arguments`: 0.68

   Causal prefix windows are poor query views, because the prefix mostly holds
   earlier episodes.

## 2. Principles for the new recipe

- Train addressing before relying on it, and measure it with an unassisted,
  fp32, tie-pessimistic gate on source-disjoint validation sites.
- The writer must be trained to fill the key slot. A key head cannot recover
  content that the slot state does not contain.
- Key training must not silently change payload semantics. Payload-preserving
  structure or distillation is part of the recipe.
- Every evaluation and every training search uses one coherent key generation.
- Reads become load-bearing gradually: supplied supports are annealed away, so
  a wrong retrieval costs task loss.

## 3. Interface changes (merged with Phase 2)

Phase 2 ([positional memory v0.7](positional-memory-v0.7.md)) already requires a
versioned migration, regenerated trajectories and rebuilt banks. The keyspace
changes ride the same migration, so we pay that cost once.

| Component | g1-live (Phase 1) | Restart target |
| --- | --- | --- |
| Writer value slots | 8 | **64** |
| Returned read slots | 8 | **64** |
| Payload layout | flat `[256,512,1024,2048]` (47% of 8 × 1,024) | **joint full-width tokens**: `[4,8,16,36]` tokens × 1,024 per space, 65,536 scalars = 64 × 1,024 (1:1) |
| Codec | one dense `Linear(8·1024 → d_s)` per space | per-space joint cross-attention: P_s learned queries over all 64 slot states |
| Reader | pooled MLP set reader | positional block-operator reader, per-space source-token counts |
| Key slot position | first write slot, before values | **last** write slot, after the 32 value slots |
| Key heads | shared `key_head` + per-space maps, BF16 | per-space direct heads with bias, **fp32** |
| Query heads | shared `query_head` + per-space maps | per-space direct query heads with bias, fp32; reader keeps its own `query_head` |
| Key width | 64 in every space | **256 in every space** (section 5); key width is independent of payload width |
| Training logit scale | fixed | learned per space for full-field objectives; ranking and gates use raw cosine |

**Joint token layout (owner decision, 23 September 2026).** The writer's 64
slot states (64 × 1,024) are reshuffled rather than compressed. Every space is
a joint mapping of all 64 slots into a few full-width tokens, with counts
ascending so the four spaces sum to the source size:

| Space | Stored tokens | Scalars | BF16 bytes | Records read (16/8/4/4) | Source tokens per read |
| ---: | ---: | ---: | ---: | ---: | ---: |
| s0 | 4 | 4,096 | 8 KB | 16 | 64 |
| s1 | 8 | 8,192 | 16 KB | 8 | 64 |
| s2 | 16 | 16,384 | 32 KB | 4 | 64 |
| s3 | 36 | 36,864 | 72 KB | 4 | 144 |

A record is 128 KB (12.8 GB per 100k records), and a query reads about
670 KB of payload. A joint dense map at this size would need billions of
parameters. Learned-query cross-attention is joint and content-adaptive at
about 2M parameters per space, and it is independent of the writer slot count.
This supersedes v0.7's per-position `[32,64,128,256]`-channel codec, which
stored about 47% of the slot states. A record read in only some spaces
contributes just those spaces' shares, by design.

Moving the key slot after the value slots does two things:
- Under the causal writer, value states no longer depend on the key slot, so
  training the key slot's embedding and heads cannot change payloads.
- The key slot can attend to the source and to every value slot it summarizes.

## 4. Stages

### R0 — Root and diagnostics
Root: `g1-live` step 39,504, the bank's exact writer with healthy oracle-read
payloads. Record its slot-versus-pooled diagnostic, fp32 score resolution and
payload-dependence baseline.

### R1 — Joint-token distillation at 8 slots

**Result (23 September 2026):** `/archive/runs/phase2-restart-20260923/r1-joint-8`,
2,000 steps. Token distillation loss fell from 0.72 to 0.07 and downstream loss
from 1.26 to 0.09. On 128 source-disjoint validation episodes with oracle
supports, answer NLL was 2.944 with correct payloads (teacher 2.949). Swapping
in another episode's supports added 0.417 (teacher 0.429), zeroing added 0.267
(teacher 0.329), and correct beat swapped in 74.2% of episodes (teacher 74.2%).
Gate passed.

From `g1-live`, keep 8 writer and read slots. Freeze the copied writer and
backbone, and train the new joint codecs and the per-space operator reader.
Payload layouts differ from the flat teacher, so distillation matches reader
states, returned tokens and fixed-plan downstream outputs rather than raw
payloads. Gate: payload removal and replacement change outputs at least as much
as the flat teacher's do.

### R2 — Expand to 64 writer and read slots

**Result (23 September 2026):** `/archive/runs/phase2-restart-20260923/r2-joint-64`,
1,000 slice-frozen steps. Old-function preservation held (payload loss 0.04,
old-slot token loss 0.004). On the same 128 validation episodes, answer NLL
was 3.021 with correct payloads, the swap gap was +0.317 (74% of the teacher's),
the zero gap was +0.192, and correct beat swapped in 68.0% of episodes. The
decoder now receives 56 read tokens that the reader has not learned to use.
R5 must recover this and then exceed it; R3 does not depend on it.

Use slice-frozen expansion for writer slots, reader target positions and
recurrent workspaces. The joint codec's learned queries do not depend on the
slot count. R2 changes nothing on the key path. Rebuild trajectory packing
with 64 read slots.

### R3 — Keyspace pretraining (the new core stage)
R3 owns every key-path change. Its initialization:
- moves the key slot to the last write position (`key_slot_position: last`);
- converts the shared 64-d key path into 256-d direct fp32 heads with bias
  (`convert_to_direct` with widths). The first 64 rows are folded exactly, the
  added rows start small, and distance-gate inputs are widened with zeros.

Moving the key slot changes what value slots attend to. R3's
payload-preservation loss against frozen R2 payloads absorbs that shift.

Train the writer to fill the key slot, and train the query side to match it.
Trainable parameters are the key-slot embedding, the direct writer and query
key heads, and the recurrent core at a low learning rate (owner decision,
section 5). The core also produces value states, so a payload-preservation loss
against frozen R2 payloads keeps value semantics.

Per-step losses:
1. **Teacher alignment (per source, no bank field).** Live writer keys are
   aligned with each space's frozen projected teacher vector: cosine plus an
   in-batch similarity-distribution KL. Four teachers, each with a distinct
   source and query view per space (section 1, item 6).
2. **Query alignment.** Live query keys at causal `memory.search` sites are
   aligned with the projected teacher query vectors in the same space.
3. **Support contrast over the full field.** Verified supports against all
   eligible records, scored with a key cache recomputed from the live writer
   every N steps (a coherent generation, not a mutable mix), with a learned
   logit scale.
4. **Lexical auxiliary** at small weight.

Data: Hotpot v2's 100k sources and 39.5k training episodes, plus the approved
public retrieval corpora in the local HF cache (MS MARCO, NQ-open, TriviaQA,
SQuAD, SearchQA) for query and source diversity. Validation stays Hotpot's
source-disjoint gate.

R3 result so far (r3-keyspace-b), on 500 level-1 validation sites against a
field of 12,130 sources, recall at 16/8/4/4:

| Step | s0 | s1 | s2 | s3 | median best rank |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.00 | 0.00 | 0.00 | 0.00 | about 4,000 |
| 500 | 0.26 | 0.19 | 0.13 | 0.14 | about 55 |
| 1000 | 0.56 | 0.44 | 0.33 | 0.33 | about 11 |
| 1500 | 0.71 | 0.61 | 0.52 | 0.52 | 4 |

At step 1500, recall@256 is about 0.95 in every space. The payload-preservation
loss stays at about 0.015.

Gate before R4, on the packed validation sites and a complete key refresh:
unassisted selected-support recall at 16/8/4/4 of at least 5% in every space
and at least 10% union (the plan's defaults). The stretch target is the pooled
probe's level (at least 50% recall@16 in s0).

### R4 — Coherent bank build
Build the 100k bank with the R3 writer: fp32 keys, positional BF16 payloads,
and one generation. Verify writer reproduction of stored keys, and report
reproduction noise across batch shapes.

### R5 — Spatial trajectory training with load-bearing reads
Train the spatial curriculum (2 loops, then 3) from R3/R4 with:
- supplied-support probability annealed from 1 to 0;
- unassisted reads always searched in fp32;
- writer replay with a nonzero key learning rate;
- a measured coherent-refresh cadence;
- the teacher-alignment loss kept at a low weight.

Evaluate the gate every 50 steps, plus stored-only transfer and payload
interventions.

R5 read limits follow the constant read-width rule (architecture section 3.2):
limit × tokens per record is the same in every space. With stored tokens
`[4, 8, 16, 36]`, a budget of C tokens per space gives limits of about
`[C/4, C/8, C/16, C/36]`. The R3 gate cutoff of 16/8/4/4 corresponds to C = 64,
with s3 rounded up from about 2. That budget is too small: the widest space
reads so few records that unassisted positives are rarely delivered, which
starved the earlier learned gates of any signal. C is chosen from the R3
read-count recall curve (any and every support within k records, per space;
logged from step 2500) together with the reader's measured cost per token. The
binding constraint is s3's recall at C/36. Candidates are C = 256 (limits
64/32/16/7) and C = 512 (limits 128/64/32/14).

## 5. Decisions

Owner decisions, 23 September 2026:

- **Writer adaptation in R3:** the recurrent core trains at a low learning rate
  together with the key-slot embedding and heads. A payload-preservation loss
  against frozen R2 payloads (serialized precision) keeps value semantics.
  Payload removal and replacement checks run at the R3 gate.
- **Public retrieval corpora:** approved, including their licenses, for R3 query
  and source diversity. The local cache has MS MARCO, NQ-open, TriviaQA, SQuAD
  and SearchQA. Each corpus keeps its own source namespace, so causal and
  authorization boundaries stay per corpus.
- **Order:** the Phase 2 migration (R1–R2) comes before R3.

- **Key width: 256 in every space.** Changing the width later means retraining
  the heads and rebuilding the bank, so the generous choice is made now. The
  source-disjoint gate still decides whether the wider keys generalize.

Key-width evidence, measured on 1,750 held-out Hotpot training sites, by
projecting a frozen teacher to width `d` and measuring all-support recall@16
(raw 768-d teacher in parentheses):

| Teacher | 32 | 64 | 128 | 256 | raw |
| --- | ---: | ---: | ---: | ---: | ---: |
| EmbeddingGemma `chunk_neighbor:content` | 0.44 | 0.61 | 0.72 | 0.75 | (0.80) |
| e5-base-v2 `chunk_neighbor:arguments` | 0.43 | 0.60 | 0.69 | 0.73 | (0.75) |

Going from 64 to 128 recovers about ten points; 128 to 256 adds about four.
Cost per record: an fp32 key is 256 bytes at 64 and 512 bytes at 128, against
2–16 KB of Phase 2 payload per space. Exact search over 100k keys in a space is
26 MB at 64 and 51 MB at 128. A backlog caution: on an earlier synthetic
fixture, wider addresses improved training fit but reduced held-out pair
retrieval, so the width choice is confirmed on the source-disjoint gate. At
256, fp32 keys take 4 KB per record across four spaces (against about 30 KB of
Phase 2 payload), and the resident exact index takes about 410 MB per 100k
records (about 4 GB per million).

## 6. Carried-over tooling

Built and tested in this session (not yet committed):
- the direct fp32 per-space key interface;
- state caching and teacher embedding scripts;
- the view benchmark;
- the unassisted gate;
- vectorized full-field losses, pessimistic ranks and standardized heads;
- the trainer's host-memory floor.

The R3 stage needs a new live-writer trainer. The cached head-only warmup is
kept as the frozen-state control.
