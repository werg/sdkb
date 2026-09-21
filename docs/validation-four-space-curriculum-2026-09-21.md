# Four-space curriculum: Spark validation, 2026-09-21

This record describes native DGX Spark work against the pinned LFM2.5-230M
backbone. It reports diagnostics, not successful knowledge-base transfer. All
corpora, banks, W&B offline files and checkpoints are under
`/mnt/external/sdkb-archive` on this machine (`/archive` in the container).
The repository contains configuration and code only.

## Frozen architecture and bank

- Middle-block consumer recurrence: two passes; writer: one pass.
- Four pooled-MLP spaces with BF16 value widths 256, 512, 1,024 and 2,048;
  64-dimensional FP32 keys. No compaction.
- Published 10,000-source SQuAD bank generation `3fea45d7e85b70967701e6a6`
  at `/archive/banks/four-space-squad-v2-10k`: 40,000 stored views, 157 complete
  atomic shards. Source manifest SHA-256:
  `d1a218d9ee16ed1a1fef4a8ce32c27a67c5a2440839b4b8a676b60dedf7f1e3c`.
- The bank writer is the final checkpoint of
  `/archive/runs/four-space-squad-pilot-v2-20260920`. The bank stage freezes
  that writer, codecs, address transforms and backbone; it trains the reader,
  bridge and query side against fetched stored payloads.

## Causal data split

The reconstruction stage used 1,024 sources and 2,048 episodes. Bank QA used
the next 1,024 eligible sources, one question each. The heldout title-located
set begins at source offset 3,072 and has 64 sources. The three source ID sets
have zero pairwise overlap. Query construction rejects a normalized answer
string in the article title or question. The QA training data SHA-256 is
`c9caa2532dc69101d369cae3d357731b61272e3d2e353e90c72c5cdf6a22294d`;
the heldout data SHA-256 is
`ce8b67948a5f670fcf604587350a93f42e3c6b72a981b09344f43ab912333f6f`.
Each source precedes its question. Stored-bank inference reads only payloads.

## Measured results

The reconstruction pilot completed 1,024 updates and one deterministic pass
over its 2,048 episodes. On 16 sampled source-disjoint validation questions,
its stored-only teacher NLL was 3.378 with correct memory, 3.391 with zero
values and 3.389 with wrong values. Free-generation exact match was 0/16.
The gate-only fork completed 2,048 updates; its correct-value NLL was 3.543
and zero-value NLL 3.538 on the same evaluation. The source-swap fork showed a
small early effect at step 256 (correct 3.328, zero 3.355), but at step 1,078
the effect reversed (correct 3.537, zero 3.535, wrong 3.517). That fork was
cooperatively stopped with a complete checkpoint.

The stored-bank stage completed 512 Muon updates, one pass over 1,024 new
questions, at `/archive/runs/four-space-bank10k-qa-novel-20260921`. The
evaluation uses 16 of the 64 heldout questions and exact learned selection
with per-space budgets [8, 4, 2, 1]. Correct-value NLL changed from 2.661
before this stage to 2.608 after it, but verified-source recall stayed 0/16.
At step 512, correct-value NLL was 2.608 and zero-value NLL was 2.610.
This is not evidence that the model used the correct source; the read plan did
not contain it.

On 32 heldout queries, median rank of the verified source among 10,000 eligible
sources was:

| Space | Writer checkpoint | Bank QA step 128 | Bank QA step 512 |
| --- | ---: | ---: | ---: |
| s0 | 5,077 | 4,927 | 3,780 |
| s1 | 4,891 | 5,453 | 4,105 |
| s2 | 4,830 | 4,687 | 5,303 |
| s3 | 5,518 | 6,091 | 5,142 |

The verified source never entered the top eight in any space in these 32
questions. Raising the frozen-bank routing-loss weight from 0.1 to 1.0 in a
separate 128-update fork also left median ranks around 4,100–5,700, with zero
top-eight hits. The next training test therefore makes both source keys and
query addressing trainable on causally eligible candidate sets before
publishing another bank generation.

## Runtime and integrity

The final 10,000-source key array contains 10.24 MB of FP32 keys. A warm-cache
probe measured roughly 0.05–0.06 seconds per space for exact SQLite key scans
and 0.0006–0.0007 seconds per space for the resident exact array. These figures
exclude cold NVMe timing and are not ANN measurements. Payload fetches remain
in SQLite and recheck visibility. Published bank readers verify shard bytes
under shared read transactions. The full test suite passed 553 tests after
the storage, corpus and local-routing preparation changes; Ruff was clean.

The first local-routing data attempt was excluded because distractors could
include later heldout sources. The corrected training set restricts every
distractor to the first 3,072 source rows. Its 1,024 positive sources and all
2,357 distinct training support IDs have zero overlap with the 64 heldout
positive source IDs. The corrected episode SHA-256 is
`9630774d00752f5019f5cfa4f56919650cc49f697efe722e9c28fbbbe98060a3`.
The live-key run is a new warm start from the Phase A checkpoint; it does not
resume either the contaminated data attempt or the frozen-bank stage.

## Live-key bank refresh

The corrected local-routing stage completed 512 Muon updates and exactly one
pass over 1,024 questions with four live source candidates each. Mean routing
loss over the first 128 updates was 1.428; over the last 128 it was 0.826.
Mean answer NLL over the same windows changed from 2.888 to 2.718. These are
training metrics, not proof of bank retrieval or value use.

Its frozen writer produced a separate complete 10,000-source bank at
`/archive/banks/four-space-squad-local-routing-10k`, generation
`0c2f794f83924025b204c123`. The 157 atomic shards contain 40,000 stored
views; the SQLite file occupies 109,105,152 bytes. No previous generation
was relabeled or overwritten.

On the same 32 heldout queries, verified-source median ranks under the new
writer/query pair are s0 739, s1 584, s2 1,746 and s3 1,178. The previous
writer/query pair ranked them at 5,077, 4,891, 4,830 and 5,518. On 16 heldout
questions, learned retrieval found a complete support in 1/16; correct-value
NLL was 2.660 versus 2.661 with zero values. With the verified source supplied,
correct-value NLL was 2.663 versus 2.662 with zero values. Thus addressing
improved markedly, while payload dependence remains unestablished. A separate
stored-only validation set from source-disjoint articles had correct-value
NLL 3.484, zero-value 3.480, wrong-value 3.486 and 0/16 free-generation
exact matches.

The next stage trains reader/query paths against this immutable generation
with a fixed-plan stored-value source-swap loss. Its 2,000 training source IDs
are disjoint from the 64 heldout source IDs. Swapped sources are eligible
comparisons, not verified insufficient records; heldout interventions remain
the decision criterion.

## Stored-bank value-contrast checkpoint

The value-contrast stage at
`/archive/runs/four-space-bank10k-value-contrast-20260921` uses a new set of
2,000 source-disjoint training questions, two microbatches per update, and
the immutable local-routing bank. It freezes the bank writer and trains the
reader, recurrent bridge and causal query paths. Each training plan includes
the verified source plus exact-search distractors, with per-space counts
[8, 4, 2, 1]. The wrong-value arm replaces only the verified stored payload
in each space, under a fixed plan. The checkpoint at step 260 was written
cooperatively, with full resume state, then training resumed.

At step 260, heldout exact retrieval still found complete support in 0/16
questions. Median verified-source ranks on 32 questions were 587, 472, 930
and 595 across the four spaces. With the verified source alone, mean
teacher NLL was 2.614 using its stored values and 2.621 with those values
zeroed. In a new matched-neighborhood evaluation that supplied the verified
source plus causal-query distractors in the same [8, 4, 2, 1] pattern as
training, stored-value NLL was 2.623 versus 2.619 with all values zeroed.
Thus the single-source gain does not survive the training-shaped read;
retrieval and value-dependent answer behavior remain unproven. The
matched-neighborhood evaluation captures one causal read plan and reuses it
for the zero-value intervention; it never re-encodes a source.

The new evaluator regression test passed, as did the full native test suite
(556 tests) and Ruff. The run remains on the external disk and continues to
its one-pass, 1,000-update endpoint for a controlled final comparison.

## Full bank pass and second-pass stop

The stored-bank value-contrast run completed one shuffled pass over all 2,000
training questions at step 1,000. On 64 heldout questions under the matched
neighborhood, mean teacher NLL was 2.8038 with correct payloads, 2.8071 with
all values zeroed, and 2.8058 when the verified source was replaced by an
eligible retrieved source in each space. Exact learned retrieval found a
complete support in 6/64 questions. The smaller first-16 subset showed a
larger 0.013–0.014 NLL effect; the full set establishes that it was optimistic.
Greedy generation on four of those questions had 0/4 exact matches, with
identical predictions under correct, zero and swapped payloads.

An additional 500 updates reached a full recovery checkpoint at step 1,500.
On the same 64 questions, learned complete-support recall rose to 9/64, but
the matched value effect reversed: correct 2.7791, zero 2.7737, swapped
2.7733 NLL. Greedy generation remained 0/8 exact, with no useful payload
intervention response. The bank-value stage is stopped at step 1,500; more
updates on this objective are not justified by these controls. Source-swap
comparisons are eligible negatives, not proof that the alternative source is
insufficient. These are teacher NLL and retrieval diagnostics, not agent
success or parameter substitution.

## Short-passage reconstruction preparation

The next writer/reader stage uses 6,000 training and 512 validation sources
with complete 28-word passages. Tokenizer lengths of source text span 35–64;
the model's source and target limits allow one extra EOS token. The prepared
corpus at `/archive/corpora/squad-short-reconstruction-20260921` has disjoint
source IDs and article titles across splits, and no target text in queries.
Its training and validation episode SHA-256 values are
`a2b53b2cbacef742392ff43eb5ed2c60b83cc503fead0ef205b6ee523ed729ca`
and `53b9fe3a2b12954a38763d806a792b8c77a02a0576dc2ad0fa896793ef06d027`.
The first launch stopped when a 64-token raw source became 65 tokens with
EOS; the corrected fresh run uses a 65-token model limit. That failure left
the source manifest untouched and did not create a trained checkpoint.

The corrected run at `/archive/runs/four-space-short-reconstruction-v2-20260921`
warm-starts the compatible four-space local-routing checkpoint and trains the
live writer and reader with oracle source delivery. Its step-0 stored-only
baseline on 16 article-disjoint validation passages had mean teacher NLL
3.6745 with correct BF16 payloads and 3.6808 with zeroed payloads, and 0/16
exact greedy reconstructions. At step 256, the same passage set scored 3.2913
and 3.3217 NLL, respectively; all 16 greedy outputs changed under the value
ablation, although exact reconstruction remained 0/16. These are early
information-flow signs, not a solved reconstruction task. A separate offline
writer phase creates the evaluation bank, after which the writer is disabled
and the stored payloads are reopened for inference. The next controlled
check was at step 1,000.

At step 1,000, a broader 64-passage stored-only evaluation scored 3.3246
teacher NLL with correct values and 3.4527 with zero values. This is a
payload-dependent teacher-forced signal under oracle delivery. Greedy exact
reconstruction remained 0/16; the correct-value outputs changed under
ablation but their mean generated-word overlap did not exceed the zero-value
control. The run is continuing through all 6,000 unique sources before the
next decision.

A following full-passage plus indexed-span stage has been prepared, not
launched, at `/archive/corpora/squad-short-span-curriculum-20260921`. It has
12,000 training episodes over the same 6,000 sources and 1,024 validation
episodes over 512 separate sources. Each source receives one full-passage
target and one exact eight-word span target; the span query contains word
positions rather than passage content. Its training data SHA-256 is
`c3c20e513ff70572c11bf242d8f21d131234a029e3a605de8f53f7d6d1e8981f`.

The whole-passage stage completed exactly one deterministic pass over all
6,000 sources at step 3,000. Mean training NLL declined from 3.401 in the
first 750 updates to 3.179 in the last 750. On all 512 source-disjoint
validation passages, the frozen writer's reopened stored payloads scored
3.1428 teacher NLL, versus 3.5138 with values zeroed. On 32 greedy
generations, exact match remained 0/32; correct-value outputs changed under
ablation, and mean generated-word overlap was 0.201 versus 0.136 for zero
values. These generated passages still contain substantial hallucinated
content. The result establishes useful payload-dependent token probabilities
under oracle delivery, but not accurate passage reconstruction.

The full-passage/indexed-span mixture started from the
step-3,000 writer as a new training stage at
`/archive/runs/four-space-short-span-20260921`. Its first stop is at 1,000
updates for a source-disjoint span and full-passage check; the bank writer
from the earlier 10,000-source QA experiments is a separate frozen
generation and is not silently relabeled with these new weights.

Before indexed-span training, 64 source-disjoint span requests scored 5.219
teacher NLL with correct stored values and 5.736 with zeroed values; greedy
exact span generation was 0/16. The mixed stage's step-1,000 checkpoint
scored 4.066 and 4.812 on the same requests, with 0/16 exact generations.
On the same 64 full-passage sources, correct-value NLL was 3.159 before the
mixed stage and 3.189 after it; the corresponding zero-value scores were
3.555 and 3.589. The mixed objective improved span token probabilities
while approximately preserving the earlier full-passage payload signal.
Neither task yet has convincing exact free generation. Training then
completed one deterministic pass over the 12,000 mixed episodes at step 6,000.

For the subsequent published-bank stage, a source-oriented manifest and
content-located span queries are prepared at
`/archive/corpora/squad-short-bank-queries-v2-20260921`. The manifest contains
6,512 distinct source versions: 6,000 training and 512 heldout sources.
Each query names its article and the first six passage words, then requests
an eight-word span starting at word seven or later. No target span appears
in its query, and source IDs are disjoint across the splits. Training query
SHA-256 is `3b63102844e8fbe6b4b1ce666aff89792678a5a199141bd3a297cc56f297cf2c`;
the standalone source-manifest SHA-256 is
`6c1f8e28a39e9c260197a2f8fce2f1d3314d8cbb17bc3dab27879e915bb10d58`.
Each source row retains its original source ID, article title, context hash,
short-corpus identity, and text hash in provenance bound to this bank manifest.
The bank was built only after the step-6,000 writer was frozen, so stored keys
and values are not silently mixed across writer versions. Validation source
text is in the bank without its question/answer labels, preserving
source-disjoint training labels.

## Target-curriculum execution upgrade

The sustained two-generation HotpotQA curriculum runs under
`/archive/runs/four-space-target-curriculum-v2-20260921`. Its scientific
budget remains two live and two stored-bank passes per generation over 39,504
episodes, with 100,000 source records per published bank generation. At step
32,419 of the first live stage it was cooperatively stopped and checkpointed
to replace serial execution with `padded-batch-v1`.

The upgrade executes two examples in one padded recurrent consumer graph,
batches all variable-length source writers for those examples, and projects
stored values only for records in the fixed oracle read plan while retaining
keys for every routing candidate. Source padding follows each source's write
slots; consumer padding follows each complete prompt/workspace/target row.
Per-row memory insertion positions and target losses preserve causal
boundaries. Backbone and MLP-reader activation checkpointing are disabled.
The effective batch remains two examples per optimizer update.

Training token IDs are materialized at
`/archive/corpora/hotpot-four-space-target-100k-v2-20260921/train.tokens.jsonl`.
Its 39,504 rows occupy 64 MB and are bound to episode SHA-256
`2d23a2c1d16efd88cffb3a49acadb1b423d35d4b6ba4dfe674f6baac519961d2`,
the pinned LFM revision, memory-only prompt contract, and configured token
limits. The raw source/query records remain authoritative.

The stop arrived after one serial microbatch of an unfinished update. The
upgrade restored model and Muon optimizer state from completed step 32,419,
discarded that uncommitted partial gradient, and replayed the corresponding
two-example position as a batch. This is an explicit learned-state resume,
not a bitwise continuation of the former execution schedule.

On the actual pinned HF model, steps 32,420 through 32,517 averaged 0.425
seconds per optimizer update while another process occupied most of the GPU,
compared with about 0.99 seconds per update for the earlier uncontended serial
path. Peak CUDA allocation was 2.39 GB after the upgrade versus 2.22 GB before
it. These are warm live-stage measurements and are not bank-build, stored-read,
or cold-NVMe throughput claims. The full suite passed 567 tests and Ruff was
clean before the actual-model restart; the actual BF16 run then exposed and
validated one codec-buffer dtype correction before any optimizer update.
An actual-model serial-versus-batched writer check on unequal source lengths
gave key cosine similarities of at least 0.9999993. Payload cosine similarities
were at least 0.9999995; the largest payload RMS difference was 0.00492 against
an RMS scale of 6.13. These are expected BF16 kernel-shape differences rather
than bitwise equality.
The subsequent 100,000-source bank build retains 64-source atomic recovery
shards and now executes each shard as four writer batches of 16; publication
still occurs only after all four space views in every shard are verified.

At step 6,000, all 512 source-disjoint indexed-span episodes scored 3.6575
teacher NLL with reopened correct BF16 payloads, 4.5099 with values zeroed,
and 6.6572 with no memory. On 32 greedy generations, exact match was 0/32;
mean generated-word overlap was 0.0865 with correct values and 0.1045 with
zero values. The NLL gain has not translated into useful exact span
generation. On all 512 full-passage episodes, the corresponding NLL values
were 3.0610, 3.5765, and 4.1853. Full-passage greedy exact match was also
0/32, with generated-word overlap 0.2033 versus 0.1511 for zero values.

The frozen bank at `/archive/banks/four-space-short-span-6512` published 102
verified shards, 6,512 logical sources, and 26,048 space-local records. Its
manifest SHA-256 is
`c45f17bc8e9ca2f9ed892990681e951418bb66fd2bec06bf2b09d14d07edb1a0`;
the writer model SHA-256 is
`73e42bd31dac87446a40f98ba202b2fc73493fa1eb4a79400e353ea114d96383`.
The SQLite file occupies about 71 MB on the external disk. This generation
is immutable and distinct from the earlier QA banks.

Before bank training, 128 heldout content-located span questions with a
supplied verified source plus exact-search distractors scored 3.8825 NLL
with correct stored values, 4.2199 with values zeroed, and 4.5753 when the
verified source was swapped out under the same plans. Greedy exact match
was 0/16. Unassisted global retrieval was poor: verified-source median
ranks across the four spaces were 3,119, 2,738, 3,705, and 3,115.5 among
6,512 eligible records. The stored-value signal supports a corpus-backed
training pilot, but neither retrieval nor exact generation is established.

The first stored-bank pilot completed one shuffled pass over all 6,000
training questions. Supplied-source heldout NLL improved after its first
1,000 updates from 3.8825 to 3.4647, while verified-source top-8 recall
remained zero in every space. Completing the pass did not fix retrieval;
heldout median ranks remained between 2,200 and 3,130. A controlled fork
raising routing weight from 0.1 to 1.0 also produced zero top-8 recall after
1,000 updates. Reader/query training against fixed poor keys is therefore not
sufficient for global addressing on this task.

A subsequent live-key stage trained the writer and query addressing paths
against each verified source plus three distinct-article training-source
candidates. The bank was then re-encoded as the separate immutable generation
`4042c6f8f9aef9b01eba627b` at
`/archive/banks/four-space-short-local-routing-6512`; its manifest SHA-256 is
`c048cf46979fbe7256bcfea28a2aabd90293af2ca536a8f24f326d666db03322`.
On 128 heldout sources, median ranks improved to 197.5, 158, 158.5, and 183.5
in s0 through s3. Exact learned reads at limits 8/4/2/1 included the verified
source for 13/128 questions. Their mean NLL was 3.8913 with stored values and
3.9718 with values zeroed; greedy exact match remained 0/16. This is useful
retrieval progress, not accurate span generation.

## Sustained target curriculum

The target run now uses the pinned HotpotQA distractor corpus rather than another
small routing fork. Raw Parquet files and the CC-BY-SA-4.0 dataset card are retained
under the external archive at dataset revision
`1908d6afbbead072334abe2965f91bd2709910ab`. The published v2 preparation at
`/archive/corpora/hotpot-four-space-target-100k-v2-20260921` contains 100,000
unique source chunks, 29,628 verified multi-hop transfer episodes, 9,876
reconstruction episodes, and 1,002 validation questions. Reconstruction is
exactly 25 percent of the 39,504-example training mixture. Its source, training,
and validation SHA-256 values are respectively
`05e66177416486db868d3d0b69641e55c7b4a8f11858183c202c8bd8b491b8c0`,
`2d23a2c1d16efd88cffb3a49acadb1b423d35d4b6ba4dfe674f6baac519961d2`,
and `fccbcee9b3010f04846adfe7bda4d010ddbbdea6368d5d468a8dec71ea7e1aa3`.

An audit rejected the first preparation before training because reconstruction
reused 244 validation source IDs. V2 has zero overlap between all 58,122 training
required-source IDs and all 2,349 validation required-source IDs. Every source
precedes its query; all required IDs exist in the bank manifest; targets are absent
from query prompts; and required support counts range from one to four.

The run at `/archive/runs/four-space-target-curriculum-v2-20260921`
performs two full live-writer passes, publishes and trains against an immutable
100,000-source bank for two passes, evaluates stored-only heldout transfer, then
repeats a live writer refresh, bank publication, stored training, and evaluation.
Each generation uses 39,504 live updates and 39,504 bank updates. It originally
used two serial accumulated examples, then moved at step 32,419 to a real batch of
two with one accumulation step; the examples-per-update budget is unchanged.
Stored training ranks 256 exact global candidates per space while fetching
only 16/8/4/4 payloads. It uses Muon, BF16, offline W&B, external artifacts,
10,000-update periodic checkpoints, and emergency full-state recovery. Bank
generations are never embedded into optimizer checkpoints.

This curriculum is Phase 0 interface and addressing pretraining for the planned
[trajectory memory v0.5](trajectory-memory-v0.5.md). It does not contain visible
memory tool calls, frequent sites across a long task, prompted writes, or recursive
memories authored by read-augmented trajectories.

The owner subsequently redirected the unstarted stored-bank stages to the v0.5
spatial path. The completed `g1-live` weights and its in-progress 100,000-source bank
are retained. Two externally stored immutable layouts now contain 9,876 trajectories,
39,504 visible search calls, 8,337,388 tokens, and 3,246,518 supervised tokens each:
one activates four sites at recurrent level one, while the other alternates sites
across levels one and two. This is a compatibility bridge with eight read slots; it
does not yet satisfy the later eight-search/four-write milestone.
