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
