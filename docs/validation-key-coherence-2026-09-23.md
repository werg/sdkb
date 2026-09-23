# Full-bank key coherence: 23 September 2026

## Observation and decision

The 100,000-source, four-space R3 run remained near random unassisted support
recall after more than 4,800 combined optimizer steps. The first address
curriculum fork continued for 497 steps with sampled negatives, a soft lexical
teacher, and stronger routing loss; sampled top-256 recall remained about
0.1–0.4%, around the 0.256% random expectation. Supplied supports kept task NLL
usable, so that loss did not demonstrate learned retrieval.

The bank journal at the stopped fork's step-497 checkpoint had complete mutable
views for 58,757 logical records. The other 41,243 still used the original
physical bank. In a deterministic sample of 1,000 revised records per space,
the median cosine between each current head and its base key was:

| Space | Median head/base cosine |
| --- | ---: |
| s0 | 0.004 |
| s1 | -0.003 |
| s2 | 0.027 |
| s3 | -0.013 |

A separate 2,000-record s0 sample comparing the first and latest journal
revisions had median cosine 0.070, across a median gap of 944 journal commits.
This is evidence of substantial key drift, not proof of its cause. The exact
search index was comparing vectors created by different writer states, which is
a strong explanation for poor global query alignment. The old run was stopped
cooperatively at step 497; its checkpoint and bank-state token match at cursor
1,687. No additional SDKB GPU job ran beside it.

## Corrective path

1. Regenerate every stored key and payload from that stopped writer checkpoint
   into a separate, resumable external journal. Publish it only after all four
   views of all 100,000 sources are refreshed.
2. Measure source-disjoint support ranks against the coherent journal, with no
   writer calls during evaluation.
3. Warm-start the full trajectory curriculum from the matching model and journal.
   Keep writer-key parameters trainable at a smaller nonzero learning rate; retain
   supervised supports, sampled and mined negatives, and soft lexical guidance.
4. Measure drift and unassisted recall at the next checkpoint. Decide the next
   whole-bank refresh interval from those measurements. If routing remains weak
   with a coherent bank, prepare a small embedding teacher with a distinct
   projection per space.

## Full-refresh result and next correction

The staged rewrite completed all 100,000 sources at journal cursor 3,250.
An exact stored-key evaluation on 128 source-disjoint validation queries used
the matching writer checkpoint, the complete mutable journal, and no source
encoding. Median support ranks were 72,192, 71,415, 70,253, and 73,422 in
spaces s0–s3, with zero recall at 128 in every space. Coherence alone therefore
did not produce a working router.

The training path also exposed a second mismatch: when a verified positive was
replayed, the routing loss scored its newly encoded live key, while the bank
search ranked the older stored key. The revised path uses stored keys for
contrastive routing and retains task gradients through live keys and gates.
A separate replayed-key stability penalty and a smaller nonzero writer-key
learning rate are prepared for the next full trajectory stage. This is a
corrective training hypothesis, not yet a measured retrieval improvement.

## Coherent-bank routing plateau and next stage

The coherent-bank stage reached step 210 with stable roughly 28–30 second
optimizer steps and 400,000 active space views. Its sampled easy routing loss
remained around 2.35 and top-256 candidate recall stayed near chance or zero
in most logged batches. The stored-key objective used raw cosine values as
softmax logits. Their bounded range gives a shallow probability distribution
over 32 sampled negatives or hundreds of mined candidates. The next stage
scales those logits by 10 for the supervised routing and soft lexical losses.
Cosine ranking and the reader's density-adaptive gates still use unscaled
scores. This is a testable training correction, not evidence of improved recall.

The earlier exact-rank probe used a short query prompt rather than the packed
`memory.search` trajectory used by this stage. Its poor ranks are a warning,
but the packed training-site unassisted recall is the primary current signal.

## Writer-address supervision gap

The sharpened stage reduced its sampled easy loss from about 2.5 to below 2
and briefly lifted top-256 recall into the 2–10% range, but recall fell back
near chance by step 260; top-16/8/4/4 selected support recall remained zero.
The hard-negative loss declined as its weight ramped, without sustained
retrieval improvement yet.

The stored-key routing loss has a required limitation: the bank key is a
serialized, detached revision, so that loss reaches the query projection but
does not directly train the writer's key projection. The next stage adds an
auxiliary contrastive loss using replayed live keys for its verified support
records and stored keys for the other candidates. Its main routing loss still
uses stored keys, and a key-stability term keeps live and stored geometry close.
This supplies direct writer-address gradients without pretending the live key
was used by the actual search. We will measure unassisted packed-trajectory
recall before treating the change as successful.

## External-journal maintenance stall

The live-key stage showed no selected-neighborhood support recall through
step 110. It then stopped producing its usual ten-step logs for more than
25 minutes while the external drive showed high read latency and sustained
I/O pressure. The process remained in kernel page waits and its physical read
counter advanced by several gigabytes. Investigation found that choosing one
maintenance source each optimizer step executed `GROUP BY record_id` across
the complete mutable journal head table, then sorted all source IDs by age.
That work was disproportionate to one maintenance refresh and amplified disk
contention. Maintenance now uses the checkpointed rotating position over the
resident sorted ID array, skipping excluded/ineligible records. It preserves
fair eventual coverage without a per-step full journal scan. A regression test
covers the new rotation and exclusions. The next stage must verify that wall
time and physical read traffic improve on Spark; this is not a keyspace result.
