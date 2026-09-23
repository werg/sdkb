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

The staged refresh is underway. No improved retrieval result is claimed here.
