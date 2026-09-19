# Fresh-world confirmation of frozen compactors

Freeze the original 400-update MLP compactor and both 200-update continuations
before generating a new 32-world split. Compare raw payloads, mean-plus-mass,
the original fitted code (`trained`), continued statistics-only (`statistics`),
and continued statistics-plus-answer-loss (`task`) on the same stored banks.
No additional fitting or selection occurs on this split.

Every compactor endpoint must match the complete source checkpoint, frozen reader
fingerprint and one-code byte budget. Build new ordinary and counterfactual banks
offline in the output directory; never add new views to historical input banks.
Disable writer and every compactor before scoring or generation. Keep full-cluster
oracle selection, exact raw subset fallback, support removal, zero/no-memory,
permission/restoration counterfactuals and first-eight-world candidate-free output.

Use the serial decoder with exact-input reuse. A separate native BF16 check from
`f5c7f8c` matched all 384 score rows and 180 generation rows against uncached
`e69de72` execution. It reused 232 score calls and 104 generation calls while
still executing the authorized stored read each time. Candidate batching is not
used: its separate profile changed NLL values despite unchanged sampled choices.

All banks and artifacts remain external. Report choice accuracy, candidate-free
actions, counterfactual sensitivity and actual value/mass bytes separately. Raw
fallback is retained, so this does not demonstrate net disk savings, global
retrieval, agent success or parameter substitution.

## Completed confirmation

Source `a88183a`; candidate-free counterfactual generation `d7c0acf`. The new
32-world corpus is disjoint from original training and development worlds.

| MLP code | Action choices /128 | Free actions /32 | Permission-change pairs /128 | Restoration-change pairs /64 |
|---|---:|---:|---:|---:|
| Raw | 128 | 32 | 128 | 64 |
| Mean plus mass | 99 | 24 | 64 | 0 |
| Original fitted | 107 | 24 | 77 | 13 |
| Continued statistics | 116 | 27 | 96 | 37 |
| Continued statistics + answer loss | 116 | 27 | 96 | 37 |

The last two columns require both original and changed **choice** answers to be
correct. All arms have zero false changes on 64 restoration-invariant pairs.
Statistics continuation improves action choices over the original fit by 7.03
points, world-bootstrap interval [3.13, 10.94]. It remains 9.38 points below raw,
interval [3.91, 15.63] for the raw advantage. Answer loss adds no choice accuracy
here, despite reducing target NLL. These intervals describe worlds, not seeds.

A separate candidate-free counterfactual check on the first eight worlds gives
both continued arms 22/32 correct permission-change pairs and 8/16 correct
restoration-change pairs, versus raw 32/32 and 16/16. No arm falsely changes an
answer on the 16 invariant restoration pairs. Ordinary free-action controls are
16/32 for both zero and no memory. Compact-code fitting helps, but still loses
conditional behavior; low feature loss or ordinary accuracy alone is insufficient.

All 1,408 no-memory/zero/drop scoring rows agree exactly across the five arms;
payload-accounting fields intentionally reflect their different raw/code paths.
The historical input bank's SHA256 is unchanged. Full rows and banks remain on
the external drive; committed summaries include report hashes and paired grids.
