# Frozen routing under increasing cross-world competition

Hold the long-trained MLP, broad full-state routing adapter, one/two-record count
classifier and existing count-confirmation bank fixed. Rank stored keys against
nested 1, 4, 16 and all 32 worlds. Smaller pools contain the true world by supplied
membership; the full 128-record bank supplies no world filter. Other worlds are
ordered by a deterministic hash independent of answers, and pools remain nested.
This reuses a known diagnostic corpus; it is not a new training confirmation.

Keep exact scan/time/domain constraints, original-plan zero-value intervention,
oracle selected supports and no memory. Generate on the first eight worlds with
24 new tokens and no answer candidates. Check that the one-world scores and
oracle/no-memory controls reproduce the previous frozen evaluation. Report all
required-record recall separately from branch-sufficient recall, and selected
world accuracy separately from generated task answers. The tiny full bank is a
cross-world routing diagnostic, not large-scale ANN or cold-disk performance.

No model fitting, writer calls, compaction or checkpoint copies occur. Existing
external bank/report/source/adapter hashes bind the run. Completed pool reports
are atomic and reusable after a stop; current partial pools recompute on restart.
Results remain external; only summaries and comparison evidence enter Git.

## Completed diagnostic

Frozen source `3dd0894`; the one-world arm reproduces all 1,280 prior score rows
exactly on the retained answer/NLL/selection fields. All 640 oracle/no-memory rows
are identical across candidate pools. The original bank hash remains unchanged.

| Competing worlds | Candidate records | Action choices /128 | Both required records /128 | Free actions /32 |
|---|---:|---:|---:|---:|
| 1, supplied | 4 | 90 | 62 | 19 |
| 4, supplied | 16 | 70 | 19 | 15 |
| 16, supplied | 64 | 56 | 0 | 13 |
| 32, full bank | 128 | 50 | 0 | 16 |

Only one full-bank action query selects records exclusively from the correct
world. Free full-bank actions equal zero-value output at 16/32 (no memory 15/32),
while oracle actions remain 32/32. Full-bank permission/restoration free answers
are 6/16 and 10/16; oracle is 16/16 each. Novel identifiers remain 0/16.
This is a clear failure of global addressing at even this small bank size.

The previous routing objective only contrasted four within-world candidates.
Next test a source-bound continuation with cross-world negatives, paired with the
same additional updates/query exposures using within-world candidates. Preserve
payload/reader/decoder state and measure actual stored retrieval after feature
fitting; a feature-space improvement alone is not a capability result.
