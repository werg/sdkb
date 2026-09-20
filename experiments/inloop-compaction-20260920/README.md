# Native recurrent temporary compaction

Single-read, single-space recurrent training now accepts interleaved mean or
synthetic temporary compaction. The complete selected group is aggregated before
its shared-state update. Code values pass through the configured storage precision;
multiplicities stay FP32. Partial persistent selections still use raw fallback.
Paired native objectives and multi-read compaction remain rejected.

Regression gates cover both MLP and attention readers, synthetic and mean codes,
chunk/backbone checkpointing, noisy/random-grouped replay, causal query invariance,
and stored-only code parity with the writer/compactor disabled. Emergency resume
with sampled recurrent depths, mixed stale/live cache, random grouping and partial
gradient accumulation reproduces uninterrupted training exactly. Full suite:
361 passed, four existing dependency/precision warnings; Ruff clean.

Actual LFM2.5 on the native Spark CUDA stack passes its model preflight. The attached
native diagnostic compares all shared producer/consumer gradients and persisted
code NLL from the original useful MLP source, with a newly initialized compactor.
With autocast weight caching, the maximum per-parameter relative L2 discrepancy is
0.002400; the uncompacted baseline has a similar 0.002368 discrepancy. With caching
disabled, compact full-graph/replay gradients differ by at most 3.73e-9 absolute /
7.70e-8 relative L2. Cached casts accumulate BF16 shared-weight contributions before
FP32 conversion; replay changes this accumulation layout. All RNG states match and
integrated/persisted compact NLL matches exactly. Default cache policy is unchanged.
The tape now restores the producer's captured cache policy even when its caller
changes policy before replay; both prior-failing CPU regressions pass exactly.

These are execution checks, not evidence of useful learned compression or storage
savings. Frozen source data, banks and native diagnostic reports live under
`/archive/probes/inloop-compaction-preflight-20260920`; `preflight.json` records hashes.

## Paired objective follow-up

Native single-read paired training is now implemented. Raw and compact losses share
the causal query/selection and noisy values; optional detached raw-teacher KL
encourages behavior preservation. Tests compare an independent two-path reference,
all shared replay gradients with oracle/learned routing, ablations and interrupted
partial-accumulation resume. Full suite: 398 passed; Ruff clean.

The paired native BF16 preflight reproduces RNG exactly and persisted compact NLL
with zero discrepancy. Maximum absolute shared-gradient discrepancy is 0.00036621,
consistent with the cached accumulation effect above. The actual model preflight
passes causal-prefix and one-loop identity checks exactly. These are numerical
checks; no paired learning outcome has been established.
