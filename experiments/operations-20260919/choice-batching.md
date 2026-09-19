# Candidate-scoring profile

From `1af20ce`, profile 40 stored queries (four worlds) on the frozen native BF16
MLP source. Each question warms both batch shapes before timing, and measurement
order alternates. Prefix/read planning is outside the measured interval. Other
GPU jobs remain active. This is a warm, contended decoder microbenchmark.

Batching up to four candidates reduces median measured scoring time from
185.75 ms to 80.28 ms. All 40 candidate choices agree, but maximum absolute full
sequence NLL deviation is 0.8553. This is not bit-exact equivalence, and the small
sample cannot establish universal ranking invariance. Existing research scoring
therefore remains serial. The helper is opt-in and experimental.

CPU regressions cover prefix and native recurrent memory, unequal candidate
lengths, serial agreement and future-padding independence. Batched scoring uses
the already captured memory; it never executes retrieval, writer or compactor.
No cold-storage or end-to-end throughput gain is claimed. Full rows and input
hashes are recorded in `choice-batching.json`; the loaded-model preflight and
training source remain unchanged.
