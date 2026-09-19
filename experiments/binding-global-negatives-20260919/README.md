# Matched continuation with cross-world training negatives

The full frozen bank retrieves zero complete action pairs, despite 62/128 under
supplied world membership. The previous address probe trained against only four
records from each query's own world. Test whether that objective misses necessary
cross-world discrimination.

Start both arms from the same completed broad full-state router. Use the existing
1,024-world training features (4,096 distinct source records) and 32 development
worlds, with immutable ID/feature alignment checks. Optimize the same four address
matrices for 800 additional Muon updates, batch 128, seed 67, source learning rate,
matched query samples, temperature 0.1 and unordered required-set likelihood.
Both arms project the same global key table. The within-world control masks all
but the query world's four records; the global arm keeps every time-eligible key.
No backbone, reader, payload code, count policy or target-conditioned query changes.

Evaluate each endpoint under both world and full-bank competition. These are
feature-space development results; subsequently build stored keys with the frozen
writer and test new-world downstream choices/free generation before claiming an
improvement in usable memory. Matrix multiplication can change floating-point
rounding relative to historical per-record dot sums, so this is a matched new
comparison rather than an exact continuation of the old arithmetic.

Initial/final/emergency small-state checkpoints remain external. Named ownership,
full optimizer/config/RNG/sampler recovery, atomic fsync, disk/memory reserves,
compute-only watchdog, W&B attempts and checkpoint-aligned JSONL apply. No periodic
full-model checkpoints or backbone copies are created.

## Completed feature comparison

Both arms completed 800 updates from `e6c13b9`. Initial development scores agree
exactly. Under full-bank competition, complete heldout action-pair recall is
45/128 after global-negative training versus 4/128 after within-world training
(initial 1/128). Paired gain is 32.03 points, world-bootstrap interval
[21.88, 42.19]. Under supplied world scope the two endpoints score 79/128 and
82/128 respectively; that difference is inconclusive. Global training-bank action
pairs reach 973/4096, so the hard objective is not solved even on training worlds.

The paired control attributes this feature improvement to candidate scope rather
than only additional updates. It does not establish useful stored inference.
Both endpoints are frozen for the separate fresh-world stored confirmation.

### Precision audit

Historical `e6c13b9` feature scoring used a matrix product inside BF16 autocast and
omitted the writer's final BF16 key normalization before FP32 storage. Its scores
were therefore a lower-precision training proxy for the actual stored FP32 cosine
search, beyond ordinary batch-shape rounding. The matched comparison and all real
stored evaluations remain recorded as executed. Current source restores the writer
normalization and explicitly computes cosine products in FP32 with highest float32
matmul precision; a BF16 regression first reproduced the mismatch. New fitting must
use a new identity/output rather than resume these old optimizer states silently.
