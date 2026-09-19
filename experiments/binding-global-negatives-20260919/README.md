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
