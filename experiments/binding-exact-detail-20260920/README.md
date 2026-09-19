# Exact-detail representation diagnostic

On the known 32-world long global-routing confirmation corpus, compare all 64
identifier questions through the same frozen decoder with: the oracle-selected
stored latent source, the identical source as text, and no memory. Generate freely
with the existing 24-token budget; no candidate strings enter generation. Assert
that stored/no-memory strings reproduce the existing completed reference exactly.

The text control has the same selected source information but a different input
representation and token budget, which is recorded. This is a diagnostic of the
exact-detail failure, not a new retrieval mechanism or a production source-text
fallback. Source commands remain inert text; all sources precede the query.
Inference writer and compactor calls are forbidden. Existing adapter/count/source,
episode and bank identities are checked before reading.

Run `scripts/evaluate_exact_detail.py`. Full artifacts live externally at
`/archive/probes/exact-detail-20260920`. No training or new model checkpoint is
needed. The source and adapters are the corrected-precision global stored study.

## Result

Exact identifier generation is **64/64 with selected text, 0/64 with the selected
latent payload, and 0/64 without evidence**. All 128 stored/no-memory prediction
strings reproduce the earlier reference exactly. Each latent source occupies
4,176 serialized payload bytes plus a 256-byte key; text prompt token ranges are
reported separately. This establishes a representation/training bottleneck in
this checkpoint, not an inherent impossibility of latent exact-detail recall.
It also shows that perfect retrieval alone would not fix the observed failure.

The earlier training-only diagnostic found memorized identifiers largely survive
zeroing payloads. The next experiment therefore varies fresh training breadth
while preserving all task families, with matching update budgets and explicit
held-out memory-removal and composition controls. Validation: 333 tests pass;
Ruff clean. Native evaluator source: `09c09c0`.
