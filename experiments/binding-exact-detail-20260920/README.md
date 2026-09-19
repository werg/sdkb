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
