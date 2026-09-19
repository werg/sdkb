# Fresh-world breadth for latent exact-detail learning

The existing decoder copies 64/64 unseen identifiers from oracle-selected source
text but 0/64 from the same stored latent sources. Earlier training-only fits
largely survive zeroing the payload. Test whether broader fresh experience reduces
this reliance on memorized associations while retaining Boolean composition.

Warm-start both arms from the same 2,000-update recurrent-core MLP checkpoint.
Use 128 versus 1,024 fresh two-entity worlds from the same deterministic training
split, with the smaller corpus an exact prefix. Preserve all five questions per
entity and their original sampling proportions; no answer-dependent source budget
or changed loss. Both arms receive 1,600 native Muon updates, four microbatches per
update, seed 79, original learning rates, BF16 serialized payloads, oracle required
sources, and the same sampled loop depths. Optimizers start fresh in both forks.

Predeclare a new 32-world held-out split `detail-breadth-heldout-20260920` and
all-world candidate-free generation for source and both final endpoints. Include
zero-payload/no-memory controls, every family, and source-removal/counterfactual
composition scores. This is an oracle latent-learning experiment; previous routing
adapters are bound to the old writer and must not be silently reused.

All data, logs, offline W&B and checkpoints go to
`/archive/runs/binding-detail-breadth-20260920`. Periodic cadence is 10,000, beyond
this experiment's duration: initial, final and emergency checkpoints only. Main
trainer stop/resume preserves complete microbatch/replay state. Both arms keep the
10 GiB disk and 8 GiB host-memory reserves. No model architecture changes, downloads
or paid collection are required. Finish the declared budgets before inspecting
held-out results; do not select intermediate checkpoints by test performance.

Before any held-out evaluation, also fix an endpoint-only intervention: complement
each hex digit in the generated endpoint name, preserving source IDs, query/time,
permission/restoration rules and all non-identifier answers. Read the offline
alternate payloads under original oracle plans. Report both-correct identifier
pairs and false changes in unrelated answers. This is a generated-data diagnostic,
not a transformation of external/private source content.
