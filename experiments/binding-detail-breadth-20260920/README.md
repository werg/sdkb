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

## Completed fresh confirmation

Both arms completed 1,600 updates and exactly the same sampled decoder-depth
schedule. Final checkpoints and actual exposure counts are retained separately.
All three frozen models (source, narrow continuation and broad continuation)
generate **128/128 oracle-selected actions, 64/64 permission facts and 64/64
restoration facts**, including their appropriate rule changes. Changing endpoint
strings causes zero spurious action/fact changes.

All three still generate **0/64 exact identifiers**, both originally and after
endpoint-only changes. Identifier target-token NLL improves from 4.906 at source
to 3.989 (narrow) and 2.799 (broad); this likelihood gain does not establish exact
generation. The source/narrow/broad models change their incorrect
identifier predictions on 5/64, 23/64 and 47/64 endpoint changes, respectively;
these changes are not successful recall. Unrelated permission changes also alter
49/64, 42/64 and 45/64 identifier predictions. Broader exposure increases sensitivity
to endpoint content without establishing exact copying or field independence.

Narrow and broad identifier supervision covers 253 versus 941 distinct questions
in 1,244 versus 1,285 sampled examples. Every held-out endpoint is absent from the
source and both continuation training corpora. The equal-update comparison does
not equate repetitions per endpoint. Results remain from one training seed and a
synthetic oracle-selection setting; no learned-global retrieval benefit is claimed.

Training source `602cb06`; evaluator `3ebc2e5`; all three full reports and banks
remain external. Initial/final checkpoints were the only saves. After completion,
verified sharing of the identical initial weight files reclaimed 1.90 GiB without
removing recovery state. The next hypothesis is to present identical training
queries with multiple explicitly versioned endpoint memories, making query-only
memorization insufficient at the same update budget.
