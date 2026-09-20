# Source-derived lexical retrieval control

The learned address model loses required records as the synthetic bank grows.
Test an ordinary lexical control on the exact same 4,096-record bank, 320 questions,
causal times and authorization scope as the corrected large-bank diagnostic.
This is a known-corpus, retrieval-only comparison; no model is trained or generated.

Offline, derive lowercased `[a-z0-9_]+` token counts from each source. Store them
with source hash, opaque ID and the bank's namespace/space/generation/domain/time
metadata in an external index, then reopen that index for ranking. A query contributes
only its visible text. Operational IDs, task labels, answer choices and required-source
annotations are not ranking features. Source commands remain inert tokenized data.

Use TF-IDF cosine with TF `1 + log(count)` and IDF
`1 + log((1 + eligible_documents) / (1 + document_frequency))`. Compute statistics
only over eligible records, so future or unauthorized documents cannot influence
scores through IDF. Report top-one and top-two separately. Ties use opaque IDs only
for deterministic ordering. Tests cover invisible-document effects, time boundaries,
source-version/time mismatches, empty overlap, opaque-ID renaming and metric semantics.

| Family | Queries | Lexical top-one all annotated | Lexical top-two all annotated | Learned all annotated |
|---|---:|---:|---:|---:|
| Action | 128 | 0 | 128 | 35 |
| Identifier | 64 | 0 | 64 | 14 |
| Permission | 64 | 0 | 64 | 18 |
| Restoration | 64 | 64 | 64 | 56 |

The learned policy takes two records for every action query and one for each other
family. Thus the action top-two count budget matches; other top-two rows get an
extra record. The lexical top-one ranking always favors the restoration source in
this fixture and is not a successful general one-record selector. Exact-ID spelling
in both source and query makes this an easy lexical fixture, not semantic retrieval
across paraphrases or unfamiliar real tasks.

The serialized lexical index is **1,643,978 bytes**, including feature and provenance
metadata. The bank's stored learned key blobs total 1,048,576 bytes, excluding their
surrounding metadata/index; latent payload blobs total 17,104,896 bytes. These are
unequal representations and byte accounting scopes. The lexical comparator retains
source terms and is not the architecture's learned 64-dimensional key. No latent
inference, generation, capacity substitution, ANN or latency result is asserted.

## Retrieval metric label correction

`complete_support` in the historical evaluator means *at least one sufficient group*.
A STOP branch may need only the permission record, whereas `all_required` means both
annotated action records were selected. The raw records preserve both fields, but
the precision-study prose called 73/128 and 37/128 “both-required-source” retrieval.
Their actual full-pair counts are **71/128 and 35/128**. Within-world training's
10/128 sufficient-group count corresponds to 5/128 full pairs. The original values
remain valid under their actual metric; raw results and generation scores are unchanged.
`retrieval-label-audit.json` pins the captured rows across the related studies.

The new control explicitly reports both metrics. For top-two action retrieval,
lexical sufficient-group/full-pair counts are 128/128 each; the learned router has
37/128 sufficient groups and 35/128 full pairs. The full suite passes 416 tests;
Ruff and the native-environment CPU diagnostic pass. The first uncommitted prototype
report used the stricter metric under an ambiguous label; the versioned final output
below corrects the labels without changing ranking.

```bash
python scripts/evaluate_lexical_routing.py \
  --bank /archive/probes/global-stored-fp32-scale-20260919/bank/bank.sqlite \
  --corpus /archive/probes/bank-scale-20260919/worlds-1024.jsonl \
  --reference /archive/probes/global-stored-fp32-scale-20260919/evaluation/pool-1024.json \
  --output /archive/probes/lexical-routing-v2-20260920
```

The complete index, query rows and source data remain external. `summary.json` pins
hashes and aggregates. A different script/source/bank requires a fresh output path.
