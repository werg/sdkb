# Prior-corpus centering of fixed dense lexical keys

## Follow-up protocol declared before execution

The uncentered dense-TF control failed at 64D (2/0/0 full action pairs across three
seeds), and its larger-width results were seed-sensitive. This follow-up tests one
specified geometric adjustment: remove average source/query vectors before cosine
search. It was motivated by those results and is not a preregistered arm of the
original dense comparison. No dimension, seed or centering strength is tuned here.

Use the same widths 64/128/256/1024 and seeds 11/23/47, frozen 4,096-record keys,
and 320 evaluation questions. Fit two means from the original source model's
finetuning corpus only: one average of normalized hashed-TF vectors over unique
source records, and one over all causal query texts. Accumulate means in float64,
store in float32. Subtract the corresponding mean and normalize each source/query
vector. Do not fit on current bank/query outcomes, use answer/required-ID annotations,
or infer source types. Training source/episode IDs are disjoint from the evaluation
corpus; hashes/counts are in `inputs.json`. Every calibration support must precede
its query. Commands remain inert text.

Reopen centered keys, metadata and the two vectors before inference. Query-time
ranking uses only its causal text and the persisted query mean, with the same strict
visibility filter. No source encoder or term table runs at inference. At 64D the
key blobs remain 1 MiB and the two mean vectors add 512 bytes. Full serialized
metadata/header bytes are also reported. This is an explicitly different fixed
encoder/calibration procedure, not equal backbone compute, learned addressing or a
capacity-substitution claim. All 12 arms will be reported. No new generation is
planned from this retrieval-only test.

```bash
python experiments/binding-centered-lexical-20260920/run.py \
  --calibration /archive/runs/binding-muon-selected-20260919/data/train.jsonl \
  --corpus /archive/probes/bank-scale-20260919/worlds-1024.jsonl \
  --reference /archive/probes/global-stored-fp32-scale-20260919/evaluation/pool-1024.json \
  --dense-root /archive/probes/dense-lexical-routing-20260920 \
  --output /archive/probes/centered-lexical-routing-20260920
```

Use one CPU BLAS thread and external output with a 10 GiB reserve. SIGTERM/SIGINT
or a cooperative stop request preserves completed arm results; after clearing the
request deliberately, rerun the same frozen command to validate/reuse them. Partial
index directories have no valid manifest and are rebuilt before use. Tests exercise
answer-annotation independence, future-source rejection, persisted-only ranking,
visibility and corrupt-vector rejection. Neither active GPU training arm changes.

Pre-launch validation: 436 tests passed, with four existing warnings; Ruff passed.
