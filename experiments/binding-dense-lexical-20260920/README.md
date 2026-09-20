# Fixed dense lexical address control

## Protocol before execution

The preceding stored sparse TF-IDF comparator retrieves both annotated sources
for all 128 action questions in the known 4,096-record fixture, versus 35/128 for
the learned 64D router. Its 1.64 MB index stores source terms and has a different
byte budget. Test whether a fixed dense source/query lexical encoder can retain
that advantage using one normalized FP32 key per record and no runtime term index.

Use the identical frozen corpus, bank and 320 reference questions from the corrected
4,096-record evaluation. No new questions or model weights are selected from outcomes.
Tokenization is the preceding generic lowercase `[a-z0-9_]+` rule. Weight each token
by `1 + log(count)`; omit IDF so no bank-wide vocabulary table is required at query
time. SHAKE256 of `seed:token` supplies deterministic Rademacher signs; sum weighted
sign vectors and normalize. Reopen serialized keys/visibility metadata before
ranking. Opaque IDs affect only deterministic tie-breaking, never key encoding.
Search only namespace/space/generation/domain-matched records strictly before query
time. Source commands are inert. No target, required ID or source-kind-specific
feature enters the encoder. Report top-one and top-two results for every task.

Declare widths **64, 128, 256, 1024**, seeds **11, 23, 47**, and an unprojected sparse
TF baseline to distinguish projection loss from the preceding IDF change. Report
all arms without choosing a favorable seed. At width 64, key blobs occupy exactly
1,048,576 bytes, matching the learned 4,096 x 64 FP32 key blobs; wider arms have larger
budgets. Include actual serialized metadata/header bytes separately. The encoder is
fixed, uses literal source/query tokens and has different compute/information
processing from the learned backbone head. Equal key bytes do not establish equal
architecture, latency or robust semantic retrieval. No generation/capability or
parameter-substitution claim is made from this retrieval-only diagnostic.

The fixed bank is `/archive/probes/global-stored-fp32-scale-20260919/bank/bank.sqlite`;
corpus `/archive/probes/bank-scale-20260919/worlds-1024.jsonl`; reference
`/archive/probes/global-stored-fp32-scale-20260919/evaluation/pool-1024.json`.
Output is `/archive/probes/dense-lexical-routing-20260920`. Data/implementation hashes
are bound in the output inputs and each result. Both sufficient-group and full
annotated-source metrics are reported; these are different on STOP action questions.

```bash
python scripts/evaluate_dense_lexical_routing.py \
  --bank /archive/probes/global-stored-fp32-scale-20260919/bank/bank.sqlite \
  --corpus /archive/probes/bank-scale-20260919/worlds-1024.jsonl \
  --reference /archive/probes/global-stored-fp32-scale-20260919/evaluation/pool-1024.json \
  --output /archive/probes/dense-lexical-routing-20260920
```

Run on CPU with one BLAS thread. A cooperative SIGTERM/SIGINT or `sdkb runs stop --output` request
stops between queries; completed arms and dense indices are validated/reused on
restart. No model checkpoint or source bank is changed. All artifacts remain on the
external disk, with a 10 GiB reserve. Tests cover deterministic encoding, byte/shape
accounting, persisted boundaries/digest rejection, opaque IDs and strict visibility.
