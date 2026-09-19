# Frozen-query read-count probe

The joint STOP continuation ended too many action reads prematurely. Test a simpler
count predictor while freezing the successful broad full-state address model:
a 64-to-2 linear classifier of its normalized causal address query. Labels are the
existing verified required-set cardinalities (one or two), never inference inputs.
No task-family field, answer, record ID, source text or future token enters the
classifier. The underlying key/query/reader/payload paths remain unchanged.

Train 200 updates of 1,280 sampled queries, seed 53, with native Muon for the
classifier matrix and AdamW for its bias. Its separate learning rate is 0.01;
there is no backbone/address update. Charge 130 new parameters. Evaluate count
confusion on the same development feature split, without presenting it as a
new downstream confirmation. It cannot predict zero or more than two records.

Keep all feature and small optimizer/sampling artifacts external, with sparse
final/emergency saves, identity-checked resume and offline W&B telemetry. If this
works, actual stored inference and the fixed-one/fixed-two controls remain required.

The normalized address-query head completed from `6840bb0`: correct counts for
9,918/10,240 training queries and 305/320 development queries. All 192 single-record
development queries are correct, but 15/128 two-record queries are under-read.
This is not yet a satisfactory policy. Compare the same classifier/budget using
the frozen reader query or the full causal state, each normalized to unit length.
The first two representations need 130 parameters; full-state input needs 2,050.
This representation comparison uses the development split; a fresh stored test
is still required after choosing a policy. No address or backbone weights change.

Both alternate representations completed from `cdd06fe`: 10,240/10,240 training
and 320/320 development count predictions are correct. Choose the reader-query
head for the next fresh stored confirmation because it matches the full-state
head's observed count accuracy with 130 rather than 2,050 parameters. This choice
is based on the development split, not an untouched final test. The earlier
address-query classifier had 305/320 correct counts. All address weights remain
identical to the broad full-state endpoint throughout this comparison.
