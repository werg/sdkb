# Fresh stored confirmation after longer global routing optimization

Freeze the 3,200-update within-world and global-negative endpoints and the earlier
800-update global-negative endpoint before generating another 32-world corpus.
Compare all three on identical serialized payloads, frozen reader/decoder and the
same one/two-record count policy. This tests candidate-scope and optimization-budget
effects without treating prior confirmation worlds as fresh again.

Evaluate supplied-world and full-bank (128 records) routing with oracle/none/zero
controls. Generate without candidates for every question in all 32 worlds from the
outset, with the same 24-token cap. Retain all original reports and verify payload,
provenance and oracle/no-memory equality. Full-bank counterfactuals use captured
original selections for both choice and free-generation pairs.

Full-feature development recall motivated this confirmation: global-negative
training reaches 86/128 complete heldout action pairs after 3,200 updates, versus
45/128 after 800 and 3/128 for the long within-world control. These are not yet
stored task outcomes; the frozen native bank comparison is the required next test.
No new model copies or periodic checkpoints are created; artifacts stay external.

A targeted-source follow-up additionally changes only the specific required rule
record. The original type-wide flip can also affect wrong-world selections; targeted
interventions isolate whether the intended evidence controls the answer. They keep
original learned selections and every unrelated stored payload fixed and must leave
predictions exactly unchanged when the target was not retrieved.

## Completed fresh stored comparison

All 128 raw payload/provenance records and 640 oracle/no-memory score rows agree
across the three arms. Choice and candidate-free action results match over all
128 action questions; generation was specified for all worlds before evaluation.

| Router | Free actions /128 | Full required pairs /128 | Free permission /64 | Free restoration /64 |
|---|---:|---:|---:|---:|
| Long within-world | 54 | 5 | 40 | 47 |
| Short global | 76 | 37 | 46 | 57 |
| Long global | 101 | 71 | 47 | 61 |

All arms' action controls are 64/128 without memory, 63/128 with zero values and
128/128 with oracle supports. The long global versus short global action gain is
19.53 points [10.16, 29.69]; versus the long within-world control, 36.72 points
[22.66, 50]. Long-global memory advantage over no memory is 28.91 [20.31, 37.50].
These world-bootstrap intervals are diagnostic, uncorrected and from one training
seed. Exact unseen endpoint generation remains 0/64, including oracle retrieval.

Under **targeted source** changes, long-global both-correct permission pairs are
73/128 versus 37/128 short-global and 7/128 long-within-world. Restoration-change
pairs are 47/64 versus 33/64 and 11/64. Invariant restoration answers falsely change
10/64, 22/64 and 8/64 respectively. Unselected target changes leave original scores
and outputs exactly unchanged. The older whole-bank type-flip results remain
separate artifacts. Improved routing now supports substantial narrow global action
transfer, with unresolved retrieval, conditional-gating and exact-detail failures.

These models were fitted by the historical BF16 feature-score proxy; the actual
serialized-bank evaluations use FP32 exact search. The precision correction is
tracked separately and does not rewrite these measurements.
