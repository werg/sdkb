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
