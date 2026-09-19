# Fresh-world confirmation of frozen compactors

Freeze the original 400-update MLP compactor and both 200-update continuations
before generating a new 32-world split. Compare raw payloads, mean-plus-mass,
the original fitted code (`trained`), continued statistics-only (`statistics`),
and continued statistics-plus-answer-loss (`task`) on the same stored banks.
No additional fitting or selection occurs on this split.

Every compactor endpoint must match the complete source checkpoint, frozen reader
fingerprint and one-code byte budget. Build new ordinary and counterfactual banks
offline in the output directory; never add new views to historical input banks.
Disable writer and every compactor before scoring or generation. Keep full-cluster
oracle selection, exact raw subset fallback, support removal, zero/no-memory,
permission/restoration counterfactuals and first-eight-world candidate-free output.

Use the serial decoder with exact-input reuse. A separate native BF16 check from
`f5c7f8c` matched all 384 score rows and 180 generation rows against uncached
`e69de72` execution. It reused 232 score calls and 104 generation calls while
still executing the authorized stored read each time. Candidate batching is not
used: its separate profile changed NLL values despite unchanged sampled choices.

All banks and artifacts remain external. Report choice accuracy, candidate-free
actions, counterfactual sensitivity and actual value/mass bytes separately. Raw
fallback is retained, so this does not demonstrate net disk savings, global
retrieval, agent success or parameter substitution.
