# Frozen reader-output identifier diagnostic

The original frozen writer exposes partial hexadecimal information to a supervised
payload classifier (137/384 held-out characters with a linear readout), while its
language decoder still generates 0/64 exact endpoints. Test whether a similarly
trained diagnostic can recover characters after the pooled reader.

Use the identical frozen source, existing 4,096-record bank, 1,024-world corpus and
32-world held-out partition from the payload-readout study. Compare payload vectors,
the first native reader output before bridge injection, and the same reader output
with zero payload values. Reader features can use the causal query; the zero-value
arm controls that information. All reads are oracle selected and stored-only, with
writer/compactor calls forbidden. Neither targets nor source text enter extraction.

For each representation fit the same linear and 256-hidden-unit character heads:
1,600 native Muon updates, batch 128, seed 83, learning rate .001. Training-only
normalization, shifted-feature controls and full optimizer/RNG resume are retained.
The reader has 8,192 features versus 2,048 payload features, so its heads have more
parameters; report this difference. The fixed `api_` prefix and six-position output
format are supplied to all classifiers. These are supervised readouts, not normal
language generation, information-theoretic bounds, or parameter substitution.

Outputs live under `/archive/probes/reader-readout-20260920`. Feature reads are
reproducible and regenerated from the frozen bank on restart; they do not invoke
the writer or create another model checkpoint. Only the small readout models save
initial/final/emergency full state, with disk/memory reserves and offline W&B.
