# Frozen reader intermediate states

Predeclared after completing the fixed-query readout study. The broad 8192-world
endpoint has 251/384 characters accessible to a linear payload head, versus
156/384 from returned reader tokens. This bounded follow-up checks two intervening
representations before choosing a model-training intervention.

- Reuse the existing frozen broad checkpoint, bank, and corpus; no writer calls,
  new banks, or backbone checkpoints. Inputs and implementation hashes are pinned.
- Capture the actual three MLP input projections, concatenated (768 scalars), and
  the final shared state before output normalization/projection (2048 scalars).
  Hooks observe the normal complete reader execution; they do not reproduce or
  alter its mathematics. Record native capture precision before FP32 head inputs.
- Each representation gets linear and MLP-256 heads: 1600 Muon updates, batch 128,
  seed 83, LR .001, training-only standardization, same optimizer settings and
  sampler as the completed study. Parameter counts are recorded per head.
- Same 1984 training identifiers and previously inspected 64 held-out identifiers;
  output format is supplied by six supervised hexadecimal classifiers. This is a
  post-hoc accessibility diagnostic, not language-generation capability or an
  independent confirmation. Dimensions and conditioning differ between heads;
  weaker readout alone does not establish irreversible information loss.
- Two concurrent children at most. External-disk outputs, offline W&B, host-memory
  reserves, native vendor Torch/CUDA. Initial/final/emergency small head snapshots
  overwrite one latest state. `--resume` preserves complete head optimizer, RNG,
  sampler, configuration and identity; stop occurs at optimizer boundaries.

Run from a frozen checkout with its `PYTHONPATH` and Git environment:

```sh
python experiments/binding-reader-stages-20260920/run.py \
  --root /archive/probes/reader-stages-20260920
```

The controller watches root and `confirmation-queue` control files, terminates and
joins children on stop, and checks pinned inputs before launch. Native preflight
also requires target-independent features, zero-value world invariance, and
bitwise equality of the unchanged final reader output with the prior extractor.
