# Source-utility curriculum pilot

The preceding output-KL fork improved general next-message likelihood but did
not improve average real-over-zero payload benefit. A post-hoc diagnostic found
that payload benefit was weaker in episodes where selected source text barely
changed the fixed parent's target likelihood. This pilot asks whether training
more often on source-useful *training* episodes strengthens stored-payload use
on the unchanged repository-heldout validation set.

The fixed R=1 joint parent scores each of the 530 original training episodes
with the same selected source text and with no source text. `run.py --prepare`
selects the upper half by per-episode mean target-NLL reduction, with a stable
episode-ID tie break. It copies those 265 original JSONL lines byte-for-byte and
retains every source/target identity, versioned payload, provenance and time
boundary. Target tokens are used only to choose this supervised training
curriculum. They are never inserted into a query, writer input, read plan or
inference path. The 119 repository-heldout validation episodes do not affect
selection.

The native R=2, writer R=1, backbone-frozen Muon stage warm-starts the same
completed joint checkpoint as the no-KL uniform-data control. Both use seed 233,
400 updates, BF16, four-example gradient accumulation, one fully live causal
read, sparse normal checkpoints and offline W&B. Dataset sampling differs by
design, so this is a curriculum comparison, not a paired per-update optimizer
comparison. All mutable artifacts, including the selected data, preflight,
complete recovery states and raw scores, live under the declared external root.
The digest lock rejects a changed parent, scores, selected data or config before
allocating the model. A cooperative stop commits full optimizer/RNG and partial
gradient state; resume uses the same locked inputs.

The predeclared held-out diagnostic is the real-over-zero and real-over-wrong
stored-value benefit at R=2 on the same 119 validation episodes, with unchanged
selected IDs. Also compare no-memory likelihood and the fixed one-pass text
teacher. A lower real-value NLL alone does not establish improved payload use.
This remains teacher-forced next-message likelihood; it does not measure patch
success, learned routing, semantic source sufficiency or capacity substitution.
