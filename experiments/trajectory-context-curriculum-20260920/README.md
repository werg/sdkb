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

The policy was chosen after inspecting the preceding study's validation
diagnostic. The held-out episodes remain absent from this fork's training data,
but this is exploratory reuse of a validation set, not a sealed final test.

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

## Completed pilot

The fixed parent completed all 1,060 train-only selected/no-text scores; its
one-pass model and episode bytes are pinned in the external input lock. The
selected 265 episodes span 172 of 187 training trajectories. The frozen native
run `1a5610b` passed the actual-LFM preflight and completed 400 Muon updates,
with one in-loop read at R=2 on every logged update. The normal cadence wrote
only initial/final complete checkpoints. The frozen evaluation checkout was
`1eaaef8`. Its held-out bank used 375 writer calls for 375 unique sources under
the 64-source writer cache cap. `results.json` pins final model, raw report and
evaluator identities without committing raw data or weights.

| R=2 held-out token-weighted NLL | Uniform 530-episode control | Selected 265-episode curriculum |
|---|---:|---:|
| Real stored values | 0.971024 | 0.988519 |
| Zeroed values, same selected IDs | 0.980714 | 0.996439 |
| Wrong values, same selected IDs | 0.982071 | 0.998763 |
| No memory | 1.089557 | 1.092959 |

The paired real-value NLL change was **worse** by 0.02742 mean episode NLL
(42-trajectory descriptive bootstrap interval −0.04190 to −0.01806 when
expressed as uniform-minus-selected improvement). Real-over-zero payload
benefit changed by −0.00129 (−0.01365 to 0.00940); there is no demonstrated
increase. Real-over-wrong changed by +0.01051 (−0.00570 to 0.02665), also
uncertain. All 119 validation episode IDs, source selections and target lengths
matched across arms. The one-pass selected/no-text scores were identical at all
238 evaluated sequences, confirming a fixed text teacher.

This does not show that high text-context utility is useless as a signal. It
shows that discarding half the training examples under this single seed and
400-update budget hurt held-out likelihood and did not improve the primary
payload-specific contrast. There is no sealed new test set or agent task score.

After both evaluators exited, verified hard-link deduplication reclaimed
1,017,349,676 external bytes from the identical curriculum-initial model file.
The parent/control source and both curriculum checkpoint sets reverified
against their manifests. `dedup.json` records the operation; final optimizer,
RNG and model state remain intact for exact resume.
