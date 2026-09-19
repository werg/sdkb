# Matched native compactability-aware learning

Compare two 400-update Muon continuations from the same useful MLP source. Both
use 128 fresh training worlds, seed 103, four-example accumulation, core-only
backbone adaptation at 5e-6 and memory modules at 1e-4. Both instantiate the same
new one-code synthetic compactor and consume the same compaction-choice RNG draw.
Only the compaction probability differs: 0 (raw control) or 0.5 (temporary codes).
This preserves identical sampled examples and recurrent depths. Single-source
queries remain raw because replacing them would not reduce the group. Two-source
actions can be replaced with one code; conditional numerator/mass and reader-rollout
loss has weight 0.1, alongside the compact task loss. The remaining examples retain
raw task training. This is the interleaved objective, not paired per-example KL.

Fresh held-out split: `compact-aware-heldout-20260920`, 32 worlds / 320 questions,
with disjoint source IDs. For each endpoint evaluate raw reads, mean-plus-mass and
its native learned code, preserving exact raw partial-selection fallback. Include
no/zero memory, support removal and changed permission/restoration/endpoint sources,
with free generation on every held-out question. Check that raw composition survives
and compact counterfactual behavior improves at the same one-code budget. A raw
control's synthetic code is untrained; report that explicitly, and retain mean as
its useful cheap comparison. Neither arm deletes raw fallback records, so no net
storage-saving claim follows.

All training artifacts go directly to `/archive/runs/binding-compact-aware-20260920`.
Save only initial/final/emergency state (`checkpoint_every: 10000`, keep 2), with
full resume state, 10 GiB free-disk reserve, host memory rails and offline W&B.
No new teacher calls or model downloads. This is one-seed MLP learning evidence;
the attention implementation has the same intervention and execution tests, but
this comparison does not yet provide a matched attention learning result.

The frozen evaluator supports explicit `--compact-method raw|mean|trained`. A code
arm uses its bank for scoring, free generation, value ablation and counterfactuals;
removing a source exercises raw fallback. Separate named code views keep per-variant
accounting unambiguous. A restart verifies existing code tensors against offline
reconstruction and reuses the raw bank without writer calls; no reconstruction
runs during inference. Regression tests exercise this and reject changed evaluation
identity. Full suite after this extension: 363 passed; Ruff clean.

`run_confirmation.py` waits for both committed 400-update endpoints and released
training locks, then runs two evaluator processes at a time. It checks frozen
config/corpus identities and treats incomplete or stopped training as a failure.
