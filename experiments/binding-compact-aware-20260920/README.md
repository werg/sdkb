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

Both endpoints completed and are recorded in `endpoints.json`. Their initial weight
hashes are identical, all 400 sampled depths match, and each final saved Python
sampler state exactly matches the declared exposure trace. Of 1,600 microbatches,
807 request compaction and 322 actually reduce a two-source action group. The raw
control's compactor remains byte-identical to initialization; the temporary arm's
compactor parameters change. These are training/provenance checks, not capability
results. Fresh confirmation is running.

After both training locks released, verified hard-link deduplication reclaimed
1,022,082,588 bytes from the duplicate initial weights. All checkpoint paths,
optimizer/RNG state and manifests remain intact; see the operations record.


## Partial confirmation: raw and mean arms complete

The learned-code arms are still running. On the declared 32 fresh worlds:

| Training | Read representation | Generated actions | Permission-change both correct | Restoration-change both correct |
|---|---|---:|---:|---:|
| Raw control | Raw | 128/128 | 128/128 | 64/64 |
| Temporary compaction | Raw | 126/128 | 122/128 | 60/64 |
| Raw control | Mean plus mass | 81/128 | 31/128 | 0/64 |
| Temporary compaction | Mean plus mass | 128/128 | 128/128 | 64/64 |

Temporary training makes this cheap full-cluster mean code substantially more
useful: action accuracy improves by 36.72 percentage points versus the matched
raw-trained mean-code control, with a world-bootstrap interval [26.56, 46.09].
The compact-trained mean arm also has no false action changes on 64 unaffected
restoration pairs or 128 endpoint-only interventions. Its no-memory/zero-value
controls get 59/128 and 44/128 actions; exact identifiers remain 0/64.

Raw-read behavior has a measurable cost: the temporary arm's raw path produces
three false action changes under endpoint-only interventions and loses some correct
rule-change pairs. This motivates the separately frozen paired-objective follow-up;
it is not evidence that both paths were perfectly preserved.

The bank contains 64 full-cluster codes per variant, 4,236 serialized bytes per
code, while retaining all raw records for partial selection. This establishes a
narrow one-seed improvement in conditional compactability on synthetic oracle reads,
not net disk savings or parameter substitution. `raw-mean-confirmation.json` pins
all four completed outputs and `mean-code-training-generation.json` records the
paired comparison. Final learned-code results will be appended separately.
