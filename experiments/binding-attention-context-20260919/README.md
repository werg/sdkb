# Attention continuation with distractor records (running)

The selected-support attention model composes permission and restoration correctly
on unseen worlds, including action counterfactuals, but action accuracy falls from
100% to 68.8% when both entities' records are exposed. Test whether training with
those competing records improves binding beyond extra selected-support training.

Both arms warm-start the identical completed attention checkpoint with fresh Muon
state, seed 31, 800 additional updates and four microbatches per update. All model,
reader, precision, learning-rate, recurrent-core trainability, sampled depth and
text-anchor settings are inherited. Both consume the same training episode file.
The sole scientific difference is `train.evidence_scope`: `required` versus
`available`. That changes evidence supplied to both latent and text-anchor paths;
true required supports, causal bounds and counterfactual labels remain intact.

The selected-support control runs first; the all-world arm follows only after it
completes. Outputs, W&B offline segments and checkpoints are external. Save cadence
is 1,000, so these arms save initially, finally or on an emergency stop. Keep two.
This study runs alongside the MLP adaptation-scope study; timings are contended,
not throughput benchmarks. No optimizer state is borrowed across the warm-start.

After both complete, create 32 common new worlds. Evaluate the source and both
final checkpoints with `scripts/evaluate_binding_context.py`: all-world records,
selected supports, zero/no memory, required-support removal and both rule flips.
Primary endpoint is all-world action accuracy together with both-correct changing
pairs and low false changes on unchanged restoration branches. Report direct facts
and identifiers separately. A gain with distractors is not cross-world retrieval,
authorization, agent execution or a parameter-substitution result.

Use the validated sequential driver in
`experiments/binding-continuation-20260919/run.py --study EXTERNAL_STUDY_ROOT`.
Stop through `sdkb runs stop --output EXTERNAL_STUDY_ROOT`; resume explicitly with
`--resume`. Source/config/data hashes are checked, and an interrupted first arm
prevents advancing to the second. Training and queued evaluation launched from frozen source `97a9ded` in
`.sdkb/attention-context`. The selected-support control runs first.

The completed source-model evaluation also shows identical fact predictions for
both entities in every opposed-rule world (16 permission, 13 restoration). The
follow-up analysis will report opposed-rule entity pairs separately, using
`scripts/analyze_entity_binding.py`, rather than crediting agreement cases as
binding. This is an additional diagnostic chosen before either continuation's
final evaluation; the declared primary action endpoint is unchanged.
