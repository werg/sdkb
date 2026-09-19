# Attention continuation with distractor records (completed)

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

The selected-support control completed all 800 updates and the all-world arm is
training. Both initial weight files have exactly the same SHA256 as the captured
source checkpoint, and both optimizers start fresh (`initialization-check.json`).
Final evaluations remain pending.

## Completed common-world result

Both arms completed their prescribed 800 updates. On the same 32 new worlds:

| Measure | Source | Extra selected-support training | All-world training |
|---|---:|---:|---:|
| All-world action accuracy | 63.3% | 59.4% | 62.5% |
| Selected-pair action accuracy | 99.2% | 100.0% | 100.0% |
| All-world permission | 71.9% | 70.3% | 70.3% |
| All-world restoration | 76.6% | 76.6% | 76.6% |
| All-world identifier choice | 35.9% | 29.7% | 26.6% |

All-world training's paired action gain is −0.8 points versus the source
(world-bootstrap 95% interval −5.5 to +4.7) and +3.1 points versus the matched
extra-training control (−0.8 to +7.0). It has both action answers correct in only
43/128 permission-flip pairs and 19/64 changing restoration pairs, and makes 9/64
spurious changes on unchanged restoration branches.

Both continued models give the same fact answer for the two entities in **all 19
opposed-permission worlds and all 15 opposed-restoration worlds**. Zero pairs are
jointly correct. The source has one correct opposed-permission pair and zero
opposed-restoration pairs. More exposure to distractors at this budget therefore
does not establish entity discrimination, despite preserving selected-pair skill.

`completed-training.json` records exact final checkpoint and environment identities;
`confirmation-summaries.json`, `paired-results.json` and `entity-stratification.json`
retain controls, shared episode/raw-report hashes and paired intervals. This is
one training seed; intervals resample worlds and are not multiplicity-corrected.
The result motivates inspecting entity information in queries and stored payloads
before another training intervention, rather than assuming that more updates alone
will fix binding. The diagnostic below is now complete.

## Stored query/payload diagnostic

`scripts/inspect_entity_queries.py` verifies the evaluation bank's checkpoint and
episode provenance, disables the writer, and compares paired fact queries using
the identical ordered candidate set. In the all-world-trained endpoint there are
**no exact query, payload or returned-token collisions** among the two entities.
Median relative query differences on opposed-rule pairs are 4.6% (permission) and
5.6% (restoration); returned-token differences are 3.7% and 5.3%. Thus the system
is not literally reading identical inputs after the entity changes.

Payload differences are much larger for opposed rules than agreeing rules:
median relative differences are 1.70 versus 0.10 for permission and 1.66 versus
0.065 for restoration. These geometric observations do not prove which identity
information is recoverable, nor establish a trained matching algorithm. The
behavioral entity-pair failure remains the relevant capability finding.
`query-diagnostic.json` records the aggregates and input hashes; no source
trajectories are re-encoded during this diagnostic.
