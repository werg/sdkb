# Matched Muon binding curricula

Status: both native-GPU MLP curricula have completed training from the pinned
pretrained model. Both post-freeze evaluations are complete; common-world stage
comparisons are complete. Attention training and its launcher evaluation are also
complete; the paired reader confirmation is in progress.
Their frozen training source checkout is `837f592`; they do not inherit AdamW optimizer
state or claim that the interrupted AdamW curricula completed.

Both use seed 23, 128 training worlds, two entities per world, and stage budgets
200/200/400/400. The sole between-arm change is oracle evidence access: relevant
supports versus every record in the query's world. Required-support identities
and conditional sufficiency labels remain intact. Text and memory receive the
same evidence within each arm. There is no learned cross-world retrieval here.

The native Muon optimizer, stabilized tiny-gate reader and profiled checkpointing
setting are shared. Canonical recipes are `recipes/looped_binding_muon_selected.yaml`
and `recipes/looped_binding_muon_all.yaml`; exact prepared identities are recorded
in `inputs.json`. Checkpoints and new run data live under
`/mnt/external/sdkb-archive/runs`. Saves occur at initialization, completion,
emergency stops and every 1,000 updates, not every 50. W&B is offline.

Each launcher performs its own post-freeze tests. A queued comparison then waits
for **both** curricula to finish, creates 32 new common worlds, and compares every
completed stage on those same questions. It includes zero-payload/no-memory,
actual-support removal, and both world-consistent rule counterfactuals. The common
dataset provides paired questions across evidence policies, unlike the separately
generated launcher test sets.

The research question remains whether stored memory supports permission,
restoration, exact identifiers and their procedural combination in unseen worlds.
Teacher NLL, probe decodability and numerical stability are not substitutes for
correct answers and appropriate counterfactual changes. One-seed synthetic results
will not establish agent success or parameter substitution.

## Completed all-context result

The all-context MLP completed its 200/200/400/400 budgets without nonfinite
gradients. Its own 32 post-freeze worlds gave the following choice accuracies:

| Task | Stored payloads | Zeroed payloads | No memory |
|---|---:|---:|---:|
| Action | 49.2% | 50.0% | 50.0% |
| Identifier | 26.6% | 28.1% | 31.3% |
| Permission | 56.3% | 56.3% | 53.1% |
| Restoration | 51.6% | 48.4% | 51.6% |
| Overall | 46.6% | 46.6% | 47.2% |

Action counterfactuals have **zero** both-correct pairs for either rule change.
Direct permission flips also have zero both-correct pairs; restoration flips have
3/64. This run does not establish useful binding or composition. Its out-of-domain
Boolean result is similarly uninformative (51.0% with payloads, 52.1% with zeros).
Aggregate reports, frozen-model identity and episode checksums are in
`all-final.json`. These worlds differ from the other launchers' test sets; the
queued common-world comparisons remain necessary for paired between-arm claims.

After training and the teacher evaluation completed, this run stopped gracefully
at an evaluation boundary and resumed from `a962739` for the remaining frozen
evaluations. Every stage's checkpoint name, model hash and optimizer-state hash
remained unchanged. This is an evaluation-only source transition, not retraining
or an optimizer reset.

## Completed selected-support result

The selected-support MLP completed the same budget. On its own 32 fresh worlds
(320 questions), choice accuracy was 69.4% with payloads, 38.1% with zeroed
payloads and 48.8% without memory. Both direct rule facts scored 100%, including
every answer-changing counterfactual pair. Identifier choice accuracy was 35.9%.

Action accuracy was only 55.5%. Permission flips produced both correct actions in
7/128 pairs. Restoration flips produced both correct actions in 48/64 changing
pairs, but also changed predictions on 63/64 branches where the correct action
should stay fixed. Removing permission barely affected action accuracy (53.1%).
The model often follows restoration without respecting permission; successful
fact recall has not become reliable conditional composition.

`selected-final.json` preserves summaries and checkpoint/data identities. These
are different worlds from the all-context launcher's evaluation, so their headline
scores are not a paired comparison. Candidate-free generation has not been run on
this final model. Both MLP evaluation migrations preserved every stage's checkpoint
name and model/optimizer-state hashes (`evaluation-migration.json`).

## Common-world selected-support stage comparison

All four selected-support checkpoints have now been evaluated on the same 32
confirmation worlds. Text bootstrap and the text recurrence bridge both achieve
100% across all four families, with correct changing and unchanged counterfactual
behavior. Their zero-payload condition still supplies source text, so its 100%
score is expected; it is not evidence of latent-memory use.

Warmup achieves 100% restoration recall but 57.8% permission recall and 50% action
accuracy. Joint training raises permission recall to 100%, while action accuracy
reaches only 54.7%. The paired action gain is 4.7 percentage points, with a
world-bootstrap 95% interval of −1.6 to +11.7 points. Permission action flips have
both answers correct in 4/128 pairs. Restoration action flips have both correct
in 43/64 changing pairs, but change predictions on **all 64 unchanged branches**.

The zero-payload action control falls from 50% to 29.7% during joint adaptation.
Consequently, the increasing gap over zeros exaggerates progress on actions; the
no-memory control stays at 50%. This is evidence of a latent-interface composition
bottleneck despite successful text interpretation and single-fact retrieval.

`selected-common-stages.json` retains each checkpoint and shared dataset identity;
`selected-common-joint-gain.json` records paired gains and false-change deltas,
including raw-report hashes. Positive false-change deltas mean worse behavior.
Intervals resample worlds, not training seeds, and are not multiplicity-corrected.
The all-context comparison is complete; matched attention confirmation remains in progress.

## Short adaptation diagnostic (completed)

A separate diagnostic compared the original AdamW selected-support warmup with
100 further joint Muon updates. Both evaluations use the corrected reader and
the same 32 validation worlds (320 queries); each checkpoint writes its own frozen
bank. This is an adaptation diagnostic, not a Muon-versus-AdamW comparison.

| Task | Warmup | +100 Muon updates | Adapted zero-payload control |
|---|---:|---:|---:|
| Action | 50.8% | 55.5% | 47.7% |
| Identifier | 40.6% | 42.2% | 40.6% |
| Permission | 50.0% | 100% | 54.7% |
| Restoration | 100% | 100% | 45.3% |

Both rule questions now pass every answer-changing counterfactual pair. Action
composition does not: permission flips produce both correct answers in only 1/128
pairs, and restoration flips in 0/64. Dropping the permission support leaves action
accuracy unchanged. Restoration changes also spuriously alter action predictions
on 33/64 branches whose correct answer should remain fixed. The reported
`appropriate_prediction_change` field counts prediction changes, so it must be
read alongside **both-correct** and false-change metrics, not treated as success.

Aggregate summaries and world-paired diagnostic intervals are the
`short-adaptation-*.json` files here. Raw rows and immutable checkpoints remain
external. These validation results motivate a reader control; they are not a final
held-out confirmation and do not justify selecting a winning training seed.

### Unconstrained generation check

`scripts/evaluate_stored_generation.py` consumed the adapted model's existing
serialized bank with its writer disabled, using the first eight validation worlds
in file order (80 questions). No candidate list was supplied to the decoder.
With a 24-token greedy budget, exact output was 19/32 for action, 16/16 for each
rule fact, and **0/16 for exact identifiers**. Zeroed memory gave 16/32 action,
11/16 permission and 6/16 restoration. This small diagnostic confirms that rule
facts can be emitted freely; it also exposes the gap between identifier choice
scoring and exact reproduction. It remains a synthetic output check, not executed
agent behavior. `short-adaptation-generation.json` retains matching-subset choice
scores, script/input hashes and output summaries; raw predictions stay external.

## Attention reader control (trained; paired confirmation running)

`recipes/looped_binding_muon_attention.yaml` changes only the selected-support
recipe's reader from pooled MLP to the existing fixed-slot attention comparator.
It retains Muon, data seed, training budgets, payload width, read slots, reader
width/rounds and recurrence. It trains its own text/bridge/latent stages from the
pinned pretrained model. Reader parameter counts and random initialization draws
differ; this is equal source access and interface capacity, not equal FLOPs or an
identical-initialization claim. Neither arm currently uses compaction. Both reader
implementations retain contribution-compaction support for later experiments.

A tensor comparison after text bootstrap found all backbone tensors exactly
identical between selected-support MLP and attention. Of 194 common tensors, 167
were identical; the 27 differing common tensors were all inside the reader.
The two architectures also have different reader-specific tensors. This narrows
the initialization difference to the reader at that boundary; it does not assert
identical reader initialization or equal parameter count. Checkpoint-manifest
hashes and the tensor-name audit are in `reader-text-initialization.json`.

Primary question: can the attention system combine permission and restoration
to choose actions while responding correctly to changing and unchanged
counterfactual branches? Permission fact accuracy alone is insufficient.

The isolated training source is `f737e7a`; prepared train/validation hashes match
the MLP runs exactly (`attention-inputs.json`). A separate queued driver,
`compare_attention.py`, waits for both selected-support systems to finish before
creating 32 new common worlds. It evaluates their text-bootstrap and final latent
stages with source removal and counterfactual controls. All artifacts are external.

### Queued evaluation source update

Before either queued comparison generated its confirmation dataset, its waiting
process was replaced with an isolated `a962739` checkout. Training and active
launcher evaluations were left on their original source. The new evaluation code
batches offline SQLite writes and reserves write transactions before deletion and
lineage checks. Actual LFM2.5 BF16 validation found identical stored record bytes
and all 88 evaluation rows versus individual writes, with no writer calls during
reads (`../operations-20260919/bank-batch-parity.json`).

The frozen comparison drivers themselves are unchanged. New logs are
`/archive/runs/muon-binding-stage-comparison-batched.log` and
`/archive/runs/muon-reader-confirmation-batched.log`. This operational change
addresses per-record external-disk commits; it does not alter evidence, model
weights, scoring, budgets or the predeclared comparison questions.

The attention curriculum subsequently stopped gracefully after warmup and its
teacher evaluation, before creating the joint-training directory. It continues
from the same checkpoint into the prescribed joint stage on `a962739`. The model,
reader, replay, recurrence, optimizer and configuration modules are byte-identical
across the source transition; earlier checkpoint hashes are retained in
`attention-source-transition.json`. The training loop and scientific settings are
unchanged. This records an explicit stage-boundary storage/operations update,
rather than changing a live checkout invisibly.

## Completed common-world evidence-policy comparison

Both MLP arms have now completed all four stage evaluations on the same 32 new
worlds. `all-common-stages.json` and `selected-common-stages.json` preserve the
shared episode checksum and each checkpoint identity.

| Stage / task | Selected supports | All world records |
|---|---:|---:|
| Text bootstrap: action | 100.0% | 50.0% |
| Text bootstrap: each direct fact / identifier | 100.0% | 100.0% |
| Joint latent: action | 54.7% | 49.2% |
| Joint latent: identifier | 34.4% | 40.6% |
| Joint latent: permission | 100.0% | 42.2% |
| Joint latent: restoration | 100.0% | 39.1% |

The text recurrence bridge preserves its own arm's bootstrap results. Thus the
all-context action failure already occurs with source text: it cannot be assigned
solely to latent encoding. The selected-support arm is the cleaner text-to-latent
composition test, because its text control passes the task.

For final action accuracy, selected minus all-context is +5.5 percentage points,
with a paired world-bootstrap 95% interval of −0.8 to +12.5 points. The strong
between-policy differences are in fact recall; neither system establishes reliable
action composition. Identifier choice has no reliable selected-support advantage.
`common-evidence-policy-comparison.json` retains all paired statistics, source-report
hashes and the common episode checksum. Evidence access differs deliberately;
these are trained-system comparisons, not an inference-only retrieval ablation.

### Completed attention launcher evaluation

On its own 32 fresh worlds, attention achieves 100% choice accuracy on actions,
permission and restoration. For actions, both answers are correct on every
permission flip (128/128) and every answer-changing restoration flip (64/64).
Restoration changes cause **zero** spurious action changes on 64 unchanged branches.
No-memory and zero-payload action scores are 40.6% and 35.2%. Removing permission
reduces action accuracy to 43.8%; removing restoration reduces it to 75.0%.

This is positive evidence of narrow stored-memory action composition, subject to
confirmation on the same worlds as MLP. Identifier choice remains weak at 28.1%
(zeroed 32.8%, no-memory 35.9%). `attention-final.json` records the complete
aggregate results and source/checkpoint/input identities. Candidate-free generation
also reproduced all 32 action answers and all 32 direct rule facts in the first eight
worlds (24-token greedy budget, stored bank, writer disabled). It reproduced zero
of sixteen identifiers. `attention-generation.json` records input and raw-result
hashes. These checks do not establish executable agent skill,
multi-entity retrieval or capacity substitution.
