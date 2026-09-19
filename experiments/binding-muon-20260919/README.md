# Matched Muon binding curricula

Status: two native-GPU curricula started from the pinned pretrained model.
Their frozen source checkout is `837f592`; they do not inherit AdamW optimizer
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

## Attention reader control (running)

`recipes/looped_binding_muon_attention.yaml` changes only the selected-support
recipe's reader from pooled MLP to the existing fixed-slot attention comparator.
It retains Muon, data seed, training budgets, payload width, read slots, reader
width/rounds and recurrence. It trains its own text/bridge/latent stages from the
pinned pretrained model. Reader parameter counts and random initialization draws
differ; this is equal source access and interface capacity, not equal FLOPs or an
identical-initialization claim. Neither arm currently uses compaction. Both reader
implementations retain contribution-compaction support for later experiments.

Primary question: can the attention system combine permission and restoration
to choose actions while responding correctly to changing and unchanged
counterfactual branches? Permission fact accuracy alone is insufficient.

The isolated training source is `f737e7a`; prepared train/validation hashes match
the MLP runs exactly (`attention-inputs.json`). A separate queued driver,
`compare_attention.py`, waits for both selected-support systems to finish before
creating 32 new common worlds. It evaluates their text-bootstrap and final latent
stages with source removal and counterfactual controls. All artifacts are external.
