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
