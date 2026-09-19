# MLP adaptation-scope continuation (running)

The completed selected-support Muon curriculum interprets source text correctly
and retrieves single rule facts, but fails conditional action composition through
stored memory. Its joint stage trains only recurrent backbone layers 4–9. Test
whether more adaptation helps, and whether restricting backbone adaptation is a
bottleneck. This remains useful to the pooled-MLP thesis even if attention succeeds.

Both arms start from the **same final selected-support MLP checkpoint** and reset
optimizer state explicitly. They receive 1,600 additional updates, seed 29,
four microbatches per update, the same existing training episodes, sampled depths
2/3, one writer pass, and the same text anchor. Native Muon and its AdamW exclusions,
learning rates, BF16 interface, reader width/slots and payload budget are retained.
The sole between-arm setting is `model.backbone_train_scope`:

- `recurrent_core`: extra-training control, preserving the current restriction.
- `all`: allow prelude, coda and embedding/output weights to adapt as well.

This is an equal-update/equal-data comparison, not equal optimizer memory, compute,
or an exact continuation of old momentum. Record trainable counts and actual runtime.
No loss reweighting, task resampling, payload resizing or reader change is included.

Run arms sequentially with external output, checkpoint cadence 1,000, retention two,
offline W&B, and the existing memory/disk guards. Initial/final/emergency saves remain
mandatory. Do not modify the completed curriculum or its sealed stage budgets.

After both finish, generate 32 new common worlds and evaluate their frozen stored
banks alongside the source checkpoint. Report all four task families, no-memory,
zero-payload, support removal, both-correct changing counterfactual pairs, and false
changes on unchanged branches. Evaluate candidate-free outputs separately from
choice scores. Do not use intermediate test results to select a training checkpoint.

The main action endpoint requires both permission sensitivity and restoration
invariance when permission forbids retry. Improved direct facts or a larger gap over
a deteriorating zero-payload control are insufficient. A successful result still
would not establish multi-entity retrieval, compaction or parameter substitution.

`prepare.py --source SOURCE_STAGE --output EXTERNAL_STUDY_ROOT` writes the two
configs and source/data hashes without launching GPU work. Source must be the
completed selected-support Muon MLP stage. All paths are arguments; platform defaults
are inherited from its validated config. Launch provenance must record the frozen
code commit used for training.

The full-backbone native preflight passed on source `2112c9f`: one-loop identity
and causal-prefix error were both zero, with finite nonzero memory/bridge gradients.
`all-preflight.json` retains the runtime and numerical record. This validates
execution, not a capability result. The sequential runner has regressions for
stops, failures, configuration drift and partial-run resume; core recovery tests
separately verify full optimizer/RNG restoration.

Run `run.py --study EXTERNAL_STUDY_ROOT` from a frozen checkout; add `--resume`
to resume explicitly. Redirect its console to the path reported by
`sdkb runs status --output EXTERNAL_STUDY_ROOT`. Stop with
`sdkb runs stop --output EXTERNAL_STUDY_ROOT`; an interrupted first arm prevents
the second from starting. Existing arm checkpoints resume; new arms warm-start
from the captured source checkpoint. Training launched sequentially from frozen source `0d6c0a4`; the recurrent-core
control runs first. Its console is external at
`/archive/runs/.sdkb-control/205b517eb14291e6f5ff1d4c/console.log`.
The full-backbone preflight used `2112c9f`; model, training and optimizer modules
are unchanged across these commits. The generation diagnostic on the earlier
checkout finished before that checkout was advanced for training.
