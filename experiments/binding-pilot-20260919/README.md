# Procedural transfer pilot — 2026-09-19

Status: selected-pair text, bridge and latent warmup completed. Joint training
failed before update 312 on a non-finite gradient; its last committed checkpoint
is update 300. The all-context pilot was stopped at update 150 of joint training
for storage migration. Both AdamW pilots are now paused; the owner's requested
Muon continuation is a new experiment. These are interim diagnostics, not a
completed successful curriculum.

The Boolean pilot left two distinct gaps. Its text-access checkpoint answered
all permission, restoration and identifier queries correctly on the unfamiliar
binding family, but only 39.8% of actions. Its latent checkpoint also failed on
identifiers. The action gap therefore cannot be attributed solely to memory.
See `prior-text-diagnostic.json` for the information-matched diagnostic.

## Fixed first experiment

Use `recipes/looped_binding.yaml`: seed 23, 128 training worlds, two entities per
world, five queries per entity (permission, restoration, identifier and actions
for both capability values). Train and validation source identities are disjoint.
Start again from the pinned pretrained LFM2.5-230M revision used by the Boolean
pilot, rather than hiding an extra warm start from that task.

Budgets: text bootstrap 200, recurrent bridge 200, frozen-backbone latent warmup
400, recurrent-core joint adaptation 400 optimizer updates; accumulation four.
Use the same native architecture, payload dimensions and NVIDIA runtime.
Validate on 32 independent worlds; final evaluation creates another 32 worlds
only after training. Report all families, both rule counterfactuals, no-memory,
zero-payload and support-removal controls. Compare all four stages.

A restoration flip should change allowed retry actions and preserve STOP actions.
A permission flip should change every action. Counterfactual banks consistently
flip a rule type for every entity in a world; queries sharing a source ID must
never see conflicting variants of that ID. Queries, IDs and timestamps stay fixed.

## Scope and continuing work

Oracle routing supplies the relevant source pair, including both rules on STOP
branches. This run isolates interpreter learning and transfer through the latent
interface. It does not yet prove entity selection among competing records.
After this run, use its results to separate action-format learning, source coding
and competing-entity binding; continue with the control or follow-up needed by
those results. Record failed branches as well as successful ones.

Live output: `/home/werg/sdkb-runs/binding-pilot-20260919` (container `/runs/...`).
Local exact operational inputs: `.sdkb/binding-pilot/`. The existing retained
`sdkb-causal-pilot` container supplies native NGC 26.03 with vendor Torch/CUDA.
W&B remains offline. Checkpoint payloads were subsequently moved to
`/mnt/external/sdkb-archive/retained-checkpoints` with verified original-path links.
The internal run tree now holds small metadata/cache files. No unrelated GPU
workload was stopped. See `../operations-20260919/` for the storage correction.

## Interim findings

The new text curriculum reaches 100% choice accuracy in every binding family when
given the relevant source pair. Giving that same checkpoint all four records in a
world lowers action accuracy to 59.4%, identifier to 57.8%, permission to 75% and
restoration to 82.8%. `text-world-context.json` records this information-access
diagnostic. It motivates a matched curriculum in which both text and latent stages
see all available records, while ground-truth support labels remain unchanged.
`all-context-inputs.json` records that pilot's exact inputs and isolated source
revision. Both runs use the same validation worlds; their post-training fresh
worlds have independently derived identities and are not a paired comparison.

The selected-pair warmup is insufficient (`warmup-diagnostic.json`):

| Query family | Stored memory | Zero payload |
|---|---:|---:|
| Action | 50.8% | 50.0% |
| Identifier | 40.6% | 39.1% |
| Permission | 50.0% | 45.3% |
| Restoration | 100% | 57.8% |

Permission flips do not cause the required action changes. Low teacher-forced
mean-token NLL therefore does not establish procedural composition.

### Locating the rule signal

`probe_payloads.py` reads the existing serialized BF16 original/counterfactual
banks. A fixed ridge readout, fitted on 16 worlds and tested on 16 disjoint worlds,
recovers both permission and restoration bits at 100%. The paired source IDs and
contexts are fixed, and both members of each held-out pair stay outside probe
training. `warmup-payload-probe.json` records the results.

`probe_reader.py` repeats that diagnostic after the frozen reader returns its soft
tokens. Both bits remain linearly recoverable at 100%, including on action queries
(`warmup-reader-probe.json`). This localizes the warmup failure downstream of lost
bit information: the deployed reader/decoder combination is not using the signal
reliably. It does not show that a downstream decoder can easily learn the diagnostic
readout, and a separately trained probe is not deployed-model success.

### Joint-training failure investigation

The selected-pair run has finite losses through update 311 but fails the non-finite
gradient guard before update 312. A normal resume from the verified update-300
checkpoint repeats the failure. No bad optimizer update was applied. An isolated
checkpoint copy and `diagnose_gradients.py` reproduce the same training sequence
without archive/tracking side effects and enable anomaly detection on the failing
update. Gate masses around 1e-38 overflowed the normalization backward. The reader
now factors a common log scale out of very small gates, preserving numerator and
absolute mass. The isolated GPU sequence then completed updates 301–320 with
finite gradients (`gradient-failure-and-fix.json`). This correction was validated
separately; do not count the incomplete original joint stage as a completed result.

### Evaluation overhead

Counterfactual reporting consumes only full-evidence rows. Previously it also
scored unused no-memory and zero-payload variants. The evaluation optimization
skips those discarded rows. A pretrained GPU check found exactly equal retained
rows and scores for five paired queries (`counterfactual-evaluation-equivalence.json`).
The recorded contended timings are a local check, not a hardware benchmark.
