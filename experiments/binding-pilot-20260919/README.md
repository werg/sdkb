# Procedural transfer pilot — 2026-09-19

Status: protocol implemented and CPU-tested; GPU pilot about to start.

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
W&B remains offline, active checkpoints stay on local storage, verified archives
use `/mnt/external/sdkb-archive`, and no unrelated GPU workload is stopped.
