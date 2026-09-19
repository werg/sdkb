# Current Spark validation

19 September 2026. This is the current machine-validation record. The earlier
[0.4 handoff](validation-v0.4.md) remains a historical CPU execution record.

## Runtime and implementation

Native ARM64 execution on DGX Spark/GB10 uses the retained NVIDIA Torch 2.11
development build with CUDA 13.2 and Transformers 5.17.0. The NVIDIA Torch/CUDA
stack was preserved. No x86 emulation was used for SDKB work.

The actual pretrained student is `LiquidAI/LFM2.5-230M`, pinned to
`40cb2ad3b3044d5a41eee083a6103c8b523afa45`. Its numerical/gradient preflight
and native GPU curricula have run. Middle-block recurrence repeats layers
`[4:10]`; the writer remains one pass and evaluation reads serialized BF16
payloads. This verifies execution of the real model, not a simulated GPU backend.

The latest full suite passed **246 tests**, with four existing dependency/runtime
warnings. Ruff passed for `src`, `tests`, and `scripts`. Core tests require no
downloads. Coverage includes full/replay gradients, causal prefixes, serialized
precision, checkpoint recovery, storage visibility and concurrent invalidation.

## Operational evidence

| Check | Result and scope |
|---|---|
| Native Muon | New binding runs use native Muon for eligible matrix parameters with AdamW for the excluded parameter groups. Both components and their actual group settings are checkpointed. |
| Emergency resume | A real BF16 LFM run stopped after one of two accumulated microbatches and reproduced the uninterrupted final model, complete optimizer state and RNG state exactly. No partial optimizer update was taken. |
| Checkpoint placement | Twenty-five retained stage/diagnostic checkpoint directories were verified and moved externally. The internal run tree fell from about 49 GiB to 52 MiB; internal free space increased by about 48 GiB. |
| Future launches | This checkout's ignored storage settings select the external run root. Configured directories must exist; relative container training output paths are rejected. Environment overrides remain available on other machines. |
| Save cadence | Defaults and current real-model recipes use 1,000 updates, plus initial, final and emergency saves. Short 200–400-update stages normally save only their initial and final states. |
| Memory and stalls | Optional host-memory reserve, CUDA allocation fraction and compute stack watchdog are implemented. New source versions log host/CUDA memory trends and checkpoint start/commit duration. |
| Checkpointing cost | Two 100-update Muon profiles had identical training metrics and final weight-file hash. Disabling activation/reader-chunk checkpointing cut median update time 18% and raised peak CUDA allocation from 2.14 to 2.53 GiB. This applies only to the measured short binding distribution, with existing GPU contention. |
| Offline bank writes | Native BF16 individual/bulk writes produced byte-identical stored records and identical results for all 88 evaluation rows, with the writer disabled during reads. Atomic batching avoids a durable commit per record. |
| Concurrent writes/deletions | Regressions reproduced check/insert and lineage/delete races. Explicit SQLite writer reservations now protect those operations; failed bulk writes roll back without exposing partial records. |

Detailed evidence: [checkpoint relocation](../experiments/operations-20260919/checkpoint-relocation.json),
[emergency resume](../experiments/operations-20260919/muon-emergency-resume.json),
[checkpointing profile](../experiments/operations-20260919/checkpointing-profile.json),
[bank equivalence](../experiments/operations-20260919/bank-batch-parity.json).
See the [bgkit adoption audit](bgkit-audit.md) and [operating guide](operations.md).

Direct external checkpoint writes can take minutes under disk contention. They
do not silently make an internal emergency copy. The optional asynchronous archive
path can stage locally, with explicit retention/relocation responsibilities. A
blocked native call must return before cooperative stopping can complete.

## Capability evidence

The [Boolean pilot](../experiments/causal-pilot-20260919/) established narrow
stored-memory dependence on its trained synthetic family, including adjusted-answer
counterfactuals. Its text and warmup controls also succeeded; more shared depth did
not establish an additional benefit. This is not general agent capability or
parameter substitution.

The [Muon binding study](../experiments/binding-muon-20260919/) is a harder test
of permission, restoration, exact identifiers and their combination:

- Both MLP curricula completed their 200/200/400/400 training budgets without
  nonfinite gradients. Earlier AdamW pilots remain explicitly interrupted records.
- The completed all-context MLP scored 46.6% on its 320 fresh binding questions,
  equal to zeroed payloads. Action counterfactuals had no both-correct pairs.
  This run does **not** establish useful binding or composition.
- A separate 100-update Muon warm-start diagnostic recovered both direct rule
  facts, including their counterfactuals, but action composition remained weak.
  Candidate-free generation reproduced zero of sixteen exact identifiers.
- The completed selected-support MLP recalled both rule facts perfectly, but action
  accuracy was 55.5%. Restoration changes spuriously changed actions in 63/64
  branches that should remain fixed; permission flips had only 7/128 both-correct
  pairs. The 69.4% overall score does not establish conditional composition.
- Common-world MLP stage comparisons are complete. Selected-support text controls
  solve every family; all-context text controls score only 50% on actions despite
  perfect direct facts. Final latent action accuracy is 54.7% versus 49.2%, with
  a paired interval including zero. The attention launcher reaches 100% action and rule-fact choice accuracy, with
  every action counterfactual pair correct and no spurious action changes on
  unchanged restoration branches. Candidate-free generation also emits all
  action and rule-fact answers on eight worlds, but zero of sixteen identifiers.
  The common-world reader confirmation also gives attention 100% action accuracy
  versus MLP 53.1% (paired gain 46.9 points, world-bootstrap interval 40.6–53.9).
  Both text controls solve every task; identifier choice remains weak.
- Exposing both entities to the frozen attention reader reduces action accuracy
  from 100% to 68.8%, with degraded counterfactual consistency. This remains above
  zero/no-memory controls, but robust entity binding is not established.

The all-context model's lower NLL versus no memory did not translate into a benefit
over zeroed payloads. Tests, falling losses, linear probes and successful execution
must not be substituted for the causal behavior measurements.

## What remains unproven

Useful multi-entity latent binding; reliable procedural composition outside the
synthetic Boolean/binding families; useful pooled-MLP composition; literal identifier reproduction; learned retrieval at scale;
behavior-preserving compaction of a useful real-student memory system; additional
capability from recurrent depth; agent execution success; and a quality/memory/latency
frontier showing substitution for resident model parameters.

These are research outcomes to establish, not implemented-module checkboxes. The
current work is the paired MLP continuation (extra training versus broader backbone
adaptation) and diagnosis of entity binding. A completed matched attention
continuation with/without distractors did not improve binding: all-world action
accuracy was 62.5% versus 63.3% for its source and 59.4% for extra selected-support
training, with no jointly correct opposed-rule entity pairs after either continuation. No paid teacher
collection, license change or public-visibility change has been initiated.


### Longer Muon MLP continuation

The fixed 1,600-update continuations now both pass selected-support action
composition: 128/128 actions, all changed action counterfactual pairs correct,
and no false changes on unchanged actions. Candidate-free action generation is
32/32 in both arms; identifier generation remains 0/16. Recurrent-core adaptation
alone suffices at this larger budget. This revises the architectural interpretation
of the earlier short-budget failure; it does not establish distractor binding.
See [the continuation record](../experiments/binding-continuation-20260919/README.md).

`evaluate_binding_context.py --learned-world --read-budget 2` also supports
world-scoped exact learned ranking of stored keys. Supplied world membership
sets eligibility; required supports are used only by the named oracle control
and support-removal intervention. Zero values and counterfactuals retain the
original read plans. CPU regression and a native BF16 148-row mechanics smoke
verify selection eligibility and intervention plans. This is not ANN or global
retrieval, and the smoke is not retrieval capability evidence.


Both longer MLP endpoints were subsequently tested with competing entities. Both
fell to 61.7% action accuracy and gave identical answers to both entities in all
opposed-rule worlds (17 permission, 19 restoration), with no jointly correct fact
pairs. The selected-support composition success therefore does not resolve binding.
A fixed-budget, matched routing-supervision study is now running from `304fb17`;
its results remain pending. Checkpoints and banks remain on the external disk.


Candidate-free generation now also supports explicit world-scoped learned ranking.
A native BF16 two-world smoke generates 60 rows from an existing bank with the
writer disabled; all selected-ID lists match the corresponding choice and control
rows. CPU regressions cover both existing oracle behavior and learned ranking.


An explicitly training-only identifier diagnostic gets 7/16 exact for the longer
core-only MLP and 13/16 for the full-backbone MLP, versus 7/16 and 12/16 with
zero payloads. Held-out identifiers remain 0/16. Training fit largely surviving
payload removal is not evidence of reading identifiers from storage.
