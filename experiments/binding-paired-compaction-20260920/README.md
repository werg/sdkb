# Paired raw/compact continuation

The preceding interleaved study's mean-code confirmation preserves 128/128 actions
and its tested rule-change pairs, versus 81/128 for the matched raw-trained control.
However, the interleaved endpoint's raw reads fall to 126/128. Test whether retaining
the raw task objective on compact-selected examples protects that path.

Warm-start again from the original useful MLP source, not the interleaved endpoint.
Use the same fresh 128-world training corpus, seed 103, 400 Muon updates, four-example
accumulation, learning rates, 50% compaction requests and sampled two/three-loop
depths. Only the scientific objective changes to `paired`; behavior KL stays zero.
Compare to the already completed interleaved endpoint. Verify identical initial
weights and the actual example/depth/compaction sampling trace after training.

The paired loss retains raw NLL and adds compact NLL at weight one on selected
examples, plus the existing contribution auxiliary at weight 0.1. Both paths share
the same causal query, selection and values. One-source groups still receive the
paired objective despite no size reduction. This changes loss scale and decoder
compute at the same optimizer-update budget; it does not isolate a free algorithmic
benefit at equal FLOPs. Report elapsed/resource observations without interpreting
contended timing as a hardware benchmark.

Declare a new 32-world, 320-question held-out split before training. Its source IDs
and endpoint strings are disjoint from both fine-tuning corpora and the previous
confirmation. Evaluate both endpoints under raw, mean-plus-mass and native learned
codes, including free generation, no/zero memory, source removal and rule/endpoint
counterfactuals. Preserve complete-group mass and exact raw subset fallback. No net
disk savings, global routing, exact identifier recall, agent-success or capacity
substitution claim follows from the earlier narrow result.

All artifacts go directly to `/archive/runs/binding-paired-compaction-20260920`.
Initial/final/emergency checkpoints only, keep two, 10 GiB disk reserve, 8 GiB host
memory reserve, native NVIDIA Torch, offline W&B. The `interleaved` directory is an
alias of its existing external run, not another copy of model weights. Training and
confirmation use frozen checkouts. The new evaluator resumes completed scoring
variants and generated strings; it never re-encodes sources during inference.


## Training endpoint

All 400 updates completed. Initial weights are SHA-identical to the interleaved
control, all 400 sampled depths match, and final saved Python sampler state exactly
matches the declared sequence of examples/live choices/compaction requests. There
were 807 paired requests and 322 reductions of two-source groups. Peak logged CUDA
allocation was 3,133,870,592 bytes. Elapsed training was 829.7 seconds under changing
contention; this is not a paired/interleaved speed comparison. `endpoint.json` pins
the final manifest, model and both frozen execution commits.

The confirmation is running on the new split. After training ownership released,
verified hard-link deduplication reclaimed 1,022,082,588 bytes from its identical
initial weights; all checkpoint paths and full recovery state remain intact.


## Partial fresh confirmation: raw reads

| Objective | Original actions | Permission-change both correct | Restoration-change both correct | False restoration changes / 64 | False endpoint changes / 128 |
|---|---:|---:|---:|---:|---:|
| Interleaved | 128/128 | 125/128 | 62/64 | 0 | 1 |
| Paired | 127/128 | 126/128 | 64/64 | 1 | 1 |

The result is mixed, not a demonstrated repair of raw-read reliability. All six
interleaved errors across supplied-memory counterfactual conditions omit required
restoration (`RETRY` instead of `RESTORE_RETRY`). The paired arm has one such
permission-intervention error and one original `STOP` case predicted as
`RESTORE_RETRY`. These are substantive action errors, not string-format mismatches.
The paired arm's one false change under an irrelevant restoration intervention
corrects that initially wrong answer; it remains a failure of the required
invariance. Both arms still generate 0/64 exact identifiers.

Mean and learned-code confirmation are pending. `raw-confirmation.json` pins the
completed counts; the one-seed, 32-world sample does not justify promoting the
paired objective on raw-read results alone.
