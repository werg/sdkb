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
