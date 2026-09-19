# Task-loss continuation of the frozen MLP compactor

The initial statistics/rollout fit retains 103/128 stored action choices, below
the same frozen reader's 128/128 raw ceiling. Test whether directly preserving
the decoder's answer loss improves the compact code. Keep every backbone,
writer, query and reader parameter frozen.

Fork the completed MLP compactor into two fresh Muon optimizers: statistics-only
continuation and statistics plus target NLL with weight one. Both use 200 updates,
batch size four, seed 61, learning rate 1e-4 and the same original training file.
The target is used only in the loss. Each code is query-independent; the cached
first-boundary query comes solely from the causal prompt. Complete the reader
aggregate once, then replay those tokens at the remaining native core boundaries.

A regression checks this cached single-read schedule and every compactor gradient
against the full causal prefix-plan execution. The task arm adds frozen decoder
backpropagation, so matching data and updates does not match compute. Initial,
final and emergency state remain external and source/config bound. Original
400-update fit records are immutable; this is a new exploratory continuation.

Compare heldout raw/compact choices and then counterfactual/candidate-free behavior.
Single-child questions retain raw fallback. No net disk savings, learned routing,
agent success or parameter-substitution claim follows from lower target NLL.
The inherited 32-world set is now a development comparison; any selected method
needs a fresh-world confirmation after selection.

## Completed development comparison

Both forks completed from `13aeb05`. Statistics-only continuation reaches 113/128
compact action choices; statistics plus answer loss reaches 111/128. Their common
starting compactor scored 103/128. Task minus statistics is −1.56 points with
paired world-bootstrap interval [−3.91, 0]. This does not establish a benefit from
the answer-loss term. Its lower compact target NLL does not imply higher choice
accuracy. Raw action accuracy remains 128/128.

All 1,728 mean-view rows and all 1,408 non-persistent raw/control rows are exactly
identical between the two forks. Only the fitted code changes. The native BF16
preflight reproduced the full causal prefix-plan task loss and every compactor
gradient exactly; base gradients remain absent. Full resume files and feature
caches are external, with no backbone copies or periodic saves.

Freeze these endpoints for a new-world comparison with the original compactor,
raw payloads and mean-plus-mass. Continue to report candidate-free answers and
counterfactual sensitivity separately from teacher NLL and choice accuracy.
