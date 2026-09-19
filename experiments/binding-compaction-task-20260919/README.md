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
