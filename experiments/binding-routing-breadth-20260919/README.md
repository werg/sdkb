# Routing feature breadth study

Follow up the frozen-feature probe with a common held-out set and a 2 × 2 design:
128 versus 1,024 newly generated training worlds; compressed versus adaptable
full-state query features. The first 128 worlds are a prefix of the broad dataset.
All four arms start from the same successful core-only MLP1600 checkpoint and use
800 native Muon updates of 1,280 queries sampled with replacement per update.
Thus the update count and query exposures match across dataset breadth. The two
feature variants have different trainable parameter counts, explicitly reported.

Both dataset splits are distinct from earlier training and diagnostic worlds.
Features use only offline source writes and the first causal prompt query. The
original reader/decoder is frozen and unused in the feature loss. This remains a
routing learnability study, not evidence of downstream stored-memory capability.
Inputs, script hash, optimizer state and sampling RNG are preserved externally;
no large backbone checkpoints are copied. Source extraction reports progress and
can be repeated after a stop. The small fitting states save only on final/stop.
Inherited CUDA allocation and shared-host-memory reserve checks are applied.
