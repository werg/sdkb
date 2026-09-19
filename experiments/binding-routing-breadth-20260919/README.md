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

## Completed feature-space result

All four arms completed from `2c05b93`; paired evaluation used `ac15b2c` and
reproduced every aggregate count exactly. Full required action-pair recall on
32 common unseen worlds is 27/128 (narrow compressed), 48/128 (narrow full-state),
22/128 (broad compressed), and 66/128 (broad full-state). For the full-state head,
breadth improves recall by 14.06 points, paired world-bootstrap interval
[3.13, 25.00]. With broad data, the full-state head beats the compressed head by
34.38 points [24.22, 43.75]. These are one-seed, unequal-parameter feature probes.
Absolute recall remains 51.56%; downstream behavior is not yet measured.

The broad full-state arm's top-one required-record rates for permission,
restoration and identifier queries are 46/64, 58/64 and 43/64. Source-kind recall
alone would hide these remaining entity-selection errors. The observed attempt
peak CUDA allocation was about 1.90 GiB, including feature extraction. All raw
features, selected-ID rows and small optimizer states remain external.
