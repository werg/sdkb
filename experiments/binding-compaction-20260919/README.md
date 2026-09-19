# Frozen-reader post-hoc compaction protocol

Start with useful oracle-composition checkpoints and freeze every base parameter.
Fit the same query-independent two-to-one synthetic-record compactor separately
for MLP and attention. The readers' source training budgets differ, so this is
not a matched architecture contest. Both receive the same compactor opportunity.

Use the original 128-world training file and a new fixed 32-world heldout split.
Training features contain only serialized stored payload pairs and causal first
boundary queries. Optimize conditional pre-normalization numerator and mass at
each raw reader state, plus a free compact-reader rollout. Cast compact values
through actual BF16 storage precision in every training forward. Use 400 Muon
updates, batch size 32, seed 59 and learning rate 1e-4; non-matrix parameters use
AdamW. A single code retains total multiplicity two.

Compare the fitted code with mean-plus-mass at the same stored byte budget.
Persist codes offline, disable both writer and compactor, then evaluate native
recurrent reads. Partial selections use exact raw fallback. Record raw and compact
behavior, support-removal controls and actual code/storage bytes. Retaining raw
fallback means no net disk savings claim. Counterfactual and candidate-free
confirmation are required before claiming retained useful composition.

All banks, feature caches, W&B files and small compactor states live externally.
Save initial, final and requested emergency state only, including named optimizer
ownership, hyperparameters, source identity and Torch/CUDA/sampling RNGs. No
backbone checkpoints are copied. Resume rejects changed run identity.
