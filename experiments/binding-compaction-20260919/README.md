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

## Completed initial stored comparison

Both fits and initial stored evaluations completed from `868a1b3`. On the same
32 new worlds, MLP action choice accuracy is 128/128 raw, 96/128 mean-plus-mass,
and 103/128 fitted. Fitted minus mean is +5.47 points, paired world-bootstrap
interval [1.56, 10.16]. Attention is 127/128 raw and 128/128 with either compact
code. The two source models have different training budgets; this result does
not isolate reader architecture.

Each compact full-pair read uses one BF16 value plus FP32 multiplicity: 4,100
logical tensor bytes, 4,236 serialized code bytes. There are 64 persisted codes
per heldout view. Permission, restoration and identifier questions select single
children and retain exact raw fallback. Their scores therefore do not establish
compact-code capability. All raw values remain in the store.

Both fits together occupy about 43 MiB externally after stored result files;
each complete compactor/optimizer resume state is about 9 MiB. The initial source
config still contains inherited training flags, but the actual optimizer owns
only SyntheticCompactor parameters and every base parameter is frozen. The
recorded launch and script identify this scope. Later source versions make those
metadata explicit and add compute-only stack diagnostics/progress reports.

Counterfactual and candidate-free confirmation is running separately from
`e69de72`, under `/archive/probes/posthoc-compaction-confirmation-20260919`.
