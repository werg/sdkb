# One fixed identifier question, many supplied memories

The eight-history continuation still generates 0/64 exact held-out identifiers.
Each of its query identities nevertheless has only eight possible targets. Test a
stronger communication bootstrap: ask exactly the same identifier question for
all 2,048 endpoint targets, with the answer determined by the supplied record.

Transform only the identifier query text in the completed 1,024-independent-world
training corpus. Keep all source IDs, source text, times, targets, required sources,
family order, other questions and source access unchanged. Assign new opaque episode
IDs to changed queries and retain parent/query-hash provenance outside neural inputs.
The fixed question is: “What endpoint is specified by the selected earlier record?
Return only the endpoint.” No target, world identity or source ID enters that question.

This explicitly relies on oracle selection of one prior source. The fixed query
cannot specify a global retrieval target; this is a communication diagnostic, not
a deployable retrieval policy. Positive exact recall would show decoding of supplied
latent content under this bootstrap, not resolution of address generalization.

Start from the same original useful MLP checkpoint as the independent-world control.
Match seed 79, 1,600 Muon updates, four-example accumulation, learning rates, task
proportions and two/three-loop sampling. The only scientific change is the identifier
question transformation. Verify identical initial weights and final sampler/depth
state. Shorter questions change token compute; equal updates are not equal FLOPs.

Before training, declare 32 fresh held-out worlds in both original and fixed-query
forms. Compare the completed independent-world control and the new endpoint on each
form, with full stored-only free generation, no/zero memory, source removal and
permission/restoration/endpoint counterfactuals. Reporting both forms distinguishes
the easier interface from transfer back to entity-specific questions. Exact endpoint
matches are primary for this diagnostic; formatting, NLL and wrong-prediction changes
remain separate. Sources/endpoints are disjoint from both training corpora and the
preceding endpoint-view/paired confirmation splits.

Use `/archive/runs/binding-fixed-query-detail-20260920` for every artifact. Save only
initial/final/emergency checkpoints, keep two, preserve complete resume state and
offline W&B, and retain the 10 GiB disk / 8 GiB host reserve settings. NVIDIA Torch
is unchanged. No data/model downloads, paid teachers or source-command execution.

Preparation uses `scripts/make_fixed_query_identifiers.py`; regression tests verify
immutable sources/targets/causal metadata, versioned query identities, deterministic
indexed output and rejection of future or misattributed answers. Full suite before
launch: 405 passed; Ruff clean.


## Completed training and numerical comparability

All 1,600 updates completed. The final sampler state matches both the independently
reconstructed plan and the independent-world control, as do all 1,600 sampled
recurrent depths. There are 1,285 identifier microbatches and 941 distinct identifier
query episodes, matching that control. Thus this budget exposes fewer than half of
the corpus's 2,048 identifier examples directly as identifier targets; the sources
also participate in other task families. Do not interpret a negative result as
failure after exhaustive copy training. `endpoint.json` records hashes and resources.

An isolated native BF16 Muon update on unchanged control data compares the older
control checkout (`602cb06`) with the new training checkout (`1c261c7`). Initial and
post-update model tensors, gradients, optimizer/state ownership, sampler and Torch/
CUDA RNG fingerprints match exactly. `code-parity.json` records the external reports
and hashes. This checks that executed code path; it is not a second full 1,600-update
control reproduction. The probe writes fingerprints instead of model checkpoints.

After training ownership released, verified hard-link deduplication reclaimed
1,017,349,676 bytes from the identical initial weights. Complete recovery state and
all paths remain intact. Fresh confirmation is running in both question forms.
