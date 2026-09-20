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
