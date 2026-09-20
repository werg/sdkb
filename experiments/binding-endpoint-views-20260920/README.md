# Memory-conditional endpoint training

The matched breadth study improves identifier NLL but leaves exact generation at
0/64. Test a remaining shortcut: each training query previously had one fixed
endpoint, so the learner could associate a query with its answer without decoding
the endpoint from memory.

Expand the prior 128-world fresh training corpus into eight alternate prior
histories per query world. Keep the query, rule bits, source times and non-identifier
answers fixed; give each history distinct endpoint strings and immutable source /
episode version IDs. Every identifier query now has eight answers, each supplied
by its corresponding prior source. Retain a base-source/view/text-hash provenance
map. These are separate oracle-conditioned histories, not a single chronological
live namespace or a learned-global-routing benchmark. Each read sees only its own
history's required sources; versions are never combined as extra evidence.

The resulting 1,024 snapshots have 10,240 questions and 2,048 endpoint-bearing sources,
matching the completed 1,024-independent-world control's corpus size. Warm-start
from the same original reader checkpoint, with the same seed 79, 1,600 Muon
updates, four microbatches, learning rates, task proportions and loop schedule.
Only the training corpus changes. This isolates conflicting query-only targets
and repeated query identities at the same update and endpoint-source-count budget.
The fixed generator produces 2,047 distinct strings because of one cross-query
six-hex collision, versus 2,048 in the control; every individual query still has
eight distinct targets. This small content difference is retained and disclosed.

Compare the frozen source, completed independent-world control and new endpoint-view
arm on a newly declared 32-world held-out split `endpoint-views-heldout-20260920`.
Use the pinned oracle evaluator with all-world free generation, no/zero payloads,
support removal, rule changes and endpoint-only changes. Do not use the already
inspected breadth confirmation for endpoint selection.

All outputs go to `/archive/runs/binding-endpoint-views-20260920`; initial/final/
emergency checkpoints only, with full main-trainer resume state and offline W&B.
No architectural changes, target-derived query inputs, downloads or teacher calls.

The new arm completed all 1,600 updates; `endpoint.json` pins its final manifest
and model hashes. Its final saved sampler state and every sampled recurrent depth
match the independent-world control exactly. The new histories therefore change
the conditional training examples at matched sampling positions and update budget.
Fresh confirmation is complete; results follow below. After training ownership released, the identical
initial weights were verified and linked to the original source, reclaiming another
1,017,349,676 bytes without dropping recovery state or changing checkpoint paths.


## Fresh held-out confirmation

| Frozen endpoint | Action exact | Identifier exact | Identifier target NLL | Action without memory | Action with zero values |
|---|---:|---:|---:|---:|---:|
| Original source | 128/128 | 0/64 | 4.7557 | 63/128 | 62/128 |
| 1,024 independent worlds | 128/128 | 0/64 | 2.7963 | 64/128 | 49/128 |
| 128 worlds × eight endpoint histories | 128/128 | 0/64 | 2.6551 | 46/128 | 56/128 |

Every arm also generates all permission/restoration answers correctly, preserves
128/128 action answers under permission changes and 64/64 affected restoration
pairs, and makes no action changes on the 64 unaffected restoration pairs or 128
endpoint-only changes. All three still generate zero exact endpoint strings under
endpoint counterfactuals. Wrong identifier predictions change with the supplied
endpoint in 10/64, 46/64 and 55/64 cases respectively; this is sensitivity, not
successful recall. Irrelevant permission changes also alter identifier predictions
in 48/64, 38/64 and 36/64 cases. Endpoint-specific behavior is not cleanly isolated.

Thus this matched continuation preserves synthetic composition and slightly lowers
identifier teacher NLL, but does not solve latent exact-detail recall. Removing the
fixed query-to-answer association alone was insufficient at this update budget.
This does not rule out a longer curriculum or different representation. The
comparison is one seed on 32 fresh worlds; it establishes neither agent success
nor parameter substitution. `confirmation.json` pins input/result/bank hashes and
all condition counts; `independent-vs-views-generation.json` records the paired
world bootstrap comparison. Raw records and predictions remain on external disk.
