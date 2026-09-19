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
