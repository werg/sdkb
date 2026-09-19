# Stored-memory confirmation of routing probes

Freeze the completed 1,024-world compressed-query and full-state-query address
probes. Compare them with their common successful core-only MLP1600 source on 32
new worlds generated after all fitting is complete. The small trained address
states overlay that verified source checkpoint; no backbone copies are written.
The independent head supplies only retrieval, preserving reader conditioning.

Use exact top-two learned ranking inside supplied world eligibility. Keep oracle
support, no-memory, zero-payload, support-removal and rule-counterfactual controls.
Generate without candidate answers on the first eight worlds, up to 24 tokens.
Then require exact equality of all serialized payload/provenance rows and all
oracle/no-memory scoring rows across the three arms. Report choice and generation
separately. This is neither global retrieval nor an agent or substitution test.

The native LFM preflight from `38a2634` passes with zero one-pass identity and
causal-prefix error, and a finite nonzero independent-routing gradient. Core
regressions cover full-graph/replay, target exclusion, stored-session parity,
exact Muon resume and frozen payload/reader behavior. The adapter loader checks
source checkpoint identity and parameter ownership. Generation checks the bank's
adapter identity so a different routing model cannot silently reuse its keys.

## Single-fact read-budget diagnostic

Prepared before seeing the full-state stored results: compare fixed read budgets
one and two on the permission/restoration subset of these same 32 worlds, using
the full-state adapter and its existing bank. Generate without candidates with
real, zero and no payloads. The routing objective rewards a correct top-one item
for these single-required-record questions, while fixed top-two execution can
also return a conflicting entity's fact. This diagnostic tests whether that extra
record interferes. It does not implement or claim an adaptive read-count policy,
and it excludes two-record action questions explicitly.

## Completed stored confirmation

All three arms completed from `23d54fc` on 32 new worlds. Action choice accuracy
is 52.34% for the source, 48.44% for compressed routing, and 60.94% for full-state
routing. Full-state minus source is +8.59 points with a paired interval
[−3.91, 20.31]; full-state minus compressed is +12.50 points [0.78, 24.22]. The
source comparison does not establish an improvement. Candidate-free actions are
14/32, 16/32 and 17/32, respectively; all zero/no-memory action controls are 16/32.
Unseen identifiers remain 0/16. This is not reliable learned action composition.

The native preservation gate passes: 640 oracle/no-memory rows and 384 serialized
payload/provenance records are identical across all three arms. Oracle action
accuracy remains 100%. No extra full-model checkpoint was written.

The planned fact-budget diagnostic completed from `0c6c875`. With the same full-state
router, reader and stored bank, one record versus two improves candidate-free
permission answers from 45/64 to 49/64 (+6.25 points [1.56, 12.50]) and restoration
from 45/64 to 59/64 (+21.88 points [12.50, 31.25]). Zero/no-memory accuracies are
unchanged. The first record is actually required in 40/64 permission and 55/64
restoration queries; answer accuracy alone overstates addressing correctness when
two entities share a rule.

Among opposed-rule worlds, one-record reads answer both entities correctly in
3/18 permission worlds and 15/19 restoration worlds, versus zero jointly correct
worlds for both families with two records. This demonstrates a narrow restoration
binding behavior and interference from the extra record. Read count was a fixed
experimental intervention on a single-fact subset; an adaptive stopping policy
remains unimplemented. The action experiment retains its two-record budget.
