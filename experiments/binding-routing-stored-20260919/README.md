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
