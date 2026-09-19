# Fresh stored confirmation of learned neighborhood cardinality

Select the 130-parameter reader-query count head using the earlier development
comparison, then freeze it. Create 32 new worlds after all fitting and policy
selection. Compare full-state routing with fixed two records, fixed one record,
and learned one/two records; also compare compressed routing with the same count
policy. Both address endpoints are the original broad 800-update probes, not the
later STOP continuations. No address/backbone/reader weights change here.

The count policy receives only the causal reader query. Required-set annotations
are used for evaluation, never to choose counts. Keep world-scoped eligibility,
exact stored-key ranking, oracle supports, zero/no-memory, support removal and
counterfactual controls. Generate without candidate answers on the first eight
worlds. Oracle supports deliberately retain their full set under every control;
the fixed-one intervention restricts learned search, not the oracle ceiling.

Require identical oracle/no-memory rows and payload/provenance bytes across all
arms. Report requested cardinalities separately from correct selected records,
choice accuracy and generated answers. This finite one/two-record classifier is
not a general learned stopping policy, global retrieval or an agent capability.
All banks and small adapter states are external; no backbone copies are made.

Native validation from `3eb4e49`: ordinary model preflight passes; count-policy
smoke has 20/20 correct cardinalities, bit-identical offline writer outputs and
zero writer calls during stored reads. The full core suite passes 281 tests.

## Completed confirmation

All four arms completed from `44927b8`. The count policy requests two records for
all 128 action queries and one for all 192 single-record queries on the fresh
32-world split. This measures count prediction, not correct entity addressing.

| Choice accuracy | Fixed two, full router | Fixed one, full router | Learned count, full router | Learned count, compressed router |
|---|---:|---:|---:|---:|
| Action | 70.31% | 52.34% | 70.31% | 49.22% |
| Permission | 67.19% | 78.13% | 78.13% | 70.31% |
| Restoration | 75.00% | 92.19% | 92.19% | 75.00% |
| Identifier | 31.25% | 29.69% | 29.69% | 31.25% |

Learned count versus fixed two improves permission by 10.94 points, paired
world-bootstrap interval [4.69, 18.75], and restoration by 17.19 [7.81, 26.56].
Action outcomes are identical. Full versus compressed routing with the same count
policy gains 21.09 action points [10.16, 32.81]. These are one training seed and
one new 32-world split; intervals do not represent training-seed variability.

Candidate-free first-eight-world results for the full router with learned counts
are action 19/32, permission 12/16, restoration 14/16 and identifier 0/16. Action
zero/no-memory controls are 16/32 and 15/32. This remains unreliable action
composition through learned routing, despite the preserved 100% oracle-action
ceiling. On action counterfactuals, both original and changed answers are correct
for 78/128 permission pairs and 38/64 restoration pairs; restoration falsely
changes 12/64 answers whose correct answer should stay unchanged.

All arms retain exactly the same 640 oracle/no-memory rows and 384 serialized
payload/provenance records. All artifacts are external, with no backbone copies.
The finite one/two classifier is useful here; arbitrary-count stopping, global
retrieval, unseen identifier recall, agent success and parameter substitution
remain unestablished.
