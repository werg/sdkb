# MLP world-scoped learned selection (protocol prepared)

The longer selected-support MLP now composes both rule facts correctly. Test whether
verified group supervision can teach its existing key/query path to select records
for the requested entity among the four records in a world. This is candidate-local
retrieval, not global retrieval or learned authorization.

Both arms warm-start the identical completed 1,600-update recurrent-core MLP
continuation, reset Muon state, and run 800 updates with seed 37. Keep the same
training worlds, core-only adaptation, reader, BF16 payload, learning rates,
accumulation, sampled depths and text anchor. Both use learned top-two selection
with 100 oracle-warmup updates. The sole between-arm difference is routing loss
weight: zero versus one. Thus this tests explicit group supervision over task-loss
training of the same discrete-selection system. Both receive identical training
examples and support labels; the control does not optimize the routing objective.
The text anchor receives required supports in both arms. Fact questions need one
record, but learned reads have a fixed two-record budget; report this distinction.

Use the existing sequential runner in `binding-continuation-20260919/run.py`.
Direct external output, cadence 1,000 plus initial/final/emergency, retention two,
offline W&B and inherited host/device guards apply. No periodic checkpoint is
expected inside this 800-update budget. Exact resumes preserve optimizer state;
the initial fork explicitly resets it in both arms.

At the fixed final endpoint, create 32 new shared worlds after both arms complete.
Evaluate both using `evaluate_binding_context.py --learned-world --read-budget 2`:
actual exact key ranking within supplied world eligibility, complete-support recall,
action/fact/identifier choice, support removal with reranking, zero payload and
counterfactuals with original plans held fixed. The named selected-pair control is
an oracle upper bound, not learned selection. Compare world-paired outcomes and
retain episode/checkpoint hashes. Evaluate the common source on the same worlds.
Do not choose an endpoint using held-out performance. One seed and synthetic worlds
cannot establish agent success, arbitrary entity binding or parameter substitution.

Preparation does not launch training. The full-graph/replay test covers the new
combination of in-loop learned top-two selection, competing records, serialized
BF16 payloads, live key/query gradients, and checkpointing on/off.
