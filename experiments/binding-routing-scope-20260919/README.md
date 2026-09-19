# Projection-only routing scope (running)

The joint routing intervention failed action selection even on a small training
subset and regressed oracle-support composition. Isolate address learning from
changes to the frozen content/reader path. Start from the same successful 1,600-update
MLP checkpoint as that joint arm. Keep its seed 37, 800-update budget, Muon settings,
data, routing weight one, top-two budget, 100-update routing warmup, and other
scientific settings. Change only `train.optimization_scope` to `routing`; the W&B
group is a separate operational label.

Only `key_head`, `address_maps`, and `query_maps` parameters train. The shared
backbone, writer slots/value codecs, query feature extractor, reader and recurrent
bridge remain frozen. Frozen child modules run in evaluation mode; the top-level
training flag still honors routing warmup. This tests the learnability of an
address metric on fixed features. It is not a new text index or an entity oracle.
CPU regression checks all other weights, serialized payloads and oracle NLL remain
bit-identical, and checks exact model-state equality after Muon save/resume.

Use the existing sequential runner in `binding-continuation-20260919/run.py`.
Output is external, cadence 1,000, retention two, initial/final/emergency saves,
offline W&B and inherited resource guards. A model-interface preflight is required
before launch; native frozen-parameter/payload invariance must be checked at the
endpoint. No training success is claimed by preparing this protocol.

After completion, create 32 new common worlds and compare the source, prior fixed
joint-routing endpoint and this endpoint. Use exact top-two ranking within supplied
world membership; retain oracle support, zero-value, no-memory, support-removal
and counterfactual controls. Report sufficient-support recall separately from the
full required pair and from source-kind selection. Run candidate-free generation
on the first eight worlds, 24 tokens. One seed, synthetic tasks and world-scoped
eligibility do not establish global retrieval or agent success.


The native model-interface preflight passed from `67933bf`: zero one-pass identity
and causal-prefix error, finite memory/bridge gradients. It checks the model's
interface before optimizer scope is applied; the separate scope regression checks
frozen paths and exact Muon resume. At the endpoint, `verify_frozen.py` checks every
non-address tensor and compares all serialized payloads plus oracle/no-memory rows
against the source on the same new worlds. Passing this check is distinct from
learning successful routing.


Launched from frozen `a60da7e`, with its common-world comparison queued from the
same checkout. Native ownership reports exactly 73,728 trainable parameters in the
three address projections; initial model bytes equal the shared source. All weights
remain on the external disk. `launch.json` and `initialization-check.json` record
the process, environment, optimizer and first update. Capability and endpoint
invariance results remain pending.
