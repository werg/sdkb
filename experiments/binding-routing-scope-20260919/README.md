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
the process, environment, optimizer and first update. Completed capability and invariance results are recorded below.


Training completed all 800 updates. The initially queued comparison exited during
a startup registration race (`not_managed` before the trainer registered). The
waiting helper now gives initial registration a bounded 60-second grace period;
stop requests, failed runs and loss of already-registered state still fail promptly.
Four regressions and the full 259-test suite pass. Evaluation is restarted from a
new frozen checkout with this orchestration fix; training code and weights are
unchanged. The failed comparison created no evaluation worlds or result files.


The comparison restarted successfully from `46bcedb`, with new worlds created only
after training finished. Its numerical model/evaluation code is unchanged from
the declared source; only the startup wait was fixed. `completed-training.json`
and `evaluation-restart.json` preserve the final state and both attempt identities.
After training released its locks, the redundant initial weight copy was verified
and deduplicated against the source, preserving the complete initial checkpoint.


## Completed result

Training source `a60da7e`; evaluation source `46bcedb`. On 32 new common worlds,
projection-only routing scores 51.56% action accuracy, versus 51.56% for joint
routing and 46.88% for the source. Paired differences are 0 points (95% world
bootstrap interval [−9.38, 7.81]) and +4.69 points ([−5.47, 14.06]), respectively.
This does not establish an action improvement. Sufficient-support recall is
35.94%, and only 9/128 action queries retrieve both required records.

The preservation test passes exactly: 206 frozen tensors, 384 serialized payload
records and 640 oracle/no-memory scoring rows match the source. Oracle action
accuracy remains 100%, versus 68.75% after joint routing adaptation. Freezing the
composition path prevents its regression but does not solve entity addressing.

Candidate-free generation on the first eight worlds gives 14/32 correct actions,
versus 15/32 with zero payloads and 16/32 without memory. Unseen identifiers remain
0/16. Permission and restoration top-one entity selection is 31/64 and 32/64.
These are one-seed synthetic results with supplied world eligibility, not global
retrieval or agent success. Compact results and exact invariant checks are in the
adjacent JSON records; full rows, banks and checkpoints remain external.
