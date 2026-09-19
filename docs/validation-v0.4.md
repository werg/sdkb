# SDKB 0.4 validation

Date: 19 September 2026. This release adds native middle-block recurrence, live
anchored re-entry, reads between shared-core passes, a conversion curriculum, and
fixed-writer depth evaluation. See [recurrence.md](recurrence.md) for research and
operational detail.

## Executed locally

**165 tests passed; 3 optional native Hugging Face tests skipped.** The environment
is Python 3.13.5, PyTorch 2.10.0+cpu, x86_64, with no CUDA or Transformers. A targeted
dependency-install attempt could not obtain the optional packages. Ruff could not
be executed locally; its existing CI check is retained. Compile-all and
`git diff --check` completed successfully.

The additional tests cover:

- Exact one-pass hidden output and input-gradient equality with the tiny parent.
- Native layer call order, one final normalization, and shared parameter identity.
- Live bridge/input/update/memory gradients, including a frozen parent.
- Query-prefix independence from changed future targets at every read boundary.
- Later-query dependence on earlier payloads; no memory effect before its arrival.
- Integrated training versus prefix-planned stored-only scoring for both readers.
- Full-graph/selective-replay gradients across multiple reads and checkpointing.
- Independent fixed writer depth, core-only freezing, explicit old-checkpoint
  conversion, fixed-parent KL, and exact resumed depth-sampling training.
- Four-stage execution/resume and a depth sweep that forbids further writer calls.
- BF16 CPU execution; PyTorch emits a mixed-dtype RMSNorm fused-dispatch warning
  and uses its available implementation. This is not a CUDA performance result.

The optional tests are the pre-existing native LFM embedding test and new random
LFM/Llama layer-splitting tests. They require the pinned Transformers extra. They
are not described as passed. No model weights are downloaded by those random-model
tests.

## End-to-end execution

`recipes/tiny_looped_smoke.yaml` ran all four stages with one optimizer step each:
text bootstrap, frozen-parent bridge training with distribution anchoring, latent
warmup, and a joint recurrent-core update. It then ran stored-only teacher evaluation,
fresh Boolean counterfactual evaluation, and multi-binding evaluation. Repeating
the launcher with `--resume` completed without repeating stages.

The initial tiny-model preflight measured one-pass maximum error **0**, causal
prefix maximum error **0**, and a nonzero two-pass state difference (~0.0347).
Writer, reader and all tested bridge parameter groups had finite nonzero gradients.
This validates live computation, not useful reasoning.

A separate command wrote two support records once and evaluated 1/2/3/4 core passes
from the same frozen checkpoint and stored bank. For the four-layer test model,
layer visits were 4/6/8/10. The sweep records memory-disabled R=1 separately and
marks depths beyond the final main training range. Its wall times include all
conditions and plan/scoring work, not production serving latency.

Metrics and execution evidence are under `experiments/recurrence-v0.4/`. One update
per stage is deliberately insufficient for a scientific capability claim. Earlier
v0.2 partial-transfer results remain historical and are not attributed to this new
recurrent decoder.

## Not executed here

Actual LFM2.5-230M or SmolLM2 checkpoint loading; native LFM/Llama tests; the Docker
build; ARM64/CUDA training; live teacher-dataset ingestion in this release; remote
GitHub CI; improved capability, capacity substitution or latency benchmarks. The
Spark launcher performs an actual-checkpoint numerical/gradient preflight before
training. Its failure is a stop condition, not something bypassed by the CPU tests.

## Source handoff

The release preserves Git history from v0.3 and includes the updated Markdown plan,
configuration files, tests and execution records. Changes are committed locally.
The connected GitHub repository was inspected and remained empty; this release
was not pushed. Use the Git bundle or non-forced publication commands in
[handoff.md](handoff.md). No model checkpoints or downloaded teacher corpora are
included in the source release.
