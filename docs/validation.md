# Bootstrap validation

This file records executable checks from the bootstrap environment. Final numerical
results and the exact implementation commit are recorded below before handoff.

## Scope

CPU Linux x86_64, Python 3.13.5, PyTorch 2.10.0+cpu. No CUDA device, no Transformers
installation and no model downloads. Real LFM weights, Spark hardware and Docker
image execution have not been validated. The optional random-LFM architecture test
is skipped without Transformers; it is configured to run in CPU CI and on Spark.

The editable package was installed without downloading dependencies, and its `elm`
CLI was executed. Core unit tests cover readers, compaction, replay, storage,
causality, training/resume, general support/query import and stored-only evaluation.
Shell scripts receive syntax checks, not a claim of GitHub publication or Docker
build success. Ruff is configured in CI but unavailable in the bootstrap environment;
syntax compilation and an AST unused-import inspection were performed locally.

## Interpretation

The short tiny-model run is an optimization and serialization smoke test. A falling
training loss alone does not demonstrate useful latent memories. The evaluated
support/removal/counterfactual conditions must establish causal dependence. The
30-step run does not yet establish that dependence, and its raw numbers are retained.

The compaction probes use fixed **random** readers and synthetic tensors, not a
trained memory language model. Their target losses demonstrate that the code can
optimize the specified objective; they do not establish semantic compaction or rank
MLP against attention. The attention probe is retained even when its held-out
redundant-cluster loss does not improve.

The SQLite timing uses only 128 records and an uncontrolled warm page cache. It is
a runtime-path check, not an NVMe, ANN, throughput or VRAM-substitution benchmark.
