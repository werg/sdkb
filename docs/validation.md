> Historical evidence/development document. Current SDKB operations: [training](training.md),
> [Spark](spark.md), [mutable-bank validation](validation-mutable-bank-v0.8.md), and
> [scale-out preparation validation](validation-scale-out-v0.9.md).

Adaptive routing changes require regression tests showing that task loss reaches
the query, distance gate, selected serialized keys and payloads, and writer
parameters through replay. Resume validation must verify the mutable bank's small
cursor/digest token and roll its revision journal back exactly; checkpoints do not
copy the bank SQLite file. Density-adaptive compaction tests
must conserve pre-normalization numerator and mass across overlapping fields. See
[adaptive memory v0.6](adaptive-memory-v0.6.md).

> Historical v0.1 bootstrap report. For the current implementation and executed
> learning/compaction study, see [validation-v0.2.md](validation-v0.2.md).

# Bootstrap validation

This file records executable checks from the bootstrap environment. Numerical results
and the implementation commit are recorded below.

## Scope

CPU Linux x86_64, Python 3.13.5, PyTorch 2.10.0+cpu. No CUDA device, no Transformers
installation and no model downloads. Real LFM weights, Spark hardware and Docker
image execution have not been validated. The optional random-LFM architecture test
is skipped without Transformers; it is configured to run in CPU CI and on Spark.

The editable package was installed without downloading dependencies, and its `sdkb`
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

## Executed results — 19 September 2026

Implementation commit: `b87e730a5b6cd7bb01bd7e0f35fe868cbabaa3f9`.
The following commit adds this report and artifacts without changing the implementation.

| Check | Result |
|---|---|
| Unit/integration collection | **75 passed, 1 skipped** (2.25 s on this CPU run) |
| Skipped test | Random LFM architecture integration; Transformers unavailable |
| Editable install and CLI | Installed and exercised |
| Tiny support/query training | 30 steps; finite gradients, checkpoint saved/reloaded |
| Tiny training loss | First step 5.624008; final step 4.131498 (different sampled tasks; not a held-out learning curve) |
| Frozen stored-only evaluation | 8 new worlds × 8 interventions; actual BF16 payload serialization/reload |
| Choice accuracy, all/none support | 0.125 / 0.125; **no memory-dependent accuracy benefit established** |
| Temporary synthetic compaction training | 12 steps; 5 steps used nonzero compaction loss |
| Compaction optimizers | 100 steps each for MLP and attention fixed-reader probes |
| SQLite harness | 128 records; exact scan and reload path exercised |
| Shell scripts | `bash -n` passed |
| Python syntax | `compileall` passed |
| Ruff | Not run locally; configured for CI |
| Spark/LFM weights/Docker | Not run; hardware/environment validation remains required |
| Remote GitHub creation/push | Not performed; available connector was read-only |

### Held-out probe losses

These losses live in each random reader's own feature space; they must **not** be
compared as a ranking between reader architectures. The sample is small and unseeded
variance beyond seed 7 has not been studied.

| Probe | Redundant-cluster loss before | After |
|---|---:|---:|
| MLP synthetic-record compactor | 0.00111555 | 0.00097559 |
| Attention synthetic-record compactor | 0.01531088 | 0.01545538 |

### Artifacts and reproduction

Raw environment, test output/JUnit, training metrics, eight-condition evaluation,
compaction probe outputs and SQLite timing are in `experiments/bootstrap/`.
Model weights and database caches are intentionally not committed.

```bash
python -m pytest -q --disable-warnings
sdkb train --config configs/tiny_cpu.yaml --output runs/reproduce-tiny --steps 30
sdkb evaluate --run runs/reproduce-tiny --count 8
sdkb train --config configs/tiny_compaction_cpu.yaml --output runs/reproduce-compact --steps 12
sdkb compact-probe --steps 100 --reader mlp --output runs/mlp-probe.json
sdkb compact-probe --steps 100 --reader attention --output runs/attention-probe.json
sdkb io-bench --path runs/reproduce-io.sqlite --records 128 --reads 8 --output runs/io.json
```

Wall-clock times and allocator counters are environment-specific. Neither timing
reproducibility nor a scientific capability result follows from deterministic CPU
unit tests. The next meaningful result is causal support dependence with the actual
pretrained student, not increasing this tiny model's step count and relabeling it
as a capacity-substitution experiment.
