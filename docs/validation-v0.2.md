> Historical evidence/development document. Current SDKB operations: [training](training.md), [Spark](spark.md), [validation](validation-v0.3.md).

# Development validation — v0.2

Date: 2026-09-19. This report supersedes the original bootstrap report for the new
code paths. The original architecture document and v0.1 evidence are retained.

## Scope and environment

Executed on x86_64 CPU, Python 3.13.5, PyTorch 2.10.0+cpu. No CUDA device or
Transformers installation was available; dependency downloads failed in this
execution environment. **No real LFM checkpoint, Spark GPU, NVIDIA container,
ARM64 runtime, disk ANN, or production throughput result is claimed.**

The tiny backend uses a native causal byte-token transformer, not a classifier fed
fact labels. Source text enters the same writer interface used by the HF adapter.
The main study used width 48, one decoder layer, four heads, two write/read slots,
a 64-coordinate BF16 stored payload, and a two-round width-48 MLP set reader.
Training seed 17 was used throughout. The LFM model ID present in a tiny config is
inactive; the actual resolved model revision is `local-tiny`.

## Correctness tests and integration exercises

**117 tests passed; one optional Transformers/LFM structural test was skipped.**
The final test transcript and JUnit report are in `experiments/development-v2/`.
Coverage now includes:

- Full-graph versus selective-producer-replay gradients for paired raw/compact
  objectives and state-dependent multi-read trajectories.
- Periodic checkpoint recovery with optimizer/RNG state and the stale training
  cache. An interrupted/resumed run is bit-identical to uninterrupted training,
  including loss rows, after injecting uncommitted cache writes and log fragments.
- Failure before checkpoint-pointer publication, checksum corruption detection,
  and compatible warm starts with a new compactor.
- Compactor-only training leaves every non-compactor tensor unchanged.
- Exact additive streaming versus materialized reads for MLP and attention,
  empty sets, uneven chunks, and revocation between passes.
- Persistent full-cluster reads, strict version/domain/time checks, deletion
  lineage, partial-selection fallback, and tests that prohibit writer and
  compactor calls during stored inference.
- Write-once/many-query datasets, source deduplication, stateful follow-up reads,
  explicit sum-versus-mean sequence scoring, and counterfactual metadata parity.
- Tiny one-loop identity, zero-gated two-loop identity, causality, and nonzero
  writer/value/reader/gate gradients, including the multi-space probe.

A seven-arm, two-step CPU matrix completed for oracle text, no memory,
source-independent shared compute, MLP memory, attention memory, one reader round,
and two scheduled reads. This validates orchestration, not comparative capability.
The complete reproduction script also passed its four-world/one-step smoke mode,
including warm-start training and offline persistent-code evaluation.

`compileall`, shell syntax, all standalone YAML configurations, and Git whitespace
checks were exercised. Ruff was not available locally; its existing CI check is
retained. Tests do not establish numerical parity on BF16 CUDA or ARM64 hardware.

## What the learning experiment actually showed

### Data and protocol

Each world contains two independent bit facts in separate natural-text supports.
Queries ask for fact A, fact B, or their XOR. Values are balanced across worlds;
world names and record IDs are opaque hashes. Required-ID annotations guide oracle
routing but are never inputs to the writer, reader, or decoder. Retrieval order is
fixed by the intervention plan, not by the answer.

The training bank has 256 worlds; the exploratory development set has 64 separately
named worlds. After choosing the bootstrap procedure, a separate confirmation set
of 256 new worlds was generated, with three queries per world. All weights are
frozen for evaluation. Each of its 512 support experiences is encoded **once**,
serialized as BF16 values, reopened, and reused across its queries. Main scores
use full answer-sequence probability including EOS, with equal-length 0/1 choices.

This is a one-seed exploratory synthetic study. It is not a preregistered benchmark,
a result on coding/agent tasks, learned routing, or evidence of capacity substitution.

### Failed controls are retained

| Training condition | Optimizer steps | Development result |
|---|---:|---|
| Latent memory, XOR only, cold start | 1,000 | XOR 50.0%; no-memory 56.25% |
| Latent memory, A/B/XOR, cold start | 1,000 | A/B/XOR all 50.0% |
| Oracle support text, A/B/XOR | 1,000 | A 96.875%; B 98.4375%; XOR 70.3125% |

These runs motivated a bootstrap rather than an unsupported claim that the initial
architecture already learned. The text control demonstrates that this tiny
controller can use explicit facts on this distribution; it does not isolate one
universal explanation for the latent cold-start failures.

### Warm-started stored memory: partial transfer

The oracle-text model initialized the shared student; its backbone was frozen.
The writer heads, memory interfaces, and reader then trained for 1,500 steps on
mixed A/B/XOR queries. There is no support text in the later latent read prompt.

| Confirmation condition | A accuracy | B accuracy | XOR accuracy |
|---|---:|---:|---:|
| Stored latent supports | 81.640625% | 82.421875% | **66.015625%** |
| No memory | 50.78125% | 52.34375% | 50.0% |
| Same read plan, zero value payloads | 50.0% | 50.0% | 49.609375% |
| Drop A support | — | — | 49.609375% |
| Drop B support | — | — | 50.390625% |

On XOR, the paired gain over no memory is 16.015625 percentage points. A 10,000-draw
world-level paired bootstrap gives [11.328125, 20.703125] percentage points. This
interval resamples these worlds, **not independent training seeds**, and does not
cover model-selection or other workloads.

Both supports matter to aggregate XOR accuracy. However, **counterfactual behavior
is not yet reliable**. Under a source-bit flip with query/IDs/time/order fixed,
both the original and counterfactual XOR answers are correct in only 32.421875%
of A-flip pairs and 31.640625% of B-flip pairs. An XOR bit flip always changes the
correct answer; many model predictions do not change appropriately.

Conclusion: a useful partial latent-transfer/joint-use signal, not solved or robust
compositional reasoning. The next capability milestone remains reliable causal
behavior across intervention pairs and harder binding/task distributions.

### Compactor-only preservation on the same confirmation set

Freeze the entire warm-started system and train only the amortized synthetic-record
compactor for 300 steps, using the training worlds' XOR tasks. Its objective combines
compact task loss, detached-teacher behavior matching, and conditional contribution
matching, while retaining the raw branch.

| XOR read representation | Accuracy | Mean target token NLL |
|---|---:|---:|
| Two original stored records | 66.015625% | 0.3395501 |
| One temporary synthetic record | 66.015625% | 0.3399583 |
| One serialized persistent synthetic record | 66.015625% | 0.3399581 |

All **71 existing model-state tensors** are exactly unchanged by this phase; the
only additional trained state is the compactor's 10 tensors. The separate weight
audit is recorded in the evidence directory. Persistence stores BF16 synthetic
values and FP32 multiplicities; the actual read does not run the compactor.

The 256 XOR pairs become 256 full-cluster codes. Single-fact queries select only
one child and deliberately fall back to its raw stored value. Consequently, the
unchanged A/B accuracy is **not** evidence that one code handles arbitrary subsets.

For a fully selected pair in this implementation:

| Accounting item | Raw pair | Persistent code |
|---|---:|---:|
| Reader input records | 2 | 1 |
| Value coordinates, BF16 | 128 | 64 |
| Value bytes plus explicit code multiplicity | 256 | 132 |
| Serialized safetensors value blobs | 400 bytes | 268 bytes |

Keys, parent markers, SQL metadata/page reads, cache behavior, and index traversal
are excluded from those payload figures. **Original records remain for partial
selection and deletion-safe fallback, so total disk usage increases.** No net disk
compression, latency gain, or VRAM frontier has been measured. The result is
behavior preservation with fewer active reader records on a narrow task family.

## Reproduction and evidence

```bash
# Fast execution check: four worlds, one step per stage.
python scripts/reproduce_boolean.py --output runs/repro-smoke --smoke

# Full diagnostic including the failed controls and the bootstrap.
python scripts/reproduce_boolean.py --output runs/repro-study
# Recover a stopped run from its last committed complete optimizer step:
python scripts/reproduce_boolean.py --output runs/repro-study --resume
```

The script runs synchronously and records commands, stage configs, training logs,
choice evaluations, interventions, and compaction. Reproduction assumes the same
software/numerical execution environment; exact floating-point trajectories across
hardware/PyTorch versions are not guaranteed.

`experiments/development-v2/` includes the actual training configs/metrics, dataset
hashes, selected checkpoint manifests, summary reports, compressed per-query rows,
weight audit, matrix smoke report, and reproduction-script smoke report. Full model
weights and mutable caches are intentionally excluded from Git.

These runs were executed while developing v0.2. Their original environment reports
identify the inherited v0.1 commit; they do **not** claim that v0.2 was committed
before the measurements. The delivered source snapshot and tests capture the
implemented paths. Future reports explicitly record working-tree dirtiness.

## Next evidence to obtain on Spark

Run `model-probe` with a pinned actual LFM checkpoint before expensive training.
Then execute the common-data matrix, inspect oracle-text ability, and compare
cold versus controlled warm-start latent training. Increase independent training
seeds, binding complexity, exact-detail queries, and counterfactual consistency.
Compaction should be evaluated on successful readers, including exceptions and
selection patterns; preserving a weak model's accuracy is not the final goal.
