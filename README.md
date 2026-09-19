# External Latent Memory

A research implementation of **transfer-trained stored soft-token memory**, recurrent
set composition, selective producer replay, and conditional cluster compaction.
The initial pretrained student is **LiquidAI/LFM2.5-230M**. Development targets a
single **NVIDIA DGX Spark**, with an offline CPU backend for correctness tests.

**Status: v0.2 research implementation with a partial CPU latent-transfer result.**
The tiny byte-model diagnostic reaches 66.0% held-out XOR accuracy versus 50.0%
without memory; either missing support returns it to chance. Counterfactual flips
remain inconsistent, so this is not robust composition. A separately trained
compactor preserves that score using one stored synthetic record instead of two.
No LFM/Spark capability or deployment-resource result has been established.

See [v0.2 validation and raw evidence](docs/validation-v0.2.md),
[development guide](docs/development-v0.2.md), and the
[implementation boundary](docs/implementation.md).

## Architecture and project map

The governing design is [docs/architecture.md](docs/architecture.md), copied from the
latest review-integrated Markdown plan. This repository does not replace its
research hypotheses with claims that they are already demonstrated.

```text
prior support trajectory
  -> shared student + learned write slots
  -> one canonical vector key + fixed canonical value slots
  -> per-space address/value transforms
  -> versioned SQLite / safetensors payload bank

causal query prefix -> query vector -> selected stored values
  -> recurrent fixed-slot pooled-MLP reader (or attention control)
  -> soft-token tool-result equivalent -> shared student -> later-task loss

training: downstream cotangents -> replay only contributing live producers
inference: stored values only; no trajectory re-encoding on the read path
```

| Component | Implemented now |
|---|---|
| Shared writer/controller | Fixed write slots, vector-key head, per-space codecs, answer loss through soft-token inputs |
| Dense set reader | Factorized conditional MLPs, output-slot identity, repeated pooled residual feedback, gated numerator/mass |
| Attention comparison | Fixed-slot cross-attention with the same residual updater and additive numerator/mass compaction targets |
| Multiscale reader | Different payload widths and neighbor counts; one common recurrent residual state |
| Selective replay | RNG/autocast replay, accumulated leaf cotangents, shared-parameter gradients, cached/live mixtures; full-graph parity tests |
| Reader memory | Chunk-checkpointed training; single-space stored-only streaming with bounded device payload staging |
| Compaction | Paired raw/compact losses, conditional contribution/KL targets, compactor-only warm starts; persistent full-cluster codes with raw partial fallback |
| Compactability | Integrated local/random/overlap grouping, mass conservation, merge regularization, storage noise, and synthetic-record replacement |
| Retrieval | Global-bank exact CPU search, complete-group supervision, scheduled state-conditioned follow-up reads with exclusion |
| Storage | Versioned immutable records, authorization/time filtering, safetensors payloads, deletion lineage; asynchronous CPU retrieval utility |
| Recurrence | Optional tied **full-stack refinement** through public model APIs; one loop preserves the original path |
| Experiments | Write-once/many-use and Boolean counterfactual datasets; full-sequence scoring, world-level paired intervals, seven-arm runner, stored-only persistent evaluation |
| Development | Atomic checkpoint sets including stale cache/RNG, compatible warm starts, hardware preflight, Spark configs/container, CI and replay tests |

Not yet implemented: disk ANN, trajectory-wide adaptive read scheduling, early-layer
asynchronous query overlap, a hybrid-state loop cache, production serving,
arbitrary-subset compact codes, nested producer dependency replay,
teacher rollout collection, task verifiers, or a measured capacity-substitution frontier.
The asynchronous retrieval utility is **not** an implemented asynchronous LLM scheduler.

## Reproduce the new CPU diagnostic

```bash
# Tests need no model download.
python -m pytest -q
# Four worlds and one optimizer step per stage: execution only.
python scripts/reproduce_boolean.py --output runs/study-smoke --smoke
# Full study, including failed cold starts, text bootstrap, and compaction.
python scripts/reproduce_boolean.py --output runs/study
```

The study keeps all weights frozen during stored evaluation, writes each source
once, and tests new worlds. The reported run uses one training seed and oracle
retrieval. It is not a measurement on the 230M student. Original values are retained
for partial-cluster fallback: fewer active read bytes is not net disk compression.

## Quick start: offline CPU reference

Use a Python environment with PyTorch installed. On CPU-only Linux:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[dev]'
python -m pytest -q
elm doctor
elm train --config configs/tiny_cpu.yaml --output runs/tiny --steps 30
elm evaluate --run runs/tiny --count 8
elm compact-probe --steps 100 --reader mlp --output runs/compact-mlp.json
elm compact-probe --steps 100 --reader attention --output runs/compact-attention.json
```

The tiny backend has random weights and a byte tokenizer. It checks the pipeline;
it is **not** a replacement for the pretrained 230M student. Output directories are
created exclusively to prevent accidental run overwrites. Resume with the same
configuration and a larger `--steps` value plus `--resume`.

## DGX Spark: first real-model run

The container intentionally preserves NVIDIA's supplied PyTorch/CUDA build rather
than installing an x86 wheel, a CPU wheel, or an arbitrary replacement Torch build.
Run these commands **on the Spark** after its vendor-supported driver and NVIDIA
Container Toolkit are operational:

```bash
./scripts/spark.sh build
./scripts/spark.sh run elm doctor --require-spark
./scripts/spark.sh run python -m pytest -q

# Resolve the actual HF checkpoint to an immutable revision for the experiment.
./scripts/spark.sh run python scripts/pin_model.py \
  --config configs/lfm25_230m_spark.yaml --output runs/lfm-pinned.yaml

# Two-step integration check, not an expensive capability training job.
./scripts/spark.sh run bash scripts/smoke_lfm.sh \
  runs/lfm-pinned.yaml runs/lfm-smoke

# Prepare common-data controls; preparation launches no training.
./scripts/spark.sh run python scripts/prepare_matrix.py \
  --config runs/lfm-pinned.yaml --output runs/lfm-matrix \
  --steps 200 --worlds 64 --eval-worlds 32 --bindings 2 --seeds 17
./scripts/spark.sh run bash runs/lfm-matrix/run_all.sh
```

The base image defaults to `nvcr.io/nvidia/pytorch:26.08-py3`; override it with
`ELM_BASE_IMAGE` when the installed driver or image availability requires another
compatible NVIDIA image. The build and `doctor` checks are gates, not an assertion
that this image has been tested on the user's machine. See [Spark setup](docs/spark.md).

`ELM_CACHE_DIR` controls the host cache mount; default is `~/.cache/elm`. No server
ports, privileged container mode, host credentials, or Hugging Face tokens are
mounted automatically. The checkpoint is downloaded on first use and is not
included in this repository.

## Student and looping choice

[LFM2.5-230M](https://huggingface.co/LiquidAI/LFM2.5-230M) is a hybrid with
convolution and attention layers, not an attention-only decoder. The adapter uses
its public `inputs_embeds` path and disables both convolution and attention cache
reuse. It conservatively caps context at 32,768 tokens for this integration.

The primary baseline uses **one loop**. The `lfm25_230m_loop2_spark.yaml` variant
reuses the entire stack with a zero-initialized gated residual refinement. It is
an experimental conversion, not a claim that the checkpoint was pretrained as a
looped model or that shared parameters imply free extra computation.

A [SmolLM2-135M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct)
configuration supplies an attention-only alternative through the same interface.
Choose between these with measured oracle-text ability, numerical stability, and
wall-clock cost—not an assumption that the smaller model will compose better.
See [model decision](docs/model-decision.md).

## Configurations

Every YAML file is standalone. Unknown sections/fields fail rather than silently
using defaults. `model.revision: main` is a bootstrap default; pin it before runs.

| Configuration | Purpose |
|---|---|
| `tiny_cpu.yaml` | Offline full pipeline |
| `tiny_compaction_cpu.yaml` | Temporary synthetic-record replacement during training |
| `lfm25_230m_spark.yaml` | Main single-space oracle-routing baseline, 8 write/read slots |
| `lfm25_230m_wide_payload_spark.yaml` | Identity storage codec; isolates the extra write/storage bottleneck |
| `lfm25_230m_direct_latent_spark.yaml` | Deliver individual canonical latent records without pooling |
| `lfm25_230m_attention_spark.yaml` | Attention reader comparison |
| `lfm25_230m_compaction_spark.yaml` | Temporary full-neighborhood compaction; no persistent index rewrite |
| `lfm25_230m_paired_compaction_spark.yaml` | Paired raw/compact task loss and local grouping |
| `lfm25_230m_overlap_spark.yaml` | Mass-conserving overlapping-field compaction training |
| `lfm25_230m_multiread_spark.yaml` | Two causally state-conditioned scheduled reads |
| `lfm25_230m_streaming_spark.yaml` | Bounded-payload stored inference; synchronous repeated disk passes |
| `lfm25_230m_oracle_text_spark.yaml` | Information-availability control |
| `lfm25_230m_shared_compute_spark.yaml` | Source-independent query/reader/soft-position control |
| `lfm25_230m_learned_spark.yaml` | Two-of-ten retrieval, known-group supervision, oracle warmup |
| `lfm25_230m_multiscale_spark.yaml` | Optional four-space branch; not required for the first experiment |
| `lfm25_230m_loop2_spark.yaml` | Gated shared-stack recurrence |
| `smollm2_135m_spark.yaml` | Attention-only alternative |

The 256-coordinate stored payload in the main configuration is deliberately a
separate bottleneck from eight 1,024-wide canonical LFM write slots. Use the wide
payload variant to diagnose information loss at that boundary.

## Training with your own support/query data

Set `train.episodes_file` to a JSONL file of causally earlier experiences and later
queries. A minimal record is:

```json
{"episode_id":"query-001","query_time":20,"supports":[{"record_id":"experience-001","created_at":10,"text":"An earlier task showed that this API requires a snapshot before retry."}],"required_ids":["experience-001"],"query":"How should I retry with this API?","answer":"Take a snapshot before retrying."}
```

The answer can be ordinary text or an action sequence. It is **never** supplied to
the query head or writer of its own support. Required IDs are oracle annotations
or routing supervision, not features passed to the writer. Source IDs are opaque;
repeated IDs must identify exactly the same content. Record time is validated.
These checks cannot detect semantic answer leakage or near-duplicate solutions;
construct and audit dataset splits accordingly.

```bash
elm train --config configs/my-experiment.yaml --output runs/my-experiment
elm evaluate-episodes --run runs/my-experiment --episodes data/heldout.jsonl
```

Use `evaluate` for the built-in transaction/retry benchmark. Use
`evaluate-episodes` for general held-out answer likelihood. Neither is a coding
verifier or an autonomous tool-execution evaluator. See [experiment protocol](docs/experiments.md).

## Development and publishing

```bash
make test
make lint
```

The provided GitHub workflows run CPU numerical tests (including a tiny random
LFM architecture when Transformers is installed). The Spark workflow is manual
and requires a separately registered ARM64 runner labeled `spark`; it never runs
untrusted pull-request code on that runner.

The bootstrap was committed locally. The available connected GitHub account was
`werg`, but its exposed connector had no write actions, and the execution
environment had no authenticated Git CLI. **No remote repository or push is
claimed.** To publish the committed checkout:

```bash
gh auth login  # only when not already authenticated
./scripts/publish_github.sh werg/external-latent-memory
```

The script creates a **private** repository, pushes `main`, verifies the remote
commit, and refuses to overwrite an existing repository or origin. With a source
archive rather than a Git checkout, restore the supplied Git bundle first; see
[handoff](docs/handoff.md).

## Research order

Start with oracle transfer and two-record composition; keep stored-only evaluation
as an invariant. Establish counterfactual dependence, then improve routing and
scale replay. Compaction can branch from the first useful writer/reader without
waiting for ANN or asynchronous execution. Recurrence, multiscale access, and
storage measurements remain separately attributable branches.

[Backlog](docs/backlog.md) distinguishes completed implementation tasks from
unresolved experiments. [AGENTS.md](AGENTS.md) states invariants for further agents.

## License and model terms

No code license has been selected on the owner's behalf. Keep the initial repository
private until its intended license is chosen. The student checkpoint has separate
[Liquid model terms](https://huggingface.co/LiquidAI/LFM2.5-230M/blob/main/LICENSE).
This project does not redistribute model weights or grant rights to third-party
models, datasets, or teacher outputs.
