# SDKB

**Spatially Superposed Differentiable Knowledge Base**

SDKB trains a small language model to write experiences into stored latent values,
compose selected values through a recurrent set reader, and use the result on later
questions or actions. Selective replay trains the writer without retaining every
source graph. Compaction learns to replace groups with synthetic records while
preserving their conditional contributions.

**Version 0.3.0 · Student `LiquidAI/LFM2.5-230M` · Target: one NVIDIA DGX Spark.**
The Python package, CLI, agent class, scripts, container and active documentation now
use SDKB (`sdkb`, `SDKBAgent`). Historical experiment logs remain immutable evidence.

## Start on the Spark

The handoff includes a complete Git-history bundle. The intended remote is
`werg/sdkb`, but the connected GitHub integration rejected publication with HTTP 403.
Use the bundle until those commits are published:

```bash
git clone sdkb-v0.3.bundle sdkb
cd sdkb
./scripts/start_spark.sh --recipe recipes/spark_smoke.yaml --output runs/spark-smoke
```

This **real-model integration run** builds the native ARM64 image when absent,
checks CUDA/BF16, pins model and dataset revisions, prepares real teacher traces,
checks memory gradients and runs two updates in each of three stages. It is not
a capability experiment. After inspecting its outputs, start the main curriculum:

```bash
./scripts/start_spark.sh --recipe recipes/starter.yaml --output runs/starter
```

| Stage | Optimizer updates | Training behavior |
|---|---:|---|
| `text_bootstrap` | 200 | Adapt the student to the recorded target format with the same selected evidence rendered as text. |
| `latent_warmup` | 400 | Freeze the backbone; train the write interface, codecs and reader through selective replay. |
| `latent_joint` | 400 | Joint training with a lower backbone learning rate and a small oracle-text anchor. |

These are initial run budgets, not promised convergence thresholds. The starter scans
up to 2,000 Hermes and 2,000 UltraChat rows with seeded source shuffling. It records
how many complete, budget-fitting examples survive preparation. No live teacher API,
experiment-tracking account or paid inference endpoint is required.

Resume the identical recipe/output:

```bash
./scripts/start_spark.sh --recipe recipes/starter.yaml --output runs/starter --resume
```

The job runs synchronously in the foreground. A graceful interruption checkpoints
at an optimizer boundary. Completed stages and evaluations are not repeated. Prepared
input/config changes are rejected on resume. Run only one launcher per output directory.

To download, pin and inspect data without loading the training model:

```bash
./scripts/start_spark.sh --recipe recipes/starter.yaml --output runs/prepared --prepare-only
# Inspect data/manifest.json and launch.json; then continue:
./scripts/start_spark.sh --recipe recipes/starter.yaml --output runs/prepared --resume
```

## Training recipes

| Recipe | Purpose |
|---|---|
| `starter.yaml` | Hermes tool conversations + UltraChat; bounded first real-data curriculum. |
| `tools.yaml` / `chat.yaml` | Separate tool-use and general conversational continuation runs. |
| `coding.yaml` | Successful SWE-smith traces; earlier-prefix memory. |
| `openhands.yaml` | Successful Nebius SWE-rebench/OpenHands coding traces. |
| `cross_experience.yaml` | Different prior SWE-smith instances in the same repository; repository-held-out validation. |
| `causal.yaml` | **Real LFM student** text-to-latent training on controlled rules, then fresh-world counterfactual and binding evaluation. |
| `spark_smoke.yaml` | Real model and starter sources, two updates per stage, four held-out targets. |
| `offline_smoke.yaml` | Project-authored fixtures, tiny CPU backend, no downloads. |

Paths are under `recipes/`. Sources/configurations/licenses and exact parsing are in
[the dataset guide](docs/datasets.md); `sdkb datasets` prints the executable catalog.
See [training](docs/training.md) and [Spark setup](docs/spark.md) for operations.

**Two separate data protocols are implemented.** Prefix memory encodes earlier
context before a later assistant target. Cross-experience memory encodes different
completed instances, never the query's own future solution. Supplied context is not
falsely labeled as a verified sufficient support set. The controlled causal suite
provides that stronger test.

## Docker choice

The default is **`nvcr.io/nvidia/pytorch:25.11-py3`**, explicitly used in
[NVIDIA's Spark fine-tuning instructions](https://build.nvidia.com/spark/unsloth/instructions).
The build script pulls `linux/arm64`, verifies architecture, resolves an immutable
image digest and preserves NVIDIA's torch/CUDA installation. An isolated venv adds
Transformers 5.17.0 and datasets 5.0.1 without replacing the GPU runtime.

SDKB does not require Unsloth model patches, FlashAttention, bitsandbytes or custom
convolution wheels. It uses the public Transformers embedding interface and the
ordinary pretrained path as its one-pass baseline. A documented base image is not
proof that the complete assembled stack has run here: build-time imports and the
real-device model/gradient preflight verify it on the Spark. `SDKB_BASE_IMAGE` is an
explicit override for another tested NVIDIA image.

## Implemented architecture

```text
source experience -> shared LFM + learned write slots
  -> canonical key and fixed values -> per-space codecs -> stored payloads

causal query -> selected stored records -> recurrent MLP / attention set reader
  -> fixed soft input tokens -> shared LFM -> later teacher target

training: consumer cotangents -> selective replay of live source producers
inference: stored payloads only; no producer encoding on the read path
```

The existing recurrent MLP/attention readers, multiscale transforms, state-dependent
scheduled reads, exact cosine search, checkpointed pooling, producer replay,
raw/compact paired objectives, merge consistency, overlap grouping, storage noise,
persistent full-cluster compaction and deletion lineage are retained. The starter
uses one space, oracle addressing and one backbone pass to isolate the learned
communication interface. These are experiment defaults, not replay memory limits.

The optional loop adapter applies gated weight-shared refinement of the hybrid
convolution/attention stack, without inconsistent cache reuse. It is experimental,
not an assertion that LFM was pretrained with recurrent depth. The attention-only
SmolLM2 comparison configuration remains available.

## What is measured

Every stage writes held-out supports once, serializes/reopens the bank, and performs
stored-only teacher evaluation. Metrics include full-target NLL, token-weighted
summaries, paired trajectory intervals, no memory, zero payloads, and fixed-key/ID
payload permutation. Reference likelihood is **not** an agent execution success rate.

After the last checkpoint freezes, new Boolean and multi-binding worlds test support
removal, adjusted-answer source counterfactuals, joint evidence and exact identifiers.
For the starter this is an out-of-domain diagnostic; `causal.yaml` supplies the relevant
controlled training family. No evaluation threshold silently changes the recipe.

Artifacts stay under the run directory: revision locks, prepared JSONL, model probe,
metrics, optimizer/model/cache checkpoints, stored banks and evaluation JSON. Input
files are hashed and indexed by offsets rather than all retained in training RAM.

## Offline development

With Python 3.11+ and CPU PyTorch installed:

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
sdkb launch --recipe recipes/offline_smoke.yaml --output runs/offline
sdkb launch --recipe recipes/offline_smoke.yaml --output runs/offline --resume
```

Core tests do not download models/data. Current execution evidence is in
[validation-v0.3.md](docs/validation-v0.3.md). The earlier tiny-model learning result
remains historical CPU evidence, not a pretrained LFM result.

[Architecture](docs/architecture.md) · [Implementation](docs/implementation.md) ·
[Dataset guide](docs/datasets.md) · [Training](docs/training.md) ·
[Migration](docs/migration.md) · [Publication/handoff](docs/handoff.md)
