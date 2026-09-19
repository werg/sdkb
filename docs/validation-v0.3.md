# SDKB 0.3 validation

Date: September 19, 2026. Implementation tested: `49bcf24a020d9bca7562cb6e31e6a7eba5107810`.
This report distinguishes executed development checks from the actual-device
preflight provided for the DGX Spark. No pretrained capability result is claimed.

## Executed

**140 tests passed; one optional Transformers/LFM structural test skipped.**
The test environment was x86_64, Python 3.13.5, torch 2.10.0+cpu. CUDA,
Transformers and datasets were unavailable. The optional structural test needs
Transformers; it is not a pretrained-model benchmark.

The renamed package was installed with `pip install --no-deps --no-build-isolation -e .`.
`sdkb --version` reports `SDKB 0.3.0`; `python -m sdkb` also works. Python sources
compiled, every shell script passed `bash -n`, and an AST/import/token scan passed.
Ruff was unavailable and is not reported as run.

The public three-stage launcher and idempotent resume were executed:

```bash
sdkb launch --recipe recipes/offline_smoke.yaml --output runs/release-validation
sdkb launch --recipe recipes/offline_smoke.yaml --output runs/release-validation --resume
```

Both returned `status: complete`. Each stage trained, checkpointed and evaluated.
The second command reused the completed stages/evaluations. Preparation scanned
60 authored fixtures, retaining 46 training and 14 validation episodes with disjoint
groups and complete targets. Each stage ran one optimizer update and evaluated two
held-out targets. The values below are an execution check only:

| Stage | Training loss | Full-evidence teacher NLL/token |
|---|---:|---:|
| Text bootstrap | 5.551743 | 5.467292 |
| Latent warmup | 5.442692 | 5.475495 |
| Latent joint | 5.437436 | 5.412609 |

The text stage receives support text; the latent stages use serialized stored values.
No difference in this one-step smoke is interpreted as a learned memory benefit.

A second persistent CPU run used four controlled training worlds and two fresh
held-out worlds. It completed actual adjusted-answer counterfactual and multi-binding
evaluation paths, plus resume. Its configuration/results are preserved; it is not
an LFM transfer result. This complements unit tests that deliberately interrupt an
evaluation between its causal and binding reports.

Tests cover schema fixtures, no future-message leakage, Unicode/indentation preservation,
complete-target limits, disjoint groups, immutable source identities, multiple prior
experiences, matched text evidence, stored-only reads with producer calls prohibited,
checkpoint/resume, reader/compaction numerical parity and selective replay gradients.
Evidence is committed in `experiments/sdkb-v0.3/`.

## Prepared, not executed in this environment

The native ARM64 NVIDIA container build, pinned HF installation, actual pretrained
LFM checkpoint, real GPU gradients/training and live upstream dataset ingestion
remain actual-machine checks. Schemas were researched from primary dataset cards;
fixture validation is not mislabeled as a downloaded-corpus run.

No code-verifier success or quality/VRAM/latency frontier is claimed. The full design
and running code are supplied so those experiments can begin rather than remain a plan.

## Spark entry point

```bash
./scripts/start_spark.sh --recipe recipes/spark_smoke.yaml --output runs/spark-smoke
./scripts/start_spark.sh --recipe recipes/starter.yaml --output runs/starter
./scripts/start_spark.sh --recipe recipes/causal.yaml --output runs/causal
```

The first command resolves an ARM64 image digest, verifies CUDA/BF16, pins model/data,
prepares real inputs, checks model/soft-memory gradients and executes two updates per
stage. The next commands run the starter and controlled real-student curricula.
Use distinct output directories and retain manifests/checkpoints/evaluation JSON.

## Publication

GitHub repository `werg/sdkb` is readable but the integration's attempted contents
write returned HTTP 403. Local commits and the full Git-history bundle are the
verified handoff. No remote publication is claimed.
