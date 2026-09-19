# Changelog

## Unreleased — Evaluation correctness and portable operations

Persisted-compaction evaluation is wired into curricula. Payload and source
counterfactuals reuse captured read plans, including native recurrent boundaries.
Unverified or uninformative learned-routing training is rejected before updates.
Fully live training avoids unused cache population and I/O.

Added run locks, detached process control, cooperative stop/resume, optional W&B,
disk-space guards, and bounded checksum-verified checkpoint archives with restore.
Spark wrappers expose run/archive mounts and named detached containers. Operational
logic is independent of Spark and Docker. See `docs/operations.md` and the separate
local validation report; historical results are unchanged.

## 0.4.0 — Pretrained recurrent-core conversion and in-loop memory

Added native prelude/core/coda conversion preserving every parent layer and the
one-pass path, with a live anchored bridge and fixed-size memory injection between
weight-shared core passes. The initial LFM split is 4/6/4, with 2–3 core passes in
the main pilot and one writer pass. SmolLM2 has a separate comparison configuration.

Added frozen-parent distribution anchoring without a second model, four-stage
conversion recipes, reproducible sampled-depth training, core-only adaptation,
explicit old-checkpoint conversion, and a fixed-bank stored-only depth sweep.

Added causal, gradient, replay, serialization-path, and resume tests. The local
CPU suite passes 165 tests; three optional native-model tests are skipped because
Transformers is unavailable. Four stages and resume run on the tiny CPU fixture.
No actual pretrained-model, CUDA, Docker, or Spark training is claimed.

The local Spark image tag is now `sdkb-spark:0.4`; the verified-source NVIDIA base
and immutable-digest/native-ARM64 build policy remain unchanged.

## 0.3.0 — SDKB and real-student trajectory curricula

Renamed package, CLI, model class, scripts, editor/container configuration and active
documentation to Spatially Superposed Differentiable Knowledge Base (SDKB).

Added explicit Hermes, UltraChat, SWE-smith, Nebius OpenHands and xLAM adapters;
causal prefix and cross-experience episode creation; tokenizer budgets; group-disjoint
selection; complete targets; source provenance; streaming sampling; and validated
random-access JSONL training input.

Added the pinned staged launcher with text bootstrap, frozen latent warmup, optional
oracle-text anchored joint training, stored-only teacher likelihood/value interventions,
and post-freeze causal and multi-binding evaluation. Added nine runnable recipes,
including actual-Spark and download-free CPU integration runs.

Changed the default Spark base to NVIDIA's documented 25.11 ARM64 PyTorch image,
resolved by digest and preserving its vendor torch/CUDA stack. Added isolated HF
dependencies and actual model/gradient preflight before the main curriculum.

Historical CPU capability results remain unchanged and are not real-LFM evidence.
