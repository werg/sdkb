# Changelog

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
