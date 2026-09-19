# Development invariants

Read `docs/architecture.md`, `docs/implementation.md`, and `docs/validation.md`
before changing behavior. Do not describe a proposed module as implemented.

1. Inference reads stored latent payloads; it must not re-encode source trajectories.
   Offline creation of a new bank by a frozen writer is a separate operation.
2. A query can use only its causal prefix, never teacher-forced targets or future
   observations. Preserve opaque IDs, provenance, time and authorization boundaries.
3. Preserve the full-graph reference. Replay must accumulate all value, key, query,
   state, gate and shared-parameter paths before the optimizer step. Reproduce RNG,
   autocast and the actual serialized-precision forward. Do not truncate read counts
   to avoid implementing storage/recomputation correctly.
4. Complete each neighborhood aggregate before its next shared-state update.
   Chunk-wise independent multi-round executions compute another model.
5. Compaction preserves conditional pre-normalization numerator and mass. Never
   average normalized cluster outputs without their masses. Attention gets the same
   opportunity to compact as the MLP.
6. A full-cluster code is not a subset-selective code. Do not silently serve a
   partial selection from one; never use learned selection as authorization.
7. Keep overlap responsibilities summing to one, including mass. Do not duplicate
   evidence or drop required field shares invisibly.
8. No x86 emulation or replacement of the NVIDIA Torch/CUDA stack on Spark.
   CPU and GPU allocations share physical memory there. Separate warm-cache timing
   from cold NVMe claims, and exact scans from ANN.
9. Tests and smoke losses are not evidence of composition or parameter substitution.
   Check counterfactual behavior and information-matched controls.
10. Keep model weights, private data, credentials, caches and training artifacts out
    of Git. Do not add a license, enable public visibility, or launch paid teacher
    collection without the owner's decision.

Run `python -m pytest -q`, `ruff check src tests scripts`, and relevant end-to-end
commands. Add regression tests before changing replay, causal masking or storage.


## SDKB 0.3 development

Use `sdkb` and `SDKBAgent`; do not edit historical result records for a rename.
Read the dataset/training guides before changing causal boundaries. Keep source and
target identities/versioned payloads intact; source commands are inert data. Preserve
NVIDIA torch/CUDA and run the model preflight for actual-HF changes. Core tests must
work without downloads. Distinguish teacher NLL, controlled transfer and agent success.
