# Correct the feature objective's stored-key precision

The earlier global-address probe formed its cosine matrix inside BF16 autocast and
omitted the writer's final BF16 normalization before FP32 key storage. Actual
stored evaluations always used the real FP32 search, so their recorded outcomes
remain valid; the optimization proxy did not exactly match that precision path.

Current source restores final key normalization and disables autocast for FP32
cosine products (`highest` float32 matmul precision). A native preflight serializes
128 projected vectors, then compares all 320 query scores/rankings to `DiskStore`:
maximum cosine error 2.384185791015625e-07 and all 320 top-two rankings identical.
This validates shared batched projected vectors, not bit identity between different
HF/linear GEMM batch shapes. The BF16 regression first fails on the old score dtype.

Rerun the same matched world/global 3,200-update experiment from the same original
router, with the same seed/data/batch/Muon settings and corrected arithmetic.
Use a new output and identity; never relabel or resume the historical checkpoints.
Then compare real stored outcomes on the existing known confirmation corpus as a
paired implementation diagnostic, not a fresh scientific confirmation. Preserve
all payload/reader/decoder controls and candidate-free generation.
