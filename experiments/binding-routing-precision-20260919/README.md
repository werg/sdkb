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

## Completed stored diagnostic

Corrected global training produces 102/128 freely generated actions, compared
with 58/128 for corrected within-world training. Both-required-source retrieval
is 73/128 versus 10/128. The paired action difference is 34.38 percentage points
(world-bootstrap 95% interval 21.88–47.66). Historical global training produced
101/128 on these same questions; this correction preserves the narrow result.

Targeted permission changes yield 74/128 pairs correct before and after the
change, versus 8/128 for within-world training. Restoration changes yield
47/64 versus 11/64; invariant restoration answers falsely change 10/64 versus
7/64. Original raw payloads and provenance are identical for all 128 records
across both corrected and historical arms. All 640 oracle/no-memory scoring
rows are identical. Exact identifier generation remains zero even with oracle
supports. Full records and checkpoints remain on the external disk.

These are paired implementation diagnostics on an already inspected corpus,
with one training seed and uncorrected world-bootstrap intervals. They do not
resolve the larger-bank degradation, exact-detail failure or general agent task
performance. Validation after adding the separate width probe: 329 tests pass,
four existing warnings; `ruff check src tests scripts` passes.

### Larger-bank follow-up

Repeat the 4,096-record exact-search diagnostic with the corrected global router,
the same expanded corpus and the original 32 query worlds. Keep all 1,024 worlds
as retrieval candidates. Output: `/archive/probes/global-stored-fp32-scale-20260919`.
This checks whether the previously observed scaling loss survives the arithmetic
correction; it is another known-corpus implementation diagnostic.

Completed: free actions fall from 102/128 at 128 records to **78/128** at 4,096;
both-required-source retrieval falls from 73/128 to 37/128. The large-bank free
action advantage is 10.94 points over no memory (world-bootstrap interval
0–21.88) and 11.72 over zero payloads (0.78–22.66). This still leaves substantial
scaling loss; the small, single-seed diagnostic does not establish robust global
retrieval. Historical large-bank free actions were 76/128.
