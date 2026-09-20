# Fresh repository-disjoint trajectory slice

The first 119 validation episodes have supported several exploratory objective
and bridge probes. Before training another source-dependent objective, prepare a
new evaluation split from the same pinned public SWE-smith revision. This is a
data-only operation: no model weights, paid teacher calls or source commands.

`prepare.py` advances the same seed-191, buffer-256 streamed order past the
original 512 scanned rows, scans the next 2,048 rows, and excludes all 89
repository groups in the original normalized sample before assigning new
group-disjoint train/validation subsets. It uses the same tokenizer bytes,
budgets, causal-prefix construction and complete-target filter. The source
revision, previous file hashes, skipped/scanned counts and exclusions are locked
before publication. A complete output is atomically published under the external
drive and verified on resume. The row budget bounds scanned rows, not downloaded
bytes or the size of the entire upstream corpus.

The new **validation** portion is reserved for a later predeclared objective and
counterfactual evaluation. Preparation alone makes no quality or agent-success
claim. Before using it, check the resulting counts, prior-group disjointness,
source revision, target completeness and repository distribution. All raw rows
and normalized data remain outside Git.
