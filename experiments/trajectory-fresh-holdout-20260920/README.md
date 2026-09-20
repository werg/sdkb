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

The 2,048-row continuation yielded 121 accepted trajectories and 342 episodes
across 30 new repository groups. It excluded 752 rows from the 89 groups used
by the original preparation and filtered 1,175 unsuccessful trajectories. The
new preparation's arbitrary train/validation partition contains 304 and 38
episodes respectively. Because **neither** new subset has been used for training
or model selection, `seal.py` reserves both as one 342-episode evaluation-only
set. It checks zero prior-group overlap, disjoint internal groups, distinct
episode IDs, complete targets and causal support times; its file and source
digests are recorded externally. This avoids treating five repository groups
and 38 episodes as the entire fresh confirmation.

The sealed set is reserved for a later predeclared objective and counterfactual
evaluation. Preparation alone makes no quality or agent-success claim. All raw
rows and normalized data remain outside Git. The dataset still comes from the
same upstream collection and bounded shuffled stream, so repository disjointness
does not establish broader distributional independence or decontamination.
