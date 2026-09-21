# External artifact cleanup record

Recorded and completed on the DGX Spark on 2026-09-21. Sizes are `du -sh`
approximations. All external paths are under `/mnt/external/sdkb-archive`.

| Artifact | Previous size | Cleanup result |
| --- | ---: | --- |
| `runs/four-space-squad-gate-fork-20260920` | 2.3 GB | Removed checkpoints; retained 9.7 MB of metrics, manifests and environment records. |
| `runs/four-space-squad-contrast-20260921` | 2.5 GB | Removed checkpoints; retained 9.0 MB of scientific records. |
| `runs/four-space-bank10k-qa-novel-20260921` | 2.2 GB | Removed checkpoints; retained 3.8 MB of scientific records. |
| `runs/four-space-bank10k-value-contrast-20260921` | 2.2 GB | Removed checkpoints; retained 9.4 MB of scientific records. |
| `banks/four-space-squad-v2-10k` | 105 MB | Retained because three committed reference configurations still use this bank. |
| `corpora/squad-short-bank-queries-20260921` | 11 MB | Removed; superseded before bank publication. |
| `corpora/hotpot-four-space-target-100k-20260921` | 147 MB | Removed; rejected for validation contamination and superseded by the source-disjoint corpus. |

The cleanup also removed one superseded 1.3 GB G1 checkpoint and two superseded
1.1 GB R2 recovery checkpoints. The final G1 writer checkpoint, current R2 recovery
checkpoint, active 100,000-source bank, current corpora and queued-stage inputs were
preserved. Host test caches, the stray lockfile and the accidentally created host
virtual environment were removed. Approximately 12.8 GB was reclaimed externally;
compact validation records remain available for the documented comparisons.
