# External artifact cleanup candidates (deferred)

Recorded on the DGX Spark on 2026-09-21. This is an inventory only; no action
is authorized by this file. Sizes are `du -sh` approximations. All paths are
under the external `/mnt/external/sdkb-archive` disk.

| Candidate | Approximate size | Why it may be retired later |
| --- | ---: | --- |
| `runs/four-space-squad-gate-fork-20260920` | 2.3 GB | Its heldout gate-only control did not improve payload use. |
| `runs/four-space-squad-contrast-20260921` | 2.5 GB | The live source-swap result reversed by step 1,078. |
| `runs/four-space-bank10k-qa-novel-20260921` | 2.2 GB | The 512-step frozen-bank QA run did not retrieve verified sources. |
| `runs/four-space-bank10k-value-contrast-20260921` | 2.2 GB | The step-1,500 stored-value result reversed on 64 heldout questions. |
| `banks/four-space-squad-v2-10k` | 105 MB | Older published writer generation, retained as a retrieval reference. |

Preserve the validation records and any checkpoint needed for direct
comparisons before deciding whether to remove an item. The current short
reconstruction run, its source corpus, and the local-routing bank are active
inputs and are not cleanup candidates.
