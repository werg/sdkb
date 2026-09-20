# Fixed-parent selected-text output distillation fork

The joint trajectory stage improved held-out next-message likelihood, but most of
its improvement also appeared with no memory. At R=2 the real stored payload beat
the zeroed payload by a small margin, while selected source text was substantially
better. This fork tests whether direct next-token distribution supervision helps
the memory path use those selected source payloads.

`run.py --prepare` creates two immutable configs and an input lock under an existing
external disk directory. Both arms warm-start the exact completed 400-update joint
checkpoint. They use the same 530 repository-heldout SWE-smith training episodes,
seed 233, fully live oracle-selected producer replay, native R=2 consumer, R=1
writer, frozen pretrained decoder, Muon, BF16, four-example accumulation and 400
updates. The sole intended experimental difference is output KL weight 0 versus 1.
The fixed selected-text R=1 teacher sees each source text and preceding target
tokens. It does not supply the memory query or any inference input. The student
reads only newly encoded stored-precision payloads during training; frozen
evaluation creates a new bank once, then scores from stored payloads only.

External output includes actual-LFM model preflights, offline W&B, metrics and
initial/final full recovery checkpoints. Intermediate checkpoints occur only on
explicit stop or resource guard. A stopped arm resumes the same checkpoint and
hyperparameters. Root and stage locks reject competing writers. The source model,
configs and train/validation bytes are digest-locked before every launch.

The planned fixed-bank validation compares real, zero, wrong and absent values
against selected-text NLL on 119 repository-heldout episodes. Primary diagnostic:
the paired real-minus-zero and real-minus-wrong loss, rather than training loss
alone. Generic recurrent compute, text-token layout and output KL give the two
paths different workloads, so this is a payload-use intervention rather than an
information-and-compute-matched capacity-substitution claim. No agent task or
patch success is measured here.

## Completed matched pilot

Both arms finished all 400 updates under frozen training commit `c8833c1`, with
one completed read at R=2 on every logged update. Their initial and final full
recovery checkpoints, W&B offline files, bank records, and raw scores live under
`/archive/runs/trajectory-output-distillation-20260920`. The evaluator checkout
was `af48f2b`. Each final checkpoint was scored from a newly serialized bank;
both banks encoded 375 unique sources with exactly 375 writer calls and a
64-source writer cache cap. The analyzer checked all 119 episode identities,
selected IDs and condition pairs. Digest-pinned aggregate results are in
`results.json`.

| R=2 held-out token-weighted NLL | No KL control | Output-KL arm |
|---|---:|---:|
| Real stored values | 0.971024 | 0.964353 |
| Zeroed values, same selected IDs | 0.980714 | 0.970088 |
| Wrong values, same selected IDs | 0.982071 | 0.971581 |
| No memory | 1.089557 | 1.088532 |

The fixed R=1 selected-text teacher scored 0.866650 NLL with text and 1.107569
without text on both final checkpoints. All 238 corresponding per-sequence scores
were exactly equal across arms. Output KL therefore improved overall real-value
NLL but improved the zero-value condition more. The paired change in real-over-zero
benefit was **−0.00303 mean episode NLL** (42-trajectory descriptive bootstrap
interval −0.01031 to 0.00509); real-over-wrong changed by −0.00419 (−0.01613 to
0.00946). This pilot does not show stronger payload-specific use. The real-value
NLL reduction alone would give the wrong impression.

The original source IDs and read plans were preserved for payload interventions.
Wrong-value permutation does not guarantee semantic disagreement. Text and latent
paths differ in token and compute budget, and the study measures teacher-forced
next-message likelihood, not generation quality, task completion, learned routing,
composition, or parameter substitution. One seed and this 400-update budget do
not establish a general result about output distillation.

After both evaluators exited, verified hard-link deduplication reclaimed
2,034,699,352 external bytes from identical joint-parent and fork-initial model
files. All five affected complete checkpoint sets reverified against their
manifests; `dedup.json` records the operation. The retained final states and
optimizer/RNG information remain intact for exact resume.
