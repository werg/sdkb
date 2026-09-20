# Real-trajectory readiness

Before latent training, test whether selected prior text improves the pretrained
student's recorded-target likelihood on a bounded causal trajectory sample.
This is a data/readout readiness check, not teacher training, cross-experience
composition, tool execution or agent success.

The first preparation scans 512 rows from the public
[SWE-smith tool trajectories](https://huggingface.co/datasets/SWE-bench/SWE-smith-trajectories),
using the existing adapter, success filter and repository-disjoint 80/20 split.
It pins the actual upstream revision before reading rows. The 256-row shuffle
buffer is not uniform sampling of the full corpus, and row count is not a network
byte limit. The manifest records rejected/retained examples and precise provenance.

Prefix memory contains only older messages; query context and targets use the
existing causal preparation rules. Source commands remain inert data. Complete
targets longer than 512 tokens are filtered, never silently truncated. Source and
prompt limits are 512 and 4096 tokens, with up to four sources and three target
turns per trajectory. No model or data artifacts enter Git.

`prepare.py` uses the verified external model cache for the pinned LFM tokenizer
and directs Hugging Face, dataset, Xet, XDG and temporary caches to the external
cache root before importing their libraries. Preparation publishes its data folder
only when complete. Repeating the same command checks the pinned declaration and
prepared-file hashes. A cooperative termination retains the input lock and cleans
the pending preparation directory; it makes no training-progress claim.

```sh
python experiments/trajectory-readiness-20260920/prepare.py \
  --root /archive/probes/trajectory-readiness-20260920 --cache /archive/cache
```

Preparation completed: 512 scanned rows, 283 unsuccessful trajectories filtered,
187 accepted training trajectories from 77 repository groups and 42 validation
trajectories from 12 disjoint groups. The prepared sets contain 530 and 119 episodes;
31/7 oversized or empty targets were skipped. Upstream revision is pinned in
`preparation-inputs.json`; the complete audit is external, with only metadata in Git.

The declared next check uses **all 119 validation episodes**, the untouched
pretrained LFM at native depth one, and `selected_text` versus `none` conditions.
No synthetic-study weights are loaded and no optimizer is constructed. Report
both token-weighted and mean-episode NLL, plus a descriptive paired bootstrap over
repository groups. Text changes token/compute budgets, so this is context utility,
not a capacity comparison. `evaluate_text.py` atomically saves each completed score,
validates its complete ordered prefix on resume, handles signal/control stops and
host pressure between scores, and rejects changed inputs before model allocation.
A regression reproduces uninterrupted rows exactly after stopping at the first
score and verifies that no source writer is ever called.

Native preflight and all 238 condition scores completed from frozen `76ab096`.
Selected-text token-weighted NLL is **1.48980**, versus **1.93726** without support;
mean-episode NLL is 1.80457 versus 2.38351. The paired mean reduction is .57894,
with a descriptive repository-bootstrap interval [.50746, .65872] over 12 groups.
This supports attempting a bounded latent-memory curriculum; it does not establish
that learned payloads retain the context benefit or that an agent solves tasks.
The next declared stage is a frozen-backbone recurrent text bridge, using Muon.
