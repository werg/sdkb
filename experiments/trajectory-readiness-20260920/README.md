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

Preparation is in progress. No native text-readiness or latent-training outcome
is asserted yet.
