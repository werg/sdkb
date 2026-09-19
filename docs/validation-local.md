# Local revalidation — 2026-09-19

Revalidated upstream `45f62681a616d255991d2fdba3cdfcfd808c3026` with the
uncommitted review fixes. This revision implements native middle-block recurrence
and inter-pass memory reads. Earlier conclusions based on the initial snapshot
that these were only proposed no longer apply.

## Checks completed

- Full core suite: **186 passed**, with four dependency warnings. Tests ran with
  CUDA hidden inside the native NVIDIA container; they required no model downloads.
- `ruff check src tests scripts`, shell syntax checks and `git diff --check`: passed.
- Actual pretrained LFM2.5-230M on GB10: single-pass identity and causal-prefix
  maximum errors were both zero; two passes changed outputs; writer, reader and
  recurrence bridge gradients were finite and nonzero.
- Actual GPU training: checkpoint after update one, resume through update two,
  replay verification enabled, then stored-bank interventions and counterfactual
  evaluation. All-live training created zero stale-cache records.
- Detached local training: start, cooperative stop, committed checkpoint, resume
  and completion exercised end to end.
- Actual W&B offline logging: stable run identity across two attempts; independent
  local segments, with no online upload.
- A small checkpoint was archived to the actual external disk, verified and
  restored. Failure, corruption, low-space and bounded-backlog paths have tests.

The final scalar-logging and archive-lock ordering adjustments also passed the
eight operations tests and Ruff after the full suite.

## Runtime and evidence

The tested runtime was native ARM64 NGC **26.03**, vendor Torch
`2.11.0a0+a6c236b9fd.nv26.03.46836102`, CUDA 13.2, Transformers 5.17.0,
and NVIDIA GB10. No NVIDIA Torch replacement or emulation was used. This does
**not** validate a fresh build of the repository's canonical NGC 25.11 image.

See [numerical reports and reproduction configuration](../experiments/local-review-20260919/README.md).
Temporary model downloads and training weights are excluded from these reports.

## Remaining research and validation boundaries

These checks establish executable training and recovery plumbing, not useful
composition, agent success or parameter substitution. The one-pair GPU smoke
counterfactual scores were zero; this tiny warmup is not a capability experiment.
Teacher NLL must remain distinct from controlled transfer and agent success.

Learned routing now rejects teacher datasets whose support annotation is merely
`provided_context`, and datasets with no competing candidates. Those inputs do
not provide the verified labels and negative candidates needed by the implemented
routing objective. Useful teacher routing still requires appropriate data/objectives.

Persistent compaction is exercised in supported prefix-read evaluation; native
inter-pass compaction remains unsupported. Long training, live upstream corpus
ingestion, online W&B resume and other operating systems were not validated here.
No cold-disk throughput or memory-capacity claim follows from these checks.

See [operations](operations.md) for portable run control, external archives and
the storage lessons adopted from bgkit.
