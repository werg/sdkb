# Fixed-query readouts after endpoint-freshness training

## Protocol before launch

The completed freshness continuations transfer some early identifier characters,
but neither generates a correct original endpoint on the sealed 64-identifier
split. The repeated-target arm mostly emits old training answers; the larger arm
usually emits new but incorrect strings. Both still copy every endpoint from
selected source text. Locate accessible information before making another model
or training-objective change.

Compare all three frozen checkpoints: original useful MLP source, 256-world
continuation, and 8,192-world continuation. For each, build a separate offline bank
of the identical 1,024-world corpus. Keep the previously sealed 32 worlds first and
unchanged; they have already been inspected, so this is a post-hoc diagnostic.
Generate the first 992 additional worlds whose source IDs and endpoint targets
avoid both writer-training corpora and the original/inverted held-out endpoints.
There is one original identifier question per endpoint, with **identical query
text** for all of them. Preserve every source, target, time and provenance field.
Nonidentifier tasks remain in the corpus but do not train these readout heads.

Use the existing frozen-feature probe with three representations: 2,048-value
stored payload, 8,192-value first reader output, and reader output from zero values.
Writer and compactor calls are forbidden during reader feature extraction. Target
tokens are never inputs. A regression verifies that zero-value features are
identical across different source worlds when the query is fixed, in addition to
existing target-independence tests. This removes the query-identity shortcut seen
in the earlier reader-readout diagnostic.

Fit both the existing linear and MLP256 six-position classifiers for every model
and representation: 1,600 native Muon updates, batch 128, seed 83, LR .001. Training
normalization is fixed from the 1,984 training payloads; report all 64 held-out
identifiers, exact-six and per-position character correctness, plus cyclically
shifted feature controls. Reader heads have more parameters than payload heads;
the fixed prefix and six-position format are supplied by the probe. These are
supervised readouts, not normal language generation, information-theoretic bounds,
agent success or capacity substitution. Negative readouts do not prove absent
information. No model is selected by this already inspected split.

## Operations

Artifacts live at `/archive/probes/freshness-readout-20260920` on the external disk.
The controller runs at most two child jobs, first offline banks, then payload,
reader and zero-reader probes. It shares study-root and `confirmation-queue` stop
controls, terminates and joins active children, and supports explicit `--resume`.
Resume acknowledges stage controls only after their ownership locks release.
An active child is rejected; its stop request is not cleared.

Offline banks use the existing atomic source/writer/config manifest and raw-byte
digest. A stop rolls back incomplete records; a committed bank resumes without
writer calls. Model forwards have a compute watchdog; SQLite writes/fsync occur
outside that guard. Host/disk reserves are checked during encoding. This is a
separate offline operation, not source re-encoding during inference.

Only small readout heads save initial/final/emergency optimizer-boundary state;
there are no new backbone checkpoints and no periodic saves. Each probe preserves
named optimizer groups, complete optimizer/RNG/config state, normalization buffers,
metric reconciliation and offline W&B attempts. This is distinct from the main
trainer's stronger partial-microbatch recovery contract. No NVIDIA stack changes.

From the frozen study checkout:

```bash
python experiments/binding-freshness-readout-20260920/run.py \
  --root /archive/probes/freshness-readout-20260920
sdkb runs stop --output /archive/probes/freshness-readout-20260920
# After the previous queue and child locks release:
python experiments/binding-freshness-readout-20260920/run.py \
  --root /archive/probes/freshness-readout-20260920 --resume
```

Results are pending until the declared budgets and all controls complete.


Preparation accepted 992 worlds after rejecting seeds 126, 333, 789 and 791.
`inputs.json` pins all source checkpoints, dataset hashes and accepted/rejected
seeds. The feature probe records whether each split is exactly constant and whether
its first held-out/training features match, so the native zero-query control can
be checked directly. Per-position counts supplement the existing aggregate metrics;
historical readout records retain their original format.
