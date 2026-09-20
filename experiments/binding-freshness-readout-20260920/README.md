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


## Completed results

All 18 heads completed 1,600 updates from frozen `f9625f7`. The collector verifies
all final sampler states against the declared 204,800-example draw schedule,
optimizer settings, metric coverage, source/data/bank identities and result/endpoint
hashes. The parent controller and every child exited with code zero.

| Frozen model | Features | Head | Train exact /1,984 | Held-out exact /64 | Hex /384 | Shifted hex /384 |
|---|---|---|---:|---:|---:|---:|
| source | payload | linear | 1876 | 0 | 126 | 29 |
| source | payload | mlp256 | 1982 | 0 | 121 | 33 |
| source | reader | linear | 651 | 0 | 81 | 22 |
| source | reader | mlp256 | 1978 | 0 | 78 | 28 |
| source | zero_reader | linear | 0 | 0 | 29 | 29 |
| source | zero_reader | mlp256 | 0 | 0 | 29 | 29 |
| worlds-256 | payload | linear | 1608 | 0 | 183 | 25 |
| worlds-256 | payload | mlp256 | 1977 | 0 | 169 | 19 |
| worlds-256 | reader | linear | 1207 | 0 | 160 | 23 |
| worlds-256 | reader | mlp256 | 1943 | 0 | 158 | 22 |
| worlds-256 | zero_reader | linear | 0 | 0 | 29 | 29 |
| worlds-256 | zero_reader | mlp256 | 0 | 0 | 29 | 29 |
| worlds-8192 | payload | linear | 1957 | 3 | 251 | 24 |
| worlds-8192 | payload | mlp256 | 1980 | 0 | 212 | 31 |
| worlds-8192 | reader | linear | 1387 | 0 | 156 | 22 |
| worlds-8192 | reader | mlp256 | 1945 | 0 | 156 | 26 |
| worlds-8192 | zero_reader | linear | 0 | 0 | 29 | 29 |
| worlds-8192 | zero_reader | mlp256 | 0 | 0 | 29 | 29 |

All shifted-feature exact counts are zero. The native zero-reader feature vectors
are exactly constant within both splits and equal across training/held-out splits
for all three models. Their inability to memorize the training labels closes the
query-identity shortcut present in the earlier variable-query diagnostic.

Payload heads contain 196,608/548,864 parameters (linear/MLP256); reader and zero
heads contain 786,432/2,121,728. The broader-training writer exposes more recoverable
characters to the same payload heads. Its linear payload head recovers three full
identifiers and 251 characters, versus zero full identifiers and 156 characters
from the wider reader output. Larger MLP heads fit more training examples without
improving held-out readout. These findings concern the tested supervised heads and
budgets; they do not prove that information is irreversibly destroyed by the reader.

For the broader model, payload linear accuracy by position is
61/48/39/32/27/44 out of 64; reader linear accuracy is 55/35/22/18/14/12. The next
bounded diagnostic will inspect the reader's input projections and its state before
final output normalization/projection. This can distinguish an early projection
limitation from later transformations or readout conditioning. No production reader
or training objective has been changed on the basis of these probes.
