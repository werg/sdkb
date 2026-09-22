# Positional interface validation — 22 September 2026

This record validates mechanics only. It is not evidence of learned memory
quality, composition, retrieval, compaction quality, or parameter substitution.

## CPU regression

The full repository suite completed after introducing the opt-in positional
layout and Transformers-version mask handling:

```text
619 passed, 4 warnings in 60.03s
```

The positional tests cover shared projection shape and gradients, complete
chunked aggregation equivalence, gradient delivery to every source and target
position, compactor mass conservation, flat-teacher initialization, explicit
8-to-32-style expansion, new-slice masks, and checkpointed/resumable distillation
and expansion runners.

## Actual-model Spark preflight

Command:

```bash
python -m sdkb.cli model-probe \
  --config configs/lfm25_230m_four_space_positional_phase2_spark.yaml \
  --output /archive/validation/positional-phase2-model-probe-20260922.json
```

Environment:

| Field | Value |
| --- | --- |
| GPU | NVIDIA GB10, compute capability 12.1 |
| Torch | `2.11.0a0+a6c236b9fd.nv26.03.46836102` |
| CUDA runtime | 13.2 |
| Transformers | 5.5.4 |
| Model revision | `40cb2ad3b3044d5a41eee083a6103c8b523afa45` |
| Precision support | BF16 reported supported |

Results:

| Check | Result |
| --- | ---: |
| Total parameters | 243,787,512 |
| One-loop parent identity maximum error | 0.0 |
| Causal-prefix maximum error | 0.0 |
| Two-loop change maximum | 0.2211313 |
| Peak CUDA allocated | 2,159,687,680 bytes |
| Peak CUDA reserved | 2,189,426,688 bytes |

Every space reported finite nonzero gradients for its positional codec, operator
input map, source-position table, target-position table, address map, and query
map. The shared fusion layer and recurrent bridge projections and gates also
reported finite nonzero gradients. The complete JSON report remains on the
external disk at the command's output path.

The parameter count confirms that the 32-position structured interface does not
instantiate the rejected 503-million-parameter dense codec. Sustained Phase 2
distillation and task training still require the completed Phase 1 checkpoint.
