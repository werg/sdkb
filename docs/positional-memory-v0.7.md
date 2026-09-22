# Position-preserving memory interface (SDKB 0.7)

This document supersedes the Phase 2 interface proposal in
`adaptive-memory-v0.6.md` and the flattened-codec descriptions in
`architecture.md`, `implementation.md`, and `training.md`. Historical Phase 1
checkpoints remain valid and keep their recorded flat interface. New Phase 2
banks use the positional interface described here.

## 1. Canonical representation

A `memory.write` trajectory appends one key slot and a configured number of
value slots after the source tokens. The existing decoder processes those
positions. With a causal decoder, value slot `j` sees the source and earlier
write slots, but not later write slots. This ordered causal latent array is an
intentional part of the writer.

For Phase 2, the writer emits 32 decoder-width value states:

```
source tokens + key slot + 32 value slots -> reused decoder transformer
```

The key slot follows the existing normalized key/address path. A shared
per-position projection maps every value state into each memory space. The
logical stored layouts are:

| Space | Positions | Channels per position | Scalars per record |
| ---: | ---: | ---: | ---: |
| 0 | 32 | 32 | 1,024 |
| 1 | 32 | 64 | 2,048 |
| 2 | 32 | 128 | 4,096 |
| 3 | 32 | 256 | 8,192 |

The persistence layer may serialize a payload contiguously as `[S*C]`, but its
manifest declares the positional layout `[S,C]`. Learned components must not
reinterpret it as one unstructured feature vector.

This replaces the legacy `flatten([S,H]) -> Linear(S*H,D)` operation with
`shared_projection([S,H]) -> [S,C] -> serialize([S*C])`. The transformer has
already mixed source information into the write states. Increasing the number
of stored positions therefore increases stored bytes and operator computation
without multiplying codec parameters by the square of the interface expansion.

## 2. Block neural operator reader

The reader, also called the combiner in conceptual discussions, maintains one
residual state for every returned target position. Each block applies one MLP
shared across all input-position/target-position pairs. A message may depend on
the projected stored input position, projected target residual, causal query,
learned source and target position embeddings, relative slot position, the
record's density-adaptive distance weight, and versioned relative metadata.

For record `i`, stored position `j`, target position `k`, and block `l`, the
implemented operator has the form:

```
h_ijkl = SiLU(A_l x_ij + B_l r_k + C_l q
              + source_l[j] + target_l[k] + relative_l[j,k])
m_ijkl = D_l h_ijkl
g_ijkl = sigmoid(G_l h_ijkl)
```

Messages are accumulated as a pre-normalization numerator and mass. A record's
weight is divided among its source positions so adding positions does not
silently multiply record multiplicity. Each complete neighborhood aggregate is
finished before the next residual update. Record chunks are only an associative
execution strategy.

After every block, the normalized aggregate and log mass update each target
residual. A small linear target-slot mixer remains after aggregation. The final
projection returns 32 decoder-width soft positions for recurrent injection.

The MLP is shared over positions, not necessarily across block depth or memory
spaces. Separate blocks learn successive computations; separate spaces may
specialize while their updates meet in one target residual stream.

## 3. Positional compactor

The legacy synthetic compactor encodes complete flat records, mean-pools them,
and emits complete flat records from one global MLP. It is not the target Phase
2 compactor.

The positional compactor uses the same operator pattern as the reader. Its
target residuals represent the slots of one or more synthetic records. Repeated
shared message blocks map child-record positions into those target positions,
then a shared projection emits stored channels. A separate head distributes the
input multiplicity across synthetic records, preserving total mass.

Overlapping local fields remain external geometric assignments. Their
responsibilities sum to one for every child, including mass. Operator metadata
uses local density, relative key offsets, source-slot identity, target-slot
identity, and recursion level where useful. Absolute key coordinates are not a
default compactor input. Persistent codes retain source lineage and cannot serve
an unauthorized partial child selection.

The initial implementation supplies the positional operator and mass contract.
Persistent multi-space recursive publication, key-relative metadata, and bank
generation garbage collection remain separate executor work and must not be
described as complete.

## 4. Local distillation and expansion

Phase 2 is a versioned warm-start migration, not an exact resume. It has two
local stages before task warmup.

### 4.1 Distill the eight-position structured interface

The Phase 1 payload widths divide exactly into eight positions with channel
widths `[32,64,128,256]`. Construct an eight-position positional student with
the same total widths as its flat teacher. Copy the backbone, writer, keyspace,
recurrence bridge, and every shape-compatible consumer parameter. Freeze those
copied parameters.

Train the new per-position codecs and operator reader on Phase 1 sources and
fixed read plans using:

1. serialized-precision payload regression from flat teacher payloads to the
   corresponding reshaped positional payloads;
2. per-round reader numerator, mass, and target-state distillation;
3. final returned-memory-token distillation;
4. fixed-plan downstream logit or hidden-state distillation; and
5. payload removal/replacement checks to prevent query-only imitation.

Teacher and student consume the same causal prefix and retrieval plan. The
student never receives future target information. Compactor training follows
after the reader is usable: match uncompacted reader contributions and downstream
behavior instead of treating the old global compactor as the sole target.

### 4.2 Expand eight positions to 32

The channel widths and shared codec projections do not change. Copy the first
eight writer and reader positions. Add 24 writer positions, source-position
embeddings, target-position embeddings, null positions, recurrent workspaces,
and slot-mixer rows and columns. Initialize them from the learned positional
distribution plus small distinct perturbations; preserve the old slot-mixer
block and initialize the new diagonal near identity.

During expansion warmup, freeze all shared and copied parameters and update only
newly introduced parameter slices. Gradient masks recorded in the run identity
enforce slice-level ownership. Distill the first eight outputs against the
eight-position student. Give new positions nonzero targets derived from the old
positional distribution, then use reconstruction, extraction, and stored-payload
dependence objectives to differentiate them. Gradually unfreeze the memory
interface after local warmup.

Because 32 reserved read-result positions change trajectory shapes and recurrent
context, regenerate Phase 2 trajectories and rebuild every bank. A migration
manifest records the teacher checkpoint, copied tensors and slices, initialized
slices, layout, channel widths, distillation data, and objective weights.

The executable preparation sequence is:

```bash
python scripts/distill_positional_interface.py \
  --teacher /archive/runs/phase1-final \
  --episodes /archive/corpora/phase1/train.jsonl \
  --output /archive/runs/phase2-positional-8 \
  --steps 1000

python scripts/expand_positional_interface.py \
  --source /archive/runs/phase2-positional-8 \
  --episodes /archive/corpora/phase1/train.jsonl \
  --output /archive/runs/phase2-positional-32 \
  --steps 500 --write-slots 32 --read-slots 32
```

Both stages write ordinary verified checkpoints, record immutable migration
inputs, support cooperative stop and resume, and keep artifacts on the explicit
output filesystem. Subsequent task training warm-starts from the 32-position
checkpoint with `configs/lfm25_230m_four_space_positional_phase2_spark.yaml`.

## 5. Validation

Before publishing a positional bank:

* test gradients for every codec, source position, target position, space,
  fusion layer, and compactor block;
* compare chunked and unchunked complete-neighborhood aggregates;
* exercise serialized BF16 payloads rather than FP32-only intermediates;
* verify full-graph and replay gradients;
* verify causal query boundaries at every read site;
* perform fixed-plan payload removal and replacement interventions;
* validate interrupted distillation and warmup recovery;
* run the actual pinned Hugging Face model preflight on Spark; and
* report parameter, activation, stored-byte, and operator-compute growth.

Unit losses and teacher agreement establish migration mechanics. They do not by
themselves establish composition, retrieval quality, or parameter substitution.
