# Research status and the remaining claim

Updated 20 September 2026. This is a synthesis of completed evidence and current
work. Individual experiment records retain their original protocols and results.
The [architecture](architecture.md) remains the research objective; implementation
and tests alone do not establish that objective.

## Where the project stands

The native pretrained model can use stored latent records to combine two simple
rules into correct actions on new synthetic worlds, when the relevant records
are supplied. Training can also make that behavior survive a mean-plus-mass
compact code. These are useful scoped results. Reliable learned addressing,
exact unseen facts, real agent execution and reduced resident-model requirements
remain open.

| Part of the idea | Executed evidence | Present limit |
|---|---|---|
| Stored-only inference | Frozen evaluations serialize/reopen banks and forbid writer calls during reads. | Offline bank construction still uses a frozen writer; it is a separate operation. |
| Combining separate evidence | The useful MLP source gets 128/128 fresh oracle-supported actions, correct changed-rule pairs and invariant branches. | This is a small authored permission/restoration distribution with oracle selection. |
| Detailed transferable memories | Selected source text copies 64/64 unseen endpoints; the same information through latent payloads yields 0/64. | More worlds, alternate histories and fixed questions have not established exact recall at their tested budgets. |
| Learning versus memorizing | A focused continuation fits all 64 training endpoints through memory. | It fails all replaced and fresh endpoints and degrades rule/action behavior; it is not promoted. |
| Learned global addressing | Corrected keys generate 102/128 actions at 128 records and 78/128 at 4,096 on the measured questions. | Retrieval still misses relevant entities; an extra sparse lexical index gets 128/128 on this literal-name fixture. |
| A compact dense lexical control | Equal 64D key blobs and three fixed seeds were tested, with all results retained. | Full action-pair retrieval is 2/0/0 out of 128; a prior-corpus centering follow-up is 0/0/0. Sparse lexical success is not an established equal-byte dense alternative. |
| Learned compactability | Temporary training lets mean and learned codes retain all tested 128 action answers and rule-change pairs; a raw-trained mean control gets 81/128. | Learned codes do not outperform mean-plus-mass. The raw path can regress; pairing objectives has mixed results. |
| Persistent compaction | Full-cluster codes, mass accounting, visibility and partial-selection raw fallback are implemented and exercised. | Raw fallback records remain stored, so these runs do not establish net disk savings or arbitrary-subset coding. |
| Capacity substitution | The real native LFM student, memory interface and recurrent compute run on Spark. | No matched quality–resident-memory–latency comparison shows that external memory replaces a larger resident model. |
| Agent usefulness | Teacher-data preparation, causal-prefix protocols and imitation evaluation are implemented. | Source commands are inert data. Teacher NLL and authored action strings are not measured tool execution or agent success. |

Sources: [fresh source controls](../experiments/binding-endpoint-freshness-20260920/README.md),
[exact detail](../experiments/binding-exact-detail-20260920/README.md),
[copy fit](../experiments/binding-copy-fit-20260920/README.md),
[routing precision](../experiments/binding-routing-precision-20260919/README.md),
[lexical generation](../experiments/binding-lexical-routing-20260920/README.md),
[dense lexical controls](../experiments/binding-dense-lexical-20260920/README.md),
[centered controls](../experiments/binding-centered-lexical-20260920/README.md),
[temporary compaction](../experiments/binding-compact-aware-20260920/README.md),
[paired compaction](../experiments/binding-paired-compaction-20260920/README.md).

## What the current experiment addresses

The [endpoint freshness study](../experiments/binding-endpoint-freshness-20260920/README.md)
is running two 4,000-update Muon continuations from the useful original MLP source.
Both consume the same task/source-count/recurrent-depth schedule and retain rule
and action tasks. One repeatedly samples 512 endpoint strings; the other samples
7,516 distinct strings, mostly once. World content also differs, and equal updates
are not equal token or FLOP budgets.

The sealed held-out split excludes training source IDs and both original and
inverted endpoint targets. Source controls have completed: perfect tested
rule/action behavior, 0/64 latent identifiers and 64/64 selected-text identifiers.
Training endpoint results are pending. Exact generation, changed-endpoint answers,
irrelevant-rule invariance and retained action composition will be assessed
together. A lower teacher-forced loss alone will not justify promotion.

The completed small-corpus diagnostic matters here: memory-dependent training fit
can coexist with predictions drawn almost entirely from the old answer vocabulary.
Likewise, a wrong prediction changing after an intervention is sensitivity, not
correct recall. Those failure modes remain explicit in the confirmation reports.

## What would support the stronger idea

These are requirements for future evidence, not implemented or completed results:

1. **Useful transferred content:** repeat exact and semantic transfer on genuinely
   new experiences, with source-matched text, no/zero/permuted memory controls and
   targeted interventions. Separate retrieval errors from writer/reader errors.
2. **Selection and composition together:** preserve that behavior in larger banks
   with competing entities and permissions, without supplying world membership or
   treating learned selection as authorization. Report the actual selected budget.
3. **A measured compaction benefit:** retain behavior while reducing total stored
   bytes, including exceptions, raw fallback, indices and metadata. Give attention
   the same compaction opportunity. Full-cluster and subset-selective claims differ.
4. **A capacity/compute comparison:** compare smaller-memory-augmented and larger
   resident models under declared information, training and inference budgets.
   Measure the entire resident footprint and end-to-end latency. Warm-cache exact
   scans on this external rotational disk are not cold NVMe or ANN results.
5. **Actual agent outcomes:** evaluate tool execution in a bounded environment
   separately from teacher imitation and synthetic answer generation. No such
   execution success is asserted by the current dataset or binding studies.

The central gap is empirical: the machinery now executes several intended paths,
but the broad capacity-substitution claim still lacks these outcomes. The current
positive results should guide tests without being relabeled as that broader proof.

## Operational state

The [bgkit audit](bgkit-audit.md) and [operations guide](operations.md) record
external placement, verified relocation/deduplication, sparse saves, emergency
microbatch recovery, complete named optimizer/RNG/config state, offline W&B,
resource reserves and cooperative run control. About 48 GiB was reclaimed from
the internal run tree, which is now about 52 MiB. Current study artifacts go directly
to the external drive.

Muon uses the installed NVIDIA Torch implementation for eligible matrix transforms,
with AdamW for embeddings, slots, fallback tokens and other excluded tensors. The
fallback-token ownership correction changes future optimizer groups; older runs
must resume with their frozen checkout or explicitly warm-start a new run. The
active freshness study stays frozen at `c58028c`.

The latest download-free suite passes 448 tests; native pretrained execution and
a real BF16 Muon emergency-resume comparison have also run. CPU/CUDA configuration
and portable operating helpers are separate from the Spark wrapper. Tested native
Spark execution does not imply performance portability to every GPU or backbone.
See [current machine validation](validation-spark.md) for scope and artifacts.
