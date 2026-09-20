# Current Spark validation

Updated 20 September 2026. This is the current machine-validation record. The earlier
[0.4 handoff](validation-v0.4.md) remains a historical CPU execution record.

## Runtime and implementation

Native ARM64 execution on DGX Spark/GB10 uses the retained NVIDIA Torch 2.11
development build with CUDA 13.2 and Transformers 5.17.0. The NVIDIA Torch/CUDA
stack was preserved. No x86 emulation was used for SDKB work.

The actual pretrained student is `LiquidAI/LFM2.5-230M`, pinned to
`40cb2ad3b3044d5a41eee083a6103c8b523afa45`. Its numerical/gradient preflight
and native GPU curricula have run. Middle-block recurrence repeats layers
`[4:10]`; the writer remains one pass and evaluation reads serialized BF16
payloads. This verifies execution of the real model, not a simulated GPU backend.

The latest full suite passed **446 tests**, with four existing dependency/runtime
warnings. Ruff passed for `src`, `tests`, and `scripts`. Core tests require no
downloads. Coverage includes full/replay gradients, causal prefixes, serialized
precision, checkpoint recovery, storage visibility and concurrent invalidation.

## Latest research boundary

The useful core-only MLP source and several Muon continuations preserve 128/128
oracle-supported action generations, including the tested rule and identifier
counterfactual conditions. Exact identifiers remain 0/64. More independent worlds,
alternate endpoint histories and one fixed identifier question have not established
exact latent recall at the tested budgets. Both models in the fixed-question
comparison copy 64/64 identifiers from the same selected text. The fixed-query
continuation also regresses direct permission answers to 59/64 in both question
forms and still gets 0/64 on its most exposed training questions. A subsequent
small-corpus fit reaches 64/64 training identifiers but 0/64 held-out identifiers,
with action/rule regressions and failed endpoint interventions. See the
[fixed-query study](../experiments/binding-fixed-query-detail-20260920/README.md).

Learned global retrieval remains weaker: corrected training reaches 102/128 free
actions at 128 records and 78/128 at 4,096 records on fixed questions. Wider address
spaces improve training fit without improving held-out retrieval. These exact-scan
diagnostics do not establish robust large-bank retrieval or an ANN frontier.

Persistent single-space compact codes work at recurrent boundaries with cumulative
partial-selection fallback, mass and visibility checks. Temporary compaction
training preserves 128/128 actions and the tested action counterfactual pairs
through mean codes, versus 81/128 for the matched raw-trained mean-code control.
Its raw path regresses to 126/128; a paired-objective follow-up has mixed raw
results and no demonstrated compact advantage. Learned codes match the mean-code
action result without improving it. Raw fallbacks remain stored. These results do
not establish net disk savings, arbitrary-subset decoding, real agent success or
parameter substitution; the scoped studies and earlier results follow below.

## Operational evidence

| Check | Result and scope |
|---|---|
| Native Muon | New binding runs use native Muon for eligible matrix parameters with AdamW for the excluded parameter groups. Both components and their actual group settings are checkpointed. |
| Emergency resume | A real BF16 LFM run stopped after one of two accumulated microbatches and reproduced the uninterrupted final model, complete optimizer state and RNG state exactly. No partial optimizer update was taken. |
| Checkpoint placement | Twenty-five retained stage/diagnostic checkpoint directories were verified and moved externally. The internal run tree fell from about 49 GiB to 52 MiB; internal free space increased by about 48 GiB. |
| Future launches | This checkout's ignored storage settings select the external run root. Configured directories must exist; relative container training output paths are rejected. Environment overrides remain available on other machines. |
| Save cadence | Defaults and real-model recipes use 1,000 updates, plus initial, final and emergency saves. Recent declared 400–4,000-update studies use a 10,000-update cadence and retain only initial/final/emergency sets. |
| Memory and stalls | Optional host-memory reserve, CUDA allocation fraction and compute stack watchdog are implemented. New source versions log host/CUDA memory trends and checkpoint start/commit duration. |
| Checkpointing cost | Two 100-update Muon profiles had identical training metrics and final weight-file hash. Disabling activation/reader-chunk checkpointing cut median update time 18% and raised peak CUDA allocation from 2.14 to 2.53 GiB. This applies only to the measured short binding distribution, with existing GPU contention. |
| Offline bank writes | Native BF16 individual/bulk writes produced byte-identical stored records and identical results for all 88 evaluation rows, with the writer disabled during reads. Atomic batching avoids a durable commit per record. |
| Concurrent writes/deletions | Regressions reproduced check/insert and lineage/delete races. Explicit SQLite writer reservations now protect those operations; failed bulk writes roll back without exposing partial records. |

Detailed evidence: [checkpoint relocation](../experiments/operations-20260919/checkpoint-relocation.json),
[emergency resume](../experiments/operations-20260919/muon-emergency-resume.json),
[checkpointing profile](../experiments/operations-20260919/checkpointing-profile.json),
[bank equivalence](../experiments/operations-20260919/bank-batch-parity.json).
See the [bgkit adoption audit](bgkit-audit.md) and [operating guide](operations.md).

Direct external checkpoint writes can take minutes under disk contention. They
do not silently make an internal emergency copy. The optional asynchronous archive
path can stage locally, with explicit retention/relocation responsibilities. A
blocked native call must return before cooperative stopping can complete.

## Capability evidence

The [Boolean pilot](../experiments/causal-pilot-20260919/) established narrow
stored-memory dependence on its trained synthetic family, including adjusted-answer
counterfactuals. Its text and warmup controls also succeeded; more shared depth did
not establish an additional benefit. This is not general agent capability or
parameter substitution.

The [Muon binding study](../experiments/binding-muon-20260919/) is a harder test
of permission, restoration, exact identifiers and their combination:

- Both MLP curricula completed their 200/200/400/400 training budgets without
  nonfinite gradients. Earlier AdamW pilots remain explicitly interrupted records.
- The completed all-context MLP scored 46.6% on its 320 fresh binding questions,
  equal to zeroed payloads. Action counterfactuals had no both-correct pairs.
  This run does **not** establish useful binding or composition.
- A separate 100-update Muon warm-start diagnostic recovered both direct rule
  facts, including their counterfactuals, but action composition remained weak.
  Candidate-free generation reproduced zero of sixteen exact identifiers.
- The completed selected-support MLP recalled both rule facts perfectly, but action
  accuracy was 55.5%. Restoration changes spuriously changed actions in 63/64
  branches that should remain fixed; permission flips had only 7/128 both-correct
  pairs. The 69.4% overall score does not establish conditional composition.
- Common-world MLP stage comparisons are complete. Selected-support text controls
  solve every family; all-context text controls score only 50% on actions despite
  perfect direct facts. Final latent action accuracy is 54.7% versus 49.2%, with
  a paired interval including zero. The attention launcher reaches 100% action and rule-fact choice accuracy, with
  every action counterfactual pair correct and no spurious action changes on
  unchanged restoration branches. Candidate-free generation also emits all
  action and rule-fact answers on eight worlds, but zero of sixteen identifiers.
  The common-world reader confirmation also gives attention 100% action accuracy
  versus MLP 53.1% (paired gain 46.9 points, world-bootstrap interval 40.6–53.9).
  Both text controls solve every task; identifier choice remains weak.
- Exposing both entities to the frozen attention reader reduces action accuracy
  from 100% to 68.8%, with degraded counterfactual consistency. This remains above
  zero/no-memory controls, but robust entity binding is not established.

The all-context model's lower NLL versus no memory did not translate into a benefit
over zeroed payloads. Tests, falling losses, linear probes and successful execution
must not be substituted for the causal behavior measurements.

## What remains unproven

Useful multi-entity latent binding; reliable procedural composition outside the
synthetic Boolean/binding families; useful pooled-MLP composition; literal identifier reproduction; learned retrieval at scale;
behavior-preserving compaction of a useful real-student memory system; additional
capability from recurrent depth; agent execution success; and a quality/memory/latency
frontier showing substitution for resident model parameters.

These are research outcomes to establish, not implemented-module checkboxes. The
current work is the paired MLP continuation (extra training versus broader backbone
adaptation) and diagnosis of entity binding. A completed matched attention
continuation with/without distractors did not improve binding: all-world action
accuracy was 62.5% versus 63.3% for its source and 59.4% for extra selected-support
training, with no jointly correct opposed-rule entity pairs after either continuation. No paid teacher
collection, license change or public-visibility change has been initiated.


### Longer Muon MLP continuation

The fixed 1,600-update continuations now both pass selected-support action
composition: 128/128 actions, all changed action counterfactual pairs correct,
and no false changes on unchanged actions. Candidate-free action generation is
32/32 in both arms; identifier generation remains 0/16. Recurrent-core adaptation
alone suffices at this larger budget. This revises the architectural interpretation
of the earlier short-budget failure; it does not establish distractor binding.
See [the continuation record](../experiments/binding-continuation-20260919/README.md).

`evaluate_binding_context.py --learned-world --read-budget 2` also supports
world-scoped exact learned ranking of stored keys. Supplied world membership
sets eligibility; required supports are used only by the named oracle control
and support-removal intervention. Zero values and counterfactuals retain the
original read plans. CPU regression and a native BF16 148-row mechanics smoke
verify selection eligibility and intervention plans. This is not ANN or global
retrieval, and the smoke is not retrieval capability evidence.


Both longer MLP endpoints were subsequently tested with competing entities. Both
fell to 61.7% action accuracy and gave identical answers to both entities in all
opposed-rule worlds (17 permission, 19 restoration), with no jointly correct fact
pairs. The selected-support composition success therefore does not resolve binding.
A fixed-budget, matched routing-supervision study ran from `304fb17`;
its negative results are summarized below. Checkpoints and banks remain on the external disk.


Candidate-free generation now also supports explicit world-scoped learned ranking.
A native BF16 two-world smoke generates 60 rows from an existing bank with the
writer disabled; all selected-ID lists match the corresponding choice and control
rows. CPU regressions cover both existing oracle behavior and learned ranking.


An explicitly training-only identifier diagnostic gets 7/16 exact for the longer
core-only MLP and 13/16 for the full-backbone MLP, versus 7/16 and 12/16 with
zero payloads. Held-out identifiers remain 0/16. Training fit largely surviving
payload removal is not evidence of reading identifiers from storage.


### Learned routing intervention

The matched 800-update routing-loss intervention reduced held-out action accuracy
from the control's 60.9% to 52.3% (paired change −8.6 points, world-bootstrap
interval [−14.1, −3.9]). It learned source-type selection: direct-fact queries always
retrieve both entities' records of the requested type, while 117/128 action queries
retrieve two restoration records. No action query retrieves its complete required
pair. Even oracle-support action accuracy regresses to 69.5%, so joint adaptation
also interferes with composition. Candidate-free action generation is 17/32 in
both continued arms; identifiers remain 0/16. This does not establish entity-aware
routing. See [the full study](../experiments/binding-routing-20260919/README.md).


Freezing the composition path and training only the three address projections
preserves 206 frozen tensors, 384 serialized payloads and 640 oracle/no-memory
rows exactly. Oracle action accuracy stays 100%. Learned action accuracy is
51.6%, with no established gain over the source or joint-routing control;
candidate-free actions are 14/32 versus 15/32 with zero payloads. This isolates
address learning as an unresolved problem without sacrificing working composition.
See [the projection-only study](../experiments/binding-routing-scope-20260919/README.md).

### Address feature diagnostics

Frozen-feature Muon probes now compare the existing compressed query with an
adaptable projection from the full causal state. With 1,024 fresh training worlds
and matched update/query-exposure budgets, full required action-pair recall is
66/128 versus 22/128 for the compressed query on 32 common held-out worlds.
The paired difference is +34.38 points [24.22, 43.75]. Increasing training breadth
from 128 to 1,024 worlds improves the full-state arm by +14.06 points [3.13, 25.00].
These are feature-space retrieval counts, not downstream action accuracy; the
full-state arm has 65,536 extra trainable parameters and only one training seed.
See [the breadth study](../experiments/binding-routing-breadth-20260919/README.md).

An opt-in independent routing head now preserves the original reader query while
using that full-state projection for address search. Native LFM preflight from
`38a2634` gives zero one-pass identity and causal-prefix error and a nonzero routing
head gradient. The frozen stored-memory confirmation completed; source-bound
small address overlays avoided writing redundant backbone checkpoints. Its protocol
and preflight are in [the stored study](../experiments/binding-routing-stored-20260919/README.md).


Stored confirmation gives 60.94% action choice accuracy for the full-state adapter,
versus 52.34% for its source and 48.44% for compressed routing. The paired gain over
source remains uncertain, and candidate-free actions are only 17/32. All 640
oracle/no-memory rows and 384 payload/provenance records remain exactly equal.
A fixed one-versus-two-record fact diagnostic improves restoration generation
from 45/64 to 59/64: 15/19 opposed-rule worlds answer both entities correctly with
one record versus zero with two. This identifies interference and narrow binding;
it does not implement a learned stopping policy or solve action routing.


The fresh learned-count confirmation predicts the one/two-record budget correctly
for all 320 queries and improves single-fact retrieval, but generated actions remain
19/32. See [the count study](../experiments/binding-count-stored-20260919/README.md).
World eligibility is still supplied in that study.

Persistent full-pair compaction now runs at native recurrent boundaries. Attention
mean and fitted codes preserve all 64 sampled candidate-free counterfactual action
answers. MLP statistics-only continuation reaches 116/128 action choices on fresh
worlds versus 128/128 raw, and only 8/16 candidate-free restoration-change pairs
retain both correct answers. Adding decoder answer loss does not improve accuracy
on this confirmation. See [initial compaction](../experiments/binding-compaction-20260919/README.md)
and [fresh confirmation](../experiments/binding-compaction-fresh-20260919/README.md).
All use stored payloads with writer/compactor disabled at inference; partial
selections use raw fallback and no net storage reduction is claimed.

Standalone bank and compact-code publication now survive interruption atomically.
Native BF16 validation interrupted after bank creation, resumed with writer calls
forbidden, and preserved every bank hash and completed report on another restart.
Evidence: `experiments/operations-20260919/compact-bank-resume.json`.


Expanding the same frozen router to all 32 worlds (128 records) exposes a global
addressing failure: action choices fall from 90/128 to 50/128, full required-pair
retrieval from 62/128 to 0/128, and free actions match the zero-value control at
16/32. Oracle/no-memory controls and the existing bank are unchanged. See the
[global routing diagnostic](../experiments/binding-global-routing-20260919/README.md).


Matched cross-world routing supervision improves fresh full-bank action choices
and candidate-free answers to 80/128 versus 55/128 for the within-world continuation;
zero/no-memory controls are 64/128 and 65/128, oracle 128/128. Counterfactual free
permission pairs retain both correct answers for 52/128, restoration for 25/64,
with 14/64 false restoration changes. This is a narrow, unreliable learned-global
memory benefit, not robust composition. See [stored confirmation](../experiments/binding-global-stored-20260919/README.md).
The longer run's [fresh stored confirmation](../experiments/binding-global-stored-long-20260919/README.md)
generates 101/128 actions versus 76/128 short-global and 54/128 long-within-world.
Targeted-source counterfactuals retain both answers for 73/128 permission pairs
and 47/64 restoration-change pairs, with 10/64 false invariant restoration changes.
Earlier type-wide flips affect the whole bank and are weaker evidence of dependence
on the intended source; targeted interventions leave unrelated payloads unchanged.

[Fixed-query bank scaling](../experiments/binding-bank-scale-20260919/README.md)
reduces free actions from 101/128 at 128 records to 91/128 at 512 and 76/128 at
4,096. The 4,096-record advantage over no memory has an interval crossing zero.
Exact identifiers remain 0/64 with oracle supports. Scalability and exact-detail
recall are still open.

A [precision audit](../experiments/binding-routing-precision-20260919/README.md)
corrected BF16 feature-score approximation to match the writer's key normalization
and FP32 search. Native comparison matches all 320 top-two rankings with maximum
cosine error 2.39e-7 on shared serialized projected vectors. Corrected fitting uses
new run identities; historical stored outcomes remain unchanged.

Corrected fitting preserves the known-corpus stored result: 102/128 free actions
for global training versus 58/128 within-world, with all 128 raw payload/provenance
records and 640 oracle/no-memory scoring rows unchanged. Targeted source changes
retain both correct answers for 74/128 permission pairs and 47/64 restoration
pairs. This is an implementation diagnostic on existing questions.

A [width diagnostic](../experiments/binding-routing-width-20260919/README.md)
raises the address dimension from 64 to 128 or 256 with matched Muon updates and
query batches. Training pair retrieval improves, while held-out full pairs fall
from 87/128 to 80/128 and 79/128. The unchanged-width endpoint reproduces model,
optimizer and sampler state exactly. The production address width remains 64;
the wider feature experiment does not establish a useful model improvement.

An [exact-detail control](../experiments/binding-exact-detail-20260920/README.md)
gives 64/64 unseen identifiers from the selected source text but 0/64 from the same
source as a stored latent payload. A separate
[frozen-payload readout](../experiments/binding-payload-readout-20260920/README.md)
recovers 137/384 held-out characters linearly (21/384 with shifted payloads), yet
0/64 complete identifiers. A larger readout fits all training identifiers but also
gets 0/64 held-out exact strings. Some character information remains accessible;
the probes do not establish a general exact-detail solution or prove information
is absent. Matched fresh-world training continuations are evaluating this failure.

### Frozen reader-output identifier readout — 20 September

The [reader readout diagnostic](../experiments/binding-reader-readout-20260920/README.md)
reuses the original frozen source, 4,096-record bank and 992/32-world readout split.
Linear heads recover 137/384 held-out hex characters from payloads, 76/384 from
first-boundary reader outputs, and 22/384 from zero-payload reader outputs. All have
0/64 exact endpoints. The reader features have 8,192 dimensions versus 2,048 payload
dimensions and larger fitted heads, so these are bounded accessibility diagnostics.
They do not prove information is absent or identify an optimal decoder.

The zero-payload MLP fits 1,929/1,984 training endpoints while retaining 0/64 exact
held-out endpoints. Query-only memorization is therefore possible in this supervised
readout; it is not evidence of memory use or proof about the main language model's
mechanism. The rerun payload baseline reproduces prior weights, normalization and
all aggregate outcomes exactly. Targets/source text never enter reader extraction;
source/compactor calls are forbidden and a causal regression changes the target
without changing features.


## Matched endpoint-history continuation (20 September)

The completed source / independent-world / alternate-history comparison on a new
32-world split preserves 128/128 generated actions and the tested rule-change
counterfactuals in every arm. Identifiers remain 0/64 exact in all arms. Alternate
histories lower identifier target NLL to 2.6551 versus 2.7963 for independent worlds,
but 55/64 wrong predictions changing with the endpoint is sensitivity, not recall;
36/64 also change under irrelevant permission interventions. See
[the frozen protocol and full result](../experiments/binding-endpoint-views-20260920/README.md).
No agent-success or capacity-substitution result follows.


## Temporary-compaction learning: mean-code result (20 September)

The matched 400-update raw/interleaved Muon comparison has completed raw and mean
confirmation on 32 fresh worlds. A mean-plus-mass code after temporary training
preserves all 128 action generations and the tested rule-change pairs, versus
81/128 actions for the raw-trained mean control. The temporary-trained raw path
gets 126/128 actions, 122/128 permission pairs and 60/64 restoration pairs, so
preservation of the original raw behavior is incomplete. Native learned-code
confirmation is still running; the paired-objective follow-up is separate.
[Full protocol, counts and paired intervals](../experiments/binding-compact-aware-20260920/README.md).
Raw subset fallback records remain stored: no net disk-saving claim.


The learned-code confirmation has now completed with the same perfect tested
compact action/counterfactual counts as mean codes. All 256 codes differ numerically
across the four banks, but all 1,920 generated strings match the corresponding mean
arm. This supports compactability induced by training; it does not show a learned-
compactor advantage over simple mean-plus-mass. The raw-path cost remains.


The matched paired-objective follow-up is complete on another fresh 32-world split.
Both objectives achieve perfect tested action/counterfactual behavior with mean and
native codes. Raw reads remain mixed: pairing gives 127/128 original actions versus
128/128 for interleaving, with some better changed-rule pairs and one new false
restoration change. Pairing adds decoder work without an observed compact benefit.
[Full result and limits](../experiments/binding-paired-compaction-20260920/README.md).

### Lexical address control — 20 September

A [retrieval-only comparator](../experiments/binding-lexical-routing-20260920/README.md)
uses stored source-token features with visibility-filtered TF-IDF statistics. At
4,096 records it retrieves both annotated sources for 128/128 action questions at
a top-two budget, versus 35/128 for the learned router. Its top-one rank always
favors restoration records in this fixture, so it fails permission/identifier
selection at a one-record budget. The index retains lexical terms and is 1.64 MB;
this is not an equal-representation or equal-storage comparison, a generation result,
or a learned-router improvement. It exposes an easy lexical structure in this
synthetic retrieval fixture. The accompanying audit corrects prose that called
sufficient-group counts (73/128 and 37/128) full-pair retrieval; the full-pair
counts are 71/128 and 35/128. Raw historical records remain unchanged.

The subsequent frozen lexical-selection action check gets **128/128** original
actions, versus 78/128 for the learned router, with the same two selected records
per action. Oracle/no-memory predictions reproduce exactly (128/128 and 64/128);
zeroed payloads give 63/128. The reader consumes stored payloads with writer calls
forbidden. This is an end-to-end literal-name fixture result using the additional
lexical index, not a learned-key improvement or fresh counterfactual-policy result.

### Deliberate copy fit — 20 September

The [small-corpus diagnostic](../experiments/binding-copy-fit-20260920/README.md)
fits 64/64 original training identifiers through stored memory, with 0/64 after
endpoint replacement and only 17/64 after irrelevant permission changes. Fresh
held-out identifiers remain 0/64; actions fall from 128/128 to 114/128, and rule
retention/counterfactual behavior regress. Most new identifier predictions belong
to the old 64-target vocabulary. Teacher-forced suffix prediction is often correct
without memory once a training answer is identified by its supplied prefix. This
is evidence of a fitted finite task with poor transfer, not a general copying rule.

A [character-question preflight](../experiments/binding-character-questions-20260920/README.md)
gets 0/96 single-character answers even with selected text at both one/three loops,
while retaining 16/16 full-endpoint copying. That proposed auxiliary interface has
not been trained and is not ready as a clean latent-memory comparison.


### Endpoint freshness study launched — 20 September

The [precommitted protocol](../experiments/binding-endpoint-freshness-20260920/README.md)
compares 256 versus 8,192 source worlds with matched 4,000-update Muon task/depth
schedules, a fixed full-endpoint question and retained rule/action tasks. Both runs
and the source confirmation launched from frozen commit `c58028c`. Initial/final/
emergency artifacts stay on external storage. The smaller arm samples 512 endpoint
strings repeatedly; the larger samples 7,516 distinct strings, mostly once. World
content also changes, and equal updates do not mean equal token/FLOP budgets.
The sealed 32-world confirmation excludes training source IDs and both original
and inverted endpoint targets. No endpoint result is asserted while training runs.


The [dense lexical follow-up](../experiments/binding-dense-lexical-20260920/README.md)
fails to retain the sparse lexical advantage at equal key-blob bytes: fixed 64D
TF projections recover only 2/128, 0/128 and 0/128 full action pairs across three
seeds. The unprojected no-IDF baseline gets 64/128 versus the earlier TF-IDF 128/128.
All declared widths/seeds are reported. This identifies losses in this fixed encoder,
not an inherent dimension limit or a learned-router improvement.

A [prior-corpus centering follow-up](../experiments/binding-centered-lexical-20260920/README.md)
also fails: all three 64D seeds retrieve 0/128 full action pairs; across larger
widths only one 1024D seed gets 1/128. Its two fitted means use only disjoint
original training sources/queries, never targets. All arms are reported and none
is promoted. The source control for the ongoing freshness study independently
retains perfect tested rule/action behavior and copies 64/64 endpoints from selected
text, versus 0/64 from stored latent memory.
