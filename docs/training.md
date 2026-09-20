# SDKB training and causal-transfer protocol

Detached runs, graceful stops, W&B and external-disk checkpoint archives are
described in [portable operations](operations.md).


> **v0.4 update:** the recommended entry points are `recipes/looped_smoke.yaml`,
> `recipes/looped_starter_muon.yaml` and `recipes/looped_causal.yaml`. They add native
> middle-block recurrence and in-loop reads. See [recurrent conversion](recurrence.md)
> for the four-stage protocol. The one-pass recipes below remain control experiments.

## Foreground staged launcher

The entry point is `sdkb launch`, normally invoked through `scripts/start_spark.sh`.
The same Python launcher is exercised in CPU integration tests; it is not a placeholder
shell command. It resolves the model revision, loads its tokenizer, prepares immutable
inputs, runs the model/gradient preflight, executes stage dependencies and evaluates
each committed stage. `--prepare-only` stops before model/GPU allocation.

The default student is the native instruction-tuned `LiquidAI/LFM2.5-230M`, not its
Base, GGUF or inference-server variant. The initial memory configuration uses one
space, oracle retrieval, one backbone pass, eight write/read slots and three pooled
reader rounds. Existing recurrence and attention/multiscale alternatives remain
independent configurable experiments rather than a compulsory bundle of changes.

## Objectives and stage transitions

`text_bootstrap` adapts the interpreter to the normalized recorded-target task using
exactly the selected support text available to the memory arm. `latent_warmup` loads
that checkpoint, freezes backbone parameters, and trains write slots, query/codecs
and reader through the frozen network. Gradient propagation into soft inputs still
works with frozen weights. `latent_joint` unfreezes with a smaller backbone learning
rate and optionally retains a same-example oracle-text loss.

For memory stages the recorded target NLL flows through the fixed soft response and
reader into the selected live source producers. Selective replay recomputes those
producers after accumulating consumer cotangents. Shared parameter gradients are
all accumulated before the optimizer step. Cached values remain their actual stored
encodings; replay is not falsely described as an unbiased all-fresh gradient.
Fresh encoding remains a **training** operation. Evaluation reads never invoke it.

The optional `oracle_anchor_weight` adds text-path target NLL on the same selected
evidence. Logs distinguish the raw memory `loss`, `oracle_anchor_nll`, and actual
weighted `optimization_loss`. A stage warm-start uses a new optimizer/cache; exact
resume restores model, optimizer, RNG and intentionally stale cache instead.

`oracle_alignment_weight` is an experimental, default-zero training objective for
native in-loop oracle memory without compaction. It adds mean `1 - cosine` between
the latent decoder's final answer-prediction states and detached selected-text
states at the corresponding next-token positions. The text path shares current
model weights and must retain a positive `oracle_anchor_weight`; it is an anchored
moving teacher, not a separately frozen model. Its depth is `oracle_anchor_loops`.
Both paths receive the same selected evidence and preceding answer tokens only;
the text states supervise a loss and never enter a memory query, writer or read.
Training rejects a schedule that has not actually read the declared evidence.
The text forward is reused for its NLL anchor. `oracle_alignment_loss` is logged
separately, included in `optimization_loss`, and its accumulated total survives
partial-update emergency recovery. Changing its weight requires a warm-start fork.
This is a learning intervention, not evidence of improved generation or an
inference-time text bypass.

`oracle_distillation_weight` is a separate, default-zero experiment for the
native R=2 memory path. It minimizes next-token KL from the selected-text R=1
distribution to the latent path's answer distribution, alongside latent target
NLL. The one-pass teacher has a frozen parent backbone and is evaluated under
`no_grad`; its answer-position outputs use only the same query, selected source
text and preceding answer tokens. Teacher states and logits never enter a query,
writer or stored read. Both paths use the same selected source IDs, and training
rejects a missed read. The supported configuration is one in-loop oracle read,
writer R=1, fully live payloads, no compaction or other text objective, and no
sampled depth. The loss adds gradients to the consumer, reader and replayed
writer, while the teacher remains detached. `oracle_distillation_kl` and the
weighted `optimization_loss` are logged separately. Emergency checkpoints retain
the partial KL total, optimizer, gradients and RNG; changing its weight starts a
new warm-start fork. A lower KL or NLL alone does not establish payload use or
agent success; compare frozen stored-only real, zero, wrong and text conditions.

Default budgets: four source chunks, 512 source tokens each, 4096 outer-prompt tokens,
512 complete target tokens, a 2048-dimensional per-source storage payload, eight
returned slots. Canonical write, storage-code and returned-read capacities are separate
config axes. Targets are never silently truncated. These bounded windows are not
an architectural limit on trajectory read counts; producer replay addresses graph
storage independently.

Gradient accumulation is four examples per optimizer update. Parameters/optimizer
state remain FP32, forward operations use BF16 autocast, and stored values use BF16.
Reader chunk checkpointing and source replay reduce activation retention, not total
parameter storage or arbitrary serving-context memory.

## Run layout and resume

```text
runs/starter/
  input-lock.json                 # recipe/base-config identity
  model-lock.json                 # actual model revision
  launch.json                     # dependencies and prepared-file checksums
  data/
    normalized.jsonl              # audit messages, source provenance and split
    train.jsonl
    validation.jsonl
    manifest.json                 # source revisions, filters, licenses and budgets
  model-probe.json
  text_bootstrap.yaml
  text_bootstrap/                 # CURRENT, checkpoint sets, metrics/cache
  text_bootstrap-evaluation.json
  latent_warmup.yaml
  latent_warmup/
  latent_joint.yaml
  latent_joint/
  fresh-causal.jsonl
  causal-evaluation.json
  fresh-multiuse.jsonl
  multiuse-evaluation.json
```

Use one immutable recipe/output pair. A changed recipe, base configuration or prepared
file hash requires a fresh output directory. `CURRENT` names a fully committed atomic
checkpoint set. Never use partially written temporary files as a resumable state.
A hard kill returns to the last committed checkpoint. A graceful stop saves after
the current complete microbatch/replay, including any partial gradient accumulation;
resume finishes that same optimizer update. Runs with no committed checkpoint are
not silently overwritten.

Completed stage/evaluation markers are independent. In particular, a failed binding
evaluation after causal evaluation is retried without retraining or being skipped.
The high-level recipe has fixed step counts; the lower-level `sdkb train --steps`
can extend a compatible run deliberately. Do not advance a stage outside its sealed
launch plan and then expect the launcher to reinterpret it as unchanged.

## Real-student causal transfer

```bash
./scripts/start_spark.sh --recipe recipes/causal.yaml --output /runs/causal
```

This uses LFM, not the tiny backend. It learns on controlled Boolean support worlds
through the same text/latent stages, then freezes all weights before preparing new-world
supports. Evaluation writes each new source once, serializes/reopens the store, and
asks multiple later questions. Conditions include both supports, no memory, zeroed
payloads, dropping A or B, and content counterfactuals with adjusted correct answers.
Multi-binding rules and exact endpoint identifiers provide another held-out family.

Known sufficient support groups belong to these generated tasks. The teacher datasets
lack that annotation, so complete-support recall is not invented for them. A larger
model excluded from post-freeze random facts is not a fair capacity-substitution
control; use equal source access when comparing the deployment frontier.

The starter recipe's causal suite is an out-of-domain diagnostic, while `causal.yaml`
trains that task family before holding out new worlds. Failure can reflect interpreter
or format mismatch as well as memory quality. No scientific metric threshold changes
or stops a recipe silently. The goal is to obtain interpretable measurements.

## Real-trajectory evaluation

### Controlled procedural curriculum

`recipes/looped_binding.yaml` uses `protocol: binding` to train on permission,
restoration, action and exact-identifier queries from generated multi-entity
worlds. The four stages and stored-only evaluation are the same as the Boolean
curriculum. `bindings` sets entities per world; `causal_train_worlds` retains its
existing name and controls the generated training world count. Train, validation
and post-freeze worlds have separate namespaces.

Oracle routing supplies the query's relevant source pair. This isolates action
interpretation and latent transfer; it does not establish learned entity selection
or correct binding among all competing records. Text controls receive exactly
the same selected sources. Both rule records are supplied for action queries,
including STOP branches, so selection cardinality cannot reveal the action.

`train.evidence_scope: available` supplies every candidate in the episode to
both text and latent arms. The original `required_ids` and `sufficient_groups`
remain the ground truth for ablations and support metrics; distractors are never
relabeled as sufficient. This option requires oracle candidate access and does
not claim learned routing. For generated binding worlds it exposes all entities
in a world, while world membership is still supplied. The default `required`
preserves existing recipes. Use `recipes/looped_binding_all.yaml` for the matched
all-context curriculum. Producer replay and serialized value precision are unchanged.

The current native-Muon variants are `recipes/looped_binding_muon_selected.yaml`
and `recipes/looped_binding_muon_all.yaml`. They use Muon for eligible matrices
and AdamW for embeddings, output heads and other excluded parameters. They start
fresh optimizer state; an AdamW checkpoint cannot become a Muon exact resume.
See [optimizer ownership and resume](operations.md#optimizers-and-exact-resume) for details.

`sdkb evaluate-transfer --binding-counterfactuals` flips permission or restoration
rules consistently throughout each world. IDs, queries, timestamps and read plans
remain fixed. Scores distinguish answer-changing branches from branches that
should remain unchanged and report each task family separately. Fresh variant
banks are written offline by the frozen writer, then consumed through stored reads.
`scripts/evaluate_causal_stages.py` supports both causal and binding curricula.

Payload interventions replay the original selected IDs and scores at every read
boundary, including native recurrent reads. They measure the value channel without
changing later routing decisions. Learned-routing counterfactual evaluation uses
these captured plans rather than requiring an oracle-trained checkpoint.

The current group-routing loss requires verified support labels and competing
candidates. The runner rejects `retrieval: learned` on `provided_context` teacher
episodes, and rejects training sets in which every candidate is required. Use oracle
training for the teacher bootstrap; learning retrieval from these data needs an
explicit utility/sufficiency supervision protocol. Supplied context is not relabeled
as sufficient merely to make the objective run.

`sdkb evaluate-teachers --run RUN --episodes FILE` measures complete next-message
likelihood using a serialized stored-only bank and fixed-ID/key value ablations.
It reports both token-weighted and per-example scores, plus trajectory-clustered
uncertainty for paired memory benefits. Optional `--generate-tokens` saves generated
text and exact-reference matches; neither is a patch verifier or agent task score.

Teacher evaluation now uses a stable sibling output directory by default. Each
environment and intervention namespace commits its raw frozen-writer records with
an input and contents manifest. A restart verifies the checkpoint model, prepared
episodes, evaluator version and committed bank bytes before reusing them; completed
namespaces do not call the writer again. Each scored condition and its original
read plan are saved together, so a stopped evaluation resumes without repeating
completed NLL calls or changing the fixed-plan payload controls. `STOP` or a
termination signal is checked between namespaces and scoring conditions, and the
launcher publishes a stage evaluation marker only after all conditions finish.
Use a new output directory if checkpoint, data, evaluator, or limits change.
Frozen writer forwards, read planning and score computation use the configured
device-aware stall watchdog; it is disarmed before progress files or SQLite
transactions are flushed to the external disk.
Offline creation retains at most 64 recently encoded sources while constructing
the original namespaces. Wrong-value controls reuse the exact serialized peer
payload bytes after all original scopes commit, preserving each original key,
source ID and time/authorization metadata. A source shared by multiple original
scopes may be encoded again if evicted; actual writer calls and peak cached
sources are reported. A completed verified bank never calls the writer on resume
or during inference.

All downloaded instructions, shell commands and code remain data. A separate execution
environment would be needed to assess tool/patch success. No live teacher endpoint or
tracking service is necessary for the implemented imitation curriculum.

## Compaction and comparisons

Compaction stages receive an additional teacher evaluation using persisted codes,
with raw fallback and payload accounting. Final causal/binding evaluations also
include persistent codes. Reports state when no profitable clusters were built.
Set `compact_records: 1` on a compaction stage for two-record groups; compactor-only
training rejects an oracle dataset where no group can be reduced. Native single-read in-loop compaction supports interleaved and paired objectives;
see the recurrent training section below.

Copy a recipe and add a stage with `compaction: true` and `init_from: latent_joint`
to initialize synthetic compaction with all existing writer/reader/controller weights
frozen. Use a new output directory. Existing configuration also supports paired
raw/compact objectives, merge consistency, overlap grouping and storage noise. The
persistent representation replaces fully selected clusters, retaining raw values for
partial-selection fallback. It is not yet arbitrary-subset compressed decoding or
net disk compression solely because active reads become smaller.

The seven-arm common-data matrix script remains available for attention, direct
latent records, no memory, source-independent extra compute, one reader round and
scheduled multiple reads. Equal information access and separate resource matching
remain necessary. The starter does not claim those comparisons have already run.

## Offline integration

```bash
python -m pip install -e '.[dev]'
sdkb launch --recipe recipes/offline_smoke.yaml --output runs/offline
sdkb launch --recipe recipes/offline_smoke.yaml --output runs/offline --resume
python -m pytest -q
```

This exercises the same preparation, three stages, stored-only likelihood and resume
using authored fixtures and a tiny byte-token model. It verifies execution, not learned
real-student capability. [Validation](validation-v0.3.md) records the tested scope.


## Isolated address training

`train.optimization_scope: routing` is an optional memory-stage scope for learned
routing with positive routing weight and raw records. Only `key_head`,
`address_maps` and `query_maps` train; payload generation, the query feature
extractor, reader and backbone remain frozen. Frozen child modules run in evaluation
mode while the top-level training flag preserves the configured routing warmup.
This preserves oracle-supported behavior and payload content while changing stored
keys and their ranking. The same replay, Muon ownership, checkpoint and exact-resume
contracts apply. Use a new warm-start fork when changing this scientific setting.


## Temporary compaction inside the recurrent reader

For a single-space, single-read native recurrent memory run, set `memory.compaction`
to `mean` or `synthetic`, `compact_records: 1`, and
`compaction_objective: interleaved`. `compaction_probability` chooses compact task
examples after `compaction_warmup`; other examples preserve the raw task objective.
Start an explicit warm-start fork when changing this objective. New synthetic
compactor parameters initialize separately; the source writer/reader state is retained.

The compact read consumes all selected records before its shared-state update and
matches the configured serialized value precision plus FP32 multiplicities. Its
auxiliary loss preserves conditional pre-normalization numerator and mass. Native
scheduled multi-read compaction remains unsupported and fails validation. Both MLP and attention readers support this same intervention. This
implementation and its gradient/storage checks do not establish useful compression;
compare held-out raw/compact counterfactual behavior at a declared update budget.

Set `compaction_objective: paired` to retain raw task loss on each compact-selected
example and add `compact_task_weight * compact_nll`, plus the contribution loss
and optional `behavior_kl_weight * KL(raw.detach() || compact)`. Both paths use the
same causal first-boundary query, selected sources and noisy values. This computes
two decoder paths and changes the loss scale; declare both when comparing budgets.
Groups too small to reduce still receive the paired objective, as in prefix mode.
Use a fresh warm-start fork; an exact resume must keep its saved objective.

## Explicit reader-capacity forks

`train.reinitialize_reader: true` is an opt-in warm-start experiment. It replaces
all reader weights with their seeded initialization while loading every compatible
non-reader weight from `--init-from`. Reader kind, width and round count may change;
stored payload, key and slot interfaces may not. Coupled compactor states are
rejected. The initialization manifest lists every reset reader parameter. Use a
matched reset-reader control when testing a larger reader, since resetting itself
changes behavior. This option is never applied during exact resume; the saved
reader, optimizer ownership and RNG state are restored normally.

## Selected producer computation for fully live oracle training

`train.selected_producers_only: true` is an explicit compute-policy fork. Only
sources already selected by the oracle evidence scope are materialized; every
selected source and scheduled read remains present. Python live/cache draws are
still consumed for all declared sources, preserving the episode/depth sampler.
The policy requires memory training, oracle selection and `live_fraction: 1`.
Cached history and learned routing require their existing full candidate behavior.
Inference and offline bank creation are unaffected.

The default remains false. Omitting an unused stochastic writer forward can change
Torch RNG consumption, so this is a recorded training configuration, not a silent
optimization or an allowed exact-resume override. Full-graph and replay execution
remain available under either policy. A native numerical/timing profile is recorded
separately before adopting the policy in future experiments.
