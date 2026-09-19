# SDKB training and causal-transfer protocol


> **v0.4 update:** the recommended entry points are `recipes/looped_smoke.yaml`,
> `recipes/looped_starter.yaml` and `recipes/looped_causal.yaml`. They add native
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
A hard kill returns to the last committed checkpoint. A graceful stop requests an
optimizer-boundary save. Runs with no committed checkpoint are not silently overwritten.

Completed stage/evaluation markers are independent. In particular, a failed binding
evaluation after causal evaluation is retried without retraining or being skipped.
The high-level recipe has fixed step counts; the lower-level `sdkb train --steps`
can extend a compatible run deliberately. Do not advance a stage outside its sealed
launch plan and then expect the launcher to reinterpret it as unchanged.

## Real-student causal transfer

```bash
./scripts/start_spark.sh --recipe recipes/causal.yaml --output runs/causal
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

`sdkb evaluate-teachers --run RUN --episodes FILE` measures complete next-message
likelihood using a serialized stored-only bank and fixed-ID/key value ablations.
It reports both token-weighted and per-example scores, plus trajectory-clustered
uncertainty for paired memory benefits. Optional `--generate-tokens` saves generated
text and exact-reference matches; neither is a patch verifier or agent task score.

All downloaded instructions, shell commands and code remain data. A separate execution
environment would be needed to assess tool/patch success. No live teacher endpoint or
tracking service is necessary for the implemented imitation curriculum.

## Compaction and comparisons

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
