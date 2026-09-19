# Teacher datasets and episode construction

Source documentation checked September 19, 2026. Upstream names/schemas/licenses
are from the linked primary cards. The parser tests use project-authored fixtures;
live downloads happen on the training machine. Model/data revisions and accepted or
filtered counts are recorded by preparation. No teacher weights or raw external
corpus is redistributed in this repository.

## Selected datasets

| Catalog key | Original source / configuration / split | Role in the curriculum |
|---|---|---|
| `hermes` | [NousResearch/hermes-function-calling-v1](https://huggingface.co/datasets/NousResearch/hermes-function-calling-v1), `func_calling`, `train` | Multi-turn function calls, schemas and observations. Starter and tools recipe. |
| `ultrachat` | [HuggingFaceH4/ultrachat_200k](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k), `default`, `train_sft` | Conversational continuation and use of earlier context. Starter and chat recipe. |
| `swe_smith` | [SWE-bench/SWE-smith-trajectories](https://huggingface.co/datasets/SWE-bench/SWE-smith-trajectories), `default`, `tool` | Coding-agent action/observation traces with resolved labels and instance identity. Prefix and cross-experience recipes. |
| `openhands` | [nebius/SWE-rebench-openhands-trajectories](https://huggingface.co/datasets/nebius/SWE-rebench-openhands-trajectories), `default`, `train` | OpenHands traces of Qwen3-Coder-480B-A35B-Instruct with resolved labels. Coding comparison. |
| `xlam` | [Salesforce/xlam-function-calling-60k](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k), default, `train` | Optional mostly single-turn schema/argument curriculum; not a cross-experience benchmark. |

The cards declare Apache-2.0 for Hermes, MIT for UltraChat/SWE-smith and CC-BY-4.0
for Nebius/xLAM. These are source-card licenses, not a blanket relicensing of
repository code embedded in trajectories. Preserve source/teacher/repository
provenance and required attribution in derivatives. The official xLAM source is
gated; it is excluded from the default recipe. LFM weights retain their own
[lfm1.0 license](https://huggingface.co/LiquidAI/LFM2.5-230M).

LFM's authors position the 230M instruction checkpoint for lightweight tool use and
data extraction. This motivates tools/chat first, followed by bounded coding
commands—not a claim that coding is impossible or that a model-card recommendation
establishes a fundamental capacity limit.

## What is distilled

The teacher target is the **recorded next assistant message**, not logits or hidden
states that the dataset does not provide. No live teacher or paid API is called.
The available per-row `model`/`teacher` identity is retained; otherwise it is recorded
as `not_recorded`. The normalized transcript and hard target drive supervised NLL
through the soft-memory interface.

Successful SWE/OpenHands rows are filtered using `resolved=true/1`. Outcome columns
only affect selection; they are not neural input features. Final `patch` and
`model_patch` fields are never injected into earlier writes/queries. Commands remain
inert text: no shell, browser, file edit, patch application or test suite is executed.
Lower teacher likelihood is not a measured software-engineering success rate.

## Explicit adapters

| Adapter | Input schema |
|---|---|
| Hermes | `conversations` containing `from`/`value`; optional `tools`, `id`, `model`. |
| UltraChat | `messages` with `role`/`content`; `prompt_id`. |
| SWE-smith | `messages` list or JSON, including double-serialized strings; `instance_id`, `traj_id`, `resolved`, `model`. |
| OpenHands | `trajectory`, `repo`, `instance_id`, `trajectory_id`, `tools`, `resolved`; structured calls and observations. |
| xLAM | `query`, `tools`, `answers`; converted to schema/user/assistant. |

Human/gpt/function aliases map to standard roles. OpenAI call argument strings and
Anthropic text/tool blocks are handled explicitly. Calls become canonical JSON
inside `<tool_call>` tags and retain linking call IDs. Tool observations retain
name/call ID where supplied. Source/target whitespace and Unicode code identifiers
are preserved. Unknown roles and unsupported visual blocks are rejected with
filter counts rather than guessed into text.

The outer prompt uses the student's actual chat template. Recorded call targets
use an explicit JSON wrapper. This is not a claim that the wrapper equals LFM's
native Pythonic tool-call format, which is documented in its model card. Native
format conversion and execution validation can be separate experiments.

## Prefix-memory protocol

For assistant target at turn `t`, choose a cutoff before `t`. Older messages are
written as independent fixed-budget source chunks. The query gets an original-task
excerpt and only recent **preceding** messages. The target and subsequent observations
are excluded from both writer and query. Assistant greetings before a first user
message are not eligible targets.

Actual student tokenization determines source chunks and prompt/answer budgets.
Source chunks preserve original characters rather than decoding split UTF-8 tokens.
A few target turns are sampled across each trajectory. Oversized complete assistant
answers are **skipped**, not truncated with a false EOS target.

The text control receives exactly the same selected chunks. If the shared text
prompt is too long, earlier selected chunks are removed and counted for both arms.
The original task/recent query can be explicitly excerpted. These are bounded
training windows, not a requirement that the eventual architecture limit its reads
because producer graphs do not fit. Nor is this already a learned whole-trajectory
lesson extractor: it is a concrete source-to-latent bootstrap curriculum.

## Cross-experience protocol

Support is taken from different already ordered instances in the same repository
(or explicit tool-schema group). Multiple attempts at the **same** issue cannot
serve as each other's support. Support draws bounded chunks from a prior task and
its recorded tail. Round-robin selection preserves multiple prior experiences when
the representation budget permits; provenance lists only retained producers.

The ordering is deterministic and explicitly **experimental**, not asserted to be
the original collection chronology. Prior experiences can include their own completed
solutions because they precede this query in the experiment. The query instance's
future solution is never a prior source. Source IDs incorporate the experimental
time so immutable identity remains consistent across serialized episodes.

Related experiences are not assumed sufficient for every later query. Real-data
episodes label them `provided_context`, with no invented sufficient-group annotations.
The controlled causal suite provides known support groups and source replacements.

## Splits and contamination controls

Split before windowing. All SWE trajectories from one repository stay in one split.
Tool schemas or normalized first-user prompt identities group the other sources.
Exact normalized transcript duplicates are removed across selected sources. This
is not semantic deduplication or external benchmark decontamination; those limitations
are stated in the manifest rather than hidden behind a generic clean-data claim.

Only UltraChat `train_sft` is used; the test and generation-ranking splits are not
training inputs. Validation is a group-held-out subset of the requested upstream
split. A deterministic 256-row shuffle buffer improves local source mixing but is
not a uniform sample of an entire large corpus. `max_rows` limits scanned rows, not
accepted examples or exact bytes downloaded. A source/split yielding no usable
examples fails before training with filter counts: increase the sample budget or
choose a new seed rather than weakening splits silently.

Each prepared episode records original dataset SHA, declared license, known teacher,
trajectory/instance identity, target index/hash, complete-target flag, source indices,
retained producer IDs and experimental/original ordering type. These metadata are
not writer inputs. Training validates unique episode IDs and consistent source IDs,
hashes input files, then uses offsets rather than keeping every episode's text in RAM.

## Stored-only measurement

Evaluation has a separate write phase, serializes payload precision, reopens the bank,
and reads with per-trajectory namespaces and causal timestamps. It includes no-memory,
zero-value and fixed-key/ID payload-permutation controls. Distinct records can share
semantics, so permutation is not mislabeled a guaranteed contradictory counterfactual.
Causal source changes with adjusted targets belong to the controlled family.

The first real-data comparison is teacher NLL at equal source access. The next is
fresh-experience causal transfer. Agent execution/task success remains a separate
future measurement requiring an environment; it is not silently inferred from either.

## Recipe example

```yaml
sources:
  - name: swe_smith
    max_rows: 4000
    shuffle_buffer: 256
preparation:
  max_supports: 4
  max_targets_per_trajectory: 3
  recent_messages: 1
  recent_tokens: 512
  task_tokens: 160
```

The base YAML separately controls source/prompt/target limits (512/4096/512 initially),
write/code/read capacity, and training behavior. Inspect `data/manifest.json` before
scaling. A supplied source revision is honored; otherwise preparation resolves a
commit SHA. Local fixtures are content-hashed and use the same conversion path.
