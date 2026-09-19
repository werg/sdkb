# Pretrained causal pilot — 2026-09-19

Status: the full 1,200-update pilot, cross-stage comparisons and depth sweep
completed successfully. No SDKB training/evaluation process remains running.
Implementation baseline: `91f2442` (following review fixes in `b35279f`).

## Observed result

The final model scored **96/96** on new-world Boolean questions, including
**32/32 XOR compositions**. XOR accuracy fell to **16/32** with no memory,
zeroed payloads, or either required support removed. Flipping either stored
source bit produced the correct changed answer in **32/32 pairs** for each bit.
The text-access control also scored 96/96 and passed both flip tests.
Latent warmup already achieved 96/96; joint training did not improve accuracy on
this test. The final XOR advantage over zero payloads is 50 percentage points,
with a world-bootstrap 95% interval of **34.4–68.8 points**. This interval does
not measure variation across training seeds.

Aggregate results, paired intervals and runtime provenance are in
[results.json](results.json). All 1,200 updates had finite logged scalar metrics.
Peak allocated GPU memory across training stages was approximately **4.39 GiB**;
this is allocator usage, not total shared physical memory or a capacity frontier.

This is evidence of causal stored-memory composition on a small two-bit task.
It is one training seed, uses oracle record selection, and scores candidate
answers by summed token log probability including EOS. It is not evidence of
free-generation agent success, learned retrieval or parameter substitution.

Transfer to the untrained binding family is incomplete:

| Task | Stored memory | No memory | Zero payloads |
|---|---:|---:|---:|
| Permission fact | 73.4% | 42.2% | 42.2% |
| Restoration fact | 89.1% | 50.0% | 50.0% |
| Bound action | 50.0% | 36.7% | 50.0% |
| Exact identifier | 29.7% | 34.4% | 31.3% |

There are 64 queries per fact/identifier family and 128 action queries, clustered
in 32 worlds. The action result has no advantage over zero payloads; always
choosing STOP would also score 50%. The Boolean result must not be generalized
to useful procedural composition or exact identifier transfer.

The fixed-bank depth sweep scored 50% at depth 1 (no inter-pass read) and 100%
at depths 2, 3 and 4. Depth 4 is outside the final training depths but showed no
accuracy gain. These are 96 questions from 32 validation worlds, separate from
the final post-freeze comparison worlds. No separately trained one-pass latent
baseline or general-language retention benchmark was run.

The next research bottleneck is broader binding/action transfer. A follow-up
should train varied multi-entity rules and test disjoint worlds and task forms,
then compare memory against the same text evidence and fixed-layout payload
controls. Repeating the present result across seeds and an attention comparator
is also needed before architecture-level claims.

## Question and fixed protocol

Can the pretrained recurrent student learn to read stored latent facts and compose
both facts on fresh Boolean worlds? This is a controlled transfer experiment,
not a test of agent task success or parameter substitution.

Use the four-stage `recipes/looped_causal.yaml` curriculum: 200 text-bootstrap
updates, 200 recurrence-bridge updates, 400 latent-warmup updates, and 400 joint
updates. Each optimizer update accumulates four examples. Training uses 128
balanced worlds with A, B and XOR questions. Seed is 17. The pretrained model is
`LiquidAI/LFM2.5-230M`, pinned to revision
`40cb2ad3b3044d5a41eee083a6103c8b523afa45`.

Before the full pilot, run two updates per stage on four worlds to measure
throughput and validate the actual optimizer, stage transitions and archival.
These throughput runs are separate checkpoints, not warm starts for the pilot.
The actual throughput record is in [throughput.json](throughput.json). Second-update
times were approximately 0.35–1.1 seconds across stages under contention; they
exclude checkpoint/archive and evaluation overhead and are not serving benchmarks.

The fixed evaluation uses 32 fresh worlds (96 questions) with a frozen writer.
Sources are written once per bank, serialized and read from storage. Report
A/B retrieval separately from XOR composition. Compare full memory, no memory,
zero payloads, each required support removed, and separately written source-bit
counterfactual banks. Counterfactuals preserve source IDs and read plans.

Evaluate the text-bootstrap checkpoint on the same fresh worlds to check that
an information-matched text reader can solve the task. Also evaluate the bridge
and latent-warmup checkpoints to localize failures. The text control differs in
representation and compute; it is not a resource-matched frontier comparison.
Report all stages, including failures, without changing training budgets after
seeing the held-out result. Any follow-up intervention gets a new run identity.

Useful composition requires XOR accuracy above the balanced 50% baseline, an
advantage over payload ablations and either-support removal, and appropriate
answer changes under flips of either source bit. World-bootstrap uncertainty
covers worlds only; one training seed does not establish seed robustness.

## Local execution

The runtime is native ARM64 NGC 26.03 from the existing
`docker-smoke-flashqla:latest` image, with the repository's isolated installer
preserving vendor Torch/CUDA. This is not validation of a fresh NGC 25.11 build.
The independent container is `sdkb-causal-pilot`; unrelated GPU jobs remain running.
Shared-device elapsed time is measured under contention.

Local generated inputs are under `.sdkb/causal-pilot/`: base config, throughput
recipe, full recipe and provenance. The base differs from the checked-in config
only by immutable model revision and operational options: offline W&B, local
checkpoint retention two, archive retention two, 10 GiB free-space reserve,
per-update logging and archive root `/archive`.

Host mounts:

- `/home/werg/sdkb-runs` -> `/runs`: active checkpoints and complete results.
- `/home/werg/.cache/sdkb` -> `/cache`: persistent model/download cache.
- `/mnt/external/sdkb-archive` -> `/archive`: verified background archives.

Commands inside the retained container use `/opt/sdkb-venv/bin/sdkb`:

```bash
sdkb runs start --recipe .sdkb/causal-pilot/throughput.yaml --output /runs/causal-throughput-20260919
sdkb runs start --recipe .sdkb/causal-pilot/recipe.yaml --output /runs/causal-pilot-20260919
sdkb runs status --output /runs/causal-pilot-20260919
sdkb runs stop --output /runs/causal-pilot-20260919
# After a checkpointed stop:
sdkb runs start --recipe .sdkb/causal-pilot/recipe.yaml --output /runs/causal-pilot-20260919 --resume
```

The stage comparison ran as a separate process in the same container:

```bash
python scripts/evaluate_causal_stages.py --run /runs/causal-pilot-20260919 --wait
```

It exits on a stopped/failed curriculum rather than restarting training. Reports
are checkpoint/input-hash bound and a repeated invocation reuses complete reports.
If a stopped curriculum is resumed, explicitly restart this comparison command.
The command passed the complete four-stage tiny CPU run and report-reuse check,
the pretrained throughput checkpoints and the full pilot. The core suite passed
all 186 tests; Ruff and diff checks passed. Four dependency warnings and two
pytest-cache ownership warnings were reported; the cache ownership was corrected.

Training console: `/home/werg/sdkb-runs/.sdkb-control/7feab66d4470e114f8be5a35/console.log`.
Comparison console: `/home/werg/sdkb-runs/causal-pilot-stage-comparison.log`.
Final comparison: `causal-pilot-20260919/stage-transfer-comparison/comparison.json`
under the host run root.

Only numerical summaries and protocol records belong in Git. Checkpoints,
prepared episodes, bank files and offline tracking logs stay in the run storage.
The retained container is now idle and preserves the tested environment. W&B
stayed offline. [Checkpoint status](checkpoint-status.json) records matching final
local/archive pointers and retention counts for all four stages.

## Reproduce the inputs and report

[inputs.json](inputs.json) preserves the exact recipe/base configuration, model
lock and prepared-file hashes. From the repository root, reconstruct the local
input files (JSON is accepted by the YAML loader):

```python
import json
from pathlib import Path

record = json.loads(Path('experiments/causal-pilot-20260919/inputs.json').read_text())
recipe_path = Path(record['recipe_path'])
recipe_path.parent.mkdir(parents=True, exist_ok=True)
recipe_path.write_text(json.dumps(record['recipe'], indent=2))
Path(record['recipe']['base_config']).write_text(json.dumps(record['base_config'], indent=2))
```

Use a fresh output directory when reproducing training. Configure archive paths
for the local machine before starting; changing inputs creates a new run identity.
The exact report extraction inside the tested container was:

```bash
python experiments/causal-pilot-20260919/summarize.py \
  --run /runs/causal-pilot-20260919 \
  --output experiments/causal-pilot-20260919/results.json
```

Depth-sweep command:

```bash
sdkb evaluate-depths --run /runs/causal-pilot-20260919/recurrent_joint \
  --episodes /runs/causal-pilot-20260919/data/validation.jsonl \
  --output /runs/causal-pilot-20260919/depth-sweep \
  --depths 1 2 3 4 --max-episodes 96
```
