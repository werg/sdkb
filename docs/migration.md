# Migration to SDKB 0.3

The name is **Spatially Superposed Differentiable Knowledge Base (SDKB)**.
Distribution/import/CLI: `sdkb`. Main model: `SDKBAgent`. Active code, tests, script
commands, Docker/editor paths, configuration examples and documentation use those
names. No old CLI alias is retained.

Existing safetensors checkpoints contain tensor keys, not pickled Python class
objects. They can be warm-started when architecture/configuration still match. Do
not silently resume changed data or moved input paths; use a fresh stage when adapting
to the new trajectory curriculum.

Historical `experiments/bootstrap` and `experiments/development-v2` files preserve
original commands, paths, labels and checksums. They are measured evidence, not active
imports. Rewriting them would misrepresent which code generated the experiments.
Git history likewise keeps its original names. Current validation is versioned separately.

The Git bundle contains the complete history. Source archives contain no external
teacher datasets, downloaded weights, runtime banks or credentials. Current GitHub
publication status is in [handoff.md](handoff.md), not inferred from a local commit.
