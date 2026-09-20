# Selected-text answer-state alignment

The completed reader-capacity intervention did not establish reliable identifier
recall. This experiment tests a richer training signal while holding architecture,
source checkpoint, data and sampling fixed.

## Declared protocol

- Both arms warm-start the same broad 8192-world, 4000-update endpoint, without
  resetting any weights. Optimizers and caches start fresh.
- 800 Muon updates, accumulation four, seed 181, sampled depths two/three, writer
  depth one, reader width 256, BF16. Memory LR .0001 and recurrent-core LR .000005.
- Control has alignment weight zero; treatment has weight one. Both retain .1
  selected-text NLL anchor at depth one. Treatment reuses the anchor forward,
  detaches its answer-prediction states for cosine alignment, and sends additional
  gradients only through the latent path. The shared current text model is an
  anchored moving teacher, not a separately frozen teacher.
- Mean `1 - cosine` over corresponding next-token positions, including EOS.
  Ground-truth preceding answer tokens appear in both causal training paths;
  target/future information never enters writer, query or read inputs. Teacher
  states are a loss target only. The actual selected evidence must match.
- Fully live oracle evidence, required supports, selected-only producer work in
  both arms. That policy has a prior native 50-update equality check; it changes
  neither selected reads nor inference payloads. No compaction or source-text
  access during latent inference.
- Same inspected 32 worlds / 320 questions and all six memory/counterfactual
  conditions. Report exact identifiers and correct original/changed pairs,
  rule/action preservation and selected-text identifier controls. Lower NLL or
  alignment loss alone cannot promote a result. One seed, exploratory scope.
- All artifacts on the external disk. Cadence 10000, initial/final/emergency only,
  keep two complete states, reserve 10 GiB disk and 8 GiB host memory, CUDA fraction
  .35, 300-second compute watchdog, offline W&B, at most two children.

`run.py --root /archive/runs/binding-text-alignment-20260920` uses the existing
verified two-child training/confirmation controller. Use its frozen checkout and
matching Python/Git environment. Root/queue STOP controls join children after
cooperative recovery. `--resume` verifies immutable inputs and resumes unfinished
stages; completed training is not rerun.

This adapts bgkit2's lesson of matching representations at aligned positions, but
uses a different teacher and representation. It is not a claim that bgkit's
frozen-encoder distillation has been reproduced, nor a capability result.
