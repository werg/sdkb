# Public-trajectory Muon curriculum: recurrent bridge

The pretrained one-pass text baseline benefits from prior context on 119 held-out
SWE-smith episodes: token-weighted NLL 1.490 with selected prior text versus 1.937
without it. This establishes a context-utility signal for a bounded curriculum,
not source sufficiency, latent-memory utility or agent success.

The first declared stage uses the same 530 training episodes, pretrained LFM
revision `40cb2ad3b3044d5a41eee083a6103c8b523afa45`, two recurrent passes, and a
**frozen pretrained backbone**. Only the native recurrent bridge receives text-path
gradients. The task NLL is supplemented by weight .1 KL to the one-pass parent;
because the backbone is frozen and one pass bypasses the bridge, this teacher
remains the original pretrained model.

200 native Muon updates, accumulation four, seed 197, memory/bridge LR .0001,
BF16. No source encoder or memory reader is trained in this stage. The declared
source text and targets retain their original causal preparation and provenance.
Repository-held-out validation is unchanged. Validate the resulting text path
before declaring or launching latent warmup; no synthetic-study weights or failed
alignment objective are imported.

All artifacts and runtime caches are external. Save initial/final/cooperative
emergency states only within this budget (cadence 10000), keep two complete sets,
reserve 10 GiB disk and 8 GiB host memory, CUDA fraction .35, watchdog 300 seconds,
offline W&B. Resume with the frozen launch checkout and unchanged configuration.
No live teacher, shell command from a trajectory, or benchmark environment runs.

Native preflight is required before this training stage starts. Outcomes remain
pending; NLL must remain separate from counterfactual transfer and agent success.
