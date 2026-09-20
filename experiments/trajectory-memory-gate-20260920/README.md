# Memory-injection gate warm-start fork

The frozen trajectory model produces substantially different reader tokens for
real and zeroed stored values, but the final answer states and next-token
distributions change much less. Its conservative native bridge has a trained
memory injection gate near 0.11 and recurrent update gate near 0.11. A read-only
gate overlay improved real-value teacher NLL slightly when **only** the memory
gate was raised to 0.30; raising the update gate harmed likelihood. Those
overlays were inspected on the same validation set, so they are exploratory
diagnostics and do not count as a trained gain.

This fork tests the separate training question. From the same completed joint
parent as the uniform-data no-KL control, it starts a fresh Muon optimizer and
sets only the bridge memory injection gate to 0.30 **after** loading the parent
weights. The recurrent update gate, decoder, writer, reader, source data, seed
233, R=2 consumer, R=1 writer, frozen backbone, BF16, four-example accumulation
and 400 updates match the control. The gate remains trainable. The one-pass
parent path bypasses the bridge, preserving its selected-text endpoint.

`run.py --prepare` locks the exact parent, control config, 530 training episodes,
119 validation episodes and fork config before model allocation. All mutable
outputs, offline W&B, caches and initial/final/full emergency checkpoints are
directly on the external disk. Normal checkpoint cadence exceeds this budget;
a cooperative stop retains optimizer, RNG, sampled state and partial gradients,
then `run.py` resumes the exact checkpoint. The initial checkpoint and
`initialization.json` record the changed gate and original value. Native actual-
LFM preflight executes with the requested gate.

Held-out confirmation will use a frozen newly serialized bank at R=2 and compare
real, zero, wrong and absent values to the uniform control using identical
episodes and original read plans. The primary metric is the paired real-over-
zero NLL change together with real-value NLL; a larger contrast caused only by
degrading the zero condition is not success. The one-pass text teacher should
remain identical. This reused validation set is exploratory, not a sealed final
test. Teacher-forced NLL is not generation, task success, learned routing or
capacity substitution.
