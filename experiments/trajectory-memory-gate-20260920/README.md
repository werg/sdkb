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

Held-out confirmation uses a frozen newly serialized bank at R=2 and compares
real, zero, wrong and absent values to the uniform control using identical
episodes and original read plans. The primary metric is the paired real-over-
zero NLL change together with real-value NLL; a larger contrast caused only by
degrading the zero condition is not success. The one-pass text teacher should
remain identical. This reused validation set is exploratory, not a sealed final
test. Teacher-forced NLL is not generation, task success, learned routing or
capacity substitution.

## Completed matched result

The native run completed all 400 Muon updates from the same joint parent, with
the requested initial gate 0.30 recorded in the initial full checkpoint. Both
the control and gate checkpoints were evaluated on the same 119 held-out
episodes, each from a new stored bank built with 375 writer calls. Their episode
identities, selected source IDs and read plans match. The one-pass selected-text
control is exactly identical across all 238 scored sequences (NLL 0.866650).

| R=2 token-weighted teacher NLL | Uniform control | Trained gate fork |
|---|---:|---:|
| Real stored values | 0.971024 | 0.972040 |
| Zeroed values | 0.980714 | 0.982207 |
| Wrong values | 0.982071 | 0.986463 |
| No memory | 1.089557 | 1.090899 |

The real-value condition worsens slightly. The paired mean episode change in
real-over-zero benefit is −0.00121 NLL (42-trajectory descriptive bootstrap
interval −0.00551 to 0.00317). The earlier read-only gate overlay's improvement
does not survive this matched training run. Do not promote the raised initial
gate as a default or infer that the bridge is a proven information bottleneck.

`results.json` pins the analyzer, model, input and evaluation digests. Complete
initial/final checkpoints, W&B logs and raw scores remain on the external disk.
The verified deduplication pass found no byte-identical weights in this fork:
the changed initial gate changes its model file, and both final weights differ.
All eight related complete checkpoint sets passed manifest verification;
`dedup.json` records the check. The validation set was reused after inspecting
read-only gate overlays; the interval is descriptive. These scores measure
teacher NLL, not controlled transfer, agent success or capacity substitution.
