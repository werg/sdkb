# Local review evidence

These are new numerical validation records, not updates to historical results.
Base revision: `45f62681a616d255991d2fdba3cdfcfd808c3026`, with uncommitted
review fixes. Model revision: `40cb2ad3b3044d5a41eee083a6103c8b523afa45`.

- [model-probe.json](model-probe.json): actual pretrained GPU preflight from
  `sdkb model-probe --config configs/lfm25_230m_looped_spark.yaml`.
- [gpu-smoke-config.json](gpu-smoke-config.json): exact short training configuration.
- [gpu-smoke.json](gpu-smoke.json): one-update stop, resume to two updates, empty
  all-live stale cache, and stored-transfer/counterfactual evaluation summary.

The preflight ran as the container user root, where Git did not resolve the
bind-mounted working tree; its Git fields are null. Training ran as the workspace
owner and records the base commit and dirty working tree. Reports are preserved
as emitted, without filling missing provenance retrospectively.

To reproduce the training boundary check in a suitable native NVIDIA environment:

```python
from sdkb.config import load_config
from sdkb.training import train

config = load_config('experiments/local-review-20260919/gpu-smoke-config.json')
train(config, '/fast/sdkb-review', stop_after=1)
train(config, '/fast/sdkb-review', resume=True)
```

The smoke used a frozen pretrained backbone and trainable memory/bridge modules.
Two updates and one counterfactual pair provide no evidence of learned
composition. See [validation scope](../../docs/validation-local.md).
