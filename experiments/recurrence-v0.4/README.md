# v0.4 recurrent-conversion execution evidence

These files record a **tiny CPU model**, not LFM2.5-230M. The retained `model_id`
configuration label is not evidence that the pretrained model was loaded; backend
is `tiny`, resolved revision is `local-tiny`, and width is 32.

The four stages ran for **one optimizer update each**, on two authored Boolean
worlds. The model probe, stored-only teacher/causal/multi-use evaluations, idempotent
resume, and fixed-writer depth sweep executed. These are correctness and execution
checks, not evidence of useful learned recurrence or a new transfer result.

Commands:

```bash
sdkb launch --recipe recipes/tiny_looped_smoke.yaml --output runs/tiny-looped
sdkb launch --recipe recipes/tiny_looped_smoke.yaml --output runs/tiny-looped --resume
sdkb evaluate-depths --run runs/tiny-looped/recurrent_joint \
  --episodes runs/tiny-looped/fresh-causal.jsonl --output runs/tiny-depths --max-episodes 3
```

The test report includes skipped native-HF tests because Transformers is unavailable
in this execution environment. CPU CI installs the pinned HF extra and is configured
to run them, but that remote CI has not been executed here.
