# Reader-capacity intervention

The completed fixed-query and intermediate readouts found weaker character access
after the reader's input projections and shared-state updates. This exploratory
experiment tests greater reader capacity with ordinary language-model supervision.
It does not change source encoding, read boundaries, loss, aggregation, precision,
or the stored/returned interface.

## Declared comparison

- Same broad freshness endpoint (4000 updates), recorded in `inputs.json`.
- Fully reset 256-wide control and fully reset 1024-wide reader. Both have three
  rounds and eight slots. All non-reader weights come from the same checkpoint;
  both start new optimizers and caches. Width is the only arm configuration change.
- 2000 updates, accumulation four, seed 149, existing 8192-world slot-major corpus;
  identical sampling/task/source-count/recurrent-depth schedule. Reader/memory
  learning rate .0001, recurrent core .000005, native Muon ownership, BF16, text
  anchor .1, live fraction one. Equal updates are not equal parameters or FLOPs.
- The reset affects both arms; this is not a clean comparison against the prior
  already-trained 256-wide endpoint. Intermediate head diagnostics motivated the
  choice but did not establish an information-theoretic bottleneck.
- Frozen stored-only confirmation after training: same previously inspected 32
  worlds / 320 questions, original and changed permission/restoration/endpoints,
  none/zero memory controls. Report exact identifiers and rule/action preservation;
  NLL or training fit alone cannot promote either arm. One seed, exploratory scope.
- All artifacts directly on external disk. Periodic cadence 10000 exceeds the
  declared budget, so normal saves are initial and final; cooperative emergencies
  preserve full microbatch, replay, optimizer, RNG and configuration state. Keep
  two complete sets, reserve 10 GiB disk and 8 GiB host RAM, .35 CUDA fraction,
  compute watchdog 300 seconds, offline W&B. Two children maximum.

## Execution and recovery

```sh
python experiments/binding-reader-capacity-20260920/run.py \
  --root /archive/runs/binding-reader-capacity-20260920
```

Use the recorded frozen checkout and matching `PYTHONPATH`/Git environment. Root
and queue STOP controls interrupt both children, which save before returning. The
controller joins them. `--resume` checks inputs, acknowledges inactive controls,
resumes partial stages and skips verified complete training endpoints before
continuing the resumable frozen confirmation. Never delete incomplete stage
folders to force a restart.

Regression coverage verifies explicit opt-in, all unchanged non-reader initial
weights, rejection of stored-interface changes and bitwise full-state Muon resume
after a reader reset. Native model preflight is required before launch.

The supplemental `text-controls.json` was declared while both arms were still
training, before endpoint results. After primary confirmation, each endpoint
also receives the same 64 selected-text identifier questions and repeated
no-memory questions, using frozen `b38564a` code. The repeated no-memory strings
must match the latent reference. These controls verify the updated controller's
text-copying path; they do not add a training objective or alter either run.
