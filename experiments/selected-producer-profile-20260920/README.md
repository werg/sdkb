# Selected producer computation: native equivalence profile

Two sequential 50-update BF16 native Muon profiles from the same broad endpoint,
with the same 8192-world corpus, seed 173, sampled depth and four microbatches.
The reference materializes every source; the opt-in policy materializes only the
already oracle-selected evidence, while retaining every Python sampler draw.
No scheduled reads are dropped, and no inference-time source encoding is added.

The all-live/oracle restriction avoids stale-cache and learned-addressing changes.
CPU regressions cover both full-graph and replay execution, storage noise and two
scheduled cumulative reads. Every final weight, optimizer and RNG value must match
the reference in the native profile before claiming numerical equivalence on LFM.
Other stochastic writer implementations can consume different Torch draws; the
option stays explicit and exact resume must retain its saved value.

Use frozen code with the existing stoppable sequential continuation runner:

```sh
python experiments/binding-continuation-20260919/run.py \
  --study /archive/profiles/selected-producers-20260920
```

All artifacts are external. Initial/final/emergency saves only; retain one complete
final state per arm and verify weight deduplication after completion. Offline W&B,
8-GiB host and 10-GiB disk reserves, .35 CUDA fraction and compute-only watchdog.
Time measurements are contended by the active capacity study, with sequential arm
order; this is not isolated throughput evidence. Capability is not evaluated here.
