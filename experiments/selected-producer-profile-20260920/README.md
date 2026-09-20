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

## Completed result

Both profiles completed all 50 updates. Every model tensor, optimizer/RNG value
and scientific training metric is **bitwise identical**. Final model SHA256:
`3e16ed5da135fd4dcc65602723af261e38711ac12f43e343da6853d16852747a`.
The complete final recovery sets remain external; verified weight hard-linking
reclaimed 1,017,349,676 duplicate bytes. Initial checkpoints were retired by the
predeclared one-set retention policy.

Observed median update time after the first five updates was 1.9674 seconds for
the reference and 1.4999 seconds for selected producers. These are **not an
isolated causal speedup estimate**: arms ran sequentially alongside capacity
training, and the reference also overlapped core tests and the native preflight.
The supported result is native numerical equivalence with fewer unused producer
calls; hardware throughput should be profiled under matched contention separately.
The option remains opt-in and the active capacity experiment is unchanged.

All 479 core tests pass, including complete partial-microbatch Muon recovery under
the new policy. Native model preflight also passes. No held-out capability result
is inferred from this performance/numerical check.
