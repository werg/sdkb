# Scale-out v0.9 validation

Date: 22 September 2026.

The complete core suite ran with `CUDA_VISIBLE_DEVICES` empty in the pinned Spark
project container, alongside but without entering the Phase 1 GPU process:

```text
635 passed, 4 third-party/runtime warnings in 64.41 seconds
```

`ruff check src tests scripts` and Python bytecode compilation also passed.

Regression coverage establishes that:

* the SQLite reference and a proxy exposing no SQLite connection satisfy the public
  stored-read contract;
* public lookup retains opaque identity, source provenance, scope, and selection
  order, including duplicate selected IDs;
* compact rebuild jobs lease only the lowest ready recursive dependency layer;
* another worker cannot claim an active lease, and a failed worker releases its job;
* publication checks lease ownership and every child cursor captured when the job
  was claimed, rejecting output if a child changed during compactor inference;
* successful publication uses the ordinary all-space journal transaction, clears
  invalidation, and clears the lease;
* compaction promotion accounting rejects behavioral changes even with large byte
  savings and rejects behavior-preserving codes that miss the byte threshold; and
* the bounded recurrent pipeline preserves its gradient reference while reporting
  request/service latency, ready-queue delay, blocking time, and pending depth.

These checks validate the adapter boundary and local coordination semantics. They do
not validate a network service, ANN recall or performance, latency hiding, or useful
learned compaction. Those require the staged evidence in
[scale-out v0.9](scale-out-v0.9.md).
