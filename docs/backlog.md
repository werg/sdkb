# Development and research backlog

Checkboxes mean completed engineering or explicitly scoped experiments, not general
scientific validation. Current native results are in `validation-spark.md`; earlier
CPU reference evidence remains in `validation-v0.2.md`.

## Current priorities — 20 September 2026

1. **Latent exact-detail learning.** The same frozen decoder copies 64/64 unseen
   endpoint strings from selected text and 0/64 from selected latent payloads.
   Matched 128/1,024-fresh-world Muon continuations preserve 128/128 oracle
   actions but still produce 0/64 exact identifiers, despite improved target NLL.
   A matched endpoint-view continuation also retains 128/128 actions and 0/64
   exact identifiers. One identical identifier question across all training targets
   also yields 0/64 latent recall while both matched models copy 64/64 from selected
   text; direct permission answers regress to 59/64. Inspect training fit and
   representation/readout learning before spending on another identical curriculum.
   Deliberate small-corpus training now fits 64/64 original identifiers but remains
   0/64 on replaced/held-out endpoints and damages action/rule behavior. Test
   position-level questions only after establishing their text interface: the first
   preflight gets 0/96 even with selected text. Use the working whole-endpoint
   question for a matched target-freshness comparison with rule/action retention.
2. **Global address generalization.** Corrected training gets 102/128 free actions
   at 128 records and 78/128 at 4,096 on fixed questions. Wider addresses improve
   training fit but reduce held-out pair retrieval. A separate lexical index gets
   128/128 action pairs and stored-latent action generations on the same 4,096-record
   fixture. Its extra source-token features expose an easy literal-name baseline;
   learned-key generalization remains unresolved.
3. **MLP conditional compaction.** Native single-read interleaved and paired training now
   passes replay, causal, stored-code and emergency-resume checks. Temporary training
   now preserves 128/128 actions and tested rule-change pairs through mean codes,
   versus 81/128 for its raw-trained mean control. Its raw path falls to 126/128;
   the paired continuation has mixed raw results and no observed compact benefit. Native learned codes match the mean-code action result without improving it. Raw fallbacks remain necessary and count toward disk use.
4. **Real task success and capacity substitution.** Synthetic composition and
   teacher NLL do not establish these. Compare information-matched systems with
   actual task verifiers and resident-weight/storage/latency accounting.

The external-disk, emergency recovery, complete optimizer state, W&B and Spark
runtime work is tracked in `bgkit-audit.md`; it is part of the running experiments.

## A. Numerical reference and runnable development

- [x] Shared student writer/query/soft-token answer path.
- [x] Factorized MLP and attention set readers with recurrent fixed slots.
- [x] Full-graph versus selective-replay gradient tests, including RNG/shared weights.
- [x] Chunked-reader gradient parity and explicit numerator/mass interfaces.
- [x] Versioned safetensors-backed store, temporal/domain filters and deletion lineage.
- [x] Stored-only evaluation with writer disabled in a regression test.
- [x] Support/query trainer, mixed cached/live writes, gradient accumulation, checkpoint/resume.
- [x] General support/query JSONL input and stored-only likelihood evaluation.
- [x] CPU CLI, environment doctor, native Spark Docker/devcontainer configuration.
- [x] Run the native image and real LFM checkpoint on DGX Spark.
- [x] Add periodic checkpoints, coordinated multi-file/cache recovery and an interrupted-run test.

## B. Oracle transfer and composition — first scientific milestone

- [x] Counterfactual retry/restoration generator and intervention evaluation.
- [x] Main/wide-payload/direct-latent/attention configuration controls.
- [x] Establish a partial tiny-model stored-memory/joint-use signal with interventions.
- [ ] Establish reliable counterfactual composition on the actual pretrained student.
  Narrow oracle-selected binding composition now succeeds; global retrieval and
  targeted counterfactual reliability remain incomplete.
- [x] Add multiple uses of one write, variable bindings and separate held-out worlds (new real task families remain open).
- [x] Add a source-independent shared-compute control to isolate extra decoder positions.
- [ ] Train and compare LFM and SmolLM oracle-text ability before attributing reader failures.
- [ ] Import licensed teacher coding trajectories and execute real task verifiers.

## C. Learned access and larger runs

- [x] Complete-plan sufficient-group objective and candidate-local top-k training.
- [x] Streaming exact CPU search, captured immutable read plans and async CPU utility.
- [x] Global heterogeneous bank evaluation, negatives outside each episode namespace.
- [x] Scheduled next-query state updates, remaining-group supervision and replay parity.
- [ ] Conditional marginal-utility sampling, plan exploration and learned invocation/stopping.
- [x] Durable mutable-record journal, age/coverage regeneration of stale or
  never-refreshed entries, checkpoint-pinned revision recovery, and explicit GC.
- [x] Add dependency-ordered, leased compact-code rebuild workers with atomic
  all-space publication and retry release.
- [ ] Add learned-utility maintenance priorities.
- [x] Define public local/network read, key-search and mutable-recovery protocols.
- [x] Add request/service/ready-queue/blocking telemetry to the bounded local
  recurrent microbatch pipeline.
- [ ] Implement a network adapter and run transaction/failure conformance against SQLite.
- [ ] Disk ANN adapter and key/payload layout benchmarks under controlled cache budgets.
- [x] Bounded device payload staging for single-space stored inference.
- [ ] Sparse nested producer dependency replay and out-of-core consumer training without read truncation.

## D. Conditional compaction — independent branch from a useful reader

- [x] Mean-plus-mass and amortized synthetic-record compactor.
- [x] Conditional numerator/mass targets plus free-rollout loss for both readers.
- [x] Temporary compact task steps and raw-task interleaving, including native single-read recurrent training.
- [x] Merge-consistency, alternate-partition, overlap and noise APIs/tests.
- [x] Integrate regrouping/overlap schedules and paired raw/compact task-loss options; comparative ablations remain open.
- [x] Preserve a partial Boolean XOR result under compactor-only training and stored full-cluster codes.
- [ ] Preserve strong behavior on harder related clusters, independent facts and exceptions.
- [x] Persistent full-cluster records with key indirection and explicit raw subset fallback.
- [x] Add held-out behavior, NLL and serialized-byte promotion accounting plus
  recursive dependency rebuild scheduling.
- [ ] Selection-conditioned child responses; retain exact raw fallback until they are implemented.
- [ ] Adaptive code sizes, exception records and measured net storage/compute savings.

## E. Recurrence and deployment

- [x] Gated tied full-stack refinement and one-loop identity/causality tests.
- [x] Optional multi-space recurrent reader.
- [ ] Train recurrent LFM and compare to the attention-only student option.
- [ ] Place early query heads and integrate asynchronous result arrival safely.
- [ ] Replace the local two-microbatch retrieval pipeline with a bounded,
  completion-driven trajectory scheduler before network-bank training. Batch ready
  continuations by recurrent level/shape, bound retained activation bytes, preserve
  fixed replay plans, and report retrieval-wait/queue-depth/GPU-idle p50/p95/p99.
- [ ] Implement/test hybrid conv/KV cache lifecycle before optimized decoding.
- [ ] Build stripped, information-matched deployment baselines and measure the frontier.

## Repository administration

- [x] Local Git repository and commits supplied in the handoff bundle.
- [x] Private create/push helper, CPU CI, manual Spark workflow and issue templates.
- [x] Publish commits to the existing remote using authenticated local Git.
- [ ] Choose code license and repository visibility policy.
- [ ] Register a trusted self-hosted Spark runner only when needed.
