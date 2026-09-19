# More candidate records, fixed query set and frozen router

Keep the longer global-negative router, one/two-record count policy, payload
writer, reader and decoder fixed. Extend the newest confirmation corpus from
32 to 128 and 1,024 worlds (128 / 512 / 4,096 records). The original first 32 worlds
and their 320 questions remain byte-identical. Only additional candidate records
are introduced; every added world is generated after endpoint freezing.

Use the entire corresponding bank for learned exact search, without a world filter.
`--query-worlds 32` limits scored questions while retaining the complete candidate
corpus. Generate every question in those 32 worlds with the same 24-token cap.
Require matching original payload/provenance and oracle/no-memory outputs across
bank sizes. Separate routing pair recall, answer choice and actual generated answers.

This is a post-confirmation distractor-scaling diagnostic. It is not a new training
selection set, ANN benchmark, cold-drive performance result, agent score or
parameter-substitution study. The external drive is a USB rotational disk. Offline
writer creation is separate from stored inference; no checkpoint or model copies
are made. New raw banks and manifests publish atomically and support restart.
