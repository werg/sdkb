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

## Completed fixed-query scaling

| Candidate records | Full action pairs /128 | Free actions /128 | Free permission /64 | Free restoration /64 |
|---|---:|---:|---:|---:|
| 128 | 71 | 101 | 47 | 61 |
| 512 | 53 | 91 | 43 | 61 |
| 4,096 | 35 | 76 | 42 | 59 |

All original 128 records, including keys, are byte-identical inside both expanded
banks. All 640 oracle/no-memory score rows agree (excluding the intentionally
changed candidate-count field). Query strings, answers, read budgets and model
weights stay fixed. Choice and generated action predictions agree.

At 512 records, the free action advantage over no memory is 21.09 points
[11.72, 30.47]. At 4,096 it is 9.38 points [−1.56, 19.53]; the interval includes
zero. The apparent advantage over zero values also remains uncertain at 4,096.
Oracle action generation stays 128/128, no memory 64/128 and zero values 63/128.
The router loses complete evidence as distractors increase. Restoration remains
strong, but general scalable global composition is not established. Exact unseen
identifiers stay 0/64 even with oracle supports. No timing claim follows from these
warm cached exact scans on the external rotational drive.
