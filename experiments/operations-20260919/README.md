# Storage and optimizer recovery validation

`checkpoint-relocation.json` records the verified movement of 25 checkpoint
directories to the external disk. Approximately 50.24 GiB of logical file sizes
were relocated; some files already shared hardlinks, so this is not the number
of unique physical bytes freed. Internal free space increased from roughly
122 to 170 GiB and the internal run tree shrank from roughly 49 GiB to 52 MiB.
Existing external archive files were reused where verified, avoiding another full
archive copy. No unrelated bgkit data or shared model caches were removed.

`muon-emergency-resume.json` records actual LFM2.5-230M BF16 CUDA validation on
the Spark. A run stopped after one of two accumulated microbatches, then resumed,
produced exactly the same final weights and complete optimizer/RNG state as an
uninterrupted two-update run. It used noisy live/stale payloads, sampled recurrence
depth and native Muon plus AdamW for excluded parameters. This validates recovery,
not task capability. Checkpoint artifacts remain on the external disk.
