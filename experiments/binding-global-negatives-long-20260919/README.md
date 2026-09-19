# Longer matched router optimization

The 800-update global-negative objective is still decreasing (sampled loss about
9.80 at update 20, 6.74 at 420 and 5.35 at 800). Global training required-pair
recall remains 973/4096. Test additional optimization before attributing failure
to the fixed address representation.

Run the same `e6c13b9` implementation from the same original broad router, seed 67,
batch 128 and learning rate, with a new explicit budget of 3,200 updates per arm.
Within-world and global-negative arms remain matched. This is a separate run;
no existing exact-resume identity or checkpoint budget is modified. With no schedule,
the first 800 updates should reproduce the shorter run's learning trajectory.

Original heldout worlds and the now-observed stored-confirmation worlds are
exploratory diagnostics for this budget decision. Freeze the longer endpoints
before any newly generated confirmation corpus. Do not substitute feature recall
for stored task performance. Small initial/final/emergency states, W&B attempts,
reconciled metrics and all artifacts remain on external storage.

## Completed feature endpoints

The first 40 logged losses in each arm exactly reproduce the shorter run's first
800 updates. At 3,200 updates, global-negative training retrieves 3,745/4,096
training action pairs and 86/128 development action pairs under full-bank
competition. The matched within-world control retrieves only 9/4,096 and 3/128
under that same global competition, despite fitting all its local training pairs.
Under supplied world scope the long development results are 98/128 global-trained
versus 100/128 world-trained. Additional optimization helps the global objective;
these frozen endpoints now go to the separate fresh stored confirmation.
