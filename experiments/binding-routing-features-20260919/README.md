# Frozen-feature routing probe

## Protocol

Use the successful core-only MLP1600 checkpoint. Extract each source's input to
the key head offline, and each prompt's input/output at the first causal query
head boundary. No target tokens, labels, record IDs or future observations enter
these features. Source IDs retain their identities and are only label/index metadata.
This diagnostic is limited to one space, one read and four eligible world records.

Fit two native Muon address models on exactly the same features: the existing
64-dimensional query plus its trainable map, versus a separately trainable query
head from the 1,024-dimensional state plus that map. Both also train key/address
heads. The second arm starts with a copy of the trained query head; initial scores
must be bit-identical. It has 65,536 more trainable parameters. The original reader
and decoder are never modified.

Budget: 800 full-batch updates, adapter learning rate and Muon settings inherited
from the source, gradient clipping 1. This is an address-learnability probe with
more query exposures per update than previous end-to-end studies, not a matched
continuation of those studies. The objective uses the same required-record PL
loss as recurrent training; a regression compares batched values and gradients
against its reference. Report training fit and 32 separately generated worlds,
full required-pair coverage, sufficient-set coverage and top-one required recall.
No downstream task accuracy or stored-inference benefit is asserted by this probe.

All features and small address/optimizer states live on the external disk.
Cooperative stop or SIGINT/SIGTERM saves the optimizer at a complete update;
restart uses the same immutable inputs and deterministic full-batch computation.
Feature extraction can be repeated if stopped before committing its cache.
No backbone copies or frequent checkpoints are written. Production training and
its full emergency recovery remain in the standard runner.

The first attempt aborted before training: BF16 single-query versus batched query
head GEMMs differed enough to change scores by at most 0.005354. Both arms now
recompute the head in the same batch shape, with its weights frozen in the
compressed-query arm. This preserves identical initial scores and the intended
trainable-parameter difference. The report records the query rounding difference
against captured single-query outputs. Feature caches from the aborted extraction
remain valid; neither arm trained in that attempt.


## Completed narrow-data probe

Initial arm scores match exactly. The frozen-versus-adaptable query-head arms own
73,728 and 139,264 trainable parameters. After 800 full-batch updates, the adaptable
head fits all 512 training action pairs; the compressed arm fits 258/512. On 32
new worlds, full-pair recall is 38/128 versus 17/128. Direct permission/restoration
first-record correctness is 34/64 and 51/64 for the full-state arm versus 30/64 and
34/64 for the compressed arm. This is a training-fit improvement with limited
held-out transfer, not solved routing. All exact counts are in `results.json`.

The arms were fit from frozen `8e1fe59` using features extracted by `19a23a2`.
The single-versus-batch normalized query maximum absolute difference is 0.001268;
both trained arms use the same batched computation. No production model interface
has changed, and no downstream evaluation is claimed.
