# Learned STOP feature probe

The stored fact-budget intervention showed interference from an extra record,
including loss of entity-specific restoration answers. Test a learned STOP option
before changing production inference.

Start both arms from the completed broad-data full-state address endpoint. Reset
optimizers and run 200 further updates, each sampling 1,280 of the same frozen
training queries (seed 47). The fixed-budget control retains the existing PL
required-group objective. The STOP arm adds a query-conditioned scalar candidate
and trains the PL probability of all required records, in either valid order,
followed by STOP. This rewards stopping after the complete group, not truncating
a multi-record question to a single record. Required IDs are training labels only.

STOP is a linear projection plus bias of the normalized address query. Initialize
its weight to zero and its bias to the training-only median midpoint between the
weakest required and strongest irrelevant score. The script records that scalar;
no held-out labels calibrate it. The extra 65 parameters are charged explicitly.
Use native Muon for matrices and AdamW for the scalar bias, with the inherited
adapter LR and optimizer settings. Both address models can continue adapting.

At diagnostic inference, return records ranked ahead of STOP, at most two. Empty
selections are allowed; ties put records before STOP. Report selected counts,
full-required and exact-required sets, and sufficient support separately. The
existing held-out feature set is a development diagnostic, not a fresh downstream
confirmation. The STOP objective has a value/gradient regression against explicit
sequence enumeration. Production stored inference still uses its fixed budget.

Features and small model/optimizer/sampling states remain external. Cooperative
stop and signals save at a complete update; scientific identity and named ownership
must match on resume. W&B logs offline with stable per-arm identity and explicit
feature-probe configuration. No backbone copies or periodic checkpoints are written.

## Completed result

Both arms completed from `4b8dfed`. STOP ends action reads too early: 23/128 held-out
action queries return nothing, 37 return one record, and only 68 return two. Full
required-pair recall drops to 43/128 versus 69/128 in the matched fixed-budget
continuation. Single-fact exact-set recall is 28/64 permission, 39/64 restoration
and 24/64 identifier. The failure is also visible in training, where 1,023/4,096
action reads return fewer than two records. These 200-update results do not justify
adding this STOP policy to production inference. The two objective losses include
different events and are not directly comparable. Scalar/config telemetry was
recorded offline in W&B; full states remain external.
