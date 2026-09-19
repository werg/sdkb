# Frozen-payload endpoint readout

Fit separate linear and 256-hidden-unit SiLU readouts of the original frozen
2,048-value latent payload, predicting six hexadecimal positions with sixteen
classes each. The fixed `api_` prefix and answer structure are supplied by this
probe, so success is not language-model generation or a production read path.
Failure does not prove the payload lacks information.

Use the corrected 4,096-record bank and its 1,024-world corpus. The first 32 worlds
(the existing, already inspected diagnostic questions) are held out completely;
train on the remaining 992 worlds, with one example per endpoint-bearing source.
The model receives only the stored payload, never source text, entity IDs, bank
keys or query/target strings. Source text validates offline labels and provenance.
Time/domain/source identities are checked at fetch. Standardization statistics
come only from training payloads and are checkpointed buffers.

Predeclare two arms: linear and MLP256, each 1,600 native Muon updates, batch 128,
seed 83 and LR 0.001. No biases or embeddings; all trainable matrices use Muon.
Report training and held-out exact-six and character accuracy, plus a one-position
cyclic shift of held-out payload rows against unchanged targets. Unequal readout
parameter budgets and the supervised answer-format advantage are explicit.

Outputs: `/archive/probes/payload-readout-20260920`. Source/config/bank/data hashes,
complete optimizer/RNG/name state, training-fitted normalization, offline W&B and
JSONL are retained. Only initial/final/emergency checkpoints; existing model/bank
files are read, and no backbone checkpoint or newly encoded source is created.

## Completed result

| Readout | Parameters | Train exact /1,984 | Held-out exact /64 | Held-out characters /384 | Shifted-payload characters /384 |
|---|---:|---:|---:|---:|---:|
| Linear | 196,608 | 1,852 | 0 | 137 | 21 |
| MLP256 | 548,864 | 1,984 | 0 | 122 | 16 |

The linear readout recovers 35.7% of held-out characters versus 5.5% with shifted
payloads; MLP256 recovers 31.8% versus 4.2%. Some character information is accessible
from the stored representation, but neither readout recovers a whole unseen
identifier. Better training fit does not translate to a better held-out readout.
These two limited classifiers do not establish how much information is absent or
what a better decoder could recover. They are not adopted into production.

Native source `ea4d3e4`; both predeclared budgets completed. Full suite: 340 passed,
four existing warnings; Ruff clean. Full identities, small optimizer endpoints and
metrics remain external. The committed input summary retains split counts instead
of repeating all 2,048 episode IDs; the complete list remains in external inputs.
