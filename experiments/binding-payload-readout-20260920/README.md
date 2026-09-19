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
