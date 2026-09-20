# Fixed-parent selected-text output distillation fork

The joint trajectory stage improved held-out next-message likelihood, but most of
its improvement also appeared with no memory. At R=2 the real stored payload beat
the zeroed payload by a small margin, while selected source text was substantially
better. This fork tests whether direct next-token distribution supervision helps
the memory path use those selected source payloads.

`run.py --prepare` creates two immutable configs and an input lock under an existing
external disk directory. Both arms warm-start the exact completed 400-update joint
checkpoint. They use the same 530 repository-heldout SWE-smith training episodes,
seed 233, fully live oracle-selected producer replay, native R=2 consumer, R=1
writer, frozen pretrained decoder, Muon, BF16, four-example accumulation and 400
updates. The sole intended experimental difference is output KL weight 0 versus 1.
The fixed selected-text R=1 teacher sees each source text and preceding target
tokens. It does not supply the memory query or any inference input. The student
reads only newly encoded stored-precision payloads during training; frozen
evaluation creates a new bank once, then scores from stored payloads only.

External output includes actual-LFM model preflights, offline W&B, metrics and
initial/final full recovery checkpoints. Intermediate checkpoints occur only on
explicit stop or resource guard. A stopped arm resumes the same checkpoint and
hyperparameters. Root and stage locks reject competing writers. The source model,
configs and train/validation bytes are digest-locked before every launch.

The planned fixed-bank validation compares real, zero, wrong and absent values
against selected-text NLL on 119 repository-heldout episodes. Primary diagnostic:
the paired real-minus-zero and real-minus-wrong loss, rather than training loss
alone. Generic recurrent compute, text-token layout and output KL give the two
paths different workloads, so this is a payload-use intervention rather than an
information-and-compute-matched capacity-substitution claim. No agent task or
patch success is measured here.
