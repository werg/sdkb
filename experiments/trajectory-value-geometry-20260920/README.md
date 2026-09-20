# Where the stored-value effect weakens

Read-only probes on the frozen 400-update uniform trajectory control and its
already serialized held-out bank. All 119 episodes use their original oracle
selected IDs, time boundaries and R=2 causal prefix queries. Real, zero and
wrong-value comparisons reuse the original read plans; the writer is explicitly
forbidden after bank creation. Raw per-episode measurements and bank files stay
under `/archive/probes/` and `/archive/runs/`; the committed JSON files hold
aggregate results and exact input/code digests. No checkpoint is modified.

| Mean quantity across 119 held-out episodes | Value |
|---|---:|
| Real reader-token RMS | 0.581 |
| Real-versus-zero reader-token delta RMS | 0.566 |
| Real-versus-zero answer-state delta / real answer-state RMS | 0.0614 |
| Selected-text-versus-no-text answer-state delta / text answer-state RMS | 0.4142 |
| Real-versus-zero next-token symmetric KL, token weighted | 0.00875 |
| Selected-text-versus-no-text next-token symmetric KL, token weighted | 0.37898 |

The reader returns markedly different tokens for real and zero stored values;
the small payload effect in teacher NLL is therefore not explained by nearly
identical returned tokens. Differences shrink before the answer-state/output
interface. The native trace confirms the pre-read core state is exactly equal
between interventions. Relative to each stage's own real activation RMS, the
real-versus-zero difference is 0.974 at reader tokens, 0.129 after bridge memory
injection, 0.0268 at answer positions after the second shared-core update, and
0.0614 at final answer states. These ratios cross different representations;
they are a diagnostic of the executed path, not information-capacity bounds.

The trained bridge memory and recurrent-update gates are both about 0.11. A
read-only gate sweep on that same checkpoint reproduced the published baseline
exactly. Raising **only the memory gate** to 0.30 changed token-weighted real NLL
from 0.971024 to 0.969668 and paired real-over-zero mean episode benefit from
0.01722 to 0.02156. Raising **only the update gate** to 0.30 worsened real NLL
to 0.986819. Setting both to 0.50 enlarged the contrast but worsened real NLL
to 1.031526, illustrating why a larger real-zero gap alone is insufficient.

These overlays were inspected on the reused validation set and were not trained.
The text and latent conditions also differ in context length and compute. This
does not establish agent success, source sufficiency, learned routing or capacity
substitution. The separate matched memory-gate warm-start study tests whether
training from memory gate 0.30 yields a real held-out improvement.
