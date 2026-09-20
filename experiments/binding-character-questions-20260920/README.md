# Character-question curriculum preparation

The 64-target copy fit succeeds on original training memories but fails on changed
or held-out endpoints and damages rule/action behavior. Its decoder can often predict
training suffixes without memory once a teacher-forced prefix identifies the answer.
Test a data-only intervention: ask for one hexadecimal position, without a multi-token
answer prefix, while retaining the original full-endpoint and rule/action tasks.

`scripts/make_identifier_character_control.py` makes two aligned corpora. Each
identifier example keeps one common full-endpoint question. Its six extra rows are
whole-endpoint repetitions in the control and first-through-sixth character questions
in the candidate. Source identities/text, selected source, times, row order and all
original nonidentifier episodes match. New queries have opaque versioned IDs and
provenance outside neural inputs. The single-character target is validated against
the actual prior selected source. Characters use a separate `identifier_character`
family; they must not be sent through an evaluator that only knows whole-endpoint
counterfactual labels.

This changes target entropy and token compute; matched updates and source draws
would not imply equal FLOPs. No model module, optimizer or inference interface has
changed. A successful auxiliary task alone would not demonstrate full-string recall.

Before declaring training budgets, run a frozen selected-text control on eight new
worlds: 16 whole identifiers and 96 character questions. `preflight-inputs.json` pins
the original useful MLP checkpoint and source corpus. Generate text/none answers
without candidates or supplied answer prefixes at both the one-loop text-anchor
depth and the three-loop evaluation depth. This tests whether the question
interface is usable before spending on the proposed comparison. All artifacts and
resumable progress remain external. Training has not been launched.
