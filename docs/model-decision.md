# Initial backbone decision

## Default: LFM2.5-230M

The [official model card](https://huggingface.co/LiquidAI/LFM2.5-230M) and
[configuration](https://huggingface.co/LiquidAI/LFM2.5-230M/raw/main/config.json)
identify a 230M-class hybrid, 1,024 hidden width, 14 layers, convolution and grouped
query attention. The model card describes 32,768 context tokens; the raw positional
configuration is larger. This adapter uses the conservative 32,768 limit, not the
larger number as a tested effective context claim.

The selected dependency is Transformers 5.17.0. Its
[LFM implementation](https://github.com/huggingface/transformers/blob/v5.17.0/src/transformers/models/lfm2/modeling_lfm2.py)
exposes `inputs_embeds`, an ordinary hidden-state output and a pure PyTorch
convolution fallback. `trust_remote_code=False` and `use_safetensors=True` are
explicit. Optional custom kernels are not prerequisites for correctness.
The model card's historical library version is not used as a reason to pin an
older dependency. The integration test must be run with the pinned dependency.

## What “looped” means in this implementation

Let F be the entire pretrained stack, E the original embedded sequence, and h its
hidden states. The adapter computes

```
h0 = F(E)
proposal_t = F(E + RMS(E) * LayerNorm(h[t-1]))
h[t] = h[t-1] + tanh(gate) * (proposal_t - h[t-1])
```

The scale is detached and local to each token, not pooled from future tokens.
The gate starts at zero, preserving the original output initially. The gate gets
a learning signal, and subsequent updates can use the proposals. One-loop execution
is exactly the unwrapped path. Each pass recomputes the causal prefix, and no KV or
convolution state is carried between passes.

This avoids editing private decoder internals, but it is a **full-stack recurrent
refinement adapter**, not a native recurrent-depth pretrained model. Extra passes
cost computation and potentially activations. They are not presented as a speedup.
No claim is made about which frontier models use recurrence.

## Alternative: SmolLM2-135M-Instruct

The [official configuration](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct/raw/main/config.json)
is Llama-style with 576-wide hidden states. This gives a simpler attention-only
cache problem for later recurrence engineering. The same public embedding adapter
supports this architecture. It may have different oracle-task ability; model size
alone is not the selection criterion.

## Decision experiment

Keep the single-space writer/reader fixed as far as dimensions permit. Compare
one-loop LFM, two-loop LFM and one/two-loop SmolLM with oracle text, oracle latent
support, and no-memory controls. Record causal dependence, target NLL, full action
correctness, memory benefit, resident parameter bytes and measured wall-clock cost.
Use checkpoints with immutable HF revisions and record dtype and kernel settings.

Neither pretrained checkpoint nor either conversion was executed on a GPU during
this bootstrap. The random-LFM integration test is an additional structural check,
not equivalent to evaluating the released checkpoint.
