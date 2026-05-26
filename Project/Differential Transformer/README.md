# Differential Transformer — WIP

**Status**: paper studied, implementation planned.

Reference: Ye et al., *Differential Transformer*, 2024 — PDF in this folder.

## Plan

The Differential Transformer is a small modification to standard attention:
instead of one softmax-attention map per head, each head computes **two**
softmax maps with separate `Q1, K1` and `Q2, K2` projections, and outputs
their **difference**:

```
DiffAttn(X) = (softmax(Q1 K1ᵀ / √d) − λ · softmax(Q2 K2ᵀ / √d)) · V
```

with a learnable scalar λ per head (re-parameterised as `λ = exp(λ_q1·λ_k1) − exp(λ_q2·λ_k2) + λ_init`
in the paper, to keep it well-behaved during training).

The intuition is that the two softmax maps "cancel" common-mode noise: the
difference operation suppresses attention to irrelevant tokens, leaving a
sparser, more focused attention pattern.

Implementation strategy: subclass `MultiHeadAttention` from my
[`Transformer_paper_code.ipynb`](../Transformer_architecture/Transformer_paper_code.ipynb)
and replace only the score-computation block. The rest of the encoder/decoder
stack is unchanged.
