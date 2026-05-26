# Vision Transformer (ViT) — WIP

**Status**: paper studied, implementation planned.

Reference: Dosovitskiy et al., *An Image is Worth 16x16 Words: Transformers
for Image Recognition at Scale*, ICLR 2021 ([arXiv:2010.11929](https://arxiv.org/abs/2010.11929)) —
PDF in this folder.

## Plan

ViT reuses the standard Transformer encoder; the new pieces vs my existing
[`Transformer_paper_code.ipynb`](../Transformer_architecture/Transformer_paper_code.ipynb) are:

1. **Patch embedding** — split an image `(B, C, H, W)` into `(H/p) · (W/p)`
   patches of size `p × p`, flatten each to a vector, project to `d_model`
   with a single `nn.Linear` (equivalently a `Conv2d` with `kernel = stride = p`).
2. **Learnable `[CLS]` token** prepended to the patch sequence; its final
   hidden state is the input to the classification head.
3. **Learnable 1-D positional embedding** (the paper finds learned ≈ 2-D
   variants on ImageNet-scale data).
4. **Classification head** = `LayerNorm` + `Linear(d_model, num_classes)`
   on the `[CLS]` token output.

Everything else (MHA, FFN, residual + LayerNorm) is unchanged from my
Transformer encoder.

Toy dataset for the sanity-check forward pass: CIFAR-10 with `p = 4`.
