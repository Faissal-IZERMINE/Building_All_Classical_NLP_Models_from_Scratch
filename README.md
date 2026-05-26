# Building classical NLP models from scratch

A personal study project: I re-implement foundational NLP/Transformer
architectures from their original papers, layer-by-layer, to make sure I can
explain every line of every component. Each model lives in its own
self-contained notebook with the paper alongside it.

> **Status**: 3 of 6 architectures implemented end-to-end. The remaining
> three are *studied* (paper read and annotated in the corresponding
> folder) but the code is not yet pushed. See section 4 below for what's
> done vs WIP.

---

## 1. Why this exists

Reading a paper once is not the same as being able to implement it. This
repo is the bridge between the two: a workspace where I rebuild each
architecture from the equations, with the paper open next to me, until the
code runs end-to-end on a small toy dataset.

Each notebook is intentionally **self-contained** — no shared library, no
external utility modules. Re-deriving the same primitives (attention,
layer norm, positional encoding, etc.) for each architecture is the point:
the friction is the learning signal.

## 2. What "from scratch" means here

- **All core layers are written from `nn.Linear` + `torch.matmul` + arithmetic.**
  No `nn.MultiheadAttention`, no `nn.TransformerEncoderLayer`. The only
  high-level building blocks I use are `nn.Embedding`, `nn.LayerNorm`,
  `nn.Dropout`, and the optimisers.
- **Shapes are commented on every tensor operation** so the dimensional
  bookkeeping is explicit (and so I can debug it).
- **Each notebook ends with a forward pass on toy data** to confirm the
  architecture is wired correctly — not full training runs.

## 3. Repository structure

```
Project/
├── Transformer_architecture/           ← done
│   └── Transformer_paper_code.ipynb       Encoder, decoder, MHA, masked MHA,
│                                          cross-attention, full encoder-decoder
│                                          stack, positional encoding,
│                                          weight tying.
├── GPT_2_architecture/                 ← done
│   ├── GPT-2-code.ipynb                   Decoder-only stack, causal mask,
│   │                                      block structure.
│   ├── GPT-2-model-architecture-picture.ppm
│   └── tiny_dataset_shakespear.txt        Tiny Shakespeare for sanity checks.
├── RNN/                                ← done (RNN + LSTM + GRU)
│   └── RNN-code.ipynb                     Vanilla RNN, then LSTM cell, then
│                                          GRU cell, all written from scratch.
├── Vision Transformers/                ← WIP
│   └── 2010.11929v2.pdf                   ViT paper studied; implementation
│                                          planned (patch embed + class token
│                                          on top of the Transformer code).
├── Differential Transformer/           ← WIP
│   └── differential-transformer.pdf       Paper studied; small attention-kernel
│                                          modification planned.
└── MAMBA architecture/                 ← WIP
    └── MAMBA-(SSM)-Paper.pdf              Paper studied; selective SSM
                                            implementation planned (the
                                            largest of the three).
```

## 4. Status

| Architecture | Paper | Implementation | Notes |
|---|:---:|:---:|---|
| Transformer (Vaswani et al., 2017) | ✅ | ✅ | Encoder, decoder, masked MHA, cross-attention, sinusoidal PE, weight tying. |
| GPT-2 (Radford et al., 2019) | ✅ | ✅ | Decoder-only, causal mask, block structure. |
| RNN family (Elman / LSTM / GRU) | ✅ | ✅ | Three cells written from scratch; minimal training demo on a toy task. |
| Vision Transformer (Dosovitskiy et al., 2021) | ✅ | ⏳ | Studied — implementation planned. |
| Differential Transformer (Ye et al., 2024) | ✅ | ⏳ | Studied — implementation planned. |
| MAMBA (Gu & Dao, 2023) | ✅ | ⏳ | Studied — implementation planned (largest scope of the three). |

(I'm marking WIP items honestly rather than claiming they're done. Each
WIP folder contains the paper I've worked through.)

## 5. How to read this repo

If you want to validate that I can implement these architectures, the
intended reading order is:

1. **Transformer** — the densest notebook; it sets the patterns the others
   reuse. Start from the bottom (`Transformer` class) and work upward
   through the building blocks.
2. **GPT-2** — short. Read alongside the Transformer notebook to see what
   changes for decoder-only.
3. **RNN** — shortest. The interesting bit is comparing my LSTM / GRU
   gating against the original equations.

For each notebook I have annotated cells with the paper-equation references
they implement.

## 6. Setup

```bash
pip install torch torchvision transformers tensorboard tqdm pandas numpy matplotlib
jupyter lab
```

That's it — there's no package to install, just open the notebooks.

## 7. References

- Vaswani et al., *Attention Is All You Need*, NeurIPS 2017
  ([arXiv:1706.03762](https://arxiv.org/abs/1706.03762)).
- Radford et al., *Language Models are Unsupervised Multitask Learners* (GPT-2), 2019.
- Hochreiter & Schmidhuber, *Long Short-Term Memory*, 1997.
- Cho et al., *Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation* (GRU), 2014.
- Dosovitskiy et al., *An Image is Worth 16x16 Words* (ViT), ICLR 2021
  ([arXiv:2010.11929](https://arxiv.org/abs/2010.11929)).
- Ye et al., *Differential Transformer*, 2024.
- Gu & Dao, *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*, 2023.
