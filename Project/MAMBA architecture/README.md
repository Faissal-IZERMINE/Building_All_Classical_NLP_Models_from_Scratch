# Mamba (Selective State Space Model) — WIP

**Status**: paper studied, implementation planned. This is the largest of
the WIP items.

Reference: Gu & Dao, *Mamba: Linear-Time Sequence Modeling with Selective
State Spaces*, 2023 — PDF in this folder.

## Plan

Mamba is not a Transformer variant — it's a different architectural family
(structured state-space models, S4-style) with a key twist: **the SSM
parameters become input-dependent** (the "selective" in selective SSM).

Concretely, an S6 block computes a recurrence

```
h_t = A_t h_{t-1} + B_t x_t
y_t = C_t h_t
```

where `A_t`, `B_t`, `C_t` are projected from `x_t` itself (hence
"selective"). Discretisation uses the ZOH formula with a learned step
size `Δ_t` that is also input-dependent.

This is harder than the other WIP items because:

1. The **naïve recurrence is O(L) but slow** on GPU. The paper introduces a
   **parallel scan** kernel (similar to Blelloch scan / `associative_scan`)
   that recovers parallelism — implementing this efficiently in PyTorch is
   non-trivial (`torch.compile` or a custom CUDA kernel).
2. The **discretisation** step (ZOH for `A`, simplified for `B`) needs
   careful numerical handling.
3. **Initialization** matters a lot — the paper uses HiPPO-LegS for `A` as
   the structured prior.

Tentative implementation strategy:

1. Start with a **dumb sequential reference implementation** (a `for` loop
   over time steps) on a small sequence-modelling toy. Confirm correctness
   against the paper's equations.
2. Replace the inner loop with `torch.cumsum`-style associative scan when I
   want speed for longer sequences. Use the recipe from the paper's
   reference repo (https://github.com/state-spaces/mamba) as the spec, but
   re-derive every line.
3. Test on a small Shakespeare LM benchmark, same toy setup as the GPT-2
   notebook.

This one is genuinely a project on its own scale — I'd rather mark it
honestly as WIP than ship a half-broken version.
