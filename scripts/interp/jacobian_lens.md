# Jacobian Lens (`jacobian_lens.py`)

A standalone, educational **Jacobian Lens** for HRM-Text, following the patterns
of `logit_lens.py`. Model loading is deferred to `utils.load_hrm`; everything
else is self-contained.

## What it does

Where the logit lens unembeds an intermediate residual directly, the Jacobian
lens first **linearly transports** it into the final-layer basis via the
corpus-averaged Jacobian, then unembeds:

```
lens_l(h) = unembed( J_l @ h ),   J_l = E_corpus[ ∂h_final / ∂h_l ]
```

Companion in spirit to Anthropic's `anthropics/jacobian-lens` ("Verbalizable
Representations Form a Global Workspace in Language Models").

## Structure (self-contained except model loading)

- `load_corpus(...)` — pulls a small corpus (default `NeelNanda/c4-10k`; swap via
  `dataset` / `split` / `text_column`).
- `estimate_jacobians(...)` — faithful dim-batched VJP fitting
  (`torch.autograd.grad`): one-hot cotangents on `h_final` summed over target
  positions, gradients meaned over source positions, averaged over the corpus.
  Keyed by **invocation index** (not layer name), since HRM is recurrent — with
  geometry's invocation-count assertion as a guard.
- `JacobianLens` — `transport`, `build_hook`, `__enter__/__exit__`,
  `begin_pass`, `save`/`load` (config-hashed cache), and a `show()` reusing the
  logit-lens layer×position entropy grid so outputs are directly comparable.
- `__main__` — Balanced defaults (`N_SEQS=128 × SEQ_LEN=128`), a `MASK_MODE` knob
  (default `prefix`, i.e. `token_type_ids=1`, matching the other interp scripts),
  fit-or-load cache, then the same demo-prompt readout as `logit_lens.py`.

## Recurrence handling — one transport per *invocation*, not per layer

HRM applies its L/H stacks repeatedly, so a given block fires several times per
forward (1B config: L-blocks 6×, H-blocks 2×). This raises a design question:
should there be a single transport matrix per layer, or a separate one per
invocation of that layer?

**Principled answer: per invocation.** The transport `J = ∂h_final/∂h` is *not* a
property of a layer's weights — it is the linearized map from *an activation at a
specific point in the computation graph* to the final hidden state, and it is
governed by **everything downstream** of that point.

- In a plain feedforward transformer, layer `l` sits at exactly one place in the
  graph, so "per layer" and "per position-in-graph" coincide — which is why the
  non-recurrent reference (`anthropics/jacobian-lens`) has one `J` per layer.
- HRM reuses the *same weights* at different depths. L[0] in the **first** L-sweep
  has all remaining cycles between it and `h_final` (Jacobian far from identity);
  L[0] in the **last** sweep has almost nothing left (Jacobian near identity). So
  `∂h_final/∂h` genuinely differs at each firing even though the producing weights
  are identical.

Collapsing invocations into a single per-layer transport would average
activations as computationally distant as an early and a late layer of a
feedforward net — the same category error as averaging the logit-lens readout
across early and late layers. Keying by invocation is therefore the faithful
generalization of the reference method (which keys by position-in-graph, which
there equals the layer).

**In the code.** `estimate_jacobians` keys `jacobians` by invocation index;
`transport(residual, invocation)` selects `jacobians[invocation]`; and
`build_hook` advances a per-firing counter so each firing uses its own `J`. Firing
counts are input-independent, so invocation indices line up across corpus
sequences and the readout prompt; this is asserted. `show()` renders each
invocation as its own row.

## Two real subtleties handled

1. **HRM gradient truncation.** The forward wraps early L-cycles in
   `torch.no_grad()` (the `L_bp_cycles` k-step-gradient training trick), which
   leaves those invocations' residuals detached and breaks fitting. That
   truncation is a *training* memory optimization that does not change forward
   *values*, so during fitting we temporarily force full gradients
   (`model.model.L_bp_cycles_padded`), restored afterward, so the lens measures
   the true forward Jacobian for every invocation.
2. **`h_final` capture** = the exact `lm_head` input (post `H_module.final_norm`),
   grabbed via a `forward_pre_hook` on `model.lm_head`.

## Readout mask — matching generation vs. matching the fit (`READOUT_MASK`)

The demo readout re-encodes *prompt + generated continuation* and runs one
capturing forward. Under PrefixLM (`token_type_ids==1` → one bidirectional block,
`==0` → causal) there are two defensible attention patterns for that forward, and
in `prefix` mode they differ — hence the `READOUT_MASK` knob:

- **`"generation"` (default).** Only the original prompt is the bidirectional
  prefix (`==1`); every generated token is causal (`==0`). This reproduces how the
  tokens were actually produced — during KV-cache decode each generated token was
  emitted under a causal mask, seeing only the prefix and earlier generated tokens,
  never *forward* to later ones. So the captured activations are the ones the model
  really computed while generating. This mirrors the fix in `logit_lens.py`.
- **`"fit"`.** The whole sequence uses `MASK_MODE` (all `==1` in prefix mode), so
  the activations are drawn from the *same* attention distribution the Jacobians
  were fitted on. The transport operator is then applied to inputs of the kind it
  was averaged over, but the activations no longer reflect how the tokens were
  generated.

There is an unavoidable asymmetry in `prefix` mode: `J` is fitted over corpus
sequences that are *fully* bidirectional, whereas `"generation"` transports
activations from a prefix/causal forward. `"generation"` prioritizes a faithful
readout (what the model computed while generating); `"fit"` prioritizes a clean
application of the transport (matching `J`'s fit distribution). In `causal` mode
both collapse to left-to-right throughout, so the knob has no effect.

## Cost

Fitting does ≈ `N_SEQS × ceil(d_model / DIM_BATCH)` backward passes through the
full recurrent graph; replicating a sequence `DIM_BATCH` times also multiplies
forward-activation memory by `DIM_BATCH`. Keep `N_SEQS` / `SEQ_LEN` / `DIM_BATCH`
modest and rely on the on-disk cache (`jacobian_lens.pt`). `MASK_MODE` must match
between fit and readout (a `J` fitted causal is invalid for a prefix readout) —
enforced by the cache config.

## Validation

Both tests pass (see the session that authored this; test scripts were in a
scratchpad, not committed):

- **Test A** — on a per-position linear map `h_final = A·h_l`, the estimated `J`
  equals `A` to `1.2e-7` (confirms cotangent placement + mean-reduction math).
- **Test B** — end-to-end on a tiny real HRM (random weights, shrunk config): 16
  invocations correctly ordered (3 L-sweeps → H-sweep per H-cycle, ×2), `[d,d]`
  finite Jacobians, save/load roundtrip, and a 96-capture readout.

## Running against the real 1B

Deferred to a GPU box. Smoke it first at reduced scale (`N_SEQS=8, SEQ_LEN=32`) to
confirm fit → cache → `show()`, then use the Balanced default. Expect mid-layer
cells to surface plausible tokens; the final H invocation ≈ identity under `J`, so
its row should track the model's own next-token predictions (the logit-lens /
model baseline).

## Knobs (top of `__main__`)

| knob | default | meaning |
|---|---|---|
| `MODEL_SOURCE` | `sapientinc/HRM-Text-1B` | HF id/dir or native checkpoint dir |
| `MASK_MODE` | `prefix` | `prefix` (token_type_ids=1) or `causal` |
| `READOUT_MASK` | `generation` | demo readout mask: `generation` (prompt prefix, generated causal — faithful) or `fit` (whole sequence uses `MASK_MODE`); no effect in causal mode |
| `DATASET` | `NeelNanda/c4-10k` | any `{split:"train","text"}` corpus |
| `N_SEQS` | `128` | corpus size for fitting |
| `SEQ_LEN` | `128` | tokens per corpus sequence |
| `DIM_BATCH` | `16` | VJP rows per backward (memory ↔ #passes) |
| `LAYER_IDS` | `[0, 7, 15]` | which blocks to probe (as in logit_lens) |
| `LENS_CACHE` | `jacobian_lens.pt` | on-disk fitted-lens cache |
