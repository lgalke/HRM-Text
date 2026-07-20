"""Jacobian Lens for HRM-Text (educational, standalone).

Companion in spirit to Anthropic's `anthropics/jacobian-lens` ("Verbalizable
Representations Form a Global Workspace in Language Models"). Where the *logit
lens* reads an intermediate residual `h_l` by applying the unembedding directly,
the *Jacobian lens* first linearly transports `h_l` into the final-layer basis
using the corpus-averaged input->output Jacobian, then unembeds:

        lens_l(h) = unembed( J_l @ h ),     J_l = E_corpus[ d h_final / d h_l ]

`J_l` is estimated over a small pretraining corpus (default `NeelNanda/c4-10k`,
easily swapped) by vector-Jacobian products: a one-hot cotangent is placed on the
final hidden state `h_final` at every (valid) target position -- summing over
target positions -- and back-propagated to `h_l`; the resulting gradient is then
*meaned* over source positions. Averaging that per-sequence estimate over the
corpus gives `J_l`.

Recurrence (HRM specific) -- one transport per *invocation*, not per layer.
HRM applies its L/H stacks repeatedly, so a given block fires several times per
forward pass (for the 1B config: L-blocks 6x, H-blocks 2x). The transport
`J = d h_final / d h` is a property of *where an activation sits in the
computation graph*, i.e. of everything downstream of it -- not of the producing
weights. Because the same weights are reused at different depths, the SAME layer
has a genuinely DIFFERENT Jacobian at each firing: L[0] in the first L-sweep has
all remaining cycles between it and h_final (far from identity), while L[0] in
the last sweep has almost nothing left (near identity). Collapsing these into one
per-layer transport would average activations as computationally distant as an
early and a late layer of a feedforward net -- the wrong thing. So, exactly as in
`logit_lens.py` / `geometry.py`, a Jacobian is keyed by *invocation index* (the
position of the hook fire within a forward). This is also the faithful
generalization of the non-recurrent reference, which keys by position-in-graph
(there == layer). Firing counts are input-independent, so invocation indices line
up across corpus sequences and the readout prompt; we assert this.

Everything here is self-contained except model loading, which is deferred to
`utils.load_hrm` (shared with the other interp scripts).

Cost warning. Fitting does ~= N_SEQS * ceil(d_model / DIM_BATCH) backward passes
through the full recurrent graph; replicating a sequence DIM_BATCH times also
multiplies forward-activation memory by DIM_BATCH. Keep N_SEQS / SEQ_LEN /
DIM_BATCH modest and rely on the on-disk cache. `MASK_MODE` must match between
fit and readout (a `J` fitted causal is invalid for a prefix readout); the cache
config guards this.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch


# (invocation, position, token) readout record -- same fields as
# logit_lens.LogitLensCapture so the visualization is directly comparable.
@dataclass
class JacobianCapture:
    layer_name: str
    position_idx: int
    token_rank: int
    token_id: int
    token_str: str
    proba_value: float
    logit_value: float
    entropy: float


# --------------------------------------------------------------------------
# Corpus: a small pretraining sample used only to estimate the Jacobian.
# --------------------------------------------------------------------------
def load_corpus(tokenizer, n_seqs: int, seq_len: int,
                dataset: str = "NeelNanda/c4-10k", split: str = "train",
                text_column: str = "text") -> List[torch.Tensor]:
    """Return up to `n_seqs` 1-D token-id tensors, each exactly `seq_len` long.

    Rows shorter than `seq_len` are skipped; longer rows are truncated. Swap
    `dataset`/`split`/`text_column` to use any other text corpus.
    """
    from datasets import load_dataset
    ds = load_dataset(dataset, split=split)
    seqs: List[torch.Tensor] = []
    for ex in ds:
        ids = tokenizer(ex[text_column], return_tensors="pt")["input_ids"][0]
        if ids.numel() < seq_len:
            continue
        seqs.append(ids[:seq_len].clone())
        if len(seqs) >= n_seqs:
            break
    if len(seqs) < n_seqs:
        print(f"[jacobian_lens] warning: only {len(seqs)}/{n_seqs} sequences of "
              f"length {seq_len} available in {dataset}:{split}.")
    return seqs


def make_batch(ids: torch.Tensor, batch_size: int, mask_mode: str,
               device: str) -> Dict[str, torch.Tensor]:
    """Replicate one sequence `batch_size` times and attach masks.

    `mask_mode="prefix"` marks the whole sequence as a bidirectional prefix
    (token_type_ids=1, the convention used by the other interp scripts);
    `"causal"` uses ordinary left-to-right attention (token_type_ids=0).
    """
    ids = ids.to(device)
    s = ids.numel()
    input_ids = ids.unsqueeze(0).expand(batch_size, s).contiguous()
    am = torch.ones(batch_size, s, dtype=torch.long, device=device)
    tt = am.clone() if mask_mode == "prefix" else torch.zeros_like(am)
    return dict(input_ids=input_ids, attention_mask=am, token_type_ids=tt)


# --------------------------------------------------------------------------
# Fitting: estimate the per-invocation Jacobians over the corpus.
# --------------------------------------------------------------------------
def estimate_jacobians(model, blocks: Sequence[Tuple[torch.nn.Module, str]],
                       seqs: Sequence[torch.Tensor], device: str,
                       dim_batch: int = 16, mask_mode: str = "prefix",
                       progress: bool = True
                       ) -> Tuple[Dict[int, torch.Tensor], Dict[int, str]]:
    """Return (jacobians, layer_names), both keyed by invocation index.

    `blocks` are the (module, layer_name) pairs to probe, in the order they will
    be hooked (must match the order used at readout time). Each `jacobians[inv]`
    is a `[d_model, d_model]` fp32 matrix estimated as the corpus mean of the
    input->output Jacobian d h_final / d h at that invocation.
    """
    d = model.config.hidden_size

    # Forward hooks record each block's output (the residual stream, a live node
    # in the autograd graph) in firing order; a pre-hook on lm_head grabs the
    # final hidden state it consumes (post H_module.final_norm).
    class _Rec:
        acts: List[torch.Tensor] = []
        names: List[str] = []
        inv: int = 0
        h_final: Optional[torch.Tensor] = None
    rec = _Rec()
    handles = []

    def make_fwd_hook(name):
        def hook(module, inputs, output):
            h = output[0] if isinstance(output, tuple) else output
            rec.acts.append(h)             # keep graph node (do NOT detach)
            rec.names.append(name)
            rec.inv += 1
        return hook

    for module, name in blocks:
        handles.append(module.register_forward_hook(make_fwd_hook(name)))

    def pre_hook(module, args):
        rec.h_final = args[0]              # [B, S, d] fed to lm_head
    handles.append(model.lm_head.register_forward_pre_hook(pre_hook))

    # HRM's forward wraps early recurrent L-cycles in `torch.no_grad()` (the
    # `L_bp_cycles` k-step-gradient training trick), which would leave those
    # invocations' residuals detached. That truncation is a *training* memory
    # optimization and does not change forward values -- but the Jacobian lens
    # needs the true forward derivative d h_final / d h_l for EVERY invocation,
    # so we temporarily force all cycles to be differentiable and restore after.
    inner = getattr(model, "model", None)
    saved_bp = getattr(inner, "L_bp_cycles_padded", None)
    if inner is not None and saved_bp is not None:
        inner.L_bp_cycles_padded = [inner.config.L_cycles] * inner.config.H_cycles

    j_sum: Dict[int, torch.Tensor] = {}
    layer_names: Dict[int, str] = {}
    expected: Optional[int] = None
    try:
        for si, ids in enumerate(seqs):
            rec.acts, rec.names, rec.inv, rec.h_final = [], [], 0, None
            batch = make_batch(ids, dim_batch, mask_mode, device)

            # Fit forward MUST run with autograd enabled (no inference_mode), so
            # the intermediate residuals are differentiable.
            with torch.enable_grad():
                model(**batch)

            fired = rec.inv
            if expected is None:
                expected = fired
                if expected == 0:
                    raise RuntimeError("No hooks fired -- pass the blocks to hook.")
                for inv in range(fired):
                    j_sum[inv] = torch.zeros(d, d, dtype=torch.float32, device=device)
                    layer_names[inv] = rec.names[inv]
            elif fired != expected:
                raise RuntimeError(
                    f"Sequence {si} fired the hooks {fired} times but the first "
                    f"fired {expected}. Invocation indices no longer align across "
                    f"forward passes (input-dependent cycle count?), so "
                    f"per-invocation Jacobians are invalid.")

            acts = rec.acts
            h_final = rec.h_final                          # [dim_batch, S, d]
            valid = batch["attention_mask"][0].bool()      # [S], same for all rows

            # ceil(d / dim_batch) VJP passes fill the full [d, d] Jacobian: pass
            # k handles output dims [d0, d0+n_rows); row (d0+b) is read out by
            # batch element b's one-hot cotangent (summed over target positions),
            # then meaned over source positions.
            for d0 in range(0, d, dim_batch):
                n_rows = min(dim_batch, d - d0)
                cot = torch.zeros_like(h_final)
                for b in range(n_rows):
                    cot[b, valid, d0 + b] = 1.0
                grads = torch.autograd.grad(h_final, acts, grad_outputs=cot,
                                            retain_graph=True, allow_unused=True)
                for inv, g in enumerate(grads):
                    if g is None:
                        continue
                    rows = g[:n_rows][:, valid, :].mean(dim=1)  # [n_rows, d]
                    j_sum[inv][d0:d0 + n_rows] += rows.float()

            del acts, h_final, grads, cot
            if device == "cuda":
                torch.cuda.empty_cache()
            if progress:
                print(f"\r[jacobian_lens] fit {si + 1}/{len(seqs)} sequences", end="")
        if progress:
            print()
    finally:
        for h in handles:
            h.remove()
        if inner is not None and saved_bp is not None:
            inner.L_bp_cycles_padded = saved_bp

    n = max(len(seqs), 1)
    jacobians = {inv: (j_sum[inv] / n).cpu() for inv in j_sum}
    return jacobians, layer_names


# --------------------------------------------------------------------------
# The lens: transport residuals through J, unembed, visualize.
# --------------------------------------------------------------------------
class JacobianLens:
    """Apply pre-fitted per-invocation Jacobians as a readout head.

    Usage mirrors logit_lens.LogitLens: register `build_hook(layer_name)` on the
    same blocks (in the same order) used for fitting, then run one forward inside
    the `with lens:` context to capture a layer(pass) x position grid.
    """

    def __init__(self, tokenizer, lm_head, jacobians: Dict[int, torch.Tensor],
                 layer_names: Dict[int, str], topk: int = 1, config: Optional[dict] = None):
        self.tokenizer = tokenizer
        self.lm_head = lm_head
        self.jacobians = jacobians          # invocation -> [d, d]
        self.layer_names = layer_names
        self.topk = topk
        self.config = config or {}
        self.captures: List[JacobianCapture] = []
        self.do_capture = False
        self._invocation = 0

    # ---- transport ----------------------------------------------------------
    def transport(self, residual: torch.Tensor, invocation: int) -> torch.Tensor:
        """Linearly map a residual [.., d] into the final-layer basis: r @ J^T."""
        j = self.jacobians[invocation].to(residual.device)
        out = residual.float() @ j.t()
        return out.to(self.lm_head.weight.dtype)

    # ---- context management -------------------------------------------------
    def __enter__(self):
        self.captures = []
        self.do_capture = True
        return self.captures

    def __exit__(self, exc_type, exc_value, traceback):
        if not self.captures:
            print("No captures were made. Register build_hook() on the blocks "
                  "before running inference.")
        if self._invocation != len(self.jacobians):
            print(f"Warning: forward fired {self._invocation} invocations but "
                  f"{len(self.jacobians)} Jacobians are loaded -- the hooked "
                  f"blocks/order may differ from fitting.")
        self.do_capture = False

    def begin_pass(self):
        """Reset the invocation counter. Call once before each capturing forward."""
        self._invocation = 0

    # ---- readout ------------------------------------------------------------
    def _readout(self, outputs, layer_name, invocation):
        """Transport `outputs` through J[invocation], unembed, and record the
        per-position top-k tokens with their probabilities and entropy."""
        transported = self.transport(outputs.detach(), invocation)
        logits = self.lm_head(transported)
        bsz, seq_len, vocab_size = logits.shape
        assert bsz == 1, "Batch size > 1 not supported at readout"
        probas = torch.softmax(logits.float(), dim=-1)
        topk_values, topk_indices = torch.topk(probas, k=self.topk, dim=-1)
        topk_logits = logits.gather(-1, topk_indices)
        entropies = -(probas * torch.log2(probas.clamp_min(1e-12))).sum(dim=-1)
        captures = []
        for i in range(seq_len):
            entropy_value = entropies[0, i].item()
            for j in range(self.topk):
                token_id = int(topk_indices[0, i, j].item())
                captures.append(JacobianCapture(
                    layer_name=layer_name,
                    position_idx=i,
                    token_rank=j,
                    token_id=token_id,
                    token_str=self.tokenizer.decode([token_id], skip_special_tokens=False),
                    proba_value=float(topk_values[0, i, j].item()),
                    logit_value=float(topk_logits[0, i, j].item()),
                    entropy=entropy_value,
                ))
        return captures

    def build_hook(self, layer_name):
        """Closure integrating layer names + invocation indexing into a hook."""
        def hook(module, inputs, outputs):
            if not self.do_capture:
                return
            out = outputs[0] if isinstance(outputs, tuple) else outputs
            self.captures.extend(self._readout(out, layer_name, self._invocation))
            self._invocation += 1
        return hook

    # ---- persistence --------------------------------------------------------
    def save(self, path, dtype=torch.float16):
        torch.save({
            "jacobians": {inv: j.to(dtype) for inv, j in self.jacobians.items()},
            "layer_names": self.layer_names,
            "config": self.config,
        }, path)
        print(f"[jacobian_lens] saved lens -> {path}")

    @classmethod
    def load(cls, path, tokenizer, lm_head, topk: int = 1):
        blob = torch.load(path, map_location="cpu")
        jac = {int(inv): j.float() for inv, j in blob["jacobians"].items()}
        names = {int(inv): n for inv, n in blob["layer_names"].items()}
        return cls(tokenizer, lm_head, jac, names, topk=topk, config=blob.get("config", {}))

    # ---- plotting (copied from logit_lens.show; directly comparable) --------
    def show(self, captures=None, text=None):
        """Visualize captures as a layer(pass) x position grid (top-1 token,
        colored by entropy). Recurrent re-applications of a block become
        separate rows, recovered from capture order."""
        if captures is None:
            captures = self.captures
        if not captures:
            print("No captures to show.")
            return

        top1 = [c for c in captures if c.token_rank == 0]

        passes = []  # list of (layer_name, {position_idx: capture})
        prev_layer, prev_pos = None, None
        for c in top1:
            starts_new_pass = (
                prev_layer is None
                or c.layer_name != prev_layer
                or (prev_pos is not None and c.position_idx <= prev_pos)
            )
            if starts_new_pass:
                passes.append((c.layer_name, {}))
            passes[-1][1][c.position_idx] = c
            prev_layer, prev_pos = c.layer_name, c.position_idx

        positions = sorted({c.position_idx for c in top1})

        import numpy as np
        import matplotlib.pyplot as plt

        n_layers, n_positions = len(passes), len(positions)
        entropy_matrix = np.full((n_layers, n_positions), np.nan)
        token_matrix = [["" for _ in range(n_positions)] for _ in range(n_layers)]

        row_labels = []
        layer_pass_count = {}
        for i, (layer_name, pos_map) in enumerate(passes):
            layer_pass_count[layer_name] = layer_pass_count.get(layer_name, 0) + 1
            row_labels.append(f"{layer_name} #{layer_pass_count[layer_name]}")
            for j, pos in enumerate(positions):
                c = pos_map.get(pos)
                if c is not None:
                    entropy_matrix[i, j] = c.entropy
                    token_matrix[i][j] = c.token_str

        vmax = np.nanmax(entropy_matrix) if not np.all(np.isnan(entropy_matrix)) else 1.0
        fig, ax = plt.subplots(figsize=(max(6, n_positions * 2.5), max(4, n_layers * 0.4)))
        im = ax.imshow(entropy_matrix, cmap="viridis", vmin=0, vmax=vmax,
                       aspect="auto", origin="lower")

        for i in range(n_layers):
            for j in range(n_positions):
                token_str = token_matrix[i][j]
                if not token_str:
                    continue
                normalized_entropy = entropy_matrix[i, j] / vmax if vmax > 0 else 0.0
                text_color = "white" if normalized_entropy < 0.5 else "black"
                ax.text(j, i, token_str, ha="center", va="center", color=text_color, fontsize=6)

        ax.set_yticks(range(n_layers))
        ax.set_yticklabels(row_labels, fontsize=8)
        ax.set_ylabel("Layer (pass)")

        if text is not None:
            input_ids = self.tokenizer(text, return_tensors="pt")["input_ids"][0]
            input_tokens = [self.tokenizer.decode([tid], skip_special_tokens=False) for tid in input_ids]
            xtick_labels = [input_tokens[p] if p < len(input_tokens) else str(p) for p in positions]
        else:
            xtick_labels = [str(p) for p in positions]
        ax.set_xticks(range(n_positions))
        ax.set_xticklabels(xtick_labels, rotation=90, fontsize=8)
        ax.set_xlabel("Position")

        fig.colorbar(im, ax=ax, label="Entropy (bits)")
        ax.set_title("Jacobian lens (top-1 token, colored by entropy)")
        fig.tight_layout()
        fig.subplots_adjust(left=0.25)
        plt.show()


def _fit_config(model_source, mask_mode, dataset, n_seqs, seq_len, block_names, hidden_size):
    """Identity of a fitted lens; a cached lens is only reused if this matches."""
    return dict(model_source=model_source, mask_mode=mask_mode, dataset=dataset,
                n_seqs=n_seqs, seq_len=seq_len, blocks=list(block_names),
                hidden_size=hidden_size)


if __name__ == "__main__":
    import os
    from utils import load_hrm

    # ---- knobs --------------------------------------------------------------
    MODEL_SOURCE = "sapientinc/HRM-Text-1B"   # HF id/dir or native checkpoint dir
    MASK_MODE = "prefix"                       # "prefix" (default) or "causal"
    DATASET = "NeelNanda/c4-10k"               # any {split:"train", "text"} corpus
    N_SEQS = 128                               # corpus size for fitting (Balanced)
    SEQ_LEN = 128                              # tokens per corpus sequence
    DIM_BATCH = 16                             # VJP rows per backward (memory<->#passes)
    LAYER_IDS = [0, 7, 15]                     # which blocks to probe (as in logit_lens)
    LENS_CACHE = os.path.join(os.path.dirname(__file__), "jacobian_lens.pt")

    device = "cuda" if torch.cuda.is_available() else \
             ("mps" if torch.backends.mps.is_available() else "cpu")
    model, tokenizer = load_hrm(MODEL_SOURCE, dtype=torch.bfloat16, device=device)

    # Blocks to probe, in a fixed order shared by fitting and readout.
    blocks = [(model.model.L_module.layers[i], f"L[{i}]") for i in LAYER_IDS] \
           + [(model.model.H_module.layers[i], f"H[{i}]") for i in LAYER_IDS]
    block_names = [name for _, name in blocks]
    cfg = _fit_config(MODEL_SOURCE, MASK_MODE, DATASET, N_SEQS, SEQ_LEN,
                      block_names, model.config.hidden_size)

    # ---- fit (or load cached) ----------------------------------------------
    lens = None
    if os.path.isfile(LENS_CACHE):
        cached = JacobianLens.load(LENS_CACHE, tokenizer, model.lm_head, topk=1)
        if cached.config == cfg:
            print(f"[jacobian_lens] reusing cached lens: {LENS_CACHE}")
            lens = cached
        else:
            print("[jacobian_lens] cached lens config differs; refitting.")
    if lens is None:
        print(f"Stage A: loading {N_SEQS} corpus sequences from {DATASET}")
        seqs = load_corpus(tokenizer, N_SEQS, SEQ_LEN, dataset=DATASET)
        print(f"Stage B: estimating Jacobians ({MASK_MODE} mask) on {device} -- "
              f"~{len(seqs)} x ceil({model.config.hidden_size}/{DIM_BATCH}) backward passes")
        jacobians, layer_names = estimate_jacobians(
            model, blocks, seqs, device, dim_batch=DIM_BATCH, mask_mode=MASK_MODE)
        lens = JacobianLens(tokenizer, model.lm_head, jacobians, layer_names,
                            topk=1, config=cfg)
        lens.save(LENS_CACHE)

    # Register the readout hooks on the very same blocks, in the same order.
    for module, name in blocks:
        module.register_forward_hook(lens.build_hook(layer_name=name))

    # ---- readout on a demo prompt (same style as logit_lens.py) ------------
    condition = "<|quad_end|><|object_ref_end|>"
    prompt = f"<|im_start|>{condition}Explain why the sky is blue.<|im_end|>"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    inputs["token_type_ids"] = torch.ones_like(inputs["input_ids"]) \
        if MASK_MODE == "prefix" else torch.zeros_like(inputs["input_ids"])

    print("Step 1: Generate a short continuation")
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=16, do_sample=False)
    decoded = tokenizer.decode(out[0], skip_special_tokens=False)
    print("Generated Text:", decoded)

    # Re-encode prompt+generation and run one capturing forward (all positions
    # at once -- no generation loop needed for the layer x position grid).
    prompt = decoded
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    inputs["token_type_ids"] = torch.ones_like(inputs["input_ids"]) \
        if MASK_MODE == "prefix" else torch.zeros_like(inputs["input_ids"])

    print("Step 2: Capturing Jacobian-lens readout")
    with torch.inference_mode():
        with lens as captures:
            lens.begin_pass()
            model(**inputs)

    print("#Captures:", len(captures))
    # Note: the final H invocation is ~identity under J, so its row should match
    # the model's own next-token predictions -- the logit-lens/model baseline.
    lens.show(captures, text=prompt)
