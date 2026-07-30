import torch

from dataclasses import dataclass
from typing import List, Dict, Optional, Sequence, Tuple


@dataclass
class GeometryStat:
    """Aggregated geometry for one layer application, over a whole corpus."""
    invocation: int
    layer_name: str
    n: int                  # number of sequences aggregated
    mean_token_cos: float   # direction change, mean over sequences
    std_token_cos: float    # spread across sequences
    mean_min_cos: float     # typical most-rotated token (per-seq min, averaged)
    mean_rel_norm: float    # write size ||delta||/||input||, mean over sequences
    std_rel_norm: float


class GeometryProbe:
    """Streaming measurement of how much each block moves the residual stream.

    A block's forward_hook exposes both sides of the residual update:
    `inputs[0]` is the hidden state going in, `outputs` the hidden state
    coming out. Per layer application we record two complementary quantities,
    reduced per sequence (masked to real tokens) then streamed into running
    accumulators keyed by *invocation index* -- the position of the hook fire
    within a forward pass. Because HRM's recurrent H/L cycles are fixed and
    internal to each module's forward, a single `model(**batch)` forward fires
    the hooks in the same order every time, so invocation index aligns across
    batches and we can aggregate over an arbitrarily large corpus in O(#layer
    applications) memory.

    Per application:
      * mean_token_cos  : mean_i cos(before_i, after_i)  -- direction rotated.
      * rel_update_norm : mean_i ||after_i - before_i|| / ||before_i||  -- the
        write size, normalized so it is comparable across depth/cycles.

    `center=True` mean-centers tokens (over real tokens) before the cosine,
    removing the residual stream's shared anisotropic component. The update
    norm is always taken on the RAW stream.

    Aggregation is per sequence (each example weighted equally); the reported
    band is the standard deviation across sequences.
    """

    def __init__(self, center=False):
        self.center = center
        self.do_capture = False
        self._invocation = 0
        self._mask: Optional[torch.Tensor] = None
        self._agg: Dict[int, dict] = {}

    # ---- context management -------------------------------------------------
    def __enter__(self):
        self._agg = {}
        self.do_capture = True
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.do_capture = False

    def begin_pass(self, mask):
        """Reset the invocation counter and set the attention mask for the
        forward pass about to run. Call once before each `model(**batch)`."""
        self._invocation = 0
        self._mask = mask

    # ---- hooks --------------------------------------------------------------
    def build_hook(self, layer_name):
        def hook(module, inputs, outputs):
            if not self.do_capture:
                return
            before = inputs[0]
            after = outputs[0] if isinstance(outputs, tuple) else outputs
            self._accumulate(layer_name, before, after)
            self._invocation += 1
        return hook

    def _accumulate(self, layer_name, before, after):
        mean_cos, min_cos, mean_rel = self._masked_metrics(before, after, self._mask)
        rec = self._agg.get(self._invocation)
        if rec is None:
            rec = dict(layer_name=layer_name, n=0.0, s_cos=0.0, sq_cos=0.0,
                       s_min=0.0, s_rel=0.0, sq_rel=0.0)
            self._agg[self._invocation] = rec
        rec["n"] += mean_cos.numel()
        rec["s_cos"] += float(mean_cos.sum())
        rec["sq_cos"] += float((mean_cos ** 2).sum())
        rec["s_min"] += float(min_cos.sum())
        rec["s_rel"] += float(mean_rel.sum())
        rec["sq_rel"] += float((mean_rel ** 2).sum())

    def _masked_metrics(self, before, after, mask):
        """before/after: [B, S, d]; mask: [B, S] (1=real, 0=pad). Returns three
        per-sequence [B] tensors: mean cosine, min cosine, mean relative norm."""
        import torch.nn.functional as F
        before = before.float()
        after = after.float()
        if mask is None:
            mask = torch.ones(before.shape[:2], dtype=torch.bool, device=before.device)
        m = mask.bool()
        mf = m.float()
        n = mf.sum(1).clamp_min(1.0)  # [B] valid tokens per sequence

        # Write size on the RAW stream.
        rel = (after - before).norm(dim=-1) / before.norm(dim=-1).clamp_min(1e-12)  # [B,S]

        # Direction change; optionally de-anisotropized by centering over real tokens.
        b, a = before, after
        if self.center:
            cnt = mf.sum(1, keepdim=True).clamp_min(1.0).unsqueeze(-1)   # [B,1,1]
            mu_b = (before * mf.unsqueeze(-1)).sum(1, keepdim=True) / cnt
            mu_a = (after * mf.unsqueeze(-1)).sum(1, keepdim=True) / cnt
            b = before - mu_b
            a = after - mu_a
        cos = (F.normalize(b, dim=-1) * F.normalize(a, dim=-1)).sum(-1)  # [B,S]

        mean_cos = (cos * mf).sum(1) / n        # [B]
        mean_rel = (rel * mf).sum(1) / n        # [B]
        min_cos = cos.masked_fill(~m, float("inf")).min(1).values  # [B]
        return mean_cos.detach().cpu(), min_cos.detach().cpu(), mean_rel.detach().cpu()

    # ---- driving ------------------------------------------------------------
    def measure(self, model, batches, device="cpu"):
        """Run batched forward passes over `batches` (each a dict of tensors)
        and return aggregated per-application stats.

        Aggregating by invocation index assumes every forward fires the hooks
        the same number of times. If HRM uses input-dependent cycle counts
        (adaptive halting), that assumption breaks and the per-application
        averages would silently mix different layer applications -- so we
        assert a constant count and fail loudly instead."""
        expected = None
        with self:
            for b, batch in enumerate(batches):
                batch = {k: v.to(device) for k, v in batch.items()}
                self.begin_pass(batch.get("attention_mask"))
                with torch.inference_mode():
                    model(**batch)
                fired = self._invocation  # hook fires this forward
                if expected is None:
                    expected = fired
                elif fired != expected:
                    raise RuntimeError(
                        f"Batch {b} fired the hooks {fired} times but batch 0 fired "
                        f"{expected}. Invocation indices no longer align across forward "
                        f"passes (input-dependent cycle count?), so per-application "
                        f"aggregation is invalid.")
            if expected == 0:
                raise RuntimeError(
                    "No hooks fired -- register build_hook() on the blocks before measuring.")
        return self.results()

    def results(self) -> List[GeometryStat]:
        out = []
        for inv in sorted(self._agg):
            r = self._agg[inv]
            n = max(r["n"], 1.0)
            mean_cos = r["s_cos"] / n
            mean_rel = r["s_rel"] / n
            std_cos = max(r["sq_cos"] / n - mean_cos ** 2, 0.0) ** 0.5
            std_rel = max(r["sq_rel"] / n - mean_rel ** 2, 0.0) ** 0.5
            out.append(GeometryStat(inv, r["layer_name"], int(r["n"]),
                                    mean_cos, std_cos, r["s_min"] / n,
                                    mean_rel, std_rel))
        return out

    # ---- plotting -----------------------------------------------------------
    @staticmethod
    def _cycles(labels):
        """Split invocations into recurrent cycles. Boundary wherever a layer
        name recurs OR the module (L/H) changes, so an L sweep and the H sweep
        after it never merge. Returns (start, end, label) with per-module
        numbering: L-cycle 1, L-cycle 2, H-cycle 1, ..."""
        def module(name):
            return name.split("[")[0]

        bounds, start, seen = [], 0, set()
        for i, name in enumerate(labels):
            if seen and (name in seen or module(name) != module(labels[start])):
                bounds.append((start, i))
                start, seen = i, set()
            seen.add(name)
        bounds.append((start, len(labels)))

        cycles, counts = [], {}
        for s, e in bounds:
            mod = module(labels[s])
            counts[mod] = counts.get(mod, 0) + 1
            cycles.append((s, e, f"{mod}-cycle {counts[mod]}"))
        return cycles

    def show(self, stats=None, title=None):
        """Line chart of per-application geometry over layer applications, with
        mean +/- std bands across the corpus. Recurrent cycles are shaded."""
        if stats is None:
            stats = self.results()
        if not stats:
            print("No geometry stats to show.")
            return

        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.transforms import blended_transform_factory

        stats = sorted(stats, key=lambda s: s.invocation)
        x = np.arange(len(stats))
        mean_cos = np.array([s.mean_token_cos for s in stats])
        std_cos = np.array([s.std_token_cos for s in stats])
        min_cos = np.array([s.mean_min_cos for s in stats])
        rel = np.array([s.mean_rel_norm for s in stats])
        std_rel = np.array([s.std_rel_norm for s in stats])
        labels = [s.layer_name for s in stats]
        n_seq = stats[0].n if stats else 0

        fig, ax = plt.subplots(figsize=(max(7, len(stats) * 0.28), 4.5))
        top = max(1.05, float((rel + std_rel).max()) * 1.1)
        bottom = min(0.0, float(min_cos.min()) - 0.05)
        ax.set_ylim(bottom, top)

        label_tf = blended_transform_factory(ax.transData, ax.transAxes)
        for k, (s, e, clabel) in enumerate(self._cycles(labels)):
            if k % 2:
                ax.axvspan(s - 0.5, e - 0.5, color="0.92", zorder=0)
            ax.text((s + e - 1) / 2, 1.01, clabel, transform=label_tf,
                    ha="center", va="bottom", fontsize=7, color="0.4")

        ax.plot(x, mean_cos, "-o", ms=3, color="C0",
                label="mean per-token cos(before, after)  [direction]")
        ax.fill_between(x, mean_cos - std_cos, mean_cos + std_cos, color="C0", alpha=0.15)
        ax.plot(x, min_cos, ":", lw=1, color="gray",
                label="per-seq min cos (most-rotated token)")
        ax.plot(x, rel, "-s", ms=3, color="C3",
                label="||delta|| / ||input||  [write size]")
        ax.fill_between(x, rel - std_rel, rel + std_rel, color="C3", alpha=0.15)

        ax.axhline(1.0, color="0.8", lw=0.8, zorder=0)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=90, fontsize=6)
        ax.set_xlabel("Layer application (processing order)")
        ax.set_ylabel("cosine similarity  /  relative update norm")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(fontsize=8, loc="lower left")
        base = title or ("centered" if self.center else "raw (uncentered)")
        ax.set_title(f"{base}  (n={n_seq} sequences)")
        fig.tight_layout()
        plt.show()


# --------------------------------------------------------------------------
# Stage A: build a corpus of self-generated sequences (amortized, batched).
# --------------------------------------------------------------------------
@torch.inference_mode()
def generate_sequences(model, tokenizer, prompts, batch_size=8,
                       max_new_tokens=64, device="cpu") -> List[Tuple[torch.Tensor, int]]:
    """Batch-generate completions and return one `(ids, prompt_len)` per prompt:
    a 1-D token-id tensor (prompt + generation, padding stripped) and how many of
    its leading tokens are the prompt (the bidirectional-prefix boundary)."""
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # decoder-only generation needs left padding
    pad_id = tokenizer.pad_token_id

    seqs: List[Tuple[torch.Tensor, int]] = []
    for i in range(0, len(prompts), batch_size):
        chunk = prompts[i:i + batch_size]
        enc = tokenizer(chunk, return_tensors="pt", padding=True).to(device)
        enc["token_type_ids"] = enc["attention_mask"].clone()  # prompt is the prefix
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False)
        for k in range(len(chunk)):
            row = out[k]
            left_pad = int((enc["attention_mask"][k] == 0).sum())  # leading pads
            prompt_len = int(enc["attention_mask"][k].sum())       # real prompt tokens
            ids = row[left_pad:]                                   # prompt + generation
            nonpad = (ids != pad_id).nonzero()                    # strip trailing pads
            if nonpad.numel():
                ids = ids[: int(nonpad[-1]) + 1]
            seqs.append((ids.detach().cpu(), prompt_len))
    return seqs


def build_batches(seqs: Sequence[Tuple[torch.Tensor, int]], pad_id: int, batch_size=8,
                  sort_by_length=True, readout_mask="generation") -> List[Dict[str, torch.Tensor]]:
    """Right-pad `(ids, prompt_len)` sequences into measurement batches.

    `readout_mask` sets the attention pattern the geometry is measured under:
      "generation" -- only the prompt is the bidirectional prefix (token_type_ids
                      ==1); the generated tail is causal (==0). This reproduces how
                      the tokens were actually produced (each generated token saw
                      only the prefix + earlier tokens during causal decode), so the
                      per-block geometry reflects the model's generation-time compute.
      "prefix"     -- whole sequence bidirectional (all real tokens ==1). Simpler,
                      but generated tokens then attend forward to tokens they never
                      saw while being generated.
    Length-sorting cuts padding waste."""
    order = sorted(range(len(seqs)), key=lambda i: seqs[i][0].numel()) if sort_by_length \
        else list(range(len(seqs)))
    batches = []
    for i in range(0, len(order), batch_size):
        idx = order[i:i + batch_size]
        L = max(seqs[j][0].numel() for j in idx)
        ids = torch.full((len(idx), L), pad_id, dtype=torch.long)
        am = torch.zeros((len(idx), L), dtype=torch.long)
        tt = torch.zeros((len(idx), L), dtype=torch.long)
        for r, j in enumerate(idx):
            s, prompt_len = seqs[j]
            ids[r, : s.numel()] = s
            am[r, : s.numel()] = 1
            # Prefix span: the whole real sequence, or just the prompt if we want
            # the generated tail to stay causal (as it was during generation).
            prefix_end = s.numel() if readout_mask == "prefix" else min(prompt_len, s.numel())
            tt[r, :prefix_end] = 1
        batches.append(dict(input_ids=ids, attention_mask=am, token_type_ids=tt))
    return batches


if __name__ == "__main__":
    from utils import load_hrm

    # A HF repo id / dir, or a native training checkpoint dir (auto-converted).
    MODEL_SOURCE = "sapientinc/HRM-Text-1B"
    # Attention pattern the geometry is measured under (see build_batches):
    #   "generation" -- prompt is the prefix, generated tail causal (faithful to
    #                   how the tokens were produced); "prefix" -- whole sequence
    #                   bidirectional (generated tokens see the future).
    READOUT_MASK = "generation"
    device = "cuda" if torch.cuda.is_available() else \
             ("mps" if torch.backends.mps.is_available() else "cpu")
    model, tokenizer = load_hrm(MODEL_SOURCE, dtype=torch.bfloat16, device=device)

    geometry = GeometryProbe(center=True)
    for i, block in enumerate(model.model.L_module.layers):
        block.register_forward_hook(geometry.build_hook(layer_name=f"L[{i}]"))
    for i, block in enumerate(model.model.H_module.layers):
        block.register_forward_hook(geometry.build_hook(layer_name=f"H[{i}]"))

    # A few prompts standing in for a larger corpus.
    condition = "<|quad_end|><|object_ref_end|>"
    questions = [
        "Explain why the sky is blue.",
        "Describe how a rainbow forms.",
        "What causes the seasons to change?",
        "How does a suspension bridge stay up?",
        "Why does ice float on water?",
        "How do vaccines train the immune system?",
        "What makes a plane able to fly?",
        "Why do we see lightning before hearing thunder?",
        "How does a battery store energy?",
        "Why do leaves change color in autumn?",
        "How does the greenhouse effect warm the planet?",
        "Why is the ocean salty?",
        "How do noise-cancelling headphones work?",
        "What causes tides in the ocean?",
        "Why do onions make you cry?",
        "How does GPS know your location?",
        "Why do stars twinkle at night?",
        "How does soap remove grease?",
        "What makes a magnet attract metal?",
        "Why does bread rise when baking?",
    ]
    prompts = [f"<|im_start|>{condition}{q}<|im_end|>" for q in questions]

    print(f"Stage A: batch-generating {len(prompts)} sequences on {device}")
    seqs = generate_sequences(model, tokenizer, prompts, batch_size=4,
                              max_new_tokens=32, device=device)

    print("Stage B: batched forward passes + streaming geometry")
    batches = build_batches(seqs, pad_id=tokenizer.pad_token_id, batch_size=4,
                            readout_mask=READOUT_MASK)
    stats = geometry.measure(model, batches, device=device)

    print(f"#Layer applications: {len(stats)}  (n={stats[0].n} sequences each)")
    geometry.show(stats)
