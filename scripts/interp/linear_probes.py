"""Linear probes on the residual stream for Danish linguistic acceptability.

Trains one logistic-regression probe per layer application on the residual
stream (block output, pooled over tokens) and reports test accuracy per layer
-- a standard "where in the network is <property> linearly decodable" curve.

Data: giannor/dala (binary: 'correct' vs 'incorrect'). We train on the `train`
split and measure on the `test` split. Both are class-balanced, so the
majority-class baseline is 0.5.

Like geometry.py / logit_lens.py, HRM applies its H/L blocks recurrently, so a
layer *name* is not unique -- features are keyed by *invocation index* (the
position of the hook fire within a forward pass), which aligns across batches
as long as every forward fires the hooks the same number of times.
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple


@dataclass
class ProbeResult:
    invocation: int
    layer_name: str
    train_acc: float
    eval_acc: Dict[str, float]   # split name -> accuracy (e.g. "val", "test")
    n_train: int
    n_eval: Dict[str, int]


class ResidualCollector:
    """Hooks each block, pools its output residual stream to one vector per
    sequence, and stacks those vectors per invocation index. `pool="mean"`
    averages over real (non-pad) tokens; `pool="last"` takes the last real
    token."""

    def __init__(self, pool="mean"):
        assert pool in ("mean", "last")
        self.pool = pool
        self.do_capture = False
        self._invocation = 0
        self._mask: Optional[torch.Tensor] = None
        self._feats: Dict[int, List[torch.Tensor]] = {}
        self._names: Dict[int, str] = {}

    def begin_pass(self, mask):
        self._invocation = 0
        self._mask = mask

    def build_hook(self, layer_name):
        def hook(module, inputs, outputs):
            if not self.do_capture:
                return
            h = outputs[0] if isinstance(outputs, tuple) else outputs  # [B, S, d]
            self._feats.setdefault(self._invocation, []).append(self._pool(h, self._mask))
            self._names[self._invocation] = layer_name
            self._invocation += 1
        return hook

    def _pool(self, h, mask):
        h = h.float()
        if mask is None:
            mask = torch.ones(h.shape[:2], dtype=torch.long, device=h.device)
        if self.pool == "mean":
            m = mask.unsqueeze(-1).float()
            feat = (h * m).sum(1) / m.sum(1).clamp_min(1.0)     # [B, d]
        else:  # last real token (right-padded: real tokens fill [0, len))
            last = mask.long().sum(1) - 1                       # [B]
            feat = h[torch.arange(h.shape[0], device=h.device), last]
        return feat.detach().cpu()

    def collect(self, model, batches, device="cpu") -> Tuple[Dict[int, torch.Tensor], Dict[int, str]]:
        """Run batched forward passes and return {invocation: [N, d] features}
        (example order == batch order) plus {invocation: layer_name}."""
        self.do_capture = True
        self._feats, self._names = {}, {}
        expected = None
        try:
            for b, batch in enumerate(batches):
                batch = {k: v.to(device) for k, v in batch.items()}
                self.begin_pass(batch.get("attention_mask"))
                with torch.inference_mode():
                    model(**batch)
                if expected is None:
                    expected = self._invocation
                elif self._invocation != expected:
                    raise RuntimeError(
                        f"Batch {b} fired the hooks {self._invocation} times but batch 0 "
                        f"fired {expected}. Invocation indices no longer align across "
                        f"forward passes (input-dependent cycle count?), so per-layer "
                        f"features would be mixed together.")
            if not expected:
                raise RuntimeError("No hooks fired -- register build_hook() on the blocks first.")
        finally:
            self.do_capture = False
        feats = {inv: torch.cat(chunks, 0) for inv, chunks in self._feats.items()}
        return feats, self._names


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def load_dala(tokenizer, split, max_examples=None) -> Tuple[List[torch.Tensor], np.ndarray]:
    """Return per-example token-id tensors and int labels (correct=1)."""
    from datasets import load_dataset
    ds = load_dataset("giannor/dala")[split]
    if max_examples is not None:
        ds = ds.select(range(min(max_examples, ds.num_rows)))
    labels = np.array([1 if l == "correct" else 0 for l in ds["label"]], dtype=np.int64)
    seqs = [tokenizer(t, return_tensors="pt")["input_ids"][0] for t in ds["text"]]
    return seqs, labels


def build_batches(seqs: Sequence[torch.Tensor], pad_id: int,
                  batch_size=16) -> List[Dict[str, torch.Tensor]]:
    """Right-pad sequences into batches, preserving order (so features align
    with the label array). Whole sequence is treated as a bidirectional prefix
    (token_type_ids = attention_mask), matching the other scripts."""
    batches = []
    for i in range(0, len(seqs), batch_size):
        chunk = seqs[i:i + batch_size]
        L = max(s.numel() for s in chunk)
        ids = torch.full((len(chunk), L), pad_id, dtype=torch.long)
        am = torch.zeros((len(chunk), L), dtype=torch.long)
        for r, s in enumerate(chunk):
            ids[r, : s.numel()] = s
            am[r, : s.numel()] = 1
        batches.append(dict(input_ids=ids, attention_mask=am, token_type_ids=am.clone()))
    return batches


# --------------------------------------------------------------------------
# Probing
# --------------------------------------------------------------------------
def train_probes(train_feats, y_train, eval_sets, names) -> List[ProbeResult]:
    """Standardize (fit on train) then fit a logistic-regression probe per
    invocation, scoring each split in `eval_sets` = {name: (feats, labels)}.
    Probes are trained once and reused across all eval splits."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    results = []
    for inv in sorted(train_feats):
        Xtr = train_feats[inv].numpy()
        scaler = StandardScaler().fit(Xtr)
        clf = LogisticRegression(max_iter=1000).fit(scaler.transform(Xtr), y_train)
        eval_acc = {name: float(clf.score(scaler.transform(feats[inv].numpy()), y))
                    for name, (feats, y) in eval_sets.items()}
        results.append(ProbeResult(
            invocation=inv,
            layer_name=names[inv],
            train_acc=float(clf.score(scaler.transform(Xtr), y_train)),
            eval_acc=eval_acc,
            n_train=len(y_train),
            n_eval={name: len(y) for name, (_, y) in eval_sets.items()},
        ))
        msg = "  ".join(f"{k}={v:.3f}" for k, v in eval_acc.items())
        print(f"  {names[inv]:<8} (inv {inv:>3})  train={results[-1].train_acc:.3f}  {msg}")
    return results


def _cycles(labels):
    """Split invocations into recurrent cycles, labelled per module (L/H).
    Boundary wherever a layer name recurs or the module changes."""
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


def select_best(results: List[ProbeResult], select_on="val") -> ProbeResult:
    """Pick the layer with the highest accuracy on the selection split."""
    return max(results, key=lambda r: r.eval_acc[select_on])


def show(results: List[ProbeResult], baseline=0.5, select_on="val", title=None):
    """Probe accuracy over layer applications, one line per eval split, with
    L/H cycles shaded and the `select_on`-selected layer marked."""
    if not results:
        print("No probe results to show.")
        return
    import matplotlib.pyplot as plt
    from matplotlib.transforms import blended_transform_factory

    results = sorted(results, key=lambda r: r.invocation)
    x = np.arange(len(results))
    train_acc = np.array([r.train_acc for r in results])
    labels = [r.layer_name for r in results]
    splits = list(results[0].eval_acc.keys())
    split_acc = {s: np.array([r.eval_acc[s] for r in results]) for s in splits}
    colors = {"val": "C1", "test": "C0"}

    fig, ax = plt.subplots(figsize=(max(7, len(results) * 0.28), 4.5))
    lo = min([baseline, train_acc.min()] + [a.min() for a in split_acc.values()]) - 0.05
    ax.set_ylim(max(0.0, lo), 1.02)

    label_tf = blended_transform_factory(ax.transData, ax.transAxes)
    for k, (s, e, clabel) in enumerate(_cycles(labels)):
        if k % 2:
            ax.axvspan(s - 0.5, e - 0.5, color="0.92", zorder=0)
        ax.text((s + e - 1) / 2, 1.01, clabel, transform=label_tf,
                ha="center", va="bottom", fontsize=7, color="0.4")

    ax.axhline(baseline, color="0.6", lw=1, ls="--", label=f"majority baseline ({baseline:.2f})")
    ax.plot(x, train_acc, ":", lw=1, color="gray", label="train accuracy")
    for s in splits:
        ax.plot(x, split_acc[s], "-o", ms=3, color=colors.get(s), label=f"{s} accuracy")

    # Mark the layer selected on `select_on` (chosen without touching test).
    if select_on in splits:
        bi = int(np.argmax(split_acc[select_on]))
        ax.axvline(bi, color="C3", lw=1.2, alpha=0.7, zorder=1)
        best = results[bi]
        note = f"selected on {select_on}: {best.layer_name}"
        if "test" in best.eval_acc:
            note += f"\ntest = {best.eval_acc['test']:.3f}"
        ax.text(bi, 0.02, note, transform=label_tf, ha="center", va="bottom",
                fontsize=7, color="C3")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_xlabel("Layer application (processing order)")
    ax.set_ylabel("probe accuracy")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    n = results[0]
    ns = ", ".join(f"{k} n={v}" for k, v in n.n_eval.items())
    ax.set_title(title or f"DALA acceptability probe  (train n={n.n_train}, {ns})")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    model_id = "sapientinc/HRM-Text-1B"
    device = "cuda" if torch.cuda.is_available() else \
             ("mps" if torch.backends.mps.is_available() else "cpu")

    # Memory ~ N * d * (#layer applications) * 4 bytes per split; lower these
    # for a quick run. None = use the full split.
    MAX_TRAIN, MAX_VAL, MAX_TEST = None, None, None
    BATCH_SIZE = 16
    POOL = "mean"  # or "last"
    # First runs: keep EVAL_TEST=False to work on train+val only (fast, and the
    # test set stays untouched during layer selection). Flip to True for the
    # final measurement once you've settled on the setup.
    EVAL_TEST = False

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16).eval().to(device)

    collector = ResidualCollector(pool=POOL)
    for i, block in enumerate(model.model.L_module.layers):
        block.register_forward_hook(collector.build_hook(layer_name=f"L[{i}]"))
    for i, block in enumerate(model.model.H_module.layers):
        block.register_forward_hook(collector.build_hook(layer_name=f"H[{i}]"))

    def features_for(split, cap):
        seqs, y = load_dala(tokenizer, split, cap)
        batches = build_batches(seqs, tokenizer.pad_token_id, BATCH_SIZE)
        feats, names = collector.collect(model, batches, device=device)
        print(f"  {split}: {len(y)} sentences")
        return feats, y, names

    print(f"Loading DALA and extracting residual-stream features on {device} (pool={POOL})")
    train_feats, y_train, names = features_for("train", MAX_TRAIN)
    val_feats, y_val, _ = features_for("val", MAX_VAL)
    eval_sets = {"val": (val_feats, y_val)}
    if EVAL_TEST:
        test_feats, y_test, _ = features_for("test", MAX_TEST)
        eval_sets["test"] = (test_feats, y_test)
    print(f"  {len(names)} layer applications, feature dim {train_feats[0].shape[1]}")

    print("Training probes:")
    results = train_probes(train_feats, y_train, eval_sets, names)

    best = select_best(results, select_on="val")
    line = f"Layer selected on val: {best.layer_name} (inv {best.invocation})  val acc {best.eval_acc['val']:.3f}"
    if "test" in best.eval_acc:
        line += f"  -> test acc {best.eval_acc['test']:.3f}"
    else:
        line += "  (set EVAL_TEST=True to measure this layer on the test split)"
    print(line)
    show(results, select_on="val")
