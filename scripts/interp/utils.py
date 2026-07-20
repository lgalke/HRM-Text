"""Shared model loading for the interpretability scripts.

`load_hrm(source)` returns a `(model, tokenizer)` pair with the ordinary
HuggingFace `HrmTextForCausalLM` interface (`model.model.L_module.layers`,
`model.model.H_module.layers`, `model.lm_head`, `model.generate`, ...), so the
geometry / logit-lens / linear-probe scripts can hook and drive it exactly the
same way regardless of where the weights came from.

`source` can be either

  * a HuggingFace repo id or a directory already in HF format
    (contains ``config.json`` with ``model_type: hrm_text``), e.g.
    ``"sapientinc/HRM-Text-1B"``; or
  * a native training checkpoint directory (contains ``all_config.yaml`` and
    ``fsdp2_*`` / ``unsharded_*`` files). These are converted to a cached HF
    export on first use via the repo's ``conversion/convert_to_hf.py`` and then
    loaded like any other HF model.

The conversion runs as a subprocess so the repo-side imports (``pretrain``,
``simple_inference_engine``, the ``utils`` *package*) resolve against the repo
root rather than this file — this module is imported as ``utils`` from
``scripts/interp/`` and would otherwise shadow that package.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
CONVERT_SCRIPT = REPO_ROOT / "conversion" / "convert_to_hf.py"


def _is_native_checkpoint(path: Path) -> bool:
    """Native training checkpoints carry the training config alongside weights."""
    return path.is_dir() and (path / "all_config.yaml").is_file()


def _export_is_complete(export_dir: Path) -> bool:
    return (export_dir / "config.json").is_file() and (export_dir / "model.safetensors").is_file()


def _convert_checkpoint(
    ckpt_path: Path,
    export_dir: Path,
    *,
    epoch: Optional[int],
    tag: Optional[str],
    use_ema: bool,
    tokenizer_path: Optional[str],
) -> None:
    """Invoke the canonical converter to materialise an HF export at `export_dir`."""
    if not CONVERT_SCRIPT.is_file():
        raise FileNotFoundError(f"Converter not found: {CONVERT_SCRIPT}")
    cmd = [
        sys.executable,
        str(CONVERT_SCRIPT),
        "--ckpt_path", str(ckpt_path),
        "--out_dir", str(export_dir),
        "--ckpt_use_ema", "true" if use_ema else "false",
    ]
    if epoch is not None:
        cmd += ["--ckpt_epoch", str(epoch)]
    if tag is not None:
        cmd += ["--ckpt_tag", tag]
    if tokenizer_path is not None:
        cmd += ["--tokenizer_path", tokenizer_path]
    print(f"[load_hrm] converting checkpoint -> {export_dir}")
    subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)
    if not _export_is_complete(export_dir):
        raise RuntimeError(f"Conversion did not produce a usable export at {export_dir}")


def resolve_to_hf_dir(
    source: str,
    *,
    epoch: Optional[int] = None,
    tag: Optional[str] = None,
    use_ema: bool = True,
    export_dir: Optional[str] = None,
    tokenizer_path: Optional[str] = None,
    force_convert: bool = False,
) -> str:
    """Return something `AutoModel.from_pretrained` can load.

    HF repo ids and HF-format directories are passed through unchanged. Native
    checkpoints are converted (once, cached) and the export directory is
    returned.
    """
    path = Path(source).expanduser()

    # Already HF format, or a hub id / non-path string -> pass through.
    if not _is_native_checkpoint(path):
        return source

    # Native checkpoint: convert to a cached HF export.
    # Key the cache by what selects the weights so distinct epochs/tags/EMA
    # don't collide. "latest" (no epoch/tag) is reconverted each run since it
    # can go stale as training produces new checkpoints.
    if epoch is not None:
        key = f"epoch_{epoch}"
    elif tag is not None:
        key = tag
    else:
        key = "latest"
    key += "_ema" if use_ema else "_raw"

    out = Path(export_dir).expanduser() if export_dir is not None else path / "hf_export" / key
    if _export_is_complete(out) and not force_convert and not key.startswith("latest"):
        print(f"[load_hrm] reusing cached export: {out}")
        return str(out)

    out.mkdir(parents=True, exist_ok=True)
    _convert_checkpoint(
        path, out, epoch=epoch, tag=tag, use_ema=use_ema, tokenizer_path=tokenizer_path
    )
    return str(out)


def load_hrm(
    source: str = "sapientinc/HRM-Text-1B",
    *,
    dtype=torch.bfloat16,
    device: Optional[str] = None,
    epoch: Optional[int] = None,
    tag: Optional[str] = None,
    use_ema: bool = True,
    export_dir: Optional[str] = None,
    tokenizer_path: Optional[str] = None,
    force_convert: bool = False,
) -> Tuple[torch.nn.Module, "AutoTokenizer"]:
    """Load an HRM-Text model + tokenizer from a HF id/dir or a native checkpoint.

    Parameters mirror the checkpoint selection knobs of the converter:
    `epoch`/`tag` pick a checkpoint (mutually exclusive; default = latest epoch),
    `use_ema` selects EMA weights. `export_dir` overrides where a converted
    checkpoint is cached; `force_convert` reconverts even if a cache exists.
    Returns the model already in `.eval()` mode and on `device` if given.
    """
    hf_dir = resolve_to_hf_dir(
        source,
        epoch=epoch,
        tag=tag,
        use_ema=use_ema,
        export_dir=export_dir,
        tokenizer_path=tokenizer_path,
        force_convert=force_convert,
    )
    tokenizer = AutoTokenizer.from_pretrained(hf_dir)
    model = AutoModelForCausalLM.from_pretrained(hf_dir, dtype=dtype).eval()
    if device is not None:
        model = model.to(device)
    return model, tokenizer


if __name__ == "__main__":
    # Smoke test: python scripts/interp/utils.py [source]
    src = sys.argv[1] if len(sys.argv) > 1 else "sapientinc/HRM-Text-1B"
    m, tok = load_hrm(src, device="cpu")
    print(type(m).__name__, "loaded;",
          len(m.model.L_module.layers), "L layers,",
          len(m.model.H_module.layers), "H layers")
