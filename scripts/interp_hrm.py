from transformers import AutoModelForCausalLM, AutoTokenizer
import torch



from dataclasses import dataclass
from typing import List

# (layer_idx, position_idx, token_id) -> logit value
@dataclass
class LogitLensCapture:
    layer_name: str
    position_idx: int
    token_rank: int
    token_id: int
    token_str: str
    proba_value: float

    # def __repr__(self):
    #     return f"LogitLensCapture(layer_name={self.layer_name}, position_idx={self.position_idx}, token_rank={self.token_rank}, token_id={self.token_id}, token_str='{self.token_str}', proba_value={self.proba_value})"

class LogitLens:
    def __init__(self, tokenizer, lm_head, topk=1):
        self.tokenizer = tokenizer
        self.lm_head = lm_head
        self.topk = topk
        self.captures : List[LogitLensCapture] = []
        self.do_capture = False

    def __enter__(self):
        self.captures = []
        self.do_capture = True
        return self.captures

    def __exit__(self, exc_type, exc_value, traceback):
        if not self.captures:
            print("No captures were made. Register LogitLens.hook() on the model layers before running inference.")
        self.do_capture = False

    def __call__(self, outputs, layer_name=None):
        logits = self.lm_head(outputs.detach())
        bsz, seq_len, vocab_size = logits.shape
        assert bsz == 1, "Batch size > 1 not supported"
        probas = torch.softmax(logits, dim=-1)
        topk_values, topk_indices = torch.topk(probas, k=self.topk, dim=-1)
        captures = [] 
        for i in range(seq_len):
            print(".", end="")
            for j in range(self.topk):
                token_id = topk_indices[0, i, j].item()
                token_str = self.tokenizer.decode([token_id], skip_special_tokens=False)
                proba_value = topk_values[0, i, j].item()
                captures.append(LogitLensCapture(
                    layer_name=layer_name,
                    position_idx=i,
                    token_rank=j,
                    token_id=token_id,
                    token_str=token_str,
                    proba_value=proba_value
                ))
        return captures

    def build_hook(self, layer_name):
        """ Closure to integrate layer names in hooks """
        def hook(module, inputs, outputs):
            if not self.do_capture: return
            captures = self(outputs, layer_name=layer_name)
            self.captures.extend(captures)
        return hook

    def show(self, captures=None, text=None):
        """ Visualize a list of logit lens captures as a layer x position grid """
        if captures is None:
            captures = self.captures
        if not captures:
            print("No captures to show.")
            return

        # HRM applies the same layer repeatedly (recurrent reasoning cycles,
        # plus successive generation steps), so a layer_name is NOT a unique
        # row -- each hook invocation ("pass") is its own row. We recover
        # pass boundaries from capture order: within one hook call, i loops
        # 0..seq_len-1, so position_idx strictly increases; a new pass starts
        # whenever layer_name changes or position_idx fails to increase.
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
        proba_matrix = np.full((n_layers, n_positions), np.nan)
        token_matrix = [["" for _ in range(n_positions)] for _ in range(n_layers)]

        row_labels = []
        layer_pass_count = {}
        for i, (layer_name, pos_map) in enumerate(passes):
            layer_pass_count[layer_name] = layer_pass_count.get(layer_name, 0) + 1
            row_labels.append(f"{layer_name} #{layer_pass_count[layer_name]}")
            for j, pos in enumerate(positions):
                c = pos_map.get(pos)
                if c is not None:
                    proba_matrix[i, j] = c.proba_value
                    token_matrix[i][j] = c.token_str

        fig, ax = plt.subplots(figsize=(max(6, n_positions * 1.5), max(4, n_layers * 0.5)))
        # origin="lower": row 0 (earliest pass) at the bottom, later passes
        # stack upward -- matches processing order bottom-to-top.
        im = ax.imshow(proba_matrix, cmap="viridis", vmin=0, vmax=1, aspect="auto", origin="lower")

        for i in range(n_layers):
            for j in range(n_positions):
                token_str = token_matrix[i][j]
                if not token_str:
                    continue
                proba = proba_matrix[i, j]
                text_color = "black" if proba > 0.5 else "white"
                ax.text(j, i, token_str, ha="center", va="center", color=text_color, fontsize=8)

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

        fig.colorbar(im, ax=ax, label="Probability")
        fig.tight_layout()
        # Leave extra room on the left for the layer-name legend.
        fig.subplots_adjust(left=0.25)
        plt.show()



if __name__ == "__main__":
    model_id = "sapientinc/HRM-Text-1B"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
    ).eval()

    print(model)
    logit_lens = LogitLens(tokenizer, model.lm_head, topk=1)
    for i, transformer_block in enumerate(model.model.L_module.layers):
        if i == 15:
            layer_name = f"L_module_layers[{i}]"
            transformer_block.register_forward_hook(logit_lens.build_hook(layer_name=layer_name))
    #model.model.L_module.final_norm.register_forward_hook(logit_lens.build_hook(layer_name="L_module_final_norm"))

    for i, transformer_block in enumerate(model.model.H_module.layers):
        if i == 15:
            layer_name = f"H_module_layers[{i}]"
            transformer_block.register_forward_hook(logit_lens.build_hook(layer_name=layer_name))
    #model.model.H_module.final_norm.register_forward_hook(logit_lens.build_hook(layer_name="H_module_final_norm"))

    # synth,cot composite — reasoning / CoT style (see Disclaimer for other modes)
    condition = "<|quad_end|><|object_ref_end|>"
    prompt = f"<|im_start|>{condition}Explain why the sky is blue.<|im_end|>"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    # Mark the prompt as a single bidirectional prefix block — see "PrefixLM mask" below.
    inputs["token_type_ids"] = torch.ones_like(inputs["input_ids"])

    print("Step 1: Generate text")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=16, do_sample=False)
    decoded = tokenizer.decode(out[0], skip_special_tokens=False)
    print("Generated Text:", decoded)

    # Step 1.5 Re-encode
    prompt = decoded
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    inputs["token_type_ids"] = torch.ones_like(inputs["input_ids"])

    print("Step 2: Capturing logit lens outputs for the prompt + generated text")
    print("Prompt+Generated Text:", prompt)
    with torch.no_grad():
        with logit_lens as captures:
            out = model.generate(**inputs, max_new_tokens=1, do_sample=False, use_cache=False)

    print("#Captures:", len(captures))
    logit_lens.show(captures, text=prompt)
