from transformers import AutoModelForCausalLM, AutoTokenizer
import torch



from dataclasses import dataclass

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

    def show(self, captures=None):
        """ Visualize a list of logit lens captures """
        if captures is None:
            captures = self.captures
        # Sort by position 
        captures.sort(key=lambda x: (x.position_idx))
        for capture in captures:
            print(capture)



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
        layer_name = f"L_module_layers[{i}]"
        transformer_block.register_forward_hook(logit_lens.build_hook(layer_name=layer_name))

    for i, transformer_block in enumerate(model.model.H_module.layers):
        layer_name = f"H_module_layers[{i}]"
        transformer_block.register_forward_hook(logit_lens.build_hook(layer_name=layer_name))

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
            out = model.generate(**inputs, max_new_tokens=16, do_sample=False)
    filtered = [c for c in captures if "15" in c.layer_name]
    logit_lens.show(filtered)
