from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_id = "sapientinc/HRM-Text-1B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype=torch.bfloat16,
).eval()

# synth,cot composite — reasoning / CoT style (see Disclaimer for other modes)
condition = "<|quad_end|><|object_ref_end|>"
prompt = f"<|im_start|>{condition}Explain why the sky is blue.<|im_end|>"

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
# Mark the prompt as a single bidirectional prefix block — see "PrefixLM mask" below.
inputs["token_type_ids"] = torch.ones_like(inputs["input_ids"])

print(model)

class LogitLens:
    def __init__(self, tokenizer, lm_head, topk=1):
        self.tokenizer = tokenizer
        self.lm_head = lm_head
        self.topk = topk

    def decode_topk(self, logits):
        bsz, seq_len, vocab_size = logits.shape
        assert bsz == 1, "Batch size > 1 not supported"
        probas = torch.softmax(logits, dim=-1)
        topk_values, topk_indices = torch.topk(probas, k=self.topk, dim=-1)
        for i in range(seq_len):
            print(f"Step {i}:")
            for j in range(self.topk):
                token_id = topk_indices[0, i, j].item()
                token_str = self.tokenizer.decode([token_id], skip_special_tokens=False)
                logit_value = topk_values[0, i, j].item()
                print(f"  Top-{j+1}: Token ID {token_id}, Token '{token_str}', Proba {logit_value:.4f}")

    def hook(self, module, inputs, outputs):
        with torch.no_grad():
            logits = self.lm_head(outputs)
            print(f"[{module}/top{self.topk}]")
            self.decode_topk(logits)

logit_lens = LogitLens(tokenizer, model.lm_head, topk=1)

for transformer_block in model.model.L_module.layers:
    transformer_block.register_forward_hook(logit_lens.hook)

for transformer_block in model.model.H_module.layers:
    transformer_block.register_forward_hook(logit_lens.hook)

with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=256, do_sample=False)
print(tokenizer.decode(out[0], skip_special_tokens=False))

