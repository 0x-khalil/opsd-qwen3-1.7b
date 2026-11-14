"""Debug: does the embedding output require grad under PEFT + gradient checkpointing?"""

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

MODEL = r"C:\Users\Khali\projects\OPSD\models\Qwen3-1.7B"

model = AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.bfloat16, attn_implementation="sdpa", device_map="cuda"
)
lcfg = LoraConfig(
    r=8, lora_alpha=16,
    target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM",
)
model = get_peft_model(model, lcfg)

emb = model.get_input_embeddings()
print("embedding forward hooks:", emb._forward_hooks)
print("has enable_input_require_grads:", hasattr(model, "enable_input_require_grads"))

model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": True})
model.train()

ids = torch.randint(0, 1000, (2, 64), device="cuda")

hidden = emb(ids)
print("embedding out requires_grad:", hidden.requires_grad)

# full forward through the backbone
core = model.get_base_model().model
hs = core(input_ids=ids, use_cache=False).last_hidden_state
print("backbone out requires_grad:", hs.requires_grad, "grad_fn:", hs.grad_fn)

out = model(input_ids=ids, labels=ids)
print("loss requires_grad:", out.loss.requires_grad, "grad_fn:", out.loss.grad_fn)
