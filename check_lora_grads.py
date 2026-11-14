"""Check that gradient checkpointing + LoRA gives grads to *all* LoRA layers (incl. layer 0)."""

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

MODEL = r"C:\Users\Khali\projects\OPSD\models\Qwen3-1.7B"

model = AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.bfloat16, attn_implementation="sdpa", device_map="cuda"
)
lcfg = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lcfg)
model.print_trainable_parameters()

# what the HF Trainer does when --gradient_checkpointing is set
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": True})
# ... and what TRL's SFTTrainer does for PEFT models (sft_trainer.py:707)
model.enable_input_require_grads()
model.train()

ids = torch.randint(0, 1000, (2, 64), device="cuda")
out = model(input_ids=ids, labels=ids)
out.loss.backward()

missing = []
for name, p in model.named_parameters():
    if p.requires_grad and (p.grad is None or p.grad.abs().sum().item() == 0):
        missing.append(name)

print(f"\nLoRA params with NO gradient: {len(missing)}")
for n in missing[:10]:
    print("   ", n)
if not missing:
    print("   -> all LoRA params receive gradients (layer 0 included)")
