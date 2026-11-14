"""Does a static KV cache speed up HF decoding enough to matter for the 50-step run?"""

import time

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = r"C:\Users\Khali\projects\OPSD\models\Qwen3-1.7B"

tok = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.bfloat16, attn_implementation="sdpa", device_map="cuda"
)
lcfg = LoraConfig(
    r=64, lora_alpha=128,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lcfg)
model.eval()

prompt = tok.apply_chat_template(
    [{"role": "user", "content": "Problem: What is 17 * 23?\n\nPlease reason step by step, and put your final answer within \\boxed{}."}],
    tokenize=False, add_generation_prompt=True, enable_thinking=False,
)
enc = tok([prompt] * 4, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")


def bench(label, **extra):
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(
            **enc, max_new_tokens=1024, do_sample=True, temperature=1.1, top_p=0.95, top_k=20,
            pad_token_id=tok.pad_token_id, **extra,
        )
    torch.cuda.synchronize()
    dt = time.time() - t0
    n = (out.shape[1] - enc["input_ids"].shape[1]) * 4
    print(f"{label:34s} {dt:6.1f}s  {n/dt:6.1f} tok/s   peak {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
    torch.cuda.reset_peak_memory_stats()


bench("dynamic cache (default)")
bench("static cache", cache_implementation="static")
bench("dynamic cache (default) again")
bench("static cache again", cache_implementation="static")
