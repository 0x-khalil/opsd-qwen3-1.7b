"""Check (a) how long the student's rollouts actually are, (b) throughput at larger batches."""

import time

import torch
from datasets import load_dataset
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
    r=64,
    lora_alpha=128,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lcfg)
model.eval()

ds = load_dataset("siyanzhao/Openthoughts_math_30k_opsd")["train"].select(range(16))

prompts = []
for r in ds:
    msgs = [{"role": "user", "content": f"Problem: {r['problem']}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}."}]
    prompts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False))


def run(batch, max_new_tokens=1024):
    enc = tok(prompts[:batch], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=1.1,
            top_p=0.95,
            top_k=20,
            pad_token_id=tok.pad_token_id,
        )
    torch.cuda.synchronize()
    dt = time.time() - t0
    plen = enc["input_ids"].shape[1]
    lengths = (out[:, plen:] != tok.pad_token_id).sum(1).tolist()
    eos = tok.eos_token_id
    hits = [(row == eos).any().item() for row in out[:, plen:]]
    print(f"batch={batch}: {dt:.1f}s  {sum(lengths)/dt:.1f} tok/s  gen_lens={lengths} eos_seen={hits}")
    return out


for b in (4, 8):
    out = run(b)
    if b == 4:
        plen = tok(prompts[:4], return_tensors="pt", padding=True, add_special_tokens=False)["input_ids"].shape[1]
        txt = tok.decode(out[0, plen:], skip_special_tokens=False)
        print("\n--- SAMPLE COMPLETION (first 1200 chars) ---\n", txt[:1200])
        print("--- END ---\n")

print("peak VRAM: %.2f GB" % (torch.cuda.max_memory_allocated() / 1e9))
