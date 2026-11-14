"""Benchmark HF generation speed for the OPSD student (plain vs PEFT, various batch sizes)."""

import time

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = r"C:\Users\Khali\projects\OPSD\models\Qwen3-1.7B"
PROMPT = "Problem: What is 17 * 23?\n\nPlease reason step by step, and put your final answer within \\boxed{}."


def bench(model, tok, batch, new_tokens=64, label=""):
    tok.padding_side = "left"
    enc = tok([PROMPT] * batch, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=new_tokens,
            do_sample=True,
            temperature=1.1,
            top_p=0.95,
            top_k=20,
            pad_token_id=tok.pad_token_id,
        )
    torch.cuda.synchronize()
    dt = time.time() - t0
    n = (out.shape[1] - enc["input_ids"].shape[1]) * batch
    print(f"{label:38s} batch={batch}  {dt:6.2f}s  {n/dt:7.1f} tok/s  (incl. prompt)")
    return n / dt


def main():
    tok = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print(torch.cuda.get_device_name(0))
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, attn_implementation="sdpa", device_map="cuda"
    )
    model.eval()
    print(f"load: {time.time() - t0:.1f}s   dtype={model.dtype}   use_cache={model.config.use_cache}")

    for b in (1, 2, 4):
        bench(model, tok, b, 64, "plain bf16")

    lcfg = LoraConfig(
        r=64,
        lora_alpha=128,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    pm = get_peft_model(model, lcfg)
    pm.eval()
    print(f"peft trainable params: {sum(p.numel() for p in pm.parameters() if p.requires_grad):,}")
    for b in (1, 2, 4):
        bench(pm, tok, b, 64, "peft r=64")

    # does eager attention help/hurt?
    model.set_attn_implementation("eager")
    for b in (1, 2):
        bench(model, tok, b, 64, "plain bf16 eager")
    model.set_attn_implementation("sdpa")

    print("\npeak VRAM: %.2f GB" % (torch.cuda.max_memory_allocated() / 1e9))


if __name__ == "__main__":
    main()
