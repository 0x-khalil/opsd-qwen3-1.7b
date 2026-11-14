"""HF-transformers evaluation for the OPSD checkpoints (vLLM-free, works on Windows).

Mirrors `eval/evaluate_math.py` but uses `transformers.generate` instead of vLLM, so it
can run on a single 8 GB GPU. A LoRA adapter produced by the OPSD trainer can be
supplied with `--adapter_path`.

Example:
    python eval/evaluate_math_hf.py --base_model_path C:\\...\\models\\Qwen3-1.7B \
        --adapter_path C:\\...\\runs\\qwen31b_3070\\checkpoint-100 \
        --dataset aime24 --num_samples 4 --enable_thinking
"""

import argparse
import json
import re
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from evaluate_math import extract_boxed_answer, grade_answer

DATASETS = {
    "math500": ("HuggingFaceH4/MATH-500", "test"),
    "aime24": ("HuggingFaceH4/aime_2024", "train"),
    "aime25": ("yentinglin/aime_2025", "train"),
    "hmmt25": ("MathArena/hmmt_feb_2025", "train"),
    "amc23": ("math-ai/amc23", "test"),
}


def build_prompt(tokenizer, problem: str, enable_thinking: bool) -> str:
    user_message = (
        f"Problem: {problem}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}."
    )
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user_message}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model_path", required=True)
    ap.add_argument("--adapter_path", default=None)
    ap.add_argument("--dataset", default="aime24", choices=list(DATASETS))
    ap.add_argument("--num_samples", type=int, default=1, help="samples per problem (Avg@n)")
    ap.add_argument("--max_new_tokens", type=int, default=8192)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--top_k", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None, help="only evaluate the first N problems")
    ap.add_argument("--enable_thinking", action="store_true")
    ap.add_argument("--output_file", default=None)
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="cuda",
    )
    if args.adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter_path)
        model = model.merge_and_unload()
        print(f"[eval] merged LoRA adapter from {args.adapter_path}")
    model.eval()

    repo, split = DATASETS[args.dataset]
    dataset = load_dataset(repo, split=split)
    if args.limit:
        dataset = dataset.select(range(min(args.limit, len(dataset))))
    print(f"[eval] {args.dataset}: {len(dataset)} problems, {args.num_samples} samples each")

    results = []
    correct = 0
    total = 0
    for idx, row in enumerate(dataset):
        problem = row["problem"]
        answer = row.get("answer") or row.get("solution")
        prompt = build_prompt(tokenizer, problem, args.enable_thinking)
        enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to("cuda")

        for _ in range(args.num_samples):
            with torch.no_grad():
                out = model.generate(
                    **enc,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k if args.top_k > 0 else 0,
                    pad_token_id=tokenizer.pad_token_id,
                )
            gen = out[0, enc["input_ids"].shape[1]:]
            text = tokenizer.decode(gen, skip_special_tokens=True)
            pred = extract_boxed_answer(text)
            ok = grade_answer(pred, answer) if pred is not None else False
            correct += int(ok)
            total += 1
            results.append({"problem_idx": idx, "predicted": pred, "ground_truth": answer, "correct": ok})
        print(f"[eval] {idx + 1}/{len(dataset)} running acc = {correct / max(total, 1):.4f}", flush=True)

    acc = correct / max(total, 1)
    print(f"\n[eval] {args.dataset} Avg@{args.num_samples} = {acc * 100:.2f}%  ({correct}/{total})")
    if args.output_file:
        Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_file, "w", encoding="utf-8") as f:
            json.dump({"dataset": args.dataset, "accuracy": acc, "results": results}, f, indent=2)
        print(f"[eval] wrote {args.output_file}")


if __name__ == "__main__":
    main()
