r"""Merge an OPSD LoRA checkpoint into the base model so it can be evaluated anywhere.

The OPSD trainer only trains LoRA adapters, so a "checkpoint" is just an
`adapter_model.safetensors` + `adapter_config.json` pair. Some evaluation
harnesses (vLLM, OpenAI-compatible servers) want a plain HF model directory.

Usage (Linux):
    python scripts/merge_lora.py \
        --base_model ~/models/Qwen3-1.7B \
        --adapter    ~/runs/qwen31b_3070/checkpoint-50 \
        --output     ~/runs/merged-50

Usage (Windows): same, with backslash paths.
"""

import argparse
import shutil
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    args = ap.parse_args()

    adapter_dir = Path(args.adapter)
    out_dir = Path(args.output)
    if not (adapter_dir / "adapter_config.json").exists():
        raise SystemExit(f"{adapter_dir} does not look like a PEFT adapter directory")

    print(f"[merge] base   = {args.base_model}")
    print(f"[merge] adapter= {adapter_dir}")
    print(f"[merge] output = {out_dir}")

    dtype = getattr(torch, args.dtype)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=dtype, device_map="cpu", low_cpu_mem_usage=True
    )
    model = PeftModel.from_pretrained(model, str(adapter_dir))
    model = model.merge_and_unload()

    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir, safe_serialization=True)

    tok = AutoTokenizer.from_pretrained(args.base_model)
    tok.save_pretrained(out_dir)

    # keep the adapter next to the merged model so the provenance is obvious
    shutil.copytree(adapter_dir, out_dir / "adapter", dirs_exist_ok=True)

    size_gb = sum(f.stat().st_size for f in out_dir.rglob("*") if f.is_file()) / 1e9
    print(f"[merge] done -> {out_dir} ({size_gb:.2f} GB)")


if __name__ == "__main__":
    main()
