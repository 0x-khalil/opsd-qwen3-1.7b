r"""Package the OPSD training artifacts for transfer to an evaluation machine.

Copies the final adapter + every periodic checkpoint into one folder, writes a
`opsd_run_config.json` with the exact hyper-parameters used, and optionally zips it.

Usage:
    python scripts/package_artifacts.py --run_dir runs/qwen31b_5090_gen1024_fixteacher \
        --base_model models/Qwen3-1.7B --zip
"""

import argparse
import json
import shutil
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, help="runs\\<run_config> directory")
    ap.add_argument("--base_model", required=True, help="path of the base model used for training")
    ap.add_argument("--out", default=None, help="output folder (default: <run_dir>_package)")
    ap.add_argument("--zip", action="store_true")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    out = Path(args.out) if args.out else run_dir.parent / f"{run_dir.name}_package"
    out.mkdir(parents=True, exist_ok=True)

    copied = []
    for ckpt in sorted(run_dir.glob("checkpoint-*")):
        target = out / ckpt.name
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(ckpt, target)
        copied.append(ckpt.name)

    final_files = ["adapter_config.json", "adapter_model.safetensors"]
    for f in final_files:
        src = run_dir / f
        if src.exists():
            shutil.copy2(src, out / f)
            copied.append(f)

    state = run_dir / "trainer_state.json"
    if state.exists():
        shutil.copy2(state, out / "trainer_state.json")

    # the Trainer dumps the resolved arguments -> use them instead of guessing
    resolved = {}
    args_bin = run_dir / "training_args.bin"
    if args_bin.exists():
        try:
            import torch

            ta = torch.load(args_bin, map_location="cpu", weights_only=False)
            for k in (
                "learning_rate",
                "max_grad_norm",
                "per_device_train_batch_size",
                "gradient_accumulation_steps",
                "max_completion_length",
                "max_length",
                "temperature",
                "top_p",
                "top_k",
                "beta",
                "lmbda",
                "num_train_epochs",
                "max_steps",
                "seed",
            ):
                if hasattr(ta, k):
                    resolved[k] = getattr(ta, k)
        except Exception as e:  # pragma: no cover
            print(f"[package] could not read training_args.bin ({e})")

    cfg = {
        "base_model": str(Path(args.base_model).resolve()),
        "method": "OPSD (on-policy self-distillation), fixed teacher = step-0 policy via disabled LoRA",
        "model": "Qwen3-1.7B",
        "peft": {"type": "lora", "r": 64, "alpha": 128,
                 "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                                    "gate_proj", "up_proj", "down_proj"]},
        "objective": {"loss": "generalized JSD (full vocabulary)", "beta": 0,
                      "jsd_token_clip": 0.05},
        "sampling": {"temperature": 1.1, "top_p": 0.95, "top_k": 20},
        "optim": {"learning_rate": 5e-6, "max_grad_norm": 0.1, "effective_batch_size": 32},
        "teacher": "student_thinking=False, teacher_thinking=True",
        "resolved_training_args": resolved,
        "checkpoints": copied,
    }
    with open(out / "opsd_run_config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    print(f"[package] copied {copied} -> {out}")
    if args.zip:
        archive = shutil.make_archive(str(out), "zip", root_dir=out)
        print(f"[package] archive: {archive}")


if __name__ == "__main__":
    main()
