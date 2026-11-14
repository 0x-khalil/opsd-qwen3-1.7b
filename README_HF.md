---
license: apache-2.0
library_name: peft
base_model: Qwen/Qwen3-1.7B
tags:
  - math
  - reasoning
  - self-distillation
  - opsd
  - qwen3
---

# OPSD on Qwen3-1.7B — on-policy self-distillation (50 steps)

Reproduction of **"Self-Distilled Reasoner: On-Policy Self-Distillation for Large Language Models"**
(Zhao et al., [arXiv:2601.18734](https://arxiv.org/abs/2601.18734)) on **Qwen3-1.7B**.

One model acts as both student and teacher: the student only sees the problem, the teacher
additionally sees the ground-truth reference solution, and the two token distributions are
matched with a **Jensen-Shannon divergence** along the student's own on-policy rollouts.
The teacher is frozen at step 0 (`--fixed_teacher`, implemented by disabling the LoRA adapters).

## Training config (matches the paper's `scripts/run_opsd_1b.sh`)

| | |
|---|---|
| base model | `Qwen/Qwen3-1.7B` |
| data | `siyanzhao/Openthoughts_math_30k_opsd` (29 434 problems + reference solutions) |
| objective | generalized JSD, full vocabulary, `beta=0` (forward KL), per-element clip `0.05` |
| PEFT | LoRA r=64, alpha=128 on `q_proj k_proj v_proj o_proj gate_proj up_proj down_proj` |
| rollouts | ≤ 1024 tokens, `temperature 1.1`, `top_p 0.95`, `top_k 20` |
| teacher | step-0 policy (LoRA disabled), thinking mode; student: non-thinking |
| optimizer | lr 5e-6 (linear), `max_grad_norm 0.1`, **effective batch 32** (4 × grad-accum 8) |
| steps | 50, checkpoint every 10 |
| hardware | 1× RTX 5090 32 GB, vLLM colocate, SDPA, bf16 |

## Results of the run

| | |
|---|---|
| training time | **39 min** (2 344 s, 50 steps, 46.7 s/step) |
| vLLM rollout speed | 600–850 tok/s (4 seqs × ~800 tok in ~4.7 s) |
| peak VRAM | 21.9 GB reserved (vLLM + training model colocated) |
| loss (mean, steps 1–5) | **+0.0086** |
| loss (mean, steps 46–50) | **−0.0030** |
| grad norm | 0.18 → 0.05 (stable) |

Loss curve (one value per optimizer step, from `bootstrap.log`):

```
+0.0098 +0.0077 +0.0077 +0.0082 +0.0096 +0.0063 +0.0054 +0.0081 +0.0035 +0.0053
+0.0031 +0.0025 +0.0030 +0.0020 +0.0019 -0.0005 +0.0013 +0.0005 +0.0007 -0.0016
+0.0009 +0.0000 -0.0012 -0.0004 -0.0008 +0.0001 -0.0012 -0.0012 -0.0017 -0.0024
-0.0021 -0.0010 -0.0020 -0.0021 -0.0025 -0.0013 -0.0013 -0.0017 -0.0024 -0.0024
-0.0022 -0.0023 -0.0027 -0.0041 -0.0026 -0.0028 -0.0031 -0.0032 -0.0029 -0.0031
```

The loss becomes slightly negative because the per-element JSD clipping caps the positive
summands (this is expected and documented upstream); the magnitude is what matters.

## Repository layout

```
qwen31b_5090_gen1024_fixteacher_package/     LoRA adapters + metadata
├── adapter_config.json, adapter_model.safetensors    final adapter (step 50)
├── checkpoint-10/ … checkpoint-50/                    adapter every 10 steps
├── opsd_run_config.json                              exact hyper-parameters used
└── bootstrap.log                                     full training log
merged-qwen31b_5090_gen1024_fixteacher/     full merged model (Qwen3-1.7B + LoRA)
├── config.json, generation_config.json, model.safetensors, tokenizer files
└── adapter/                                       copy of the adapter (provenance)
```

## How to use

**LoRA adapter (recommended)**

```python
from transformers import AutoModelForCausalLM
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B", torch_dtype="bfloat16")
model = PeftModel.from_pretrained(base, "0x-Khalil/opsd-qwen3-1.7b",
        subfolder="qwen31b_5090_gen1024_fixteacher_package/checkpoint-50")
```

**Merged model** — a plain HF checkpoint, usable directly:

```python
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    "0x-Khalil/opsd-qwen3-1.7b", subfolder="merged-qwen31b_5090_gen1024_fixteacher",
    torch_dtype="bfloat16")
```

**Evaluation** (same protocol as the paper: vLLM, thinking mode, Avg@12, `math-verify`):

```bash
git clone https://github.com/0x-khalil/opsd-qwen3-1.7b
cd opsd-qwen3-1.7b/eval
python evaluate_math.py \
    --base_model <path-to-merged-model> \
    --dataset aime24 --val_n 12 --temperature 1.0 --tensor_parallel_size 1 --enable_thinking
```

## Reproduction notes

This run used a port of the upstream code to a single 32 GB GPU
(<https://github.com/0x-khalil/opsd-qwen3-1.7b>): the full-vocabulary JSD is computed in
token chunks with gradient checkpointing (mathematically identical to upstream, see
`test_loss_equivalence.py`) so that it fits next to a colocated vLLM engine. Everything
else — objective, sampling, PEFT setup, teacher construction — is unchanged.

## Citation

```bibtex
@article{zhao2026self,
  title={Self-Distilled Reasoner: On-Policy Self-Distillation for Large Language Models},
  author={Zhao, Siyan and Xie, Zhihui and Liu, Mengchen and Huang, Jing and Pang, Guan and Chen, Feiyu and Grover, Aditya},
  journal={arXiv preprint arXiv:2601.18734},
  year={2026}
}
```
