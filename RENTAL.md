# Renting a GPU for the OPSD run — step by step

You need three things before you press "rent":

1. an **HF account + write token** (the artifacts are published there),
2. a **rental** of an RTX 5090 with ≥ 60 GB disk,
3. **this repo** on the box (via git or scp).

The whole thing is then **one command**:

```bash
HF_TOKEN=hf_xxx HF_REPO_ID=your-user/opsd-qwen3-1.7b bash scripts/bootstrap_rental.sh
```

It installs the environment, downloads the model + dataset, trains 50 steps,
merges/packs the checkpoints and uploads everything to the Hub.

---

## Step 1 — Hugging Face token (5 min, do this first)

1. Create an account: <https://huggingface.co/join>
2. Create a token with **write** permission: <https://huggingface.co/settings/tokens>
   (type *Write* → *Fine-grained* or *Read & Write* is enough).
3. Copy it (`hf_...`). Keep it secret-ish: it is only used to push to your repo.

## Step 2 — rent the GPU

Whatever platform you use (the "credits" ones are all the same shape), pick:

| setting | value | why |
| --- | --- | --- |
| GPU | **RTX 5090 (32 GB)** | sm_120, ~104 TFLOPS bf16 |
| image | Ubuntu 22.04 / 24.04 | python 3.10+ and CUDA drivers included |
| disk | **≥ 64 GB** | model 3.4 GB + venv ~20 GB + dataset + checkpoints |
| network | any decent one | ~5 GB of downloads |
| extras | none | |

The platform gives you either an **SSH command** (`ssh root@<ip> -p <port>`) or a
**web terminal**. Both work.

> **Tip:** if the platform supports disk snapshots, take one after the setup finishes —
> otherwise a second run re-downloads ~6 GB.

## Step 3 — get the code onto the box

**Option A (recommended): GitHub.** From your Windows machine:

```powershell
cd C:\Users\Khali\projects\OPSD\OPSD_repo
git remote add origin https://github.com/<you>/OPSD.git   # create the repo on github.com first
git push -u origin master
```

then on the box:

```bash
git clone https://github.com/<you>/OPSD.git && cd OPSD
```

**Option B: copy the folder over** (from your Windows PowerShell):

```powershell
scp -r -P <port> C:\Users\Khali\projects\OPSD\OPSD_repo root@<ip>:/root/
```

then on the box: `cd /root/OPSD_repo`.

## Step 4 — run it (inside tmux, so a closed tab does not kill the job)

```bash
apt-get install -y tmux          # if missing
tmux new -s opsd
```

inside tmux:

```bash
export HF_TOKEN=<your-token>
export HF_REPO_ID=your-user/opsd-qwen3-1.7b
bash scripts/bootstrap_rental.sh 2>&1 | tee -r bootstrap.out
```

Detach with `Ctrl-b d` (the job keeps running), re-attach with `tmux attach -t opsd`.

> Do **not** put the token in a file that you commit. `export` in the shell is fine;
> `hf auth login` (asks for the token interactively) is even better.

### Recommended: dry run first

```bash
SMOKE=1 HF_TOKEN=... HF_REPO_ID=... bash scripts/bootstrap_rental.sh
```

`SMOKE=1` does a **1-step** version of the whole pipeline (install → train → merge →
upload) in ~10-15 min, and uploads a tiny smoke repo. If that succeeds, run the real
thing without `SMOKE=1`. This is worth it: it turns a 3-hour "it died at minute 40"
into a 10-minute check.

## Step 5 — watch it

In a second SSH session:

```bash
tail -f ~/runs/logs/bootstrap.log     # the pipeline log
nvidia-smi                            # GPU usage
ls ~/runs/qwen31b_5090_gen1024_fixteacher/   # checkpoints appear every 10 steps
```

Timeline (5090, vLLM):

| | |
| --- | --- |
| install + download | 10–20 min |
| training 50 steps | ~45–60 min |
| merge + pack + upload | ~10 min |
| **total** | **~1 h 15 min** (≈ $1–2) |

Add `RUN_EVAL=1` for the AIME24/25 + HMMT25 evaluation (+~2 h → +$2–3).

## Step 6 — what lands in the HF repo

```
https://huggingface.co/your-user/opsd-qwen3-1.7b
├── qwen31b_5090_gen1024_fixteacher_package/
│   ├── adapter_config.json, adapter_model.safetensors   (final adapter)
│   ├── checkpoint-10/ … checkpoint-50/                   (LoRA adapters)
│   ├── trainer_state.json
│   └── opsd_run_config.json                              (exact hyper-parameters)
└── merged-qwen31b_5090_gen1024_fixteacher/               (full merged model, 3.4 GB)
```

Evaluate it anywhere (needs vLLM + the base model for the adapter, or just the merged
model):

```bash
# A) use the adapter directly (upstream eval script)
python eval/evaluate_math.py --base_model ~/models/Qwen3-1.7B \
    --checkpoint_dir <hf-repo>/qwen31b_5090_gen1024_fixteacher_package/checkpoint-50 \
    --dataset aime24 --val_n 12 --temperature 1.0 --tensor_parallel_size 1 --enable_thinking

# B) use the merged model
python eval/evaluate_math.py --base_model <hf-repo>/merged-qwen31b_5090_gen1024_fixteacher \
    --dataset aime24 --val_n 12 --temperature 1.0 --tensor_parallel_size 1 --enable_thinking
```

## Step 7 — stop paying

Shut the instance down in the platform UI as soon as the upload finishes (billing is
per second/hour of *running* instance). Download nothing — it is all on the Hub.

---

## Troubleshooting

| symptom | fix |
| --- | --- |
| `torch.version.cuda != "12.8"` | the bootstrap aborts; re-run the torch step: `pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128 --extra-index-url https://pypi.org/simple` |
| vLLM: "external_launcher ... torch.distributed not initialized" | `VLLM_BACKEND=mp bash scripts/bootstrap_rental.sh` (or use the `torchrun` path, which is the default) |
| CUDA out of memory | lower `VLLM_MEM_UTIL=0.35`, or `BATCH_SIZE=2 GRAD_ACCUM=16` |
| "no kernel image is available" | flash-attn on sm_120: the trainer auto-falls back to SDPA (it prints the fallback message) |
| training died mid-way | re-run the same command — it resumes from the newest `checkpoint-*` automatically |
| upload fails | check the token has *write* permission and that `HF_REPO_ID` is `user/repo` form |
| the shell dies / SSH drops | you forgot tmux: `tmux new -s opsd` before starting |

## Cost control

* Stop the instance right after the upload (the run is ~1 h).
* If you are unsure, run with `SMOKE=1` first (10–15 min) to validate the pipeline.
* `KEEP_VENV=0` (default) deletes the ~20 GB venv at the end; keep it with `KEEP_VENV=1`
  if you plan to re-run.
* A snapshot of the disk (if the platform has them) lets you re-run without re-downloading.
