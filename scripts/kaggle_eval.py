"""Kaggle (2x T4) evaluation of the OPSD Qwen3-1.7B run.

Self-contained: installs vLLM, pulls the base model + our adapters from the Hub,
evaluates base vs. every checkpoint on AIME24 / AIME25 / HMMT25 (Avg@12, thinking
mode, the paper's protocol) and uploads all results back to the Hub repo.

Run it as a notebook cell or as a script:

    python kaggle_eval.py
"""

import glob
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

# ----------------------------------------------------------------- config ---
# The HF token is NOT stored in this file (GitHub secret scanning rejects that).
# Pass it via the environment, e.g. in the notebook cell:
#     import os; os.environ["HF_TOKEN"] = "hf_..."
TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN") or ""
HF_REPO = "0x-Khalil/opsd-qwen3-1.7b"                   # where the weights live
GITHUB = "https://github.com/0x-khalil/opsd-qwen3-1.7b"  # eval code
BASE_ID = "Qwen/Qwen3-1.7B"
PKG = "qwen31b_5090_gen1024_fixteacher_package"

WORK = Path("/kaggle/working/opsd")
REPO_DIR = WORK / "repo"
MODELS = WORK / "models"
RESULTS = WORK / "results"
LOGS = WORK / "logs"

DATASETS = ["aime24", "aime25", "hmmt25"]
CKPT_STEPS = [50, 40, 30, 20, 10]      # priority order
VAL_N = 12                              # paper: Avg@12
MAX_NEW_TOKENS = 16384                  # thinking mode (paper uses 38912)
DEADLINE = time.time() + 9 * 3600       # stop starting new jobs after this

os.environ.setdefault("HF_TOKEN", TOKEN)
# NOTE: do NOT set HF_HUB_ENABLE_HF_TRANSFER here -- with huggingface_hub 0.36 it
# makes the downloader require the `hf_transfer` package, which is not installed
# on Kaggle -> ValueError. The default (Xet) downloader is fast enough.
if not TOKEN:
    print("!! HF_TOKEN is not set -> the upload at the end will fail.")
    print("   set it in the notebook cell: os.environ['HF_TOKEN'] = 'hf_...'")
for p in (WORK, REPO_DIR, MODELS, RESULTS, LOGS):
    p.mkdir(parents=True, exist_ok=True)


def sh(cmd, **kw):
    print(f"\n$ {cmd}", flush=True)
    return subprocess.run(cmd, shell=True, check=False, **kw)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ------------------------------------------------------------ 1. packages ---
DRY_RUN = os.environ.get("DRY_RUN") == "1"
if not DRY_RUN:
    log("installing packages (vllm, transformers, math-verify, huggingface_hub)")
    # transformers MUST be pinned to 4.57.x: vllm 0.11 calls
    # `tokenizer.all_special_tokens_extended`, which transformers 5.x removed
    # (a fresh Kaggle image would otherwise pull transformers 5.x and every vLLM
    # job dies with "Qwen2Tokenizer has no attribute all_special_tokens_extended").
    sh(
        'pip install --no-cache-dir -q "vllm==0.11.0" "transformers==4.57.1" '
        '"tokenizers>=0.22,<0.23" "huggingface_hub<1.0" "math-verify==0.8.0" 2>&1 | tail -8'
    )
    sh("python -c \"import transformers,vllm,tokenizers;"
       "print('transformers',transformers.__version__,'vllm',vllm.__version__,"
       "'tokenizers',tokenizers.__version__)\"")
else:
    log("DRY RUN -> no installs, no downloads")

USE_VLLM = True
if not DRY_RUN:
    USE_VLLM = subprocess.run("python -c 'import vllm;print(vllm.__version__)'",
                               shell=True, capture_output=True).returncode == 0
log(f"vLLM available: {USE_VLLM}")
if not USE_VLLM:
    log("!! vLLM unavailable -> falling back to HF transformers (much slower, reduced scope)")

# ---------------------------------------------------------------- 2. code ---
if not DRY_RUN and not (REPO_DIR / "eval" / "evaluate_math.py").exists():
    sh(f"git clone --depth 1 {GITHUB} {REPO_DIR}")

# -------------------------------------------------------------- 3. models ---
if not DRY_RUN:
    from huggingface_hub import snapshot_download  # noqa: E402

    log("downloading the base model ...")
    snapshot_download(BASE_ID, local_dir=str(MODELS / "base"),
                      allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja"],
                      max_workers=8)
    log("downloading the OPSD adapters ...")
    snapshot_download(HF_REPO, local_dir=str(MODELS / "pkg"),
                      allow_patterns=[f"{PKG}/*"], max_workers=8)

ADAPTERS = {s: MODELS / "pkg" / PKG / f"checkpoint-{s}" for s in CKPT_STEPS}
for s, p in ADAPTERS.items():
    if not (p / "adapter_model.safetensors").exists():
        log(f"WARNING: checkpoint-{s} not found ({p})")

BASE = MODELS / "base"

# ------------------------------------------------------------ 3.5 preflight --
# Build the tokenizer the way vLLM does -> catches version mismatches in
# seconds instead of failing 8 jobs in a row.
if USE_VLLM and not DRY_RUN:
    try:
        pf = subprocess.run(
            [sys.executable, "-c",
             "from vllm.transformers_utils.tokenizer import get_tokenizer;"
             "t=get_tokenizer(%r, trust_remote_code=True);"
             "print('tokenizer ok:', type(t).__name__)" % str(BASE)],
            capture_output=True, text=True)
        log("preflight: " + (pf.stdout.strip() or pf.stderr.strip()[-400:]))
        if pf.returncode != 0:
            log("!! the vLLM tokenizer preflight failed -> the jobs would all crash.")
            log("   falling back to the HF transformers path")
            USE_VLLM = False
    except Exception:
        log("preflight raised:\n" + traceback.format_exc())
        log("   falling back to the HF transformers path")
        USE_VLLM = False

# ------------------------------------------------------------- 4. the jobs ---
def vllm_cmd(dataset, adapter, out_json):
    cmd = [
        sys.executable, "eval/evaluate_math.py",
        "--base_model", str(BASE),
        "--dataset", dataset,
        "--val_n", str(VAL_N),
        "--temperature", "1.0",
        "--top_p", "1.0",
        "--top_k", "-1",
        "--min_p", "0.0",
        "--presence_penalty", "0.0",
        "--max_new_tokens", str(MAX_NEW_TOKENS),
        "--tensor_parallel_size", "1",
        "--gpu_memory_utilization", "0.80",
        "--enable_thinking",
        "--output_file", str(out_json),
    ]
    if adapter:
        cmd += ["--checkpoint_dir", str(adapter)]
    return cmd


def hf_cmd(dataset, adapter, out_json):
    cmd = [
        sys.executable, "eval/evaluate_math_hf.py",
        "--base_model_path", str(BASE),
        "--dataset", dataset,
        "--num_samples", "2",
        "--limit", "15",
        "--max_new_tokens", "6144",
        "--temperature", "1.0",
        "--batch_size", "8",
        "--enable_thinking",
        "--output_file", str(out_json),
    ]
    if adapter:
        cmd += ["--adapter_path", str(adapter)]
    return cmd


def job_name(dataset, step):
    return f"{dataset}_{'base' if step is None else 'step' + str(step)}"


def build_jobs():
    """(gpu, [(dataset, step or None), ...]) -> one list of jobs per GPU."""
    if USE_VLLM:
        # GPU 0: the base model on every dataset   |   GPU 1: step-50 on every dataset
        g0 = [(d, None) for d in DATASETS]
        g1 = [(d, 50) for d in DATASETS]
        # then the remaining checkpoints on AIME24 (split over both GPUs)
        g0 += [("aime24", s) for s in (10, 30)]
        g1 += [("aime24", s) for s in (20, 40)]
        return [(0, g0), (1, g1)]
    # fallback: only aime24, base + step 50, one per GPU
    return [(0, [("aime24", None)]), (1, [("aime24", 50)])]


def worker(gpu, jobs):
    for dataset, step in jobs:
        if time.time() > DEADLINE:
            log(f"deadline reached, skipping {job_name(dataset, step)}")
            continue
        adapter = ADAPTERS.get(step) if step else None
        if step and (not adapter or not (adapter / "adapter_model.safetensors").exists()):
            log(f"skip {job_name(dataset, step)}: adapter missing")
            continue
        out_json = RESULTS / f"{job_name(dataset, step)}.json"
        cmd = (vllm_cmd if USE_VLLM else hf_cmd)(dataset, adapter, out_json)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env["TOKENIZERS_PARALLELISM"] = "false"
        logf = LOGS / f"{job_name(dataset, step)}.log"
        log(f"GPU{gpu} -> {job_name(dataset, step)}  (log: {logf})")
        with open(logf, "w", encoding="utf-8") as f:
            rc = subprocess.Popen(cmd, cwd=str(REPO_DIR), env=env, stdout=f,
                                  stderr=subprocess.STDOUT).wait()
        log(f"GPU{gpu} <- {job_name(dataset, step)} finished rc={rc}")
        if rc != 0:
            log(f"   !! failed, last lines:\n" + "\n".join(
                open(logf, encoding="utf-8", errors="replace").readlines()[-15:]))


log("starting the evaluation workers")
import threading  # noqa: E402

if DRY_RUN:
    for gpu, jobs in build_jobs():
        for dataset, step in jobs:
            adapter = ADAPTERS.get(step) if step else None
            out_json = RESULTS / f"{job_name(dataset, step)}.json"
            cmd = (vllm_cmd if USE_VLLM else hf_cmd)(dataset, adapter, out_json)
            print(f"GPU{gpu}: " + " ".join(cmd))
    log("DRY RUN -> nothing was executed")
    raise SystemExit(0)

threads = []
try:
    for gpu, jobs in build_jobs():
        t = threading.Thread(target=worker, args=(gpu, jobs), daemon=False)
        t.start()
        threads.append(t)
    for t in threads:
        t.join()
except Exception:
    log("the evaluation phase raised:\n" + traceback.format_exc())
log("all evaluation jobs finished")

# --------------------------------------------------------- 5. summarise ---
summary = []
for f in sorted(glob.glob(str(RESULTS / "*.json"))):
    try:
        d = json.load(open(f, encoding="utf-8"))
    except Exception as e:
        log(f"cannot read {f}: {e}")
        continue
    summary.append({
        "file": os.path.basename(f),
        "dataset": d.get("dataset"),
        "model": Path(str(d.get("base_model", ""))).name,
        "adapter": "yes" if d.get("lora") or "checkpoint" in str(f) else "no",
        "val_n": d.get("val_n"),
        "avg_at_n": d.get("average_at_n_pct"),
        "pass_at_n": d.get("pass_at_n_pct"),
        "majority_at_n": d.get("majority_vote_at_n_pct"),
        "format_rate": d.get("format_rate"),
    })

if summary:
    lines = ["| eval | Avg@%s | Pass@%s | Maj@%s | format |" % (VAL_N, VAL_N, VAL_N),
             "|---|---|---|---|---|"]
    for s in sorted(summary, key=lambda x: (str(x["dataset"]), str(x["file"]))):
        lines.append("| {f} | {a} | {p} | {m} | {fr} |".format(
            f=s["file"].replace(".json", ""),
            a=("%.2f%%" % s["avg_at_n"]) if s["avg_at_n"] is not None else "-",
            p=("%.2f%%" % s["pass_at_n"]) if s["pass_at_n"] is not None else "-",
            m=("%.2f%%" % s["majority_at_n"]) if s["majority_at_n"] is not None else "-",
            fr=("%.1f%%" % s["format_rate"]) if s["format_rate"] is not None else "-"))
    (RESULTS / "summary.md").write_text(
        "# OPSD Qwen3-1.7B evaluation (Kaggle, 2x T4)\n\n"
        f"Protocol: thinking mode, temperature 1.0, top_p 1.0, top_k off, min_p 0, "
        f"Avg@{VAL_N}, max_new_tokens {MAX_NEW_TOKENS}, graded with math-verify.\n\n"
        + "\n".join(lines) + "\n", encoding="utf-8")
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n=== SUMMARY ===")
    print("\n".join(lines))
else:
    log("no result files were produced - check the logs in " + str(LOGS))

# ------------------------------------------------------------ 6. upload ---
try:
    from huggingface_hub import HfApi

    api = HfApi(token=TOKEN)
    api.upload_folder(folder_path=str(RESULTS), repo_id=HF_REPO, repo_type="model",
                      path_in_repo="eval_results",
                      commit_message="OPSD Qwen3-1.7B eval results (Kaggle 2x T4)")
    if any(Path(LOGS).glob("*.log")):
        api.upload_folder(folder_path=str(LOGS), repo_id=HF_REPO, repo_type="model",
                          path_in_repo="eval_results/logs",
                          commit_message="OPSD Qwen3-1.7B eval logs (Kaggle 2x T4)")
    log(f"uploaded results -> https://huggingface.co/{HF_REPO}/tree/main/eval_results")
except Exception as e:
    log(f"UPLOAD FAILED: {e}")
    log(f"copy the results manually from {RESULTS}")
