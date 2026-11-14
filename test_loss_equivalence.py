"""Numerical equivalence test: chunked JSD loss vs. the upstream reference implementation.

Run:  python test_loss_equivalence.py
"""

import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, r"C:\Users\Khali\projects\OPSD\OPSD_repo")

from opsd_trainer import OPSDTrainer  # noqa: E402


def reference_loss(head, s_hidden, t_hidden, mask, beta, temperature, token_clip, top_k):
    """Exactly what upstream `generalized_jsd_loss` computes (full-vocab, single shot)."""
    student_logits = head(s_hidden).float() / temperature
    teacher_logits = head(t_hidden).float() / temperature
    if top_k is not None and top_k > 0:
        _, idx = torch.topk(teacher_logits, k=top_k, dim=-1)
        student_logits = torch.gather(student_logits, dim=-1, index=idx)
        teacher_logits = torch.gather(teacher_logits, dim=-1, index=idx)
    s_logp = F.log_softmax(student_logits, dim=-1)
    t_logp = F.log_softmax(teacher_logits, dim=-1)
    if beta == 0:
        jsd = F.kl_div(s_logp, t_logp, reduction="none", log_target=True)
    elif beta == 1:
        jsd = F.kl_div(t_logp, s_logp, reduction="none", log_target=True)
    else:
        beta_t = torch.tensor(beta, dtype=s_logp.dtype)
        mix = torch.logsumexp(
            torch.stack([s_logp + torch.log1p(-beta_t), t_logp + torch.log(beta_t)]), dim=0
        )
        kl_t = F.kl_div(mix, t_logp, reduction="none", log_target=True)
        kl_s = F.kl_div(mix, s_logp, reduction="none", log_target=True)
        jsd = beta_t * kl_t + (1 - beta_t) * kl_s
    if token_clip is not None:
        jsd = jsd.clamp(max=token_clip)
    return jsd[mask].sum() / mask.sum()


def chunked_loss(head, s_hidden, t_hidden, mask, beta, temperature, token_clip, top_k, chunk):
    total = None
    G = mask.shape[1]
    for start in range(0, G, chunk):
        end = min(start + chunk, G)
        part = OPSDTrainer._chunk_loss_fn(
            head,
            s_hidden[:, start:end],
            t_hidden[:, start:end],
            mask[:, start:end],
            torch.zeros_like(s_hidden[:, start:end, 0], dtype=torch.long),
            temperature,
            beta,
            top_k,
            token_clip,
            False,
        )
        total = part if total is None else total + part
    return total / mask.sum()


def main():
    torch.manual_seed(0)
    B, G, H, V = 2, 37, 64, 512
    head = nn.Linear(H, V, bias=False)
    s_hidden = torch.randn(B, G, H, dtype=torch.float32, requires_grad=True)
    t_hidden = torch.randn(B, G, H, dtype=torch.float32)

    mask = torch.ones(B, G, dtype=torch.bool)
    mask[0, -5:] = False  # pretend the last tokens are padding
    mask[1, -9:] = False

    ok = True
    for beta in (0.0, 0.5, 1.0):
        for temperature in (1.0, 1.1):
            for token_clip in (None, 0.05):
                for top_k in (None, 16):
                    ref = reference_loss(head, s_hidden, t_hidden, mask, beta, temperature, token_clip, top_k)
                    ref.backward()
                    ref_grad = s_hidden.grad.clone()
                    s_hidden.grad = None

                    for chunk in (1, 8, 37, 64):
                        ch = chunked_loss(
                            head, s_hidden, t_hidden, mask, beta, temperature, token_clip, top_k, chunk
                        )
                        ch.backward()
                        ch_grad = s_hidden.grad.clone()
                        s_hidden.grad = None
                        dv = (ref - ch).abs().item()
                        dg = (ref_grad - ch_grad).abs().max().item()
                        good = dv < 1e-5 and dg < 1e-4
                        ok &= good
                        if not good or chunk == 8:
                            print(
                                f"beta={beta} T={temperature} clip={token_clip} top_k={top_k} chunk={chunk}: "
                                f"loss_ref={ref.item():.8f} loss_chunk={ch.item():.8f} dloss={dv:.2e} dgrad={dg:.2e} "
                                f"{'OK' if good else 'MISMATCH'}"
                            )
    print("\nALL EQUIVALENT" if ok else "\nMISMATCH DETECTED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
