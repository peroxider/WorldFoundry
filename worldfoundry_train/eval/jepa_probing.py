"""Probing metrics for the W22.7 A/B experiment (design §9.2).

Four metrics comparing the proxy-loss baseline (Arm A) against the true JEPA
objective (Arm B):

1. ``masked_reconstruction_cosine`` — cosine between predictor output and
   target-encoder features at masked positions (B should be ≫ chance).
2. ``linear_probe_accuracy`` — linear probe on frozen features.
3. ``token_variance`` / ``effective_rank`` — collapse diagnostics (the key
   signal: Arm A is expected to drift toward an identity/constant solution;
   acceptance gate: B keeps variance stable + cosine ≥ 0.6).
4. ``knn_hit_rate`` — k-NN label-consistency of representations.

All functions are pure torch, CPU-runnable, and side-effect free; experiment
drivers wire them to checkpoints and append results via ``append_jsonl``
(records carry the checkpoint ``step`` so curves align across arms).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

__all__ = [
    "masked_reconstruction_cosine",
    "linear_probe_accuracy",
    "token_variance",
    "effective_rank",
    "knn_hit_rate",
    "collapse_metrics",
    "append_jsonl",
    "read_jsonl",
]


def masked_reconstruction_cosine(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> float:
    """Mean cosine similarity between predictions and targets.

    Args:
        pred: ``(M, D)`` predictor output at masked positions.
        target: ``(M, D)`` target-encoder features (stop-grad) at the same
            positions, unnormalized (V-JEPA convention, design §6.2).

    Returns:
        Mean per-token cosine in [-1, 1]. Chance level ≈ 0 for random
        high-dimensional features; the §9.2 gate expects Arm B ≥ 0.6.
    """
    if pred.shape != target.shape:
        raise ValueError(
            f"pred {tuple(pred.shape)} != target {tuple(target.shape)}"
        )
    if pred.shape[0] == 0:
        raise ValueError("no masked tokens (M=0)")
    return float(F.cosine_similarity(pred, target, dim=-1).mean().item())


def linear_probe_accuracy(
    train_feats: torch.Tensor,
    train_labels: torch.Tensor,
    test_feats: torch.Tensor,
    test_labels: torch.Tensor,
    epochs: int = 200,
    lr: float = 0.1,
    seed: int = 0,
) -> float:
    """Train a softmax linear probe on frozen features; return test accuracy.

    Features are standardized with train statistics (fit on train only, no
    leakage). The probe is a single bias-free... no — a single Linear layer
    trained with Adam on cross-entropy; deterministic given ``seed``.
    """
    if train_feats.ndim != 2 or test_feats.ndim != 2:
        raise ValueError("features must be (N, D)")
    n_classes = int(max(train_labels.max(), test_labels.max()).item()) + 1
    mu = train_feats.mean(dim=0, keepdim=True)
    sd = train_feats.std(dim=0, keepdim=True).clamp_min(1e-6)
    xtr = (train_feats - mu) / sd
    xte = (test_feats - mu) / sd

    g = torch.Generator().manual_seed(seed)
    head = torch.nn.Linear(train_feats.shape[1], n_classes)
    with torch.no_grad():  # deterministic init
        head.weight.copy_(
            torch.randn(head.weight.shape, generator=g) * 0.01
        )
        head.bias.zero_()
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        loss = F.cross_entropy(head(xtr), train_labels)
        loss.backward()
        opt.step()
    pred = head(xte).argmax(dim=-1)
    return float((pred == test_labels).float().mean().item())


def token_variance(feats: torch.Tensor) -> float:
    """Mean per-dimension variance across tokens — collapse alarm (§9.2-3).

    Args:
        feats: ``(B, N, D)`` or ``(N, D)`` encoder features.

    Returns:
        ``mean_d Var_tokens(feats[..., d])``, averaged over batch. A
        collapsing (constant-output) encoder drives this toward 0; Arm A is
        expected to decay, Arm B to stay stable.
    """
    if feats.ndim == 2:
        feats = feats.unsqueeze(0)
    if feats.ndim != 3 or feats.shape[1] < 2:
        raise ValueError(
            f"feats must be (B, N>=2, D) or (N>=2, D), got {tuple(feats.shape)}"
        )
    return float(feats.var(dim=1, unbiased=False).mean().item())


def effective_rank(feats: torch.Tensor) -> float:
    """Effective rank (Roy & Vetterli 2007) of the feature matrix (§9.2-3).

    ``erank = exp(H(p))`` with ``p_i = s_i / Σs`` over singular values s of
    the centered ``(M, D)`` feature matrix. Ranges [1, min(M, D)]; a
    collapsed representation → 1, an isotropic one → min(M, D).
    """
    x = feats.reshape(-1, feats.shape[-1]).double()
    if x.shape[0] < 2:
        raise ValueError("need at least 2 tokens for effective rank")
    x = x - x.mean(dim=0, keepdim=True)
    s = torch.linalg.svdvals(x)
    total = s.sum()
    if float(total) == 0.0:
        return 1.0  # fully collapsed
    p = (s / total).clamp_min(1e-12)
    entropy = float(-(p * p.log()).sum().item())
    return math.exp(entropy)


def collapse_metrics(feats: torch.Tensor) -> dict[str, float]:
    """Convenience bundle for §9.2 metric 3: both collapse signals."""
    return {
        "token_variance": token_variance(feats),
        "effective_rank": effective_rank(feats),
    }


def knn_hit_rate(
    query_feats: torch.Tensor,
    key_feats: torch.Tensor,
    query_labels: torch.Tensor,
    key_labels: torch.Tensor,
    k: int = 1,
) -> float:
    """k-NN label consistency (§9.2-4).

    For each query, find its k nearest keys by cosine similarity; a hit is a
    neighbor whose label matches the query's (e.g. same source video,
    different time segment). Returns the hit fraction in [0, 1].
    """
    if query_feats.ndim != 2 or key_feats.ndim != 2:
        raise ValueError("features must be (N, D)")
    if k < 1 or k > key_feats.shape[0]:
        raise ValueError(f"k={k} out of range for {key_feats.shape[0]} keys")
    sim = F.cosine_similarity(
        query_feats.unsqueeze(1), key_feats.unsqueeze(0), dim=-1
    )  # (Q, K)
    nn_idx = sim.topk(k, dim=1).indices  # (Q, k)
    hits = (key_labels[nn_idx] == query_labels.unsqueeze(1)).any(dim=1)
    return float(hits.float().mean().item())


# ---------------------------------------------------------------------------
# JSONL recording (aligned to checkpoint step; see design §9.2 acceptance)
# ---------------------------------------------------------------------------

def append_jsonl(path: str | Path, record: dict) -> None:
    """Append one metrics record as a JSON line. Parent dirs are created.

    Records SHOULD carry ``step`` (optimizer-step counter of the checkpoint
    the metrics were computed on) so curves align across arms.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[dict]:
    """Read back records written by :func:`append_jsonl` (sorted by step)."""
    path = Path(path)
    if not path.is_file():
        return []
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return sorted(records, key=lambda r: r.get("step", 0))
