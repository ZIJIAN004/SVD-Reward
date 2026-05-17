"""
Per-instance SVD reward.

For each TSP instance, fit a low-rank subspace from a pool of "good" tour
embeddings of THAT instance, then score every tour of THAT instance by its
orthogonal residual to the subspace.  Higher reward = closer to good manifold.

There is intentionally no persistent state — every batch (or every test
instance) refits its own SVD from a small anchor pool.  This matches the
intended GRPO usage where POMO produces ~100 rollouts per instance and the
top-K by length act as anchors.
"""

import numpy as np
import torch


# ────────────────────────────────────────────────────────────────────────────
#  numpy version (offline pipeline / debugging)
# ────────────────────────────────────────────────────────────────────────────
def fit_instance_subspace_np(z_anchors: np.ndarray, rank: int = 16):
    """SVD on one instance's anchor embeddings.  Returns (mean, basis, sv)."""
    mu = z_anchors.mean(axis=0)
    centered = z_anchors - mu
    _, S, Vt = np.linalg.svd(centered, full_matrices=False)
    k = min(rank, len(S))
    return mu, Vt[:k], S[:k]


def per_instance_reward_np(
    z: np.ndarray,
    instance_ids: np.ndarray,
    anchor_mask: np.ndarray,
    rank: int = 16,
) -> np.ndarray:
    """
    For each instance, fit local SVD on z[anchor_mask & instance==i],
    then return reward = -residual_norm for every tour of that instance.

    Args:
        z:            (M, D)
        instance_ids: (M,)    int instance label per tour
        anchor_mask:  (M,)    bool, True for tours used to fit the subspace
        rank:         SVD rank (clipped to anchor count - 1)
    Returns:
        rewards:      (M,)    higher is better (numpy float)
    """
    if isinstance(z, torch.Tensor):
        z = z.detach().cpu().numpy()
    instance_ids = np.asarray(instance_ids)
    anchor_mask = np.asarray(anchor_mask, dtype=bool)

    rewards = np.zeros(len(z), dtype=np.float32)
    for inst in np.unique(instance_ids):
        inst_mask = instance_ids == inst
        anchor_inst = inst_mask & anchor_mask

        z_anchor = z[anchor_inst]
        if len(z_anchor) < 2:
            # Cannot fit a meaningful subspace — leave reward at 0.
            continue

        mu, basis, _ = fit_instance_subspace_np(z_anchor, rank=rank)
        z_inst = z[inst_mask]
        centered = z_inst - mu
        proj = (centered @ basis.T) @ basis
        rewards[inst_mask] = -np.linalg.norm(centered - proj, axis=1)

    return rewards


# ────────────────────────────────────────────────────────────────────────────
#  torch version (GRPO online use — keeps everything on GPU, batched SVD)
# ────────────────────────────────────────────────────────────────────────────
def per_instance_reward_torch(
    z: torch.Tensor,            # (B, K, D)  K rollouts per instance
    anchor_idx: torch.Tensor,   # (B, A)     indices of anchors within each instance's K
    rank: int = 16,
    return_diag: bool = False,
):
    """
    Fully batched per-instance SVD reward.  All B instances in parallel.

    Returns:
        rewards: (B, K)  higher is better
        diag:    dict (only if return_diag=True), see compute_svd_diagnostics
                 docstring for fields.
    """
    B, K, D = z.shape
    A = anchor_idx.shape[1]

    # Gather anchors per instance: (B, A, D)
    batch_idx = torch.arange(B, device=z.device).unsqueeze(1).expand(B, A)
    z_anchor = z[batch_idx, anchor_idx]                          # (B, A, D)

    mu = z_anchor.mean(dim=1, keepdim=True)                      # (B, 1, D)
    centered_anchor = z_anchor - mu                              # (B, A, D)

    # Batched SVD.  Vt: (B, min(A,D), D); S: (B, min(A,D))
    _, S, Vt = torch.linalg.svd(centered_anchor, full_matrices=False)
    k = min(rank, Vt.shape[1])
    basis = Vt[:, :k, :]                                         # (B, k, D)

    centered_z = z - mu                                          # (B, K, D)
    coeff = torch.einsum("bkd,brd->bkr", centered_z, basis)      # (B, K, k)
    proj = torch.einsum("bkr,brd->bkd", coeff, basis)            # (B, K, D)
    residual = (centered_z - proj).norm(dim=-1)                  # (B, K)
    reward = -residual

    if not return_diag:
        return reward

    # ── Diagnostics ────────────────────────────────────────────────────────
    sv_sq = S.square()                                            # (B, min(A,D))
    total_var = sv_sq.sum(dim=1).clamp(min=1e-12)                 # (B,)

    # Variance explained by the chosen rank — high means rank captures
    # most of the anchor-subspace structure.
    explained_var_ratio = (sv_sq[:, :k].sum(dim=1) / total_var).mean()

    # 95% rank: smallest k* with cumulative variance ≥ 95% — suggests the
    # "correct" rank choice; if rank_95 >> rank, you're truncating signal.
    cum_var = sv_sq.cumsum(dim=1) / total_var.unsqueeze(1)
    rank_95 = (cum_var < 0.95).sum(dim=1).float() + 1.0           # (B,)

    # Effective rank via singular-value entropy (Roy & Vetterli 2007):
    # exp(H(s_normalized)). Low ≈ subspace is degenerate / collapsed;
    # high ≈ anchors span many directions equally.
    s_norm = S / S.sum(dim=1, keepdim=True).clamp(min=1e-12)
    entropy = -(s_norm * s_norm.clamp(min=1e-12).log()).sum(dim=1)
    eff_rank = entropy.exp()                                      # (B,)

    # Residual sanity check: anchors define the subspace so should have
    # residual ≈ 0; non-anchors carry the actual reward signal.
    anchor_mask = torch.zeros(B, K, dtype=torch.bool, device=z.device)
    anchor_mask.scatter_(1, anchor_idx, True)
    anchor_res     = residual[anchor_mask].mean()
    non_anchor_res = residual[~anchor_mask].mean()

    # anchor_to_signal: how close anchor residuals are to non-anchor residuals.
    # Ideal: << 1 (anchors define subspace, non-anchors deviate). If ≈ 1,
    # anchors and non-anchors are indistinguishable in the subspace — the
    # SVD has no discriminative power.
    anchor_to_signal = anchor_res / non_anchor_res.clamp(min=1e-8)

    diag = {
        'svd_explained_var_ratio': explained_var_ratio.detach(),
        'svd_rank_95':             rank_95.mean().detach(),
        'svd_effective_rank':      eff_rank.mean().detach(),
        'svd_anchor_residual':     anchor_res.detach(),
        'svd_non_anchor_residual': non_anchor_res.detach(),
        'svd_anchor_to_signal':    anchor_to_signal.detach(),
    }
    return reward, diag


def topk_anchor_idx(lengths: torch.Tensor, top_k: int) -> torch.Tensor:
    """Indices of the top-k shortest tours per instance.  lengths: (B, K)."""
    return lengths.topk(top_k, dim=1, largest=False).indices     # (B, top_k)
