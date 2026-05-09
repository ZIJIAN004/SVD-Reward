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
) -> torch.Tensor:
    """
    Fully batched per-instance SVD reward.  All B instances in parallel.

    Returns:
        rewards: (B, K)  higher is better
    """
    B, K, D = z.shape
    A = anchor_idx.shape[1]

    # Gather anchors per instance: (B, A, D)
    batch_idx = torch.arange(B, device=z.device).unsqueeze(1).expand(B, A)
    z_anchor = z[batch_idx, anchor_idx]                          # (B, A, D)

    mu = z_anchor.mean(dim=1, keepdim=True)                      # (B, 1, D)
    centered_anchor = z_anchor - mu                              # (B, A, D)

    # Batched SVD.  Vt: (B, min(A,D), D)
    _, _, Vt = torch.linalg.svd(centered_anchor, full_matrices=False)
    k = min(rank, Vt.shape[1])
    basis = Vt[:, :k, :]                                         # (B, k, D)

    centered_z = z - mu                                          # (B, K, D)
    coeff = torch.einsum("bkd,brd->bkr", centered_z, basis)      # (B, K, k)
    proj = torch.einsum("bkr,brd->bkd", coeff, basis)            # (B, K, D)
    residual = (centered_z - proj).norm(dim=-1)                  # (B, K)
    return -residual


def topk_anchor_idx(lengths: torch.Tensor, top_k: int) -> torch.Tensor:
    """Indices of the top-k shortest tours per instance.  lengths: (B, K)."""
    return lengths.topk(top_k, dim=1, largest=False).indices     # (B, top_k)
