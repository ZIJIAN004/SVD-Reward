"""
Example: how to use SVDReward as the reward signal in a GRPO training loop.

This is a standalone sketch — adapt to your actual POMO + GRPO codebase.
"""

import torch
from torch_geometric.loader import DataLoader

from config import Config
from model import TourAutoEncoder
from svd_reward import SVDReward
from tsp_data import make_pyg_data


# ── 1. Load pretrained autoencoder + SVD subspace ────────────────────
def load_reward_pipeline(ckpt_path: str, svd_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = Config(**ckpt["config"])
    ae = TourAutoEncoder(cfg).to(device)
    ae.load_state_dict(ckpt["model"])
    ae.eval()

    svd = SVDReward.load(svd_path)
    return ae, svd


# ── 2. Compute reward for a batch of POMO rollouts ──────────────────
@torch.no_grad()
def compute_svd_reward(
    ae: TourAutoEncoder,
    svd: SVDReward,
    coords_batch,       # (B, N, 2)  problem instances
    tours_batch,        # (B, G, N)  G rollouts per instance  (node indices)
    device: torch.device,
    temperature: float = 1.0,
):
    """
    Args:
        coords_batch:  (B, N, 2)  coordinates
        tours_batch:   (B, G, N)  G completions per instance
    Returns:
        rewards:  (B, G)  normalized reward for each completion
    """
    B, G, N = tours_batch.shape
    coords_np = coords_batch.cpu().numpy()
    tours_np = tours_batch.cpu().numpy()

    pyg_list = []
    for b in range(B):
        for g in range(G):
            pyg_list.append(make_pyg_data(coords_np[b], tours_np[b, g]))

    loader = DataLoader(pyg_list, batch_size=256)
    embeddings = []
    for batch in loader:
        batch = batch.to(device)
        z = ae.encode(batch)
        embeddings.append(z.cpu())
    embeddings = torch.cat(embeddings, dim=0)   # (B*G, D)

    raw_reward = svd.reward(embeddings)         # (B*G,)  numpy
    raw_reward = torch.tensor(raw_reward, dtype=torch.float32).view(B, G)

    # Per-instance normalization (GRPO advantage style)
    mean = raw_reward.mean(dim=1, keepdim=True)
    std = raw_reward.std(dim=1, keepdim=True).clamp(min=1e-8)
    return (raw_reward - mean) / std * temperature


# ── 3. Sketch: GRPO policy gradient step ─────────────────────────────
def grpo_step_sketch(policy, optimizer, coords, tours, log_probs, ae, svd, device):
    """
    coords:    (B, N, 2)
    tours:     (B, G, N)      sampled rollouts
    log_probs: (B, G)         log π(tour | instance)
    """
    # --- SVD-based advantage ---
    advantage = compute_svd_reward(ae, svd, coords, tours, device)  # (B, G)

    # --- Policy gradient ---
    loss = -(log_probs * advantage.to(device)).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


# ── 4. Optional: hybrid reward (SVD structure + tour length) ─────────
def hybrid_reward(ae, svd, coords_batch, tours_batch, device,
                  alpha: float = 0.5):
    """
    reward = α · SVD_reward + (1−α) · cost_reward

    α controls the trade-off:
      α=1  →  pure structure reward
      α=0  →  pure tour-length reward
    """
    B, G, N = tours_batch.shape
    coords_np = coords_batch.cpu().numpy()
    tours_np = tours_batch.cpu().numpy()

    from tsp_data import tour_length
    import numpy as np

    lengths = np.array([
        [tour_length(coords_np[b], tours_np[b, g]) for g in range(G)]
        for b in range(B)
    ])
    cost_reward = torch.tensor(-lengths, dtype=torch.float32)  # (B, G)

    svd_reward = compute_svd_reward(ae, svd, coords_batch, tours_batch, device)

    # Normalize both to z-scores per instance, then combine
    def znorm(r):
        m = r.mean(dim=1, keepdim=True)
        s = r.std(dim=1, keepdim=True).clamp(min=1e-8)
        return (r - m) / s

    return alpha * znorm(svd_reward) + (1 - alpha) * znorm(cost_reward)
