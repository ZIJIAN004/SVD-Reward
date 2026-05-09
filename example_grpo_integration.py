"""
Example: per-instance SVD reward in a GRPO training loop.

Each instance has G POMO rollouts.  We pick the top-k shortest as anchors,
fit a per-instance SVD on the fly, and score every rollout by its residual
to that instance's local subspace.  No global SVD, no offline anchor file.

Adapt to your actual POMO + GRPO codebase.
"""

import torch
from torch_geometric.data import Batch

from config import Config
from model import TourAutoEncoder
from svd_reward import per_instance_reward_torch, topk_anchor_idx
from tsp_data import make_pyg_data, tour_length


# ── 1. Load pretrained autoencoder ───────────────────────────────────
def load_encoder(ckpt_path: str, device: torch.device) -> TourAutoEncoder:
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = Config(**ckpt["config"])
    ae = TourAutoEncoder(cfg).to(device)
    ae.load_state_dict(ckpt["model"])
    ae.eval()
    return ae


# ── 2. Encode all (instance, rollout) tours into a (B, G, D) tensor ──
@torch.no_grad()
def encode_rollouts(
    ae: TourAutoEncoder,
    coords_batch: torch.Tensor,    # (B, N, 2)
    tours_batch: torch.Tensor,     # (B, G, N)  node indices
    device: torch.device,
) -> torch.Tensor:
    B, G, N = tours_batch.shape
    coords_np = coords_batch.cpu().numpy()
    tours_np = tours_batch.cpu().numpy()

    pyg_list = []
    for b in range(B):
        for g in range(G):
            # instance_id is set so the encoder's internal logic can stay generic;
            # per-instance grouping at reward time is done via the (B, G) reshape.
            pyg_list.append(make_pyg_data(coords_np[b], tours_np[b, g], instance_id=b))

    big_batch = Batch.from_data_list(pyg_list).to(device)
    z = ae.encode(big_batch)                       # (B*G, D)
    return z.view(B, G, -1)                        # (B, G, D)


# ── 3. Per-instance SVD reward (top-k by length as anchors) ──────────
@torch.no_grad()
def compute_svd_reward(
    ae: TourAutoEncoder,
    coords_batch: torch.Tensor,    # (B, N, 2)
    tours_batch: torch.Tensor,     # (B, G, N)
    device: torch.device,
    rank: int = 16,
    top_k: int = 50,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Returns:
        advantage:  (B, G)  per-instance z-scored reward, ready for GRPO.
    """
    B, G, N = tours_batch.shape

    # Lengths needed for choosing anchors.
    coords_np = coords_batch.cpu().numpy()
    tours_np = tours_batch.cpu().numpy()
    lengths = torch.tensor(
        [[tour_length(coords_np[b], tours_np[b, g]) for g in range(G)]
         for b in range(B)],
        dtype=torch.float32, device=device,
    )                                               # (B, G)

    # Encode → per-instance SVD on top-k by length → residual reward.
    z = encode_rollouts(ae, coords_batch, tours_batch, device)   # (B, G, D)
    anchor_idx = topk_anchor_idx(lengths, top_k=min(top_k, G // 2))
    raw = per_instance_reward_torch(z, anchor_idx, rank=rank)    # (B, G)

    # GRPO advantage style: per-instance z-score.
    mean = raw.mean(dim=1, keepdim=True)
    std = raw.std(dim=1, keepdim=True).clamp(min=1e-8)
    return (raw - mean) / std * temperature


# ── 4. Sketch: GRPO policy gradient step ─────────────────────────────
def grpo_step_sketch(policy, optimizer, coords, tours, log_probs, ae, device):
    """
    coords:    (B, N, 2)
    tours:     (B, G, N)      sampled rollouts
    log_probs: (B, G)         log π(tour | instance)
    """
    advantage = compute_svd_reward(ae, coords, tours, device)    # (B, G)
    loss = -(log_probs * advantage).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


# ── 5. Optional: hybrid reward (SVD structure + tour length) ─────────
def hybrid_reward(ae, coords_batch, tours_batch, device,
                  alpha: float = 0.5, rank: int = 16, top_k: int = 50):
    """
    advantage = α · SVD_advantage + (1−α) · cost_advantage   (both z-scored per instance)
    """
    B, G, N = tours_batch.shape
    coords_np = coords_batch.cpu().numpy()
    tours_np = tours_batch.cpu().numpy()

    lengths = torch.tensor(
        [[tour_length(coords_np[b], tours_np[b, g]) for g in range(G)]
         for b in range(B)],
        dtype=torch.float32, device=device,
    )
    cost_reward = -lengths                                       # (B, G)

    svd_adv = compute_svd_reward(
        ae, coords_batch, tours_batch, device, rank=rank, top_k=top_k,
    )

    def znorm(r):
        m = r.mean(dim=1, keepdim=True)
        s = r.std(dim=1, keepdim=True).clamp(min=1e-8)
        return (r - m) / s

    return alpha * svd_adv + (1 - alpha) * znorm(cost_reward)
