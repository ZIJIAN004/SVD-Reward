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


# ── 3. Reward helpers (lengths + z-score + per-instance SVD) ─────────
def _per_batch_tour_lengths(coords_batch, tours_batch, device):
    """Tour length for every (instance, rollout). Returns (B, G) on device."""
    B, G, _ = tours_batch.shape
    coords_np = coords_batch.cpu().numpy()
    tours_np = tours_batch.cpu().numpy()
    return torch.tensor(
        [[tour_length(coords_np[b], tours_np[b, g]) for g in range(G)]
         for b in range(B)],
        dtype=torch.float32, device=device,
    )


def _znorm(r):
    """Per-instance z-score along the rollout dimension."""
    m = r.mean(dim=1, keepdim=True)
    s = r.std(dim=1, keepdim=True).clamp(min=1e-8)
    return (r - m) / s


# ── 4. Hybrid GRPO advantage = SVD + cost (recommended default) ──────
@torch.no_grad()
def compute_hybrid_advantage(
    ae: TourAutoEncoder,
    coords_batch: torch.Tensor,    # (B, N, 2)
    tours_batch: torch.Tensor,     # (B, G, N)
    device: torch.device,
    alpha: float = 0.5,            # 0=pure cost (baseline POMO), 1=pure SVD
    rank: int = 16,
    top_k: int = 50,
    temperature: float = 1.0,
    return_diag: bool = False,
):
    """
    advantage = α · SVD_advantage + (1−α) · cost_advantage,
                both per-instance z-scored.

    α=0.5 default keeps the POMO cost-baseline signal at half strength and
    adds the SVD subspace-similarity signal — the two signals ADD, they
    don't replace each other. Use α=0 to ablate to pure cost; α=1 to
    ablate to pure SVD.

    Returns:
        advantage:  (B, G)
        diag:       dict (only if return_diag=True) — see below.
    """
    B, G, _ = tours_batch.shape

    lengths = _per_batch_tour_lengths(coords_batch, tours_batch, device)  # (B, G)
    cost_adv = _znorm(-lengths)                                           # (B, G)

    z = encode_rollouts(ae, coords_batch, tours_batch, device)            # (B, G, D)
    anchor_idx = topk_anchor_idx(lengths, top_k=min(top_k, G // 2))

    if return_diag:
        raw, svd_diag = per_instance_reward_torch(
            z, anchor_idx, rank=rank, return_diag=True)
    else:
        raw = per_instance_reward_torch(z, anchor_idx, rank=rank)

    svd_adv = _znorm(raw) * temperature
    advantage = alpha * svd_adv + (1.0 - alpha) * cost_adv

    if not return_diag:
        return advantage

    # Signal-mix monitoring — answers "are cost and svd carrying the same
    # information, or genuinely complementary?"
    # corr=±1 → redundant (no info gained by hybrid); corr≈0 → orthogonal.
    cs_centered = cost_adv - cost_adv.mean(dim=1, keepdim=True)
    sv_centered = svd_adv  - svd_adv.mean(dim=1, keepdim=True)
    num = (cs_centered * sv_centered).sum(dim=1)
    den = (cs_centered.norm(dim=1) * sv_centered.norm(dim=1)).clamp(min=1e-8)
    cost_svd_corr = (num / den).mean()

    diag = {
        **svd_diag,
        'cost_svd_corr':       cost_svd_corr.detach(),
        'cost_adv_abs_mean':   cost_adv.abs().mean().detach(),
        'svd_adv_abs_mean':    svd_adv.abs().mean().detach(),
        'alpha':               torch.tensor(alpha, device=device),
    }
    return advantage, diag


# ── 5. SVD-only advantage (for ablation; not the recommended default) ─
@torch.no_grad()
def compute_svd_only_advantage(
    ae, coords_batch, tours_batch, device,
    rank: int = 16, top_k: int = 50, temperature: float = 1.0,
):
    """Pure SVD advantage — equivalent to compute_hybrid_advantage(alpha=1)."""
    return compute_hybrid_advantage(
        ae, coords_batch, tours_batch, device,
        alpha=1.0, rank=rank, top_k=top_k, temperature=temperature,
    )


# ── 6. GRPO policy gradient step (uses hybrid by default) ────────────
def grpo_step_sketch(
    policy, optimizer, coords, tours, log_probs, ae, device,
    alpha: float = 0.5, rank: int = 16, top_k: int = 50,
):
    """
    coords:    (B, N, 2)
    tours:     (B, G, N)      sampled rollouts
    log_probs: (B, G)         log π(tour | instance)

    Default: hybrid advantage (α=0.5 — SVD + cost ADD).
    Pass alpha=0 to get baseline POMO behavior; alpha=1 for SVD-only.
    """
    advantage = compute_hybrid_advantage(
        ae, coords, tours, device, alpha=alpha, rank=rank, top_k=top_k,
    )                                                            # (B, G)
    loss = -(log_probs * advantage).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()
