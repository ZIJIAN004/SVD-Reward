"""
POMO TSP training with SVD-Reward hybrid advantage.

Pipeline integration: run after the AE is pretrained (see run_pipeline.py
Step 2). The AE is frozen; per training batch we:
  1. Rollout B·P TSP tours with POMOModel
  2. GPU-batched encode all rollouts via AE (two-stage: InstanceEncoder
     runs once per instance, h_inst replicated P× for TourEncoder)
  3. compute hybrid advantage = α·znorm(−svd_residual) + (1−α)·znorm(R)
  4. REINFORCE: loss = −(advantage · Σ_t log π_t).mean()

Alpha=0 falls back to standard POMO (cost-baseline only); α=1 is pure SVD;
α∈(0,1) is hybrid (recommended default 0.5).

Logged each period: cost_svd_corr (≈±1 → SVD redundant with length;
≈0 → orthogonal signal added).
"""

import sys
sys.stdout.reconfigure(line_buffering=True)

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_sched

from model import TourAutoEncoder
from config import Config as AEConfig
from pomo_tsp import TSPEnv, POMOModel
from svd_reward import per_instance_reward_torch, topk_anchor_idx


# ────────────────────────────────────────────────────────────────────────
#  Config
# ────────────────────────────────────────────────────────────────────────

@dataclass
class POMOTrainConfig:
    # Problem
    num_nodes:        int = 100
    pomo_size:        int = 100

    # Schedule
    total_epoch:      int = 100
    train_episodes:   int = 10_000        # episodes per epoch (instances)
    eval_episodes:    int =  1_000
    train_batch_size: int =   64
    eval_batch_size:  int =  256

    # POMO model
    embedding_dim:     int = 128
    encoder_layer_num: int =   6
    head_num:          int =   8
    qkv_dim:           int =  16
    ff_hidden_dim:     int = 512
    logit_clipping:    float = 10.0

    # Optimization
    lr:           float = 1e-4
    weight_decay: float = 1e-6
    lr_milestones: tuple = (90, 95)
    lr_gamma:     float = 0.1

    # SVD hybrid
    svd_alpha: float = 0.5     # 0 = baseline POMO; 1 = pure SVD; 0.5 = balanced
    svd_rank:  int   = 16
    svd_top_k: int   = 50
    svd_knn_k: int   = 10

    log_period_sec: float = 30.0


# ────────────────────────────────────────────────────────────────────────
#  GPU-batched AE encode (two-stage: InstanceEncoder once per instance)
# ────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def _instance_graph_inputs(node_xy: torch.Tensor, knn_k: int):
    """B graphs, N nodes each. KNN edges with per-instance offset."""
    B, N, _ = node_xy.shape
    device = node_xy.device

    x_inst = node_xy.reshape(-1, 2).contiguous()
    dists = torch.cdist(node_xy, node_xy)
    dists.diagonal(dim1=1, dim2=2).fill_(float('inf'))
    _, knn_idx = dists.topk(knn_k, dim=2, largest=False)              # (B, N, k)

    src = torch.arange(N, device=device).view(1, N, 1).expand(B, N, knn_k)
    inst_offset = (torch.arange(B, device=device) * N).view(B, 1, 1)
    src_g = (src + inst_offset).reshape(-1)
    dst_g = (knn_idx + inst_offset).reshape(-1)
    inst_ei = torch.stack([
        torch.cat([src_g, dst_g], dim=0),
        torch.cat([dst_g, src_g], dim=0),
    ], dim=0).contiguous()
    return x_inst, inst_ei


@torch.no_grad()
def _tour_graph_inputs(selected_list: torch.Tensor):
    """B·P graphs, N nodes each. Hamilton-cycle edges with per-rollout offset."""
    B, P, N = selected_list.shape
    device = selected_list.device

    batch_tour = torch.arange(B * P, device=device).repeat_interleave(N)
    graph_offset = (torch.arange(B * P, device=device) * N).view(B, P, 1)

    src = selected_list
    dst = torch.roll(selected_list, shifts=-1, dims=-1)
    src_g = (src + graph_offset).reshape(-1)
    dst_g = (dst + graph_offset).reshape(-1)
    tour_ei = torch.stack([
        torch.cat([src_g, dst_g], dim=0),
        torch.cat([dst_g, src_g], dim=0),
    ], dim=0).contiguous()
    return tour_ei, batch_tour


@torch.no_grad()
def encode_rollouts(ae: TourAutoEncoder, node_xy: torch.Tensor,
                     selected_list: torch.Tensor, knn_k: int) -> torch.Tensor:
    """Two-stage encode: InstanceEncoder runs once per instance; TourEncoder
    over B·P rollouts. Saves ~100× FLOPs and memory vs naive B·P-graph batch.
    Returns z: (B, P, D)."""
    B, P, N = selected_list.shape

    x_inst, inst_ei = _instance_graph_inputs(node_xy, knn_k=knn_k)
    h_inst_per_inst = ae.instance_encoder(x_inst, inst_ei)             # (B·N, hidden)

    hidden = h_inst_per_inst.shape[-1]
    h_inst = (h_inst_per_inst.view(B, N, hidden)
              .unsqueeze(1).expand(B, P, N, hidden)
              .reshape(-1, hidden).contiguous())                       # (B·P·N, hidden)

    tour_ei, batch_tour = _tour_graph_inputs(selected_list)
    z_flat = ae.tour_encoder(h_inst, tour_ei, batch_tour)              # (B·P, D)
    return z_flat.view(B, P, -1)


# ────────────────────────────────────────────────────────────────────────
#  Hybrid advantage
# ────────────────────────────────────────────────────────────────────────

def _znorm(r: torch.Tensor) -> torch.Tensor:
    m = r.mean(dim=1, keepdim=True)
    s = r.std(dim=1, keepdim=True).clamp(min=1e-8)
    return (r - m) / s


@torch.no_grad()
def compute_hybrid_advantage(z: torch.Tensor, rewards: torch.Tensor,
                              alpha: float, rank: int, top_k: int,
                              return_diag: bool = False):
    """α·znorm(−svd_residual) + (1−α)·znorm(R). Returns (advantage[, diag])."""
    B, P = rewards.shape
    cost_adv = _znorm(rewards)
    anchor_idx = topk_anchor_idx(-rewards, top_k=min(top_k, P // 2))

    if return_diag:
        raw_svd, svd_diag = per_instance_reward_torch(
            z, anchor_idx, rank=rank, return_diag=True)
    else:
        raw_svd = per_instance_reward_torch(z, anchor_idx, rank=rank)
    svd_adv = _znorm(raw_svd)
    advantage = alpha * svd_adv + (1.0 - alpha) * cost_adv

    if not return_diag:
        return advantage

    cs_c = cost_adv - cost_adv.mean(dim=1, keepdim=True)
    sv_c = svd_adv  - svd_adv.mean(dim=1, keepdim=True)
    num = (cs_c * sv_c).sum(dim=1)
    den = (cs_c.norm(dim=1) * sv_c.norm(dim=1)).clamp(min=1e-8)
    cost_svd_corr = (num / den).mean()
    diag = {
        **svd_diag,
        'cost_svd_corr':     cost_svd_corr.detach(),
        'cost_adv_abs_mean': cost_adv.abs().mean().detach(),
        'svd_adv_abs_mean':  svd_adv.abs().mean().detach(),
    }
    return advantage, diag


# ────────────────────────────────────────────────────────────────────────
#  AE loader
# ────────────────────────────────────────────────────────────────────────

def load_ae(ckpt_path: str, device: torch.device) -> TourAutoEncoder:
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = AEConfig(**ckpt["config"])
    ae = TourAutoEncoder(cfg).to(device)
    ae.load_state_dict(ckpt["model"])
    ae.eval()
    for p in ae.parameters():
        p.requires_grad_(False)
    return ae


# ────────────────────────────────────────────────────────────────────────
#  Training loop
# ────────────────────────────────────────────────────────────────────────

def _train_epoch(model, ae, env, optimizer, cfg, device, log_period_sec):
    model.train()
    episode = 0
    losses, dists, corr_log = [], [], []
    t_period = time.time()

    while episode < cfg.train_episodes:
        B = min(cfg.train_batch_size, cfg.train_episodes - episode)
        episode += B

        env.load_problems(B, device=device)
        coords = env.reset()
        model.pre_forward(coords)

        state, _, done = env.pre_step()
        prob_list = torch.zeros(B, cfg.pomo_size, 0, device=device)
        while not done:
            selected, prob, _ = model(state)
            state, reward, done = env.step(selected)
            prob_list = torch.cat((prob_list, prob[:, :, None]), dim=2)

        reward_f = reward.float()                                       # (B, P)

        if cfg.svd_alpha > 0.0 and ae is not None:
            z = encode_rollouts(ae, env.node_xy, env.selected_node_list,
                                knn_k=cfg.svd_knn_k)
            advantage, diag = compute_hybrid_advantage(
                z=z, rewards=reward_f,
                alpha=cfg.svd_alpha, rank=cfg.svd_rank, top_k=cfg.svd_top_k,
                return_diag=True)
            corr_log.append(diag['cost_svd_corr'].item())
        else:
            advantage = reward_f - reward_f.mean(dim=1, keepdim=True)

        log_prob = prob_list.log().sum(dim=2)                           # (B, P)
        loss = -(advantage * log_prob).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        dists.append((-reward_f.max(dim=1).values).mean().item())       # best-of-P per instance

        if time.time() - t_period > log_period_sec or episode >= cfg.train_episodes:
            corr_str = (f"  cost_corr={np.mean(corr_log[-50:]):+.3f}"
                        if corr_log else "")
            print(f"  ep:{episode:6d}/{cfg.train_episodes}  "
                  f"loss={np.mean(losses[-50:]):+.4f}  "
                  f"dist={np.mean(dists[-50:]):.4f}{corr_str}",
                  flush=True)
            t_period = time.time()

    return np.mean(dists), (np.mean(corr_log) if corr_log else None)


@torch.no_grad()
def _eval_epoch(model, env, cfg, device):
    model.eval()
    episode = 0
    dists = []
    while episode < cfg.eval_episodes:
        B = min(cfg.eval_batch_size, cfg.eval_episodes - episode)
        episode += B
        env.load_problems(B, device=device)
        coords = env.reset()
        model.pre_forward(coords)
        state, _, done = env.pre_step()
        while not done:
            selected, _, _ = model(state)
            state, reward, done = env.step(selected)
        dists.append((-reward.max(dim=1).values).mean().item())
    return float(np.mean(dists))


def train_pomo(ae_ckpt_path: str,
               cfg: POMOTrainConfig = None,
               save_dir: str = "checkpoints",
               run_tag: str = None):
    """Train POMO with SVD-hybrid advantage. ae_ckpt_path can be None for
    pure-baseline POMO (cfg.svd_alpha is ignored in that case)."""
    if cfg is None:
        cfg = POMOTrainConfig()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    ae = None
    if ae_ckpt_path is not None and cfg.svd_alpha > 0.0:
        ae = load_ae(ae_ckpt_path, device)
        print(f"AE loaded from {ae_ckpt_path}  (α={cfg.svd_alpha})", flush=True)
    else:
        print(f"Baseline POMO (no SVD; α={cfg.svd_alpha})", flush=True)

    model = POMOModel(
        embedding_dim=cfg.embedding_dim,
        encoder_layer_num=cfg.encoder_layer_num,
        head_num=cfg.head_num,
        qkv_dim=cfg.qkv_dim,
        ff_hidden_dim=cfg.ff_hidden_dim,
        logit_clipping=cfg.logit_clipping,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=cfg.lr,
                           weight_decay=cfg.weight_decay)
    scheduler = lr_sched.MultiStepLR(optimizer, milestones=list(cfg.lr_milestones),
                                      gamma=cfg.lr_gamma)
    env = TSPEnv(problem_size=cfg.num_nodes, pomo_size=cfg.pomo_size)

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    tag = run_tag or f"pomo_a{cfg.svd_alpha}"
    run_dir = save_dir / tag
    run_dir.mkdir(parents=True, exist_ok=True)

    train_curve, eval_curve, corr_curve = [], [], []
    t0 = time.time()

    for ep in range(1, cfg.total_epoch + 1):
        print(f"\n[ep {ep:3d}/{cfg.total_epoch}]  elapsed={time.time()-t0:6.0f}s", flush=True)
        train_dist, train_corr = _train_epoch(
            model, ae, env, optimizer, cfg, device, cfg.log_period_sec)
        eval_dist = _eval_epoch(model, env, cfg, device)
        scheduler.step()

        train_curve.append(train_dist)
        eval_curve.append(eval_dist)
        if train_corr is not None:
            corr_curve.append(train_corr)

        print(f"  train_dist={train_dist:.4f}  eval_dist={eval_dist:.4f}"
              + (f"  cost_corr={train_corr:+.3f}" if train_corr is not None else ""),
              flush=True)

        torch.save({
            'model':    model.state_dict(),
            'cfg':      cfg.__dict__,
            'epoch':    ep,
            'eval':     eval_dist,
            'train':    train_dist,
        }, run_dir / "ckpt_last.pt")

    # Save curves for plotting.
    np.savez(run_dir / "curves.npz",
             train=np.array(train_curve),
             eval=np.array(eval_curve),
             corr=np.array(corr_curve) if corr_curve else np.zeros(0))
    print(f"\nDone. Best eval dist = {min(eval_curve):.4f} (epoch "
          f"{int(np.argmin(eval_curve))+1})", flush=True)
    return train_curve, eval_curve, corr_curve


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--ae-ckpt', type=str, default=None,
                   help='AE checkpoint; omit for baseline POMO (α effectively 0).')
    p.add_argument('--alpha', type=float, default=0.5)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--episodes', type=int, default=10_000)
    p.add_argument('--save-dir', type=str, default='checkpoints')
    p.add_argument('--tag', type=str, default=None)
    args = p.parse_args()

    cfg = POMOTrainConfig(
        total_epoch=args.epochs,
        train_episodes=args.episodes,
        svd_alpha=args.alpha,
    )
    train_pomo(ae_ckpt_path=args.ae_ckpt, cfg=cfg,
               save_dir=args.save_dir, run_tag=args.tag)
