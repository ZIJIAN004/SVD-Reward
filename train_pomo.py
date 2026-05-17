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
import torch.nn.functional as F
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
    svd_alpha:       float = 0.3     # modulation strength on top of cost_adv
    svd_rank:        int   = 16
    svd_top_k:       int   = 50
    svd_knn_k:       int   = 10
    svd_temperature: float = 1.0     # softmax sharpness over non-anchors (lower = sharper)

    # Online AE training (SupCon contrastive on POMO reward ordering)
    online_ae:       bool  = False
    ae_lr:           float = 1e-5    # smaller than POMO's lr (1e-4) — gentle update
    ae_loss_weight:  float = 0.1     # weight of contrastive term in the joint loss
    ae_temperature:  float = 0.1     # InfoNCE softness
    ae_k_pos:        int   = 20      # # positives per instance for contrastive
    ae_k_neg:        int   = 20      # # negatives per instance for contrastive

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


def encode_rollouts(ae: TourAutoEncoder, node_xy: torch.Tensor,
                     selected_list: torch.Tensor, knn_k: int) -> torch.Tensor:
    """Two-stage encode: InstanceEncoder runs once per instance; TourEncoder
    over B·P rollouts. Saves ~100× FLOPs and memory vs naive B·P-graph batch.

    No @torch.no_grad() — caller decides. For frozen AE wrap in torch.no_grad();
    for online AE training keep grad on so contrastive loss can backprop.

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
                              temperature: float = 1.0,
                              return_diag: bool = False):
    """
    Hybrid advantage = cost_adv + α · svd_signal.

    Design: anchors (top-k by reward) get svd_signal = 0 — their in-sample
    residual is ≈ 0 and just noises the gradient. Non-anchors do softmax
    over themselves (with temperature T), scaled so per-non-anchor mean
    weight is 1, then centered to mean 0 so the signal is purely a
    redistribution among non-anchors.

      anchor:     advantage = cost_adv (unchanged, pure POMO baseline)
      non-anchor: advantage = cost_adv + α · (softmax redistribution)
                  "close to anchor cluster" → positive correction
                                              → less punishment
                  "far from anchor cluster"  → negative correction
                                              → more punishment

    Returns:
        advantage: (B, P)
        diag:      dict (only if return_diag=True)
    """
    B, P = rewards.shape
    device = z.device
    cost_adv = _znorm(rewards)                                            # (B, P)

    top_k_eff = min(top_k, P // 2)
    anchor_idx = topk_anchor_idx(-rewards, top_k=top_k_eff)               # (B, A)

    anchor_mask = torch.zeros(B, P, dtype=torch.bool, device=device)
    anchor_mask.scatter_(1, anchor_idx, True)
    non_anchor_mask = ~anchor_mask                                        # (B, P)
    naf = non_anchor_mask.float()

    if return_diag:
        raw_svd, svd_diag = per_instance_reward_torch(
            z, anchor_idx, rank=rank, return_diag=True)
    else:
        raw_svd = per_instance_reward_torch(z, anchor_idx, rank=rank)

    # Softmax over non-anchors only. raw_svd = -residual; higher = closer
    # to subspace. Anchors are masked to -inf so their softmax weight is 0.
    logits = (raw_svd / temperature).masked_fill(anchor_mask, float('-inf'))
    sm = torch.softmax(logits, dim=1)                                     # (B, P), sums to 1 over non-anchors

    n_non_anchors = naf.sum(dim=1, keepdim=True).clamp(min=1.0)
    # (sm · n - 1) for non-anchors → mean 0 over non-anchors, range roughly
    # [-1, n-1]. Anchors are 0 (since sm=0 there and we mask).
    svd_signal = (sm * n_non_anchors - 1.0) * naf

    advantage = cost_adv + alpha * svd_signal

    if not return_diag:
        return advantage

    # Pearson(cost_adv, svd_signal) over non-anchors only — anchors
    # contribute 0 to svd_signal so including them dilutes the metric.
    cs_mean = (cost_adv * naf).sum(dim=1, keepdim=True) / n_non_anchors
    sv_mean = (svd_signal * naf).sum(dim=1, keepdim=True) / n_non_anchors
    cs_c = (cost_adv - cs_mean) * naf
    sv_c = (svd_signal - sv_mean) * naf
    num = (cs_c * sv_c).sum(dim=1)
    den = (cs_c.norm(dim=1) * sv_c.norm(dim=1)).clamp(min=1e-8)
    cost_svd_corr = (num / den).mean()

    diag = {
        **svd_diag,
        'cost_svd_corr':       cost_svd_corr.detach(),
        'cost_adv_abs_mean':   cost_adv.abs().mean().detach(),
        'svd_signal_abs_mean': svd_signal.abs().mean().detach(),
        'svd_signal_max':      svd_signal.max().detach(),
    }
    return advantage, diag


# ────────────────────────────────────────────────────────────────────────
#  Online AE training: SupCon-style contrastive loss
# ────────────────────────────────────────────────────────────────────────

def contrastive_supcon_loss(z: torch.Tensor, rewards: torch.Tensor,
                             k_pos: int, k_neg: int,
                             temperature: float = 0.1) -> torch.Tensor:
    """
    Supervised-contrastive loss using POMO reward ordering as the supervision.

    For each instance:
      - positives = top-k_pos rollouts by reward (= shortest tours)
      - negatives = top-k_neg rollouts by reverse reward (= longest tours)
      - For each positive anchor, pull other positives close, push negatives away.
    No external quality labels needed — the reward ordering within each instance
    IS the supervision signal.

    Args:
        z:           (B, P, D)
        rewards:     (B, P)         higher = better tour (e.g., -length)
        k_pos:       positives per instance
        k_neg:       negatives per instance
        temperature: InfoNCE temperature (lower = sharper)

    Returns: scalar loss (mean over all anchors).
    """
    B, P, D = z.shape

    _, pos_idx = rewards.topk(k_pos, dim=1, largest=True)             # (B, k_pos)
    _, neg_idx = rewards.topk(k_neg, dim=1, largest=False)             # (B, k_neg)

    bp_pos = torch.arange(B, device=z.device).unsqueeze(1).expand(B, k_pos)
    bp_neg = torch.arange(B, device=z.device).unsqueeze(1).expand(B, k_neg)
    z_pos = F.normalize(z[bp_pos, pos_idx], dim=-1)                    # (B, k_pos, D)
    z_neg = F.normalize(z[bp_neg, neg_idx], dim=-1)                    # (B, k_neg, D)

    # Cosine similarities (B, k_pos, k_pos) and (B, k_pos, k_neg), scaled by 1/τ.
    sim_pp = torch.bmm(z_pos, z_pos.transpose(1, 2)) / temperature
    sim_pn = torch.bmm(z_pos, z_neg.transpose(1, 2)) / temperature

    # Mask self-similarity in positives (an anchor isn't its own positive pair).
    diag = torch.eye(k_pos, device=z.device, dtype=torch.bool)
    sim_pp = sim_pp.masked_fill(diag, float('-inf'))

    # InfoNCE: −log( Σ_pos exp(sim/τ)  /  ( Σ_pos exp + Σ_neg exp ) )
    log_num = torch.logsumexp(sim_pp, dim=2)                            # (B, k_pos)
    log_den = torch.logsumexp(torch.cat([sim_pp, sim_pn], dim=2), dim=2)
    return -(log_num - log_den).mean()


# ────────────────────────────────────────────────────────────────────────
#  AE loader
# ────────────────────────────────────────────────────────────────────────

def load_ae(ckpt_path: str, device: torch.device,
            freeze: bool = True):
    """Load pretrained AE. freeze=True (default) puts it in eval + no-grad
    for use as a fixed reward model; freeze=False keeps it trainable for
    online contrastive fine-tuning.

    Returns (ae, ae_cfg_dict) — ae_cfg_dict is the original AE Config dict
    so the online-trained variant can be saved in the same format.
    """
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg_dict = ckpt["config"]
    cfg = AEConfig(**cfg_dict)
    ae = TourAutoEncoder(cfg).to(device)
    ae.load_state_dict(ckpt["model"])
    if freeze:
        ae.eval()
        for p in ae.parameters():
            p.requires_grad_(False)
    else:
        ae.train()                          # BN running stats track POMO distribution
        # Parameters keep requires_grad=True (default).
    return ae, cfg_dict


# ────────────────────────────────────────────────────────────────────────
#  Training loop
# ────────────────────────────────────────────────────────────────────────

def _train_epoch(model, ae, env, optimizer, ae_optimizer, cfg, device, log_period_sec):
    model.train()
    if cfg.online_ae and ae is not None:
        ae.train()                          # BN running stats follow POMO distribution

    episode = 0
    losses, dists, corr_log, ae_loss_log = [], [], [], []
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

        ae_loss = None
        if cfg.svd_alpha > 0.0 and ae is not None:
            if cfg.online_ae:
                # Encode WITH grad — gradient will flow into AE through the
                # contrastive loss. Advantage uses z.detach() so REINFORCE
                # only updates POMO, not AE.
                z = encode_rollouts(ae, env.node_xy, env.selected_node_list,
                                    knn_k=cfg.svd_knn_k)
                with torch.no_grad():
                    advantage, diag = compute_hybrid_advantage(
                        z=z.detach(), rewards=reward_f,
                        alpha=cfg.svd_alpha, rank=cfg.svd_rank, top_k=cfg.svd_top_k,
                        temperature=cfg.svd_temperature,
                        return_diag=True)
                ae_loss = contrastive_supcon_loss(
                    z=z, rewards=reward_f,
                    k_pos=cfg.ae_k_pos, k_neg=cfg.ae_k_neg,
                    temperature=cfg.ae_temperature,
                )
            else:
                with torch.no_grad():
                    z = encode_rollouts(ae, env.node_xy, env.selected_node_list,
                                        knn_k=cfg.svd_knn_k)
                    advantage, diag = compute_hybrid_advantage(
                        z=z, rewards=reward_f,
                        alpha=cfg.svd_alpha, rank=cfg.svd_rank, top_k=cfg.svd_top_k,
                        temperature=cfg.svd_temperature,
                        return_diag=True)
            corr_log.append(diag['cost_svd_corr'].item())
        else:
            advantage = reward_f - reward_f.mean(dim=1, keepdim=True)

        log_prob = prob_list.log().sum(dim=2)                           # (B, P)
        pomo_loss = -(advantage * log_prob).mean()

        if ae_loss is not None:
            total_loss = pomo_loss + cfg.ae_loss_weight * ae_loss
            ae_loss_log.append(ae_loss.item())
        else:
            total_loss = pomo_loss

        optimizer.zero_grad()
        if ae_optimizer is not None:
            ae_optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        if ae_optimizer is not None:
            ae_optimizer.step()

        losses.append(pomo_loss.item())
        dists.append((-reward_f.max(dim=1).values).mean().item())       # best-of-P per instance

        if time.time() - t_period > log_period_sec or episode >= cfg.train_episodes:
            extra = ""
            if corr_log:
                extra += f"  cost_corr={np.mean(corr_log[-50:]):+.3f}"
            if ae_loss_log:
                extra += f"  ae_loss={np.mean(ae_loss_log[-50:]):+.4f}"
            print(f"  ep:{episode:6d}/{cfg.train_episodes}  "
                  f"loss={np.mean(losses[-50:]):+.4f}  "
                  f"dist={np.mean(dists[-50:]):.4f}{extra}",
                  flush=True)
            t_period = time.time()

    return (np.mean(dists),
            (np.mean(corr_log) if corr_log else None),
            (np.mean(ae_loss_log) if ae_loss_log else None))


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
    ae_cfg_dict = None
    ae_optimizer = None
    if ae_ckpt_path is not None and cfg.svd_alpha > 0.0:
        ae, ae_cfg_dict = load_ae(ae_ckpt_path, device, freeze=not cfg.online_ae)
        mode_str = "ONLINE (contrastive fine-tune)" if cfg.online_ae else "FROZEN"
        print(f"AE loaded from {ae_ckpt_path}  (α={cfg.svd_alpha}, {mode_str})",
              flush=True)
        if cfg.online_ae:
            ae_optimizer = optim.Adam(ae.parameters(), lr=cfg.ae_lr)
            print(f"AE optimizer: Adam(lr={cfg.ae_lr}, "
                  f"loss_weight={cfg.ae_loss_weight}, "
                  f"k_pos={cfg.ae_k_pos}, k_neg={cfg.ae_k_neg}, "
                  f"τ={cfg.ae_temperature})", flush=True)
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

    train_curve, eval_curve, corr_curve, ae_loss_curve = [], [], [], []
    t0 = time.time()

    for ep in range(1, cfg.total_epoch + 1):
        print(f"\n[ep {ep:3d}/{cfg.total_epoch}]  elapsed={time.time()-t0:6.0f}s", flush=True)
        train_dist, train_corr, train_ae_loss = _train_epoch(
            model, ae, env, optimizer, ae_optimizer,
            cfg, device, cfg.log_period_sec)
        eval_dist = _eval_epoch(model, env, cfg, device)
        scheduler.step()

        train_curve.append(train_dist)
        eval_curve.append(eval_dist)
        if train_corr is not None:
            corr_curve.append(train_corr)
        if train_ae_loss is not None:
            ae_loss_curve.append(train_ae_loss)

        suffix = ""
        if train_corr is not None:
            suffix += f"  cost_corr={train_corr:+.3f}"
        if train_ae_loss is not None:
            suffix += f"  ae_loss={train_ae_loss:.4f}"
        print(f"  train_dist={train_dist:.4f}  eval_dist={eval_dist:.4f}{suffix}",
              flush=True)

        torch.save({
            'model':    model.state_dict(),
            'cfg':      cfg.__dict__,
            'epoch':    ep,
            'eval':     eval_dist,
            'train':    train_dist,
        }, run_dir / "ckpt_last.pt")
        if cfg.online_ae and ae is not None:
            torch.save({
                'model':  ae.state_dict(),
                'config': ae_cfg_dict,
                'epoch':  ep,
            }, run_dir / "ae_ckpt_last.pt")

    # Save curves for plotting.
    np.savez(run_dir / "curves.npz",
             train=np.array(train_curve),
             eval=np.array(eval_curve),
             corr=np.array(corr_curve) if corr_curve else np.zeros(0),
             ae_loss=np.array(ae_loss_curve) if ae_loss_curve else np.zeros(0))
    print(f"\nDone. Best eval dist = {min(eval_curve):.4f} (epoch "
          f"{int(np.argmin(eval_curve))+1})", flush=True)
    return train_curve, eval_curve, corr_curve


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--ae-ckpt',  type=str,   default=None,
                   help='AE checkpoint; omit for baseline POMO (α effectively 0).')
    p.add_argument('--alpha',    type=float, default=0.3,
                   help='SVD modulation strength on top of cost_adv.')
    p.add_argument('--svd-temp', type=float, default=1.0,
                   help='softmax temperature for non-anchor SVD weighting '
                        '(lower=sharper, higher=more uniform).')
    p.add_argument('--epochs',   type=int,   default=100)
    p.add_argument('--episodes', type=int,   default=10_000)
    p.add_argument('--save-dir', type=str,   default='checkpoints')
    p.add_argument('--tag',      type=str,   default=None)
    p.add_argument('--online-ae', action='store_true',
                   help='Fine-tune the AE online with a SupCon contrastive loss '
                        'using POMO reward ordering as supervision.')
    p.add_argument('--ae-lr',         type=float, default=1e-5)
    p.add_argument('--ae-loss-weight', type=float, default=0.1)
    p.add_argument('--ae-temp',       type=float, default=0.1)
    p.add_argument('--ae-k-pos',      type=int,   default=20)
    p.add_argument('--ae-k-neg',      type=int,   default=20)
    args = p.parse_args()

    cfg = POMOTrainConfig(
        total_epoch=args.epochs,
        train_episodes=args.episodes,
        svd_alpha=args.alpha,
        svd_temperature=args.svd_temp,
        online_ae=args.online_ae,
        ae_lr=args.ae_lr,
        ae_loss_weight=args.ae_loss_weight,
        ae_temperature=args.ae_temp,
        ae_k_pos=args.ae_k_pos,
        ae_k_neg=args.ae_k_neg,
    )
    train_pomo(ae_ckpt_path=args.ae_ckpt, cfg=cfg,
               save_dir=args.save_dir, run_tag=args.tag)
