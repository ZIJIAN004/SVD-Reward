"""
End-to-end pipeline (per-instance SVD version):
  1. Train GNN autoencoder (quality-agnostic, reconstructs tour edges)
  2. Encode test instances; for each, fit a local SVD on its good-solution
     embeddings and score every tour by -‖orthogonal residual‖
  3. Visualize reward vs tour length, separation, and a sample SVD spectrum
  4. Train POMO TSP policy with SVD-hybrid advantage (the actual experiment;
     side-by-side baseline POMO run for comparison).
  5. Plot eval curves comparing hybrid vs baseline.
"""

import sys
# Force line-buffered stdout so progress prints don't get stuck in pipe/file
# buffers when run under nohup/SLURM/redirection.
sys.stdout.reconfigure(line_buffering=True)

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader
from pathlib import Path

from config import Config
from tsp_data import generate_dataset
from svd_reward import per_instance_reward_np, fit_instance_subspace_np
from train import train


def collect_embeddings_with_iid(model, data_list, batch_size, device):
    loader = DataLoader(data_list, batch_size=batch_size)
    zs, iids = [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            z = model.encode(batch)
            zs.append(z.cpu())
            iids.append(batch.instance_id.cpu())
    return torch.cat(zs, dim=0), torch.cat(iids, dim=0)


def rank_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """ROC AUC via rank-sum formula (O(N log N), no sklearn)."""
    labels = labels.astype(bool)
    n_pos = int(labels.sum())
    n_neg = labels.size - n_pos
    if n_pos == 0 or n_neg == 0:
        return float('nan')
    order = np.argsort(scores, kind='mergesort')
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, scores.size + 1)
    pos_rank_sum = ranks[labels].sum()
    return float((pos_rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def z_class_separation(z: np.ndarray, labels: np.ndarray) -> float:
    """Mean inter-class distance / mean intra-class distance in z space.

    >1 means good and bad embeddings are MORE separated across classes than
    within — a necessary (but not sufficient) condition for SVD reward to
    distinguish quality. ≈1 means quality-agnostic AE collapsed both classes
    onto the same manifold.
    """
    labels = labels.astype(bool)
    z_good = z[labels]
    z_bad  = z[~labels]
    if len(z_good) < 2 or len(z_bad) < 2:
        return float('nan')
    mu_g, mu_b = z_good.mean(axis=0), z_bad.mean(axis=0)
    inter = float(np.linalg.norm(mu_g - mu_b))
    intra_g = float(np.linalg.norm(z_good - mu_g, axis=1).mean())
    intra_b = float(np.linalg.norm(z_bad  - mu_b, axis=1).mean())
    intra = (intra_g + intra_b) / 2
    return inter / max(intra, 1e-8)


def main():
    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Train autoencoder ─────────────────────────────────────
    print("=" * 60)
    print("Step 1 / 3 : Train GNN autoencoder")
    print("=" * 60)
    model, _, _, _ = train(cfg)

    ckpt = torch.load(save_dir / "best_model.pt", map_location=device)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    # ── Step 2: Per-instance SVD on test set ──────────────────────────
    print("\n" + "=" * 60)
    print("Step 2 / 3 : Per-instance SVD evaluation on test set")
    print("=" * 60)
    test_data, test_lengths, test_good, test_iids = generate_dataset(
        cfg.num_test_instances, cfg.num_nodes,
        cfg.num_good_solutions, cfg.num_random_solutions,
        seed=cfg.seed + 2, knn_k=cfg.knn_k,
    )
    test_emb, test_iids_t = collect_embeddings_with_iid(
        model, test_data, cfg.batch_size, device
    )
    test_emb_np = test_emb.numpy()
    test_iids_np = test_iids_t.numpy().reshape(-1)
    test_good_np = np.array(test_good, dtype=bool)
    test_lengths_np = np.array(test_lengths, dtype=np.float32)

    rewards = per_instance_reward_np(
        test_emb_np, test_iids_np, test_good_np, rank=cfg.svd_rank
    )

    g_rew, b_rew = rewards[test_good_np], rewards[~test_good_np]
    g_len, b_len = test_lengths_np[test_good_np], test_lengths_np[~test_good_np]

    # Per-instance correlation between reward and length (in-instance ranking
    # is what GRPO actually needs).
    per_instance_corr = []
    for inst in np.unique(test_iids_np):
        m = test_iids_np == inst
        if m.sum() < 2:
            continue
        c = np.corrcoef(rewards[m], test_lengths_np[m])[0, 1]
        if not np.isnan(c):
            per_instance_corr.append(c)
    cond_corr = float(np.mean(per_instance_corr))
    marg_corr = float(np.corrcoef(rewards, test_lengths_np)[0, 1])

    # ── Additional quality diagnostics (key to the "is the AE quality-aware?"
    # question — without these, low cond_corr can't be diagnosed) ───────────
    # AUC: P(reward_good > reward_bad) over all (g, b) pairs.
    auc_global = rank_auc(rewards, test_good_np)
    # Per-instance AUC: the GRPO-relevant version (in-instance ranking).
    per_inst_aucs = []
    for inst in np.unique(test_iids_np):
        m = test_iids_np == inst
        a = rank_auc(rewards[m], test_good_np[m])
        if not np.isnan(a):
            per_inst_aucs.append(a)
    auc_per_inst = float(np.mean(per_inst_aucs))

    # z-space separation: is the encoder ITSELF separating good vs bad,
    # independent of the SVD step?
    z_sep_global = z_class_separation(test_emb_np, test_good_np)
    per_inst_seps = []
    for inst in np.unique(test_iids_np):
        m = test_iids_np == inst
        s = z_class_separation(test_emb_np[m], test_good_np[m])
        if not np.isnan(s):
            per_inst_seps.append(s)
    z_sep_per_inst = float(np.mean(per_inst_seps))

    print(f"\n{'':─<70}")
    print(f"  Good (NN+2opt)  reward = {g_rew.mean():.4f} ± {g_rew.std():.4f}  "
          f"length = {g_len.mean():.2f} ± {g_len.std():.2f}")
    print(f"  Bad  (random)   reward = {b_rew.mean():.4f} ± {b_rew.std():.4f}  "
          f"length = {b_len.mean():.2f} ± {b_len.std():.2f}")
    print(f"  Reward separation (good − bad mean): {g_rew.mean() - b_rew.mean():.4f}")
    print()
    print(f"  ── reward ↔ length ──")
    print(f"  Marginal     corr(reward, length) = {marg_corr:.4f}")
    print(f"  Per-instance corr(reward, length) = {cond_corr:.4f}  "
          f"(this is what GRPO sees)")
    print()
    print(f"  ── good vs bad classification (is the AE quality-aware?) ──")
    print(f"  AUC(reward; good vs bad) global    = {auc_global:.4f}  "
          f"(0.5=random, 1.0=perfect)")
    print(f"  AUC(reward; good vs bad) per-inst  = {auc_per_inst:.4f}")
    print(f"  z-space separation     global      = {z_sep_global:.4f}  "
          f"(>1 = good/bad separated in z)")
    print(f"  z-space separation     per-inst    = {z_sep_per_inst:.4f}")
    print(f"{'':─<70}")
    print(f"  Read me: cond_corr & per-inst AUC need to be high for SVD")
    print(f"  reward to add real signal beyond cost. If z-sep per-inst ≈ 1,")
    print(f"  the AE is quality-agnostic (training assumption broken) — try")
    print(f"  contrastive loss or training only on good tours.")
    print(f"{'':─<70}")

    # ── Step 3: Visualize ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 3 / 3 : Visualize")
    print("=" * 60)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    axes[0].hist(g_rew, bins=40, alpha=0.7, label="Good (NN+2opt)", color="#2ecc71")
    axes[0].hist(b_rew, bins=40, alpha=0.7, label="Bad (random)", color="#e74c3c")
    axes[0].set_xlabel("Reward (−‖e⊥‖)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Reward Distribution (per-instance SVD)")
    axes[0].legend()

    axes[1].scatter(g_len, g_rew, s=8, alpha=0.4, c="#2ecc71", label="Good")
    axes[1].scatter(b_len, b_rew, s=8, alpha=0.4, c="#e74c3c", label="Bad")
    axes[1].set_xlabel("Tour Length")
    axes[1].set_ylabel("Reward")
    axes[1].set_title(f"Reward vs Length  (marg ρ={marg_corr:.3f}, cond ρ={cond_corr:.3f})")
    axes[1].legend()

    # SVD spectrum of one sample test instance (first one).
    inst0 = int(np.unique(test_iids_np)[0])
    m0 = (test_iids_np == inst0) & test_good_np
    _, _, sv = fit_instance_subspace_np(test_emb_np[m0], rank=cfg.svd_rank)
    axes[2].bar(range(len(sv)), sv, color="#3498db")
    axes[2].set_xlabel("Component index")
    axes[2].set_ylabel("Singular value")
    axes[2].set_title(f"SVD Spectrum (test instance {inst0}, good anchors)")

    plt.tight_layout()
    fig_path = save_dir / "evaluation.png"
    plt.savefig(fig_path, dpi=150)
    print(f"Figure saved → {fig_path}")
    plt.close()

    # ── Step 4: Actual experiment — POMO with SVD-hybrid vs baseline ──
    print("\n" + "=" * 60)
    print("Step 4 / 5 : POMO training (hybrid SVD+cost  vs  baseline cost-only)")
    print("=" * 60)
    from train_pomo import train_pomo, POMOTrainConfig

    ae_ckpt_full = save_dir / "best_model.pt"
    pomo_cfg = POMOTrainConfig(num_nodes=cfg.num_nodes)
    # Light defaults for the sanity-check run inside the pipeline; for the
    # full experiment call train_pomo() directly with longer schedule.
    pomo_cfg.total_epoch    = 30
    pomo_cfg.train_episodes = 5_000
    pomo_cfg.eval_episodes  = 1_000

    print("\n>>> baseline POMO (α=0, cost reward only)")
    pomo_cfg_base = POMOTrainConfig(**{**pomo_cfg.__dict__, 'svd_alpha': 0.0})
    train_b, eval_b, _ = train_pomo(
        ae_ckpt_path=None, cfg=pomo_cfg_base,
        save_dir=str(save_dir / "pomo_runs"), run_tag="baseline_a0.0",
    )

    print("\n>>> hybrid POMO (α=0.5, SVD + cost)")
    pomo_cfg_hyb = POMOTrainConfig(**{**pomo_cfg.__dict__, 'svd_alpha': 0.5})
    train_h, eval_h, corr_h = train_pomo(
        ae_ckpt_path=str(ae_ckpt_full), cfg=pomo_cfg_hyb,
        save_dir=str(save_dir / "pomo_runs"), run_tag="hybrid_a0.5",
    )

    # ── Step 5: Plot eval curves ──────────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 5 / 5 : Comparison plot")
    print("=" * 60)
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    epochs = np.arange(1, len(eval_b) + 1)
    ax[0].plot(epochs, eval_b, marker='o', markersize=3, label='baseline (α=0)',
               color="#888888")
    ax[0].plot(epochs, eval_h, marker='o', markersize=3, label='hybrid (α=0.5)',
               color="#3498db")
    ax[0].set_xlabel("Epoch"); ax[0].set_ylabel("Avg tour distance")
    ax[0].set_title("POMO eval distance")
    ax[0].grid(True); ax[0].legend()

    if len(corr_h) > 0:
        ax[1].plot(np.arange(1, len(corr_h) + 1), corr_h,
                   marker='o', markersize=3, color="#e74c3c")
        ax[1].axhline(0, linestyle='--', color='gray', alpha=0.5)
        ax[1].set_xlabel("Epoch"); ax[1].set_ylabel("cost_svd_corr (per-instance Pearson)")
        ax[1].set_title("SVD ↔ cost signal correlation\n(≈±1 redundant; ≈0 orthogonal)")
        ax[1].set_ylim(-1.05, 1.05); ax[1].grid(True)

    plt.tight_layout()
    pomo_fig = save_dir / "pomo_comparison.png"
    plt.savefig(pomo_fig, dpi=150)
    plt.close()
    print(f"Comparison plot saved → {pomo_fig}")

    print("\n" + "=" * 60)
    print("Pipeline complete.")
    print(f"  AE checkpoint           : {ae_ckpt_full}")
    print(f"  AE evaluation plot      : {fig_path}")
    print(f"  Baseline POMO best eval : {min(eval_b):.4f}")
    print(f"  Hybrid POMO best eval   : {min(eval_h):.4f}")
    print(f"  Comparison plot         : {pomo_fig}")
    print("=" * 60)


if __name__ == "__main__":
    main()
