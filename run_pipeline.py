"""
End-to-end pipeline:
  1. Train GNN autoencoder (quality-agnostic, reconstructs tour edges)
  2. Encode good solutions → SVD → positive subspace
  3. Evaluate: projection residual as reward on test set
  4. Visualize reward vs tour length
"""

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader
from pathlib import Path

from config import Config
from tsp_data import generate_dataset
from model import TourAutoEncoder
from svd_reward import SVDReward
from train import train


def collect_embeddings(model, data_list, batch_size, device):
    loader = DataLoader(data_list, batch_size=batch_size)
    zs = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            z = model.encode(batch)
            zs.append(z.cpu())
    return torch.cat(zs, dim=0)


def main():
    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = Path(cfg.save_dir)

    # ── Step 1: Train autoencoder ─────────────────────────────────────
    print("=" * 60)
    print("Step 1 / 4 : Train GNN autoencoder")
    print("=" * 60)
    model, train_data, train_good = train(cfg)

    ckpt = torch.load(save_dir / "best_model.pt", map_location=device)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    # ── Step 2: Build SVD subspace from good training solutions ───────
    print("\n" + "=" * 60)
    print("Step 2 / 4 : Build SVD subspace")
    print("=" * 60)
    good_data = [d for d, g in zip(train_data, train_good) if g]
    good_emb = collect_embeddings(model, good_data, cfg.batch_size, device)
    print(f"Good-solution embeddings: {good_emb.shape}")

    svd = SVDReward(rank=cfg.svd_rank)
    svd.fit(good_emb)
    svd.save(save_dir / "svd_reward.npz")

    # ── Step 3: Evaluate on test set ──────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 3 / 4 : Evaluate on test data")
    print("=" * 60)
    test_data, test_lengths, test_good = generate_dataset(
        cfg.num_test_instances, cfg.num_nodes,
        cfg.num_good_solutions, cfg.num_random_solutions,
        seed=cfg.seed + 2,
    )
    test_emb = collect_embeddings(model, test_data, cfg.batch_size, device)
    rewards = svd.reward(test_emb)

    test_lengths = np.array(test_lengths)
    test_good = np.array(test_good)

    g_rew, b_rew = rewards[test_good], rewards[~test_good]
    g_len, b_len = test_lengths[test_good], test_lengths[~test_good]

    print(f"\n{'':─<55}")
    print(f"  Good (NN+2opt)  reward = {g_rew.mean():.4f} ± {g_rew.std():.4f}  "
          f"length = {g_len.mean():.2f} ± {g_len.std():.2f}")
    print(f"  Bad  (random)   reward = {b_rew.mean():.4f} ± {b_rew.std():.4f}  "
          f"length = {b_len.mean():.2f} ± {b_len.std():.2f}")
    corr = np.corrcoef(rewards, test_lengths)[0, 1]
    print(f"  Corr(reward, length) = {corr:.4f}  "
          f"(negative = reward prefers shorter tours)")
    print(f"{'':─<55}")

    sep_gap = g_rew.mean() - b_rew.mean()
    print(f"  Reward separation (good − bad mean): {sep_gap:.4f}")

    # ── Step 4: Visualize ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 4 / 4 : Visualize")
    print("=" * 60)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    axes[0].hist(g_rew, bins=40, alpha=0.7, label="Good (NN+2opt)", color="#2ecc71")
    axes[0].hist(b_rew, bins=40, alpha=0.7, label="Bad (random)", color="#e74c3c")
    axes[0].set_xlabel("Reward (−‖e⊥‖)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Reward Distribution")
    axes[0].legend()

    axes[1].scatter(g_len, g_rew, s=8, alpha=0.4, c="#2ecc71", label="Good")
    axes[1].scatter(b_len, b_rew, s=8, alpha=0.4, c="#e74c3c", label="Bad")
    axes[1].set_xlabel("Tour Length")
    axes[1].set_ylabel("Reward")
    axes[1].set_title(f"Reward vs Length  (ρ = {corr:.3f})")
    axes[1].legend()

    sv = svd.singular_values
    axes[2].bar(range(len(sv)), sv, color="#3498db")
    axes[2].set_xlabel("Component index")
    axes[2].set_ylabel("Singular value")
    axes[2].set_title("SVD Spectrum (good solutions)")

    plt.tight_layout()
    fig_path = save_dir / "evaluation.png"
    plt.savefig(fig_path, dpi=150)
    print(f"Figure saved → {fig_path}")
    plt.close()

    # ── Summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Pipeline complete.")
    print(f"  Model checkpoint : {save_dir / 'best_model.pt'}")
    print(f"  SVD reward file  : {save_dir / 'svd_reward.npz'}")
    print(f"  Evaluation plot  : {fig_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
