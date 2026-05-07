import numpy as np
import torch
from pathlib import Path


class SVDReward:
    """
    Build a low-rank subspace from good-solution embeddings via SVD.
    Reward = negative orthogonal residual norm (closer to subspace → higher reward).
    """

    def __init__(self, rank: int = 16):
        self.rank = rank
        self.mean: np.ndarray = None
        self.basis: np.ndarray = None          # (k, D)
        self.singular_values: np.ndarray = None

    # ------------------------------------------------------------------
    def fit(self, embeddings) -> "SVDReward":
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.detach().cpu().numpy()

        self.mean = embeddings.mean(axis=0)
        centered = embeddings - self.mean

        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        k = min(self.rank, len(S))
        self.basis = Vt[:k]
        self.singular_values = S[:k]

        explained = np.sum(S[:k] ** 2) / np.sum(S ** 2)
        print(f"SVD  rank={k}  explained_var={explained:.4f}  "
              f"top-3 σ={S[:3].round(2)}")
        return self

    # ------------------------------------------------------------------
    def residual(self, embeddings) -> np.ndarray:
        """‖e⊥‖ for each embedding — lower is closer to the good-solution subspace."""
        if isinstance(embeddings, torch.Tensor):
            emb = embeddings.detach().cpu().numpy()
        else:
            emb = np.asarray(embeddings)

        centered = emb - self.mean
        proj = (centered @ self.basis.T) @ self.basis   # (N, D)
        res = centered - proj
        return np.linalg.norm(res, axis=1)

    def reward(self, embeddings):
        """Higher = better.  Returns numpy array or torch Tensor matching input type."""
        r = self.residual(embeddings)
        neg = -r
        if isinstance(embeddings, torch.Tensor):
            return torch.tensor(neg, dtype=torch.float32, device=embeddings.device)
        return neg

    # ------------------------------------------------------------------
    def reward_for_grpo(self, embeddings, temperature: float = 1.0):
        """
        Normalized reward suitable for GRPO advantage computation.
        Centers and scales within the batch so that mean=0, std≈1.
        """
        r = self.reward(embeddings)
        if isinstance(r, torch.Tensor):
            return (r - r.mean()) / (r.std() + 1e-8) * temperature
        return (r - r.mean()) / (r.std() + 1e-8) * temperature

    # ------------------------------------------------------------------
    def save(self, path):
        np.savez(
            Path(path),
            mean=self.mean,
            basis=self.basis,
            singular_values=self.singular_values,
            rank=np.array(self.rank),
        )

    @classmethod
    def load(cls, path) -> "SVDReward":
        data = np.load(path)
        obj = cls(rank=int(data["rank"]))
        obj.mean = data["mean"]
        obj.basis = data["basis"]
        obj.singular_values = data["singular_values"]
        return obj
