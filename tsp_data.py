import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected


def generate_instance(n: int, rng: np.random.Generator = None) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()
    return rng.random((n, 2)).astype(np.float32)


def tour_length(coords: np.ndarray, tour: np.ndarray) -> float:
    rolled = np.roll(tour, -1)
    return float(np.sum(np.linalg.norm(coords[tour] - coords[rolled], axis=1)))


def nearest_neighbor(coords: np.ndarray, start: int = 0) -> np.ndarray:
    n = len(coords)
    visited = np.zeros(n, dtype=bool)
    tour = [start]
    visited[start] = True
    for _ in range(n - 1):
        cur = tour[-1]
        dists = np.linalg.norm(coords - coords[cur], axis=1)
        dists[visited] = np.inf
        nxt = int(np.argmin(dists))
        tour.append(nxt)
        visited[nxt] = True
    return np.array(tour)


def two_opt(coords: np.ndarray, tour: np.ndarray, max_iter: int = 100) -> np.ndarray:
    tour = tour.copy()
    n = len(tour)
    improved = True
    it = 0
    while improved and it < max_iter:
        improved = False
        it += 1
        for i in range(n - 1):
            for j in range(i + 2, n):
                if j == n - 1 and i == 0:
                    continue
                ci, ci1 = coords[tour[i]], coords[tour[i + 1]]
                cj, cj1 = coords[tour[j]], coords[tour[(j + 1) % n]]
                d_old = np.linalg.norm(ci - ci1) + np.linalg.norm(cj - cj1)
                d_new = np.linalg.norm(ci - cj) + np.linalg.norm(ci1 - cj1)
                if d_new < d_old - 1e-10:
                    tour[i + 1:j + 1] = tour[i + 1:j + 1][::-1]
                    improved = True
    return tour


def random_tour(n: int, rng: np.random.Generator = None) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()
    tour = np.arange(n)
    rng.shuffle(tour)
    return tour


def tour_to_edge_index(tour: np.ndarray) -> torch.Tensor:
    """Hamiltonian cycle → undirected edge_index (2, 2n)."""
    n = len(tour)
    src, dst = [], []
    for i in range(n):
        u, v = int(tour[i]), int(tour[(i + 1) % n])
        src += [u, v]
        dst += [v, u]
    return torch.tensor([src, dst], dtype=torch.long)


def knn_edge_index(coords: np.ndarray, k: int) -> torch.Tensor:
    """Undirected KNN graph over node coordinates (used by the instance stream)."""
    n = len(coords)
    k = min(k, n - 1)
    dists = np.linalg.norm(coords[:, None] - coords[None, :], axis=-1)
    np.fill_diagonal(dists, np.inf)
    knn_idx = np.argsort(dists, axis=1)[:, :k]
    src = np.repeat(np.arange(n), k)
    dst = knn_idx.flatten()
    edge_index = torch.tensor(np.stack([src, dst]), dtype=torch.long)
    return to_undirected(edge_index)


def compute_edge_labels(tour: np.ndarray, n: int) -> torch.Tensor:
    """Binary labels for all upper-triangle edges: 1 if in tour, 0 otherwise."""
    tour_edges = set()
    for i in range(n):
        u, v = int(tour[i]), int(tour[(i + 1) % n])
        tour_edges.add((min(u, v), max(u, v)))
    labels = []
    for i in range(n):
        for j in range(i + 1, n):
            labels.append(1.0 if (i, j) in tour_edges else 0.0)
    return torch.tensor(labels, dtype=torch.float32)


def make_pyg_data(
    coords: np.ndarray,
    tour: np.ndarray,
    instance_id: int = -1,
    knn_k: int = 10,
) -> Data:
    n = len(tour)
    return Data(
        x=torch.tensor(coords, dtype=torch.float32),
        edge_index=tour_to_edge_index(tour),
        instance_edge_index=knn_edge_index(coords, k=knn_k),
        target=compute_edge_labels(tour, n),
        tour_len=torch.tensor([tour_length(coords, tour)], dtype=torch.float32),
        instance_id=torch.tensor([instance_id], dtype=torch.long),
    )


def generate_dataset(
    num_instances: int,
    num_nodes: int,
    num_good: int = 50,
    num_random: int = 50,
    seed: int = 42,
    knn_k: int = 10,
    progress_every: int = 10,
):
    """
    Returns:
        data_list:    list[Data]   each carries .instance_id (long tensor)
        lengths:      list[float]
        is_good:      list[bool]   True for NN+2opt solutions, False for random
        instance_ids: list[int]    which instance each tour belongs to

    Prints a progress line every `progress_every` instances so the script
    doesn't look dead during the slow 2opt phase (set to 0 to silence).
    """
    import time as _time
    rng = np.random.default_rng(seed)
    data_list, lengths, is_good, instance_ids = [], [], [], []
    t0 = _time.time()

    for inst in range(num_instances):
        coords = generate_instance(num_nodes, rng)

        # When num_good > num_nodes, allow repeated start points (different
        # tie-breaks during 2opt may still produce diverse local optima).
        replace = num_good > num_nodes
        starts = rng.choice(num_nodes, size=num_good, replace=replace)
        for s in starts:
            tour = nearest_neighbor(coords, start=int(s))
            tour = two_opt(coords, tour)
            d = make_pyg_data(coords, tour, instance_id=inst, knn_k=knn_k)
            data_list.append(d)
            lengths.append(d.tour_len.item())
            is_good.append(True)
            instance_ids.append(inst)

        for _ in range(num_random):
            tour = random_tour(num_nodes, rng)
            d = make_pyg_data(coords, tour, instance_id=inst, knn_k=knn_k)
            data_list.append(d)
            lengths.append(d.tour_len.item())
            is_good.append(False)
            instance_ids.append(inst)

        if progress_every and ((inst + 1) % progress_every == 0 or inst + 1 == num_instances):
            elapsed = _time.time() - t0
            rate = (inst + 1) / max(elapsed, 1e-6)
            eta = (num_instances - inst - 1) / max(rate, 1e-6)
            print(f"  [data] {inst+1:4d}/{num_instances} instances  "
                  f"elapsed={elapsed:6.1f}s  ETA={eta:6.1f}s",
                  flush=True)

    return data_list, lengths, is_good, instance_ids
