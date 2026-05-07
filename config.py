from dataclasses import dataclass


@dataclass
class Config:
    # TSP
    num_nodes: int = 50

    # Data generation
    num_train_instances: int = 2000
    num_val_instances: int = 200
    num_test_instances: int = 200
    num_good_solutions: int = 2      # NN+2opt per instance
    num_random_solutions: int = 4    # random permutations per instance

    # GNN encoder
    node_feat_dim: int = 2           # (x, y)
    hidden_dim: int = 128
    embedding_dim: int = 64
    num_gnn_layers: int = 4
    dropout: float = 0.1

    # Edge decoder
    decoder_hidden_dim: int = 128

    # Training
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-5
    num_epochs: int = 50
    patience: int = 10

    # SVD subspace
    svd_rank: int = 16

    # Paths
    save_dir: str = "checkpoints"
    seed: int = 42
