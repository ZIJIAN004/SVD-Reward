import torch
import torch.nn as nn
from torch_geometric.nn import GINConv, global_mean_pool


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int,
                 num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        layers = []
        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.BatchNorm1d(dims[i + 1]))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class GINStack(nn.Module):
    """Stack of GIN layers + BN + ReLU + Dropout, returning per-node features."""

    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()
        self.proj = nn.Linear(in_dim, hidden_dim)
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.convs.append(GINConv(mlp))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        h = self.proj(x)
        for conv, bn in zip(self.convs, self.bns):
            h = conv(h, edge_index)
            h = bn(h)
            h = torch.relu(h)
            h = self.drop(h)
        return h


class InstanceEncoder(nn.Module):
    """Stream 1: encode the problem instance (KNN graph over coords).
    Per-node embeddings carry instance-aware geometry, fed as init features into Stream 2.
    """

    def __init__(self, node_feat_dim: int, hidden_dim: int,
                 num_layers: int, dropout: float):
        super().__init__()
        self.gnn = GINStack(node_feat_dim, hidden_dim, num_layers, dropout)

    def forward(self, x, instance_edge_index):
        return self.gnn(x, instance_edge_index)


class TourEncoder(nn.Module):
    """Stream 2: propagate instance-aware node features along the tour Hamiltonian
    cycle, then mean-pool to a graph-level embedding z.
    """

    def __init__(self, in_dim: int, hidden_dim: int, embedding_dim: int,
                 num_layers: int, dropout: float):
        super().__init__()
        self.gnn = GINStack(in_dim, hidden_dim, num_layers, dropout)
        self.head = MLP(hidden_dim, hidden_dim, embedding_dim,
                        num_layers=2, dropout=dropout)

    def forward(self, h_node, tour_edge_index, batch):
        h = self.gnn(h_node, tour_edge_index)
        z = global_mean_pool(h, batch)
        return self.head(z)


class EdgeDecoder(nn.Module):
    """Predict tour edges from instance-aware node features and graph embedding z."""

    def __init__(self, node_feat_dim: int, z_dim: int, hidden_dim: int,
                 dropout: float = 0.1):
        super().__init__()
        self.node_mlp = MLP(node_feat_dim + z_dim, hidden_dim, hidden_dim,
                            num_layers=2, dropout=dropout)

    def forward(self, z, node_feat, batch):
        z_broad = z[batch]
        h = self.node_mlp(torch.cat([node_feat, z_broad], dim=-1))

        scores = []
        for b in range(z.shape[0]):
            mask = batch == b
            hb = h[mask]
            n = hb.shape[0]
            ri, ci = torch.triu_indices(n, n, offset=1, device=h.device)
            s = (hb[ri] * hb[ci]).sum(dim=-1)
            scores.append(s)
        return torch.cat(scores)


class TourAutoEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.instance_encoder = InstanceEncoder(
            cfg.node_feat_dim, cfg.hidden_dim,
            cfg.num_instance_gnn_layers, cfg.dropout,
        )
        self.tour_encoder = TourEncoder(
            cfg.hidden_dim, cfg.hidden_dim, cfg.embedding_dim,
            cfg.num_gnn_layers, cfg.dropout,
        )
        self.decoder = EdgeDecoder(
            cfg.hidden_dim, cfg.embedding_dim,
            cfg.decoder_hidden_dim, cfg.dropout,
        )

    def forward(self, data):
        h_inst = self.instance_encoder(data.x, data.instance_edge_index)
        z = self.tour_encoder(h_inst, data.edge_index, data.batch)
        scores = self.decoder(z, h_inst, data.batch)
        return z, scores

    @torch.no_grad()
    def encode(self, data):
        h_inst = self.instance_encoder(data.x, data.instance_edge_index)
        return self.tour_encoder(h_inst, data.edge_index, data.batch)
