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


class TourEncoder(nn.Module):
    """GIN encoder: tour graph → graph-level embedding z."""

    def __init__(self, node_feat_dim: int, hidden_dim: int, embedding_dim: int,
                 num_layers: int = 4, dropout: float = 0.1):
        super().__init__()
        self.node_proj = nn.Linear(node_feat_dim, hidden_dim)

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
        self.head = MLP(hidden_dim, hidden_dim, embedding_dim,
                        num_layers=2, dropout=dropout)

    def forward(self, x, edge_index, batch):
        h = self.node_proj(x)
        for conv, bn in zip(self.convs, self.bns):
            h = conv(h, edge_index)
            h = bn(h)
            h = torch.relu(h)
            h = self.drop(h)
        z = global_mean_pool(h, batch)      # (B, hidden_dim)
        z = self.head(z)                    # (B, embedding_dim)
        return z


class EdgeDecoder(nn.Module):
    """From graph embedding z + raw coordinates, predict tour edges (inner-product)."""

    def __init__(self, coord_dim: int, z_dim: int, hidden_dim: int,
                 dropout: float = 0.1):
        super().__init__()
        self.node_mlp = MLP(coord_dim + z_dim, hidden_dim, hidden_dim,
                            num_layers=2, dropout=dropout)

    def forward(self, z, coords, batch):
        z_broad = z[batch]                           # (N_total, z_dim)
        h = self.node_mlp(torch.cat([coords, z_broad], dim=-1))  # (N_total, H)

        scores = []
        for b in range(z.shape[0]):
            mask = batch == b
            hb = h[mask]                             # (n, H)
            n = hb.shape[0]
            ri, ci = torch.triu_indices(n, n, offset=1, device=h.device)
            s = (hb[ri] * hb[ci]).sum(dim=-1)        # (n*(n-1)/2,)
            scores.append(s)
        return torch.cat(scores)                     # (B * n*(n-1)/2,)


class TourAutoEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.encoder = TourEncoder(
            cfg.node_feat_dim, cfg.hidden_dim, cfg.embedding_dim,
            cfg.num_gnn_layers, cfg.dropout,
        )
        self.decoder = EdgeDecoder(
            cfg.node_feat_dim, cfg.embedding_dim,
            cfg.decoder_hidden_dim, cfg.dropout,
        )

    def forward(self, data):
        z = self.encoder(data.x, data.edge_index, data.batch)
        scores = self.decoder(z, data.x, data.batch)
        return z, scores

    @torch.no_grad()
    def encode(self, data):
        return self.encoder(data.x, data.edge_index, data.batch)
