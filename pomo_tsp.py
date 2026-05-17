"""
Self-contained POMO TSP env + attention model.

Kept inside SVD-Reward so the project can run the full pipeline (pretrain
AE → train POMO with SVD reward → eval) without cross-project imports.

Architecture matches Kwon et al. 2020 (POMO): standard attention encoder
(stacked self-attention over the N node embeddings) + single-step decoder
that conditions on (first_node, last_node) embeddings and runs masked
softmax over remaining nodes. P pomo rollouts per instance, each starting
from a different node so the policy learns symmetry-invariant scoring.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ────────────────────────────────────────────────────────────────────────
#  TSP environment (tensor-based; no PyG, no Python rollout loop)
# ────────────────────────────────────────────────────────────────────────

class TSPEnv:
    """B instances × P pomo rollouts × N step decision process.

    Random uniform [0,1]² coords; tour returns to start. Reward at done =
    −tour_length (so higher = better). Exposes `node_xy` and
    `selected_node_list` for downstream reward (SVD) computation.
    """

    def __init__(self, problem_size: int, pomo_size: int):
        self.N = problem_size
        self.P = pomo_size
        self.B = None
        self.device = None

    def load_problems(self, batch_size: int, device: torch.device):
        self.B = batch_size
        self.device = device
        self.node_xy = torch.rand(batch_size, self.N, 2, device=device)
        self.BATCH_IDX = torch.arange(batch_size, device=device)[:, None].expand(batch_size, self.P)
        self.POMO_IDX  = torch.arange(self.P, device=device)[None, :].expand(batch_size, self.P)

    def reset(self):
        self.selected_count = 0
        self.current_node = None
        self.selected_node_list = torch.zeros((self.B, self.P, 0), dtype=torch.long,
                                               device=self.device)
        self.ninf_mask = torch.zeros(self.B, self.P, self.N, device=self.device)
        return self.node_xy

    def pre_step(self):
        return self._state(), None, False

    def step(self, selected: torch.Tensor):
        """selected: (B, P) long. Returns (state, reward_or_None, done)."""
        self.selected_count += 1
        self.current_node = selected
        self.selected_node_list = torch.cat(
            (self.selected_node_list, selected[:, :, None]), dim=2)
        self.ninf_mask[self.BATCH_IDX, self.POMO_IDX, selected] = float('-inf')

        done = (self.selected_count == self.N)
        reward = -self._tour_length() if done else None
        return self._state(), reward, done

    def _state(self):
        return {
            'BATCH_IDX':    self.BATCH_IDX,
            'POMO_IDX':     self.POMO_IDX,
            'count':        self.selected_count,
            'current_node': self.current_node,
            'ninf_mask':    self.ninf_mask,
        }

    def _tour_length(self) -> torch.Tensor:
        idx = self.selected_node_list[:, :, :, None].expand(-1, -1, -1, 2)
        seq = self.node_xy[:, None, :, :].expand(-1, self.P, -1, -1)
        ordered = seq.gather(dim=2, index=idx)
        rolled = ordered.roll(dims=2, shifts=-1)
        return ((ordered - rolled) ** 2).sum(-1).sqrt().sum(-1)


# ────────────────────────────────────────────────────────────────────────
#  POMO Attention Model
# ────────────────────────────────────────────────────────────────────────

def _reshape_heads(x, head_num):
    B, N, _ = x.shape
    return x.view(B, N, head_num, -1).transpose(1, 2)        # (B, h, N, d_k)


def _mha(q, k, v, mask=None):
    # q: (B, h, P, d_k); k,v: (B, h, N, d_k); mask: (B, P, N) or None
    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / (d_k ** 0.5)   # (B, h, P, N)
    if mask is not None:
        scores = scores + mask[:, None, :, :]                      # broadcast over heads
    attn = F.softmax(scores, dim=-1)
    out = torch.matmul(attn, v)                                    # (B, h, P, d_k)
    return out.transpose(1, 2).contiguous().view(out.size(0), out.size(2), -1)


def _gather(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """x: (B, N, D); idx: (B, P) long → (B, P, D)."""
    B, P = idx.shape
    D = x.size(-1)
    return x.gather(dim=1, index=idx.unsqueeze(-1).expand(B, P, D))


class _EncoderLayer(nn.Module):
    def __init__(self, embedding_dim, head_num, qkv_dim, ff_hidden_dim):
        super().__init__()
        self.head_num = head_num
        self.Wq = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.mha_combine = nn.Linear(head_num * qkv_dim, embedding_dim)
        self.norm1 = nn.InstanceNorm1d(embedding_dim, affine=True)
        self.ff = nn.Sequential(
            nn.Linear(embedding_dim, ff_hidden_dim),
            nn.ReLU(),
            nn.Linear(ff_hidden_dim, embedding_dim),
        )
        self.norm2 = nn.InstanceNorm1d(embedding_dim, affine=True)

    def forward(self, x):
        q = _reshape_heads(self.Wq(x), self.head_num)
        k = _reshape_heads(self.Wk(x), self.head_num)
        v = _reshape_heads(self.Wv(x), self.head_num)
        mh = self.mha_combine(_mha(q, k, v))
        x = self.norm1((x + mh).transpose(1, 2)).transpose(1, 2)
        x = self.norm2((x + self.ff(x)).transpose(1, 2)).transpose(1, 2)
        return x


class POMOModel(nn.Module):
    """Standard POMO TSP model (Kwon et al. 2020).

    Encoder: 6 transformer-style self-attention layers over (B, N, D).
    Decoder: at each step, query = Wq_first(first_node) + Wq_last(last_node);
             attend over all node embeddings with visited-node mask;
             produce softmax over remaining nodes; sample (train) or argmax
             (eval). Returns selected, prob, entropy for each (B, P).
    """

    def __init__(self, embedding_dim=128, encoder_layer_num=6, head_num=8,
                 qkv_dim=16, ff_hidden_dim=512, logit_clipping=10):
        super().__init__()
        self.head_num = head_num
        self.logit_clipping = logit_clipping
        self.sqrt_d = embedding_dim ** 0.5

        self.embed = nn.Linear(2, embedding_dim)
        self.encoder = nn.ModuleList([
            _EncoderLayer(embedding_dim, head_num, qkv_dim, ff_hidden_dim)
            for _ in range(encoder_layer_num)
        ])
        self.Wq_first = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wq_last  = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.mha_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        self._enc = None         # cached encoded nodes (B, N, D)
        self._k = self._v = None # cached projected keys/values per head
        self._first_node = None  # first node per pomo: (B, P)

    def pre_forward(self, coords: torch.Tensor):
        """Encode the instance once. coords: (B, N, 2)."""
        x = self.embed(coords)
        for layer in self.encoder:
            x = layer(x)
        self._enc = x
        self._k = _reshape_heads(self.Wk(x), self.head_num)
        self._v = _reshape_heads(self.Wv(x), self.head_num)
        self._single_k = x.transpose(1, 2)
        self._first_node = None

    def forward(self, state):
        B = state['BATCH_IDX'].size(0)
        P = state['BATCH_IDX'].size(1)
        dev = state['BATCH_IDX'].device

        if state['count'] == 0:
            # POMO branching: each rollout starts from a different node.
            selected = torch.arange(P, device=dev)[None, :].expand(B, -1)
            self._first_node = selected
            prob = torch.ones(B, P, device=dev)
            entropy = torch.zeros(B, P, device=dev)
            return selected, prob, entropy

        enc_first = _gather(self._enc, self._first_node)             # (B, P, D)
        enc_last  = _gather(self._enc, state['current_node'])        # (B, P, D)
        q = _reshape_heads(self.Wq_first(enc_first), self.head_num) \
          + _reshape_heads(self.Wq_last(enc_last), self.head_num)    # (B, h, P, d_k)

        mh = self.mha_combine(_mha(q, self._k, self._v, mask=state['ninf_mask']))  # (B, P, D)
        score = torch.matmul(mh, self._single_k) / self.sqrt_d        # (B, P, N)
        score = self.logit_clipping * torch.tanh(score) + state['ninf_mask']
        probs = F.softmax(score, dim=-1)

        entropy = -(probs * probs.clamp(min=1e-20).log()).sum(dim=-1)

        if self.training:
            # Multinomial sampling; retry if a zero-prob action was somehow drawn.
            while True:
                with torch.no_grad():
                    selected = probs.reshape(B * P, -1).multinomial(1).squeeze(1).view(B, P)
                prob = probs[state['BATCH_IDX'], state['POMO_IDX'], selected]
                if (prob != 0).all():
                    break
        else:
            selected = probs.argmax(dim=-1)
            prob = probs[state['BATCH_IDX'], state['POMO_IDX'], selected]
        return selected, prob, entropy
