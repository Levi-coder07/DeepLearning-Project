import torch
import torch.nn.functional as F
from torch import nn

DIM = 512
PATCH_DIM = 768


def scorer(hidden):
    return nn.Sequential(nn.Linear(2 * DIM, hidden), nn.ReLU(), nn.Linear(hidden, 1))


def attention(heads):
    return nn.MultiheadAttention(DIM, heads, kdim=PATCH_DIM, vdim=PATCH_DIM, batch_first=True)


class GlobalMLP(nn.Module):
    def __init__(self, hidden=1536):
        super().__init__()
        self.mlp = scorer(hidden)

    def forward(self, global_emb, patches, t0, t1):
        t = torch.stack([t0, t1], 1)
        image = global_emb[:, None].expand_as(t)
        return self.mlp(torch.cat([image, t], 2)).squeeze(2)


class AttentionFixed(nn.Module):
    def __init__(self, hidden=256, heads=8):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, DIM) * 0.02)
        self.attention = attention(heads)
        self.mlp = scorer(hidden)

    def forward(self, global_emb, patches, t0, t1):
        t = torch.stack([t0, t1], 1)
        query = F.normalize(self.query, dim=-1).expand(len(t), -1, -1)
        z, _ = self.attention(query, patches, patches)
        z = F.normalize(z, dim=-1).expand_as(t)
        return self.mlp(torch.cat([z, t], 2)).squeeze(2)


class AttentionGuided(nn.Module):
    def __init__(self, hidden=256, heads=8):
        super().__init__()
        self.attention = attention(heads)
        self.mlp = scorer(hidden)

    def forward(self, global_emb, patches, t0, t1):
        t = torch.stack([t0, t1], 1)
        z, _ = self.attention(t, patches, patches)
        z = F.normalize(z, dim=-1)
        return self.mlp(torch.cat([z, t], 2)).squeeze(2)

class MeanPatches(nn.Module):
    """Ablation: attention with uniform weights, i.e. the projected mean of the patches."""

    def __init__(self, hidden=256):
        super().__init__()
        self.value = nn.Linear(PATCH_DIM, DIM)
        self.out = nn.Linear(DIM, DIM)
        self.mlp = scorer(hidden)

    def forward(self, global_emb, patches, t0, t1):
        t = torch.stack([t0, t1], 1)
        z = F.normalize(self.out(self.value(patches.mean(1))), dim=-1)[:, None].expand_as(t)
        return self.mlp(torch.cat([z, t], 2)).squeeze(2)
