import torch
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

    def forward(self, globalemb, patches, t0, t1):
        t = torch.stack([t0, t1], 1)
        z, _ = self.attention(self.query.expand(len(t), -1, -1), patches, patches)
        return self.mlp(torch.cat([z.expand_as(t), t], 2)).squeeze(2)


class AttentionGuided(nn.Module):
    def __init__(self, hidden=256, heads=8):
        super().__init__()
        self.attention = attention(heads)
        self.mlp = scorer(hidden)

    def forward(self, globalemb, patches, t0, t1):
        t = torch.stack([t0, t1], 1)
        z, _ = self.attention(t, patches, patches)
        return self.mlp(torch.cat([z, t], 2)).squeeze(2)