import pandas as pd
import torch
from torch import nn

from src.data.prepare import OUT_DIR
from src.models.zero_shot import accuracy
from src.train import Features, fit, predict


class TextOnly(nn.Module):
    def init(self, hidden=256):
        super().init()
        self.mlp = nn.Sequential(nn.Linear(512, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, global_emb, patches, t0, t1):
        return self.mlp(torch.stack([t0, t1], 1)).squeeze(2)


def main():
    features = Features(with_images=False)
    model = fit(TextOnly, features, seed=0)
    test = pd.read_csv(OUT_DIR / "test.csv")
    test["pred"] = predict(model, features, features.split("test")).cpu().numpy()
    print(accuracy(test))


if name == "main":
    main()