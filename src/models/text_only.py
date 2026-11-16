import pandas as pd
import torch
from torch import nn

from src.data.prepare import OUT_DIR
from src.eval import accuracy
from src.train import Features, fit, predict, save


class TextOnly(nn.Module):
    def __init__(self, hidden=256):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(512, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, global_emb, patches, t0, t1):
        return self.mlp(torch.stack([t0, t1], 1)).squeeze(2)


def main():
    features = Features(with_images=False, device="cpu")
    model = fit(TextOnly, features, seed=0)
    save(model, "text_only", 0)
    test = pd.read_csv(OUT_DIR / "test.csv")
    print(accuracy(test, predict(model, features, features.split("test")).cpu().numpy()))


if __name__ == "__main__":
    main()
