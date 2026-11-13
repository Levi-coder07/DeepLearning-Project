import copy

import pandas as pd
import torch
import torch.nn.functional as F

from src.data.prepare import OUT_DIR
from src.features.extract import FEATURES_DIR


class Features:
    def init(self, with_images=True, device="cuda"):
        texts = torch.load(FEATURES_DIR / "texts.pt")
        self.text_index = {s: i for i, s in enumerate(texts["statements"])}
        self.text_emb = texts["emb"].to(device)
        self.image_index, self.global_emb, self.patches = {}, None, None
        if with_images:
            images = torch.load(FEATURES_DIR / "images.pt")
            self.image_index = {p: i for i, p in enumerate(images["paths"])}
            self.global_emb = images["global"].to(device)
            self.patches = images["patches"].to(device)
        self.device = device

    def split(self, name):
        df = pd.read_csv(OUT_DIR / f"{name}.csv")
        rows = pd.DataFrame({
            "image": df.image.map(self.image_index).fillna(-1),
            "option_0": df.option_0.map(self.text_index),
            "option_1": df.option_1.map(self.text_index),
            "label": df.label,
        })
        return torch.tensor(rows.to_numpy(), dtype=torch.long, device=self.device)

    def inputs(self, rows):
        image, option_0, option_1 = rows[:, 0], rows[:, 1], rows[:, 2]
        global_emb = self.global_emb[image].float() if self.global_emb is not None else None
        patches = self.patches[image].float() if self.patches is not None else None
        return global_emb, patches, self.text_emb[option_0].float(), self.text_emb[option_1].float()


@torch.no_grad()
def predict(model, features, rows, batch_size=1024):
    model.eval()
    return torch.cat([model(features.inputs(batch)).argmax(1) for batch in rows.split(batch_size)])


def fit(make_model, features, epochs=20, lr=1e-3, batch_size=256, seed=0):
    torch.manual_seed(seed)
    model = make_model().to(features.device)
    train, val = features.split("train"), features.split("val")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    best_acc, best_state = 0, None

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        order = torch.randperm(len(train), device=features.device)
        for rows in train[order].split(batch_size):
            loss = F.cross_entropy(model(features.inputs(rows)), rows[:, 3])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(rows)

        acc = (predict(model, features, val) == val[:, 3]).float().mean().item()
        print(f"epoch {epoch + 1}: loss {total_loss / len(train):.4f}, val acc {acc:.4f}")
        if acc > best_acc:
            best_acc, best_state = acc, copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    return model