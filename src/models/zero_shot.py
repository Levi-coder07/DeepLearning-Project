import pandas as pd
import torch
import torch.nn.functional as F

from src.data.prepare import OUT_DIR, ROOT
from src.features.extract import FEATURES_DIR

RESULTS_DIR = ROOT / "results"


def lookup(features, keys, values):
    index = {key: i for i, key in enumerate(keys)}
    rows = torch.tensor(values.map(index).to_numpy())
    return F.normalize(features[rows].float(), dim=-1)


def predict(df, images, texts):
    image = lookup(images["global"], images["paths"], df.image)
    option_0 = lookup(texts["emb"], texts["statements"], df.option_0)
    option_1 = lookup(texts["emb"], texts["statements"], df.option_1)
    return ((image * option_1).sum(-1) > (image * option_0).sum(-1)).long().numpy()


def accuracy(df):
    correct = df.pred == df.label
    return pd.concat([
        pd.Series({"all": correct.mean()}),
        correct.groupby(df.source).mean(),
        correct.groupby(df.category).mean(),
    ]).mul(100).round(1)


def main():
    images = torch.load(FEATURES_DIR / "images.pt")
    texts = torch.load(FEATURES_DIR / "texts.pt")
    RESULTS_DIR.mkdir(exist_ok=True)

    table = {}
    for split in ["val", "test"]:
        df = pd.read_csv(OUT_DIR / f"{split}.csv")
        df["pred"] = predict(df, images, texts)
        df.to_csv(RESULTS_DIR / f"zero_shot_{split}.csv", index=False)
        table[split] = accuracy(df)

    print(pd.DataFrame(table))


if __name__ == "__main__":
    main()
