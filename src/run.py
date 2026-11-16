import argparse
import time

import pandas as pd
import torch

from src.data.prepare import OUT_DIR
from src.models.heads import AttentionFixed, AttentionGuided, GlobalMLP, MeanPatches
from src.models.text_only import TextOnly
from src.eval import accuracy
from src.models.zero_shot import RESULTS_DIR
from src.train import Features, fit, predict, save

MODELS = {
    "text_only": TextOnly,
    "global_mlp": GlobalMLP,
    "attention_fixed": AttentionFixed,
    "attention_guided": AttentionGuided,
    "mean_patches": MeanPatches,
}


def shuffle_images(rows, seed=0):
    """Give every chart the features of another chart (no chart keeps its own)."""
    charts = rows[:, 0].unique()
    generator = torch.Generator().manual_seed(seed)
    charts = charts[torch.randperm(len(charts), generator=generator).to(charts.device)]
    mapping = torch.empty(int(charts.max()) + 1, dtype=torch.long, device=rows.device)
    mapping[charts] = charts.roll(1)
    shuffled = rows.clone()
    shuffled[:, 0] = mapping[rows[:, 0]]
    return shuffled


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=list(MODELS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    args = parser.parse_args()

    features = Features()
    test = pd.read_csv(OUT_DIR / "test.csv")
    test_rows = features.split("test")
    shuffled_rows = shuffle_images(test_rows)
    cuda = features.device == "cuda"

    results_path, predictions_path = RESULTS_DIR / "models.csv", RESULTS_DIR / "predictions_test.csv"
    predictions = pd.read_csv(predictions_path) if predictions_path.exists() else test.copy()
    results = []
    for name in args.models:
        for seed in args.seeds:
            print(f"== {name}, seed {seed}")
            if cuda:
                torch.cuda.reset_peak_memory_stats()
                base = torch.cuda.memory_allocated()
            start = time.time()
            model = fit(MODELS[name], features, seed=seed)
            seconds = time.time() - start
            memory = (torch.cuda.max_memory_allocated() - base) / 2**20 if cuda else float("nan")
            save(model, name, seed)

            pred = predict(model, features, test_rows).cpu().numpy()
            predictions[f"{name}_seed{seed}"] = pred
            shuffled = (predict(model, features, shuffled_rows) == test_rows[:, 3]).float().mean().item() * 100
            params = sum(p.numel() for p in model.parameters())
            results.append({
                "model": name, "seed": seed, "params": params, "seconds": seconds, "memory_mb": memory,
                **accuracy(test, pred), "shuffled": round(shuffled, 1),
            })

    # Keep earlier runs of other models/seeds; replace the ones run now.
    results = pd.DataFrame(results)
    if results_path.exists():
        old = pd.read_csv(results_path)
        rerun = old.set_index(["model", "seed"]).index.isin(results.set_index(["model", "seed"]).index)
        results = pd.concat([old[~rerun], results], ignore_index=True)
    order = {name: i for i, name in enumerate(MODELS)}
    results = results.sort_values(["model", "seed"], key=lambda c: c.map(order) if c.name == "model" else c)
    RESULTS_DIR.mkdir(exist_ok=True)
    results.to_csv(results_path, index=False)
    predictions.to_csv(predictions_path, index=False)

    summary = results.drop(columns="seed").groupby("model", sort=False)
    print(summary.mean().round(1).T.to_string())
    print(summary.std().round(1).T.to_string())


if __name__ == "__main__":
    main()
