import argparse

import pandas as pd

from src.data.prepare import OUT_DIR
from src.models.heads import AttentionFixed, AttentionGuided, GlobalMLP
from src.models.zero_shot import RESULTS_DIR, accuracy
from src.train import Features, fit, predict, save

MODELS = {
    "global_mlp": GlobalMLP,
    "attention_fixed": AttentionFixed,
    "attention_guided": AttentionGuided,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=list(MODELS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    args = parser.parse_args()

    features = Features()
    test = pd.read_csv(OUT_DIR / "test.csv")
    test_rows = features.split("test")

    results = []
    for name in args.models:
        for seed in args.seeds:
            print(f"== {name}, seed {seed}")
            model = fit(MODELS[name], features, seed=seed)
            save(model, name, seed)
            test["pred"] = predict(model, features, test_rows).cpu().numpy()
            params = sum(p.numel() for p in model.parameters())
            results.append({"model": name, "seed": seed, "params": params, **accuracy(test)})

    results = pd.DataFrame(results)
    RESULTS_DIR.mkdir(exist_ok=True)
    results.to_csv(RESULTS_DIR / "models.csv", index=False)
    print(results.groupby("model").mean(numeric_only=True).drop(columns="seed").round(1))


if __name__ == "__main__":
    main()