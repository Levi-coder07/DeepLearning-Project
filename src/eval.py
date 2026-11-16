import numpy as np
import pandas as pd


def accuracy(df, pred=None):
    """Top-1 accuracy (%) overall, per source and per category.

    `df` is a split (rows of data/processed/<split>.csv); `pred` holds the chosen
    statement (0 or 1) per row and defaults to the `pred` column of `df`.
    """
    pred = df.pred if pred is None else pd.Series(pred, index=df.index)
    if len(pred) != len(df) or not pred.isin([0, 1]).all():
        raise ValueError("pred must hold one 0/1 prediction per row of df")
    correct = pred == df.label
    return pd.concat([
        pd.Series({"all": correct.mean()}),
        correct.groupby(df.source).mean(),
        correct.groupby(df.category).mean(),
    ]).mul(100).round(1)


def correct(df, model, seeds=(0, 1, 2)):
    """Fraction of seeds (0 to 1) that choose the right statement, per row.

    `df` holds the `<model>_seed<s>` prediction columns of results/predictions_test.csv.
    """
    return np.mean([df[f"{model}_seed{s}"] == df.label for s in seeds], axis=0)


def paired_bootstrap(df, a, b, n=10_000, seed=0):
    """Accuracy difference a - b (points) with a 95% CI and a two-sided p-value.

    Both models are scored on the same rows, averaged over seeds. Statements of
    the same chart are not independent, so the bootstrap resamples charts
    (the `image` column), not rows.
    """
    diff = pd.Series(correct(df, a) - correct(df, b), index=df.index)
    by_chart = diff.groupby(df.image.values).agg(["sum", "count"]).to_numpy()
    idx = np.random.default_rng(seed).integers(0, len(by_chart), (n, len(by_chart)))
    sums = by_chart[idx].sum(1)
    boot = sums[:, 0] / sums[:, 1] * 100
    p = min(1.0, 2 * min((boot <= 0).mean(), (boot >= 0).mean()))
    return pd.Series({"diff": diff.mean() * 100, "low": np.percentile(boot, 2.5),
                      "high": np.percentile(boot, 97.5), "p": max(p, 1 / n)})
