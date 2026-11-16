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
