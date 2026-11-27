
import argparse
import csv
import html
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def normalize(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def pca(x, components):
    centered = x - x.mean(axis=0)
    covariance = np.einsum("ni,nj->ij", centered, centered, optimize=False)
    values, vectors = np.linalg.eigh(covariance)
    order = np.argsort(values)[::-1][:components]
    projected = np.einsum("ni,ij->nj", centered, vectors[:, order], optimize=False)
    return projected, np.maximum(values[order], 0) / max(values.sum(), 1e-12)


def lexical_features(records, train, vocabulary_size):
    # Fit vocabulary and IDF on training statements only. Deduplicate repeated text.
    tokenize = lambda text: re.findall(r"\b\w+\b", text.lower())
    statements = sorted({row[key] for row in train for key in ("option_0", "option_1")})
    counts = Counter(token for text in statements for token in set(tokenize(text)))
    vocabulary = sorted(counts, key=lambda token: (-counts[token], token))[:vocabulary_size]
    index = {token: i for i, token in enumerate(vocabulary)}
    idf = np.array([np.log((1 + len(statements)) / (1 + counts[token])) + 1 for token in vocabulary])
    x = np.zeros((len(records), len(vocabulary)))
    for i, record in enumerate(records):
        for token, count in Counter(tokenize(record["statement"])).items():
            if token in index:
                x[i, index[token]] = (1 + np.log(count)) * idf[index[token]]
    return normalize(x), vocabulary


def plot_svg(coords, records, path, title, subtitle):
    panels = ("source", "truth") if "statement" in records[0] else ("source",)
    colors = ["#2563eb", "#ef4444", "#16a34a", "#9333ea"]
    low, high = coords.min(axis=0), coords.max(axis=0)
    scaled = (coords - low) / np.maximum(high - low, 1e-12)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="650" viewBox="0 0 1200 650">',
             '<rect width="1200" height="650" fill="white"/>',
             f'<text x="30" y="32" font-family="sans-serif" font-size="22">{html.escape(title)}</text>',
             f'<text x="30" y="58" font-family="sans-serif" font-size="13">{html.escape(subtitle)}</text>']
    for panel, field in enumerate(panels):
        left = 55 + panel * 590
        labels = sorted({str(record[field]) for record in records})
        palette = dict(zip(labels, colors))
        parts.append(f'<text x="{left}" y="95" font-family="sans-serif" font-size="17">Colored by {field}</text>')
        parts.append(f'<rect x="{left}" y="115" width="510" height="440" fill="#f8fafc" stroke="#cbd5e1"/>')
        for xy, record in zip(scaled, records):
            tooltip = html.escape(str(record.get("statement", record["image"])))
            parts.append(f'<circle cx="{left + 10 + xy[0] * 490:.2f}" cy="{545 - xy[1] * 420:.2f}" r="2.5" fill="{palette[str(record[field])]}" fill-opacity="0.45"><title>{tooltip}</title></circle>')
        for i, label in enumerate(labels):
            x = left + i * 165
            parts.append(f'<circle cx="{x}" cy="585" r="5" fill="{palette[label]}"/><text x="{x + 12}" y="590" font-family="sans-serif" font-size="13">{html.escape(label)}</text>')
    parts.append('<text x="30" y="630" font-family="sans-serif" font-size="13">Exploratory projection; overlap or separation does not establish visual grounding. Hover for examples.</text></svg>')
    path.write_text("\n".join(parts))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", choices=["lexical", "clip-text", "clip-image"], default="clip-image")
    parser.add_argument("--method", choices=["pca", "tsne"], default="pca")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--per-source", type=int, default=300, help="charts per source; all pairs for sampled charts are kept")
    parser.add_argument("--vocabulary-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--perplexity", type=float, default=30)
    parser.add_argument("--output", type=Path, default=ROOT / "results/projections")
    args = parser.parse_args()
    if args.per_source < 1 or args.vocabulary_size < 2:
        parser.error("per-source must be positive and vocabulary-size at least 2")
    cache_path = ROOT / "data/features" / ("images.pt" if args.features == "clip-image" else "texts.pt")
    if args.features != "lexical" and not cache_path.exists():
        parser.error(f"Missing {cache_path}. Run feature extraction first, or use --features lexical for a wording-only experiment.")
    with (ROOT / f"data/processed/{args.split}.csv").open() as f:
        rows = list(csv.DictReader(f))
    rng = np.random.default_rng(args.seed)
    images = set()
    for source in sorted({row["source"] for row in rows}):
        paths = sorted({row["image"] for row in rows if row["source"] == source})
        images.update(rng.choice(paths, size=min(args.per_source, len(paths)), replace=False).tolist())
    records = []
    seen_images = set()
    for pair_id, row in enumerate(rows):
        if row["image"] not in images:
            continue
        base = {key: row[key] for key in ("image", "source", "category")}
        if args.features == "clip-image":
            if row["image"] not in seen_images:
                records.append({"image": row["image"], "source": row["source"]})
                seen_images.add(row["image"])
        else:
            for option in (0, 1):
                records.append({**base, "pair_id": pair_id, "option": option,
                                "truth": "correct" if option == int(row["label"]) else "incorrect",
                                "statement": row[f"option_{option}"]})
    if len(records) < 3:
        parser.error("Need at least three observations")
    vocabulary = None
    if args.features == "lexical":
        with (ROOT / "data/processed/train.csv").open() as f:
            train = list(csv.DictReader(f))
        x, vocabulary = lexical_features(records, train, args.vocabulary_size)
    else:
        import torch
        cache = torch.load(cache_path, map_location="cpu", weights_only=True)
        keys, feature_key, record_key = (("paths", "global", "image") if args.features == "clip-image"
                                         else ("statements", "emb", "statement"))
        lookup = {key: i for i, key in enumerate(cache[keys])}
        indices = [lookup[record[record_key]] for record in records]
        x = normalize(cache[feature_key][indices].float().numpy())
    coords, explained = pca(x, 2)
    if args.method == "tsne":
        from sklearn.manifold import TSNE
        if not 0 < args.perplexity < len(records):
            parser.error("perplexity must be positive and below the observation count")
        reduced, _ = pca(x, min(50, x.shape[1], len(x) - 1))
        coords = TSNE(n_components=2, perplexity=args.perplexity, init="pca",
                      learning_rate="auto", random_state=args.seed).fit_transform(reduced)
    args.output.mkdir(parents=True, exist_ok=True)
    stem = f"{args.features}_{args.method}_{args.split}"
    with (args.output / f"{stem}.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[*records[0], "x", "y"])
        writer.writeheader()
        writer.writerows({**record, "x": float(xy[0]), "y": float(xy[1])} for record, xy in zip(records, coords))
    summary = {"features": args.features, "method": args.method, "split": args.split,
               "seed": args.seed, "charts": len(images), "observations": len(records),
               "dimensions": x.shape[1], "pca_explained_variance_ratio": explained.tolist(),
               "perplexity": args.perplexity if args.method == "tsne" else None,
               "vocabulary": vocabulary,
               "note": "Text plots retain repeated statements across pairs; observations are not independent. PCA is fitted on the selected split for exploration, not evaluation."}
    (args.output / f"{stem}.json").write_text(json.dumps(summary, indent=2))
    subtitle = (f"PC1 {explained[0]:.1%}, PC2 {explained[1]:.1%} variance; " if args.method == "pca" else f"perplexity {args.perplexity}; ")
    subtitle += f"{len(records):,} observations / {len(images):,} charts; seed {args.seed}"
    plot_svg(coords, records, args.output / f"{stem}.svg", f"{args.features}: {args.method.upper()} ({args.split})", subtitle)
    print(f"Saved {args.output / stem}.[svg,csv,json]\n{subtitle}")


if __name__ == "__main__":
    main()
