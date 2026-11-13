import csv
import random
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_CSV = ROOT / "data/raw/data/chart_mix_reproduce.csv"
OUT_DIR = ROOT / "data/processed"

SOURCES = ["chartqa", "figureqa", "dvqa"]
CHARTS_PER_SOURCE = 3000
SEED = 0

COMPARISON = re.compile(
    r"\b(than|most|least|median|maximum|minimum|highest|lowest|largest|smallest"
    r"|biggest|greatest|roughest|smoothest)\b"
)
NUMBER = re.compile(r"\d|\b(zero|one|two|three|four|five|six|seven|eight|nine|ten)\b")


def category(statement):
    statement = statement.lower()
    if COMPARISON.search(statement):
        return "comparison"
    if NUMBER.search(statement):
        return "lookup"
    return "other"


def load_charts():
    charts = defaultdict(set)
    with open(RAW_CSV, newline="") as f:
        for row in csv.DictReader(f):
            image = row["img_path"].removeprefix("./")
            pos, neg = row["caption"].strip(), row["neg_caption"].strip()
            if image.split("/")[2] in SOURCES and pos and neg and pos != neg:
                charts[image].add((pos, neg))
    return charts


def main():
    rng = random.Random(SEED)
    charts = load_charts()
    splits = {"train": [], "val": [], "test": []}

    for source in SOURCES:
        images = sorted(image for image in charts if image.split("/")[2] == source)
        images = rng.sample(images, CHARTS_PER_SOURCE)
        n_train, n_val = int(0.7 * len(images)), int(0.15 * len(images))
        split_images = {
            "train": images[:n_train],
            "val": images[n_train:n_train + n_val],
            "test": images[n_train + n_val:],
        }

        for split, images in split_images.items():
            for image in images:
                for pos, neg in sorted(charts[image]):
                    label = rng.randint(0, 1)
                    options = [neg, pos] if label else [pos, neg]
                    splits[split].append({
                        "image": image,
                        "source": source,
                        "category": category(pos),
                        "option_0": options[0],
                        "option_1": options[1],
                        "label": label,
                    })

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for split, rows in splits.items():
        with open(OUT_DIR / f"{split}.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        n_charts = len({row["image"] for row in rows})
        print(f"{split}: {len(rows)} instances, {n_charts} charts")


if __name__ == "__main__":
    main()
