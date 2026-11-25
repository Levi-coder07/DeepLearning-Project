from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


OUT = Path("results/figures")
OUT.mkdir(parents=True, exist_ok=True)

COLORS = {
    "zero_shot": "#8B95A5",
    "text_only": "#6C8EBF",
    "global_mlp": "#E69F00",
    "attention_fixed": "#CC6677",
    "attention_guided": "#009E73",
    "mean_patches": "#8E6CBB",
}
LABELS = {
    "zero_shot": "Zero-shot",
    "text_only": "Text-only",
    "global_mlp": "Global MLP",
    "attention_fixed": "Fixed attention",
    "attention_guided": "Guided attention",
    "mean_patches": "Mean pooling",
}

mpl.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "bold",
        "figure.dpi": 140,
        "savefig.dpi": 400,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    }
)


def finish(fig, stem):
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"{stem}.{ext}", facecolor="white")
    plt.close(fig)


def overall_accuracy():
    models = ["zero_shot", "text_only", "global_mlp", "attention_fixed", "attention_guided", "mean_patches"]
    means = np.array([52.9, 57.6, 59.8, 58.7, 60.3, 57.1])
    stds = np.array([0.0, 0.6, 0.2, 0.3, 0.1, 1.7])
    y = np.arange(len(models))
    fig, ax = plt.subplots(figsize=(7.2, 4.1))
    bars = ax.barh(y, means - 50, left=50, color=[COLORS[m] for m in models], height=0.68)
    ax.errorbar(means, y, xerr=stds, fmt="none", ecolor="#263238", capsize=3, lw=1.2)
    ax.axvline(50, color="#4B5563", lw=1, ls="--")
    for bar, value, sd in zip(bars, means, stds):
        suffix = f" ± {sd:.1f}" if sd else ""
        ax.text(value + sd + 0.25, bar.get_y() + bar.get_height() / 2, f"{value:.1f}{suffix}%", va="center", fontsize=9)
    ax.set_yticks(y, [LABELS[m] for m in models])
    ax.invert_yaxis()
    ax.set_xlim(50, 63)
    ax.set_xlabel("Test accuracy (%)")
    ax.set_title("Text-guided attention achieves the best overall accuracy")
    ax.text(50.05, len(models) - 0.05, "Chance", color="#4B5563", fontsize=8, va="bottom")
    ax.grid(axis="x", color="#E5E7EB", lw=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    finish(fig, "01_overall_accuracy")


def dataset_breakdown():
    models = ["zero_shot", "text_only", "global_mlp", "attention_fixed", "attention_guided", "mean_patches"]
    datasets = ["ChartQA", "DVQA", "FigureQA"]
    values = np.array(
        [
            [51.5, 58.0, 49.3],
            [52.1, 67.9, 52.9],
            [52.7, 73.9, 52.8],
            [54.3, 70.6, 51.5],
            [53.6, 75.3, 52.4],
            [53.9, 67.0, 50.8],
        ]
    )
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 4.5), sharey=True)
    y = np.arange(len(models))
    for j, (ax, dataset) in enumerate(zip(axes, datasets)):
        vals = values[:, j]
        ax.barh(y, vals - 48, left=48, color=[COLORS[m] for m in models], height=0.66)
        best = int(np.argmax(vals))
        for i, v in enumerate(vals):
            ax.text(v + 0.35, i, f"{v:.1f}", va="center", fontsize=8, fontweight="bold" if i == best else "normal")
        ax.axvline(50, color="#4B5563", lw=1, ls="--")
        ax.set_title(dataset)
        ax.set_xlim(48, 79)
        ax.grid(axis="x", color="#E5E7EB", lw=0.8)
        ax.set_axisbelow(True)
        ax.set_xlabel("Accuracy (%)")
    axes[0].set_yticks(y, [LABELS[m] for m in models])
    axes[0].invert_yaxis()
    fig.suptitle("The main gains come from DVQA; ChartQA and FigureQA remain difficult", fontweight="bold", y=1.01)
    fig.tight_layout()
    finish(fig, "02_dataset_breakdown")


def shuffled_control():
    models = ["text_only", "global_mlp", "attention_fixed", "attention_guided", "mean_patches"]
    matched = np.array([57.6, 59.8, 58.7, 60.3, 57.1])
    shuffled = np.array([57.6, 54.8, 52.6, 53.5, 55.1])
    x = np.arange(len(models))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    ax.bar(x - width / 2, matched, width, label="Matched chart", color="#167D9A")
    ax.bar(x + width / 2, shuffled, width, label="Shuffled chart", color="#C5CBD3")
    for i, (a, b) in enumerate(zip(matched, shuffled)):
        ax.text(i - width / 2, a + 0.25, f"{a:.1f}", ha="center", fontsize=8)
        ax.text(i + width / 2, b + 0.25, f"{b:.1f}", ha="center", fontsize=8)
        if i > 0:
            ax.text(i, min(a, b) - 1.05, f"−{a-b:.1f} pp", ha="center", fontsize=8, color="#9B2C2C")
    ax.axhline(50, color="#4B5563", lw=1, ls="--")
    ax.set_ylim(49, 63)
    ax.set_ylabel("Test accuracy (%)")
    ax.set_xticks(x, [LABELS[m].replace(" ", "\n") for m in models])
    ax.set_title("Shuffling charts hurts every image-based model")
    ax.legend(frameon=False, ncols=2, loc="upper left")
    ax.grid(axis="y", color="#E5E7EB", lw=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    finish(fig, "03_shuffled_image_control")


def cost_tradeoff():
    models = ["text_only", "global_mlp", "attention_fixed", "attention_guided", "mean_patches"]
    time = np.array([6, 7, 89, 96, 9])
    accuracy = np.array([57.6, 59.8, 58.7, 60.3, 57.1])
    memory = np.array([0.9, 0.9, 2.0, 2.0, 0.9])
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    for m, x, y, mem in zip(models, time, accuracy, memory):
        ax.scatter(x, y, s=260 * mem, color=COLORS[m], edgecolor="white", linewidth=1.2, zorder=3)
        offset = (5, 7) if m != "attention_fixed" else (5, -15)
        ax.annotate(LABELS[m], (x, y), xytext=offset, textcoords="offset points", fontsize=9)
    ax.set_xscale("log")
    ax.set_xticks([6, 10, 30, 100], ["6", "10", "30", "100"])
    ax.set_xlim(4.5, 140)
    ax.set_ylim(56.2, 61.2)
    ax.set_xlabel("Training time (seconds, log scale)")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("Small accuracy gains require much more training time")
    ax.text(0.99, 0.02, "Bubble area ∝ peak GPU memory", transform=ax.transAxes, ha="right", color="#4B5563", fontsize=8)
    ax.grid(color="#E5E7EB", lw=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    finish(fig, "04_accuracy_cost_tradeoff")


def resolution_ablation():
    labels = ["Overall", "ChartQA", "FigureQA", "Lookup"]
    guided_224 = np.array([60.3, 53.6, 52.4, 58.0])
    guided_448 = np.array([60.8, 56.4, 51.9, 60.2])
    delta = guided_448 - guided_224
    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7.6, 4.1))
    for i, (a, b) in enumerate(zip(guided_224, guided_448)):
        ax.plot([a, b], [i, i], color="#9CA3AF", lw=2, zorder=1)
    ax.scatter(guided_224, y, s=85, color="#6C8EBF", label="224×224", zorder=2)
    ax.scatter(guided_448, y, s=85, color=COLORS["attention_guided"], label="448×448 tiles", zorder=2)
    for i, (b, d) in enumerate(zip(guided_448, delta)):
        sign = "+" if d >= 0 else ""
        ax.text(max(guided_224[i], b) + 0.45, i, f"{sign}{d:.1f} pp", va="center", fontsize=9, color="#166534" if d > 0 else "#9B2C2C")
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(49.5, 63.8)
    ax.set_xlabel("Guided-attention accuracy (%)")
    ax.set_title("Higher resolution helps reading-heavy subsets, but not FigureQA")
    ax.legend(frameon=False, ncols=2, loc="upper left")
    ax.grid(axis="x", color="#E5E7EB", lw=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    finish(fig, "05_resolution_ablation")


if __name__ == "__main__":
    overall_accuracy()
    dataset_breakdown()
    shuffled_control()
    cost_tradeoff()
    resolution_ablation()
    print(f"Wrote 15 files to {OUT}")
