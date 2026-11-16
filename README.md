# Global and Text-Guided Regional Representations for Chart–Text Matching

Deep Learning course project - Levi Castillon, Luisa Villanueva (FGV).

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data

[Vision4Chart](https://huggingface.co/datasets/Junteng/Vision4Chart) (Liu et al., 2025). Downloads the CSV and the ChartQA, FigureQA and DVQA images (~10 GB) into `data/raw/`:

```bash
python scripts/download_data.py
```

Image paths in the CSV (`img_path`) are relative to `data/raw/`.

Build the train/val/test splits (3,000 charts per source, 70/15/15 split by chart) into `data/processed/`:

```bash
python src/data/prepare.py
```

Statistics, split checks and the manual review of category rules are in `notebooks/01_data.ipynb`.

## Features

Frozen CLIP ViT-B/16 (OpenAI weights) features for every chart (global embedding and 196 patch tokens) and statement, saved to `data/features/`. Charts are padded to a square before resizing so no part is cropped.

```bash
python -m src.features.extract
```

## Baselines

- `zero_shot`: picks the statement with the higher CLIP similarity to the chart. No training.
- `text_only`: MLP (512 → 256 → 1) on the statement embedding only, to detect linguistic shortcuts (57.6% on test). Needs only `data/features/texts.pt` and runs on CPU.

```bash
python -m src.models.zero_shot
python -m src.models.text_only
```

Preprocessing, feature checks and zero-shot results are in `notebooks/02_features_baselines.ipynb`.

## Models

Trainable heads on frozen CLIP features (`src/models/heads.py`), the three main heads with ~1.58M parameters:

- `global_mlp`: MLP on the global embedding and the statement.
- `attention_fixed`: a learned query attends over the patches.
- `attention_guided`: the statement attends over the patches.
- `mean_patches` (ablation): uniform weights over the patches.

```bash
python -m src.run --seeds 0 1 2
python -m src.run --models mean_patches --seeds 0 1 2   # one model only
```

`src.run` trains these models and `text_only` with each seed. Results (accuracy, shuffled-image accuracy, parameters, training time, peak GPU memory) are saved to `results/models.csv` (rerunning a model replaces only its rows), per-statement test predictions to `results/predictions_test.csv`, and weights to `checkpoints/<model>_seed<seed>.pt`. All models are trained with `src/train.py` (AdamW, cross-entropy over the two statements, best epoch on val).

## Results

Full analysis, attention maps and error examples: `notebooks/03_results.ipynb`. Test accuracy (%), mean over seeds 0, 1 and 2 (± std for the overall score):

| | zero_shot | text_only | global_mlp | attention_fixed | attention_guided | mean_patches* |
|---|---|---|---|---|---|---|
| **all** | 52.9 | 57.6 ± 0.6 | 59.8 ± 0.2 | 58.7 ± 0.3 | **60.3 ± 0.1** | 57.1 ± 1.7 |
| chartqa | 51.5 | 52.1 | 52.7 | **54.3** | 53.6 | 53.9 |
| dvqa | 58.0 | 67.9 | 73.9 | 70.6 | **75.3** | 67.0 |
| figureqa | 49.3 | **52.9** | 52.8 | 51.5 | 52.4 | 50.8 |
| comparison | 51.5 | 58.3 | **58.6** | 57.2 | 58.0 | 56.1 |
| lookup | 53.2 | 55.9 | 56.4 | 57.4 | **58.0** | 56.8 |
| other | 57.7 | 58.9 | 72.7 | 68.1 | **75.3** | 62.2 |
| shuffled images | - | 57.6 | 54.8 | 52.6 | 53.5 | 55.1 |
| parameters | 0 | 132k | 1.58M | 1.58M | 1.58M | 0.92M |
| training time | - | 6 s | 7 s | 89 s | 96 s | 9 s |
| peak GPU memory | - | 0.9 GB | 0.9 GB | 2.0 GB | 2.0 GB | 0.9 GB |

\* Ablation: the attention head with uniform weights (projected mean of the 196 patches).

**Sanity checks.** `text_only` (57.6%) is well above zero-shot (52.9%), so the statements contain linguistic shortcuts. Adding the image (`global_mlp`) adds 2.2 points, mostly on DVQA (+6.1) and on `other` statements (+13.8). In the shuffled-image control, every test chart is replaced by a different chart. All image models then fall below `text_only` (52.6–55.1%), so their predictions do depend on the matching chart.

**Significance.** Seed std only measures training noise. To account for the test sample too, `paired_bootstrap` in `src/eval.py` averages each model over the 3 seeds and resamples test **charts** (1,350, not the 13,123 correlated statement pairs) 10,000 times. The intervals below are 95% intervals of the paired difference.

**RQ1: Does learned regional aggregation help?** No. Text-independent attention (`attention_fixed`) is 1.0 ± 0.5 points below the global embedding and less stable across seeds. The drop is borderline significant (95% CI over charts −2.0 to −0.03, p = 0.045). Its only gains are on ChartQA (+1.7) and lookups (+1.0), both within the seed variation.

**RQ2: Does text conditioning help?** Yes. Guided attention beats fixed attention by 1.6 ± 0.4 points (95% CI over charts 0.7 to 2.5, p = 0.001). It has the highest score and is the most stable across seeds, but its +0.6 ± 0.3 over `global_mlp` is not significant (CI −0.3 to 1.5, p = 0.18). The effect is subtle in how it works. Guided attention is almost uniform: its entropy is about 5.05 nats, against 5.28 for uniform and 2.8 for the fixed query. The maps for the two candidate statements are nearly identical (total variation distance 0.03–0.05). Even so, the small statement-dependent differences matter:
- When both candidates share one query (the mean of the two statements), accuracy drops to about 53%. This shared query is out of distribution for the scorer, so the drop is an upper bound on the effect.
- Uniform pooling (`mean_patches`) is 3.2 ± 1.8 points worse than guided attention.

So conditioning on the statement helps, but not through visible localization in the attention maps.

**RQ3: When does text guidance help?** The gains are not where we expected. Guided minus fixed attention is +0.9 on comparisons, +0.6 on lookups and +7.1 on `other`. Almost all of the `other` gain comes from DVQA statements about chart structure (legend colors, bar labels). Only the DVQA and `other` gains are significant; the comparison gain is not (p ≈ 0.1). The hypothesis that comparison statements benefit most is not supported. On ChartQA, guided attention is not better than fixed attention (−0.7 ± 1.2).

**Cost.** All three heads have about 1.58M parameters. Attention over the patches makes training about 13× slower and needs about 2× the peak GPU memory. Peak memory is the extra memory during training, on top of the features kept on the GPU. Most of the 0.9 GB for the cheap models comes from gathering batch features.

**Limitations.** Every model uses frozen CLIP ViT-B/16 features at 224×224. This resolution is likely too low to read small text and exact values (an encoding bottleneck), which may explain why FigureQA and ChartQA stay near chance for all models. Attention maps are qualitative and are not evidence of causal grounding.

## Layout

```
notebooks/    exploration and analysis
scripts/      data download
src/data/     loading, cleaning, splits
src/features/ CLIP feature extraction
src/models/   baselines and attention models
src/eval.py   accuracy overall, per source and per category
src/train.py  training loop
src/run.py    trains and evaluates all trainable models
```
