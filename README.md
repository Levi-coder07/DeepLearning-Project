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
- `text_only`: MLP on the statement embedding only, to detect linguistic shortcuts. Needs only `data/features/texts.pt` and runs on CPU.

```bash
python -m src.models.zero_shot
python -m src.models.text_only
```

Preprocessing, feature checks and zero-shot results are in `notebooks/02_features_baselines.ipynb`.

## Models

Trainable heads on frozen CLIP features (`src/models/heads.py`), all with ~1.58M parameters:

- `global_mlp`: MLP on the global embedding and the statement.
- `attention_fixed`: a learned query attends over the patches.
- `attention_guided`: the statement attends over the patches.

```bash
python -m src.run --seeds 0 1 2
```

Results are saved to `results/models.csv` and weights to `checkpoints/<model>_seed<seed>.pt`. All models are trained with `src/train.py` (AdamW, cross-entropy over the two statements, best epoch on val).

## Layout

```
notebooks/    exploration and analysis
scripts/      data download
src/data/     loading, cleaning, splits
src/features/ CLIP feature extraction
src/models/   baselines and attention models
src/train.py  training loop
src/run.py    trains and evaluates the attention models
```
