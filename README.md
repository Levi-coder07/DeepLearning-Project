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

## Layout

```
configs/     experiment configs
notebooks/   exploration and analysis
scripts/     data download
src/data/    loading, cleaning, splits
src/features/ CLIP feature extraction
src/models/  baselines and attention models
```
