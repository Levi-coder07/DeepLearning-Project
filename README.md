# Global and Text-Guided Regional Representations for Chart–Text Matching

Deep Learning course project — Levi Castillon, Luisa Villanueva (FGV).

Given a chart and two statements, choose the correct statement. We investigate whether aggregating regional CLIP features helps compared with a whole-chart embedding, and whether conditioning that aggregation on the statement improves performance.

## Approach

All experiments use frozen CLIP ViT-B/16 with OpenAI weights. Only the scoring heads are trained, which keeps the encoder constant across comparisons and allows features to be cached.

| Model | Approach | Purpose |
|---|---|---|
| `zero_shot` | Similarity between whole-chart and statement embeddings; no training | Pretrained baseline |
| `text_only` | MLP on each statement embedding; no image | Check linguistic shortcuts |
| `global_mlp` | MLP on whole-chart and statement embeddings | Trained global baseline |
| `attention_fixed` | A learned, statement-independent query pools patch features | Test regional aggregation |
| `attention_guided` | Each statement queries the patch features | Test text conditioning |
| `mean_patches` | Uniform pooling of patch features followed by projection | Pooling ablation |

The global representation has 512 values per chart; regional heads receive 196 × 768 patch values. The three main image heads have about 1.58M trainable parameters, but their input sizes and computational costs differ.

Training uses a correct and an incorrect caption from Vision4Chart, with cross-entropy over their two scores. We do not fine-tune CLIP with the paper’s NegCLIP method; our experiments compare how heads use existing frozen representations.

## Main findings

Test accuracy (%), mean ± standard deviation over seeds 0, 1, and 2; zero-shot is evaluated once. Chance is 50%.

| Model | Accuracy |
|---|---:|
| Zero-shot | 52.9 |
| Text-only | 57.6 ± 0.6 |
| Global MLP | 59.8 ± 0.2 |
| Fixed attention | 58.7 ± 0.3 |
| Guided attention | **60.3 ± 0.1** |
| Mean pooling | 57.1 ± 1.7 |

- **Text conditioning helps:** guided attention beats fixed attention by 1.6 percentage points (95% chart-bootstrap CI: 0.7 to 2.5). Its advantage over the global MLP is not statistically significant at standard resolution.
- **Regional features have a cost:** attention heads take about 89–96 seconds to train versus 7 seconds for the global MLP, with about twice the extra peak GPU memory. These measurements exclude feature extraction.
- **Dataset shortcuts matter:** text-only exceeds chance, and replacing charts with other charts reduces image-model accuracy. Neither check alone demonstrates precise visual grounding.
- **Higher resolution gives limited gains:** four tiles provide 784 patches and raise guided attention to 60.8 ± 1.0%, with gains on ChartQA and lookup statements. The overall improvement over standard-resolution guided attention is not significant.

Low accuracy does not establish that information is absent from CLIP features: the heads may also fail to extract it. Attention maps are qualitative, not evidence of causal grounding.

See [detailed results](docs/results.md) for source/category breakdowns, controls, statistical comparisons, and resolution experiments. The [results notebook](notebooks/03_results.ipynb) contains attention maps and error examples.

## Reproduce the experiments

Run commands from the repository root. Feature extraction currently requires a CUDA GPU; training uses CUDA when available and otherwise CPU.

### 1. Install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Download and prepare data

We use [Vision4Chart](https://huggingface.co/datasets/Junteng/Vision4Chart), selecting 3,000 charts each from ChartQA, FigureQA, and DVQA. Splits are 70/15/15 by chart, so a chart cannot appear in multiple splits.

```bash
python scripts/download_data.py
python src/data/prepare.py
```

The download includes the CSV and images (~10 GB) in `data/raw/`. CSV image paths are relative to that directory; prepared splits are written to `data/processed/`.

### 3. Extract features and run models

Charts are padded to a square before resizing to avoid cropping. Features are saved to `data/features/`.

```bash
python -m src.features.extract
python -m src.models.zero_shot
python -m src.run --seeds 0 1 2
```

The runner trains all five trainable heads with AdamW and selects the best epoch on validation accuracy. To run a subset, use `--models`, for example:

```bash
python -m src.run --models attention_fixed attention_guided --seeds 0 1 2
```

The text-only baseline can also run separately on CPU with only the processed splits and `data/features/texts.pt`:

```bash
python -m src.models.text_only
```

### 4. Optional: higher-resolution tiles

After standard feature extraction, resize each padded chart to 448×448 and encode its four quadrants. This produces 784 patches per chart (~11 GB of patch features). Tile patches stay in CPU memory and are transferred by batch during training.

```bash
python -m src.features.extract --tiles
python -m src.run --tiles --models attention_fixed attention_guided mean_patches --seeds 0 1 2
```

### Exploratory projections

To project the **trained guided head**, including downloading a small test image
sample and extracting frozen CLIP patches on CUDA, Apple MPS, or CPU:

```bash
python -m src.project_guided --per-source 30
```

This loads `attention_guided_seed0.pt` and the existing `texts.pt` cache. Each
candidate statement queries its chart's 196 patches; the projected vector is the
normalized 512-dimensional attention output **before** concatenation with text
and scoring. Both PCA and t-SNE plots are saved as PNG/SVG, with coordinates,
raw guided vectors, a checkpoint hash, and subset accuracy in
`results/projections/`. Chart downloads use HTTP ZIP ranges; sampled patches are
cached separately from the full experiment features. No weights are trained.
Use `--checkpoint-seed`, `--seed`, `--batch-size`, or `--device` to change the run.
The default uses 30 charts per source and keeps all their statement pairs.

Analyze actual model failures using the saved candidate scores:

```bash
python -m src.analyze_projection_errors
```

This reuses both projection coordinates, colors pairs by model success/failure,
and shows failed pairs with the rejected correct statement connected to the
chosen incorrect statement. It also exports pair decisions, source/category
breakdowns, and examples ranked by incorrect score advantage. Statement truth
and model success are distinct labels; score margins are not probabilities.

Project cached whole-chart or statement CLIP embeddings without retraining:

```bash
python -m src.project --features clip-image --method pca
python -m src.project --features clip-text --method tsne
```

If feature caches are unavailable, run a wording-only PCA using TF-IDF (vocabulary
and IDF fitted on training statements):

```bash
python -m src.project --features lexical --method pca
```

Each run samples up to 300 charts per source from the test split, retaining both
candidates for every sampled pair. Image plots use one observation per chart.
Outputs in `results/projections/` include an SVG colored by source (and statement
correctness for text), coordinates with metadata in CSV, and settings in JSON.
Use `--per-source`, `--split`, and `--seed` to change the sample. t-SNE first
reduces normalized features to at most 50 PCs. Projections are exploratory:
separation does not demonstrate correct chart reasoning, and repeated statements
are not independent observations. Lexical features describe wording, not CLIP.

### Experiment outputs

| Path | Contents |
|---|---|
| `results/models.csv` | Accuracy, shuffled-image accuracy, parameters, training time, and extra peak GPU memory |
| `results/predictions_test.csv` | Test-pair predictions for each model and seed |
| `checkpoints/<model>_seed<seed>.pt` | Trained head weights; tiled models use a `_tiles` suffix |

Rerunning a model/seed replaces its result rows while preserving other runs. Raw images, image features, and generated result CSVs are not included in this checkout; follow the steps above to regenerate them.

## Repository guide

| Path | Contents |
|---|---|
| [01_data.ipynb](notebooks/01_data.ipynb) | Data statistics, split checks, and category-rule review |
| [02_features_baselines.ipynb](notebooks/02_features_baselines.ipynb) | Preprocessing, feature checks, and zero-shot evaluation |
| [03_results.ipynb](notebooks/03_results.ipynb) | Experimental analysis, attention maps, and errors |
| [src/data/](src/data/) | Data preparation and chart-level splits |
| [src/features/](src/features/) | Frozen CLIP feature extraction |
| [src/models/](src/models/) | Baselines and trainable heads |
| [src/train.py](src/train.py) | Training and prediction |
| [src/run.py](src/run.py) | Multi-model, multi-seed experiments |
| [src/eval.py](src/eval.py) | Accuracy breakdowns and paired chart bootstrap |

## Reference

Liu et al. (2025), [On the Perception Bottleneck of VLMs for Chart Understanding](https://aclanthology.org/2025.findings-emnlp.573/). We use their Vision4Chart data; this project studies aggregation heads rather than reproducing their encoder fine-tuning.
