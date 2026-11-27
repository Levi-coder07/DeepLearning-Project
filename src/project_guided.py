
import argparse
import csv
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from src.project import ROOT, pca


def download_images(paths, workers=4):
    from remotezip import RemoteZip
    for source in sorted({path.split('/')[2] for path in paths}):
        wanted = [path for path in paths if path.split('/')[2] == source]
        missing = [path for path in wanted if not (ROOT / 'data/raw' / path).exists()]
        if not missing:
            continue
        url = f'https://huggingface.co/datasets/Junteng/Vision4Chart/resolve/main/data/vision4math_pic/{source}.zip'
        print(f'Downloading {len(missing)} {source} charts via HTTP ranges', flush=True)
        def fetch_partition(partition):
            # ZIP readers maintain seek state: give each download thread its own.
            with RemoteZip(url, initial_buffer_size=1024 * 1024) as archive:
                members = archive.namelist()
                for path in partition:
                    suffix = path.split(f'{source}/', 1)[1]
                    matches = [name for name in members if name == suffix or name.endswith('/' + suffix)]
                    if len(matches) != 1:
                        raise ValueError(f'Ambiguous/missing ZIP member for {path}: {matches[:5]}')
                    dest = ROOT / 'data/raw' / path
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    content = archive.read(matches[0])
                    temporary = dest.with_suffix(dest.suffix + '.part')
                    temporary.write_bytes(content)
                    temporary.replace(dest)
                    print(f'  Downloaded {path}', flush=True)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(fetch_partition, [missing[i::workers] for i in range(workers) if missing[i::workers]]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--per-source', type=int, default=30)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--checkpoint-seed', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--device', choices=['auto', 'cpu', 'mps', 'cuda'], default='auto')
    args = parser.parse_args()
    if args.per_source < 1 or args.batch_size < 1:
        parser.error('Sample and batch sizes must be positive')
    os.environ.setdefault('HF_HOME', str(ROOT / 'data/models/huggingface'))
    os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
    os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'results/.matplotlib'))
    import numpy as np
    import torch
    import torch.nn.functional as F
    from src.models.heads import AttentionGuided

    torch.manual_seed(args.seed)
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    device = args.device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f'Using {device}; PyTorch {torch.__version__}', flush=True)
    with (ROOT / 'data/processed/test.csv').open() as f:
        rows = list(csv.DictReader(f))
    rng = np.random.default_rng(args.seed)
    paths = []
    for source in sorted({row['source'] for row in rows}):
        choices = sorted({row['image'] for row in rows if row['source'] == source})
        paths.extend(rng.choice(choices, min(args.per_source, len(choices)), replace=False).tolist())
    paths = sorted(paths)
    selected = set(paths)
    records = []
    for pair_id, row in enumerate(rows):
        if row['image'] in selected:
            for option in (0, 1):
                records.append({**{k: row[k] for k in ('image', 'source', 'category')},
                                'pair_id': pair_id, 'option': option,
                                'truth': 'correct' if option == int(row['label']) else 'incorrect',
                                'statement': row[f'option_{option}']})
    checkpoint = ROOT / f'checkpoints/attention_guided_seed{args.checkpoint_seed}.pt'
    head = AttentionGuided().to(device).eval()
    head.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True), strict=True)
    texts = torch.load(ROOT / 'data/features/texts.pt', map_location='cpu', weights_only=True)
    text_index = {s: i for i, s in enumerate(texts['statements'])}
    out = ROOT / 'results/projections'
    out.mkdir(parents=True, exist_ok=True)
    cache = ROOT / f'data/features/projection_patches_n{args.per_source}_seed{args.seed}.pt'
    if cache.exists():
        stored = torch.load(cache, map_location='cpu', weights_only=True)
        if stored['paths'] != paths:
            raise ValueError('Sample cache paths do not match')
        patches = stored['patches']
    else:
        download_images(paths)
        import open_clip
        from PIL import Image
        from src.features.extract import pad_square, MODEL
        clip, _, preprocess = open_clip.create_model_and_transforms(
            MODEL, pretrained='openai', cache_dir=str(ROOT / 'data/models'))
        clip = clip.to(device).eval()
        clip.visual.output_tokens = True
        chunks = []
        with torch.inference_mode():
            for start in range(0, len(paths), args.batch_size):
                images = []
                for path in paths[start:start + args.batch_size]:
                    with Image.open(ROOT / 'data/raw' / path) as image:
                        images.append(preprocess(pad_square(image.convert('RGB'))))
                _, tokens = clip.visual(torch.stack(images).to(device))
                chunks.append(tokens.half().cpu())
                print(f'Encoded {min(start + args.batch_size, len(paths))}/{len(paths)} charts', flush=True)
        patches = torch.cat(chunks)
        torch.save({'paths': paths, 'patches': patches, 'model': MODEL, 'pretrained': 'openai'}, cache)
        del clip
    image_index = {path: i for i, path in enumerate(paths)}
    chunks, logits = [], []
    with torch.inference_mode():
        for start in range(0, len(records), 64):
            batch = records[start:start + 64]
            p = patches[[image_index[r['image']] for r in batch]].float().to(device)
            t = F.normalize(texts['emb'][[text_index[r['statement']] for r in batch]].float().to(device), dim=-1)
            z, _ = head.attention(t[:, None], p, p, need_weights=True)
            z = F.normalize(z[:, 0], dim=-1)
            logits.extend(head.mlp(torch.cat([z, t], dim=-1)).flatten().cpu().tolist())
            chunks.append(z.cpu())
    z = torch.cat(chunks).numpy()
    # Check extraction against the unmodified paired forward pass.
    with torch.inference_mode():
        pair = records[:2]
        p = patches[[image_index[pair[0]['image']]]].float().to(device)
        t = F.normalize(texts['emb'][[text_index[r['statement']] for r in pair]].float().to(device), dim=-1)
        reference = head(None, p, t[:1], t[1:]).flatten().cpu().numpy()
        if not np.allclose(reference, logits[:2], atol=1e-4, rtol=1e-4):
            raise AssertionError('Extracted representation scores differ from paired forward pass')
    for record, logit in zip(records, logits):
        record['score'] = logit
    # Both candidates remain adjacent; the scorer is unchanged from the checkpoint.
    scores = np.asarray(logits).reshape(-1, 2)
    labels = np.array([r['option'] if r['truth'] == 'correct' else 1 - r['option'] for r in records[::2]])
    accuracy = float((scores.argmax(1) == labels).mean())
    assert np.isfinite(z).all() and np.allclose(np.linalg.norm(z, axis=1), 1, atol=1e-5)
    stem = f'guided_seed{args.checkpoint_seed}_n{args.per_source}_sample{args.seed}'
    np.savez_compressed(out / f'{stem}_representations.npz', guided=z)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    summary = {'representation': 'L2-normalized text-guided attention output z before concatenation with text and scorer',
               'checkpoint': str(checkpoint.relative_to(ROOT)),
               'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
               'charts': len(paths), 'pairs': len(records) // 2, 'vectors': list(z.shape),
               'sample_seed': args.seed, 'device': device, 'sample_accuracy': accuracy,
               'evaluation_scope': 'full test set' if len(records) == 2 * len(rows) else 'test subset',
               'note': 'Single guided checkpoint. Repeated statements and pairs within charts are correlated.'}
    reduced, explained = pca(z.astype(np.float64), min(50, len(z) - 1))
    summary['pca_first_two_explained_variance'] = explained[:2].tolist()
    for method in ('pca', 'tsne'):
        coords = reduced[:, :2] if method == 'pca' else TSNE(
            perplexity=min(30, len(z) - 1), init='pca', learning_rate='auto',
            random_state=args.seed).fit_transform(reduced)
        assert np.isfinite(coords).all()
        with (out / f'{stem}_{method}.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[*records[0], 'x', 'y'])
            writer.writeheader()
            writer.writerows({**record, 'x': xy[0], 'y': xy[1]} for record, xy in zip(records, coords))
        fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
        for ax, field in zip(axes, ('source', 'truth', 'category')):
            for label in sorted({record[field] for record in records}):
                mask = np.array([record[field] == label for record in records])
                ax.scatter(*coords[mask].T, s=8, alpha=.4, label=label, rasterized=True)
            ax.set_title(f'Colored by {field}')
            ax.legend(fontsize=8)
            ax.set_xlabel('PC1' if method == 'pca' else 't-SNE 1')
            ax.set_ylabel('PC2' if method == 'pca' else 't-SNE 2')
        variance = f'; first two PCs explain {explained[:2].sum():.1%}' if method == 'pca' else '; perplexity 30; PCA pre-reduction to 50 dimensions'
        fig.suptitle(f'Guided patch pooling: {method.upper()} | {len(paths)} test charts, {len(records)} candidates{variance}', fontsize=12)
        fig.savefig(out / f'{stem}_{method}.png', dpi=160)
        fig.savefig(out / f'{stem}_{method}.svg')
        plt.close(fig)
    (out / f'{stem}_summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
