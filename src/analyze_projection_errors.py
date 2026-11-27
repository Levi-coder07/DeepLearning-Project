
import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path

from src.project import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stem', default='guided_seed0_n30_sample0')
    args = parser.parse_args()
    out = ROOT / 'results/projections'
    os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'results/.matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    def read(method):
        with (out / f'{args.stem}_{method}.csv').open() as f:
            return list(csv.DictReader(f))

    rows = read('pca')
    groups = defaultdict(list)
    for row in rows:
        groups[row['pair_id']].append(row)
    decisions = {}
    for pair_id, pair in groups.items():
        pair.sort(key=lambda row: int(row['option']))
        assert len(pair) == 2 and [r['option'] for r in pair] == ['0', '1']
        assert {r['truth'] for r in pair} == {'correct', 'incorrect'}
        assert pair[0]['image'] == pair[1]['image']
        # argmax picks option 0 on ties, as in the original model evaluation.
        chosen = max(pair, key=lambda row: float(row['score']))
        correct = next(row for row in pair if row['truth'] == 'correct')
        wrong = next(row for row in pair if row['truth'] == 'incorrect')
        decisions[pair_id] = {'pair_id': pair_id, 'image': correct['image'],
                              'source': correct['source'], 'category': correct['category'],
                              'success': chosen['truth'] == 'correct', 'chosen_option': chosen['option'],
                              'correct_statement': correct['statement'], 'incorrect_statement': wrong['statement'],
                              'correct_score': float(correct['score']), 'incorrect_score': float(wrong['score']),
                              'margin': float(correct['score']) - float(wrong['score'])}
    pairs = list(decisions.values())
    assert len(pairs) * 2 == len(rows)
    summary = {'pairs': len(pairs), 'successes': sum(p['success'] for p in pairs),
               'failures': sum(not p['success'] for p in pairs),
               'ties': sum(p['margin'] == 0 for p in pairs)}
    vectors = np.load(out / f'{args.stem}_representations.npz')['guided'].astype(np.float64)
    assert len(vectors) == len(rows)
    by_pair = defaultdict(list)
    for i, row in enumerate(rows):
        by_pair[row['pair_id']].append(i)
    for pair_id, indices in by_pair.items():
        a, b = vectors[indices]
        decisions[pair_id]['guided_cosine_similarity'] = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    summary['guided_pair_cosine_similarity'] = {}
    for outcome, success in [('success', True), ('failure', False)]:
        values = np.array([p['guided_cosine_similarity'] for p in pairs if p['success'] == success])
        summary['guided_pair_cosine_similarity'][outcome] = {
            'mean': float(values.mean()), 'median': float(np.median(values)),
            'q25': float(np.quantile(values, .25)), 'q75': float(np.quantile(values, .75))}
    # Rewrite decisions to include the original-space similarity measure.
    with (out / f'{args.stem}_decisions.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(pairs[0]))
        writer.writeheader()
        writer.writerows(sorted(pairs, key=lambda p: p['margin']))
    for field in ('source', 'category'):
        summary[field] = {}
        for label in sorted({p[field] for p in pairs}):
            group = [p for p in pairs if p[field] == label]
            summary[field][label] = {'pairs': len(group), 'failures': sum(not p['success'] for p in group),
                                     'accuracy': sum(p['success'] for p in group) / len(group)}
    for method in ('pca', 'tsne'):
        records = read(method)
        pair_indices = defaultdict(list)
        for i, record in enumerate(records):
            pair_indices[record['pair_id']].append(i)
        coords = np.array([[float(r['x']), float(r['y'])] for r in records])
        success = np.array([decisions[r['pair_id']]['success'] for r in records])
        truth = np.array([r['truth'] == 'correct' for r in records])
        fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
        for mask, label, color in ((success, 'Model succeeds (both candidates)', '#94a3b8'),
                                   (~success, 'Model fails (both candidates)', '#dc2626')):
            axes[0].scatter(*coords[mask].T, s=12, alpha=.45, c=color, label=label)
        axes[0].set_title('All pairs: colored by model outcome')
        for pair_id in decisions:
            indices = pair_indices[pair_id]
            if not decisions[pair_id]['success']:
                axes[1].plot(*coords[indices].T, color='#94a3b8', linewidth=.5, alpha=.35, zorder=0)
        for mask, label, color, marker in ((~success & truth, 'Correct statement rejected', '#2563eb', 'o'),
                                           (~success & ~truth, 'Incorrect statement chosen', '#dc2626', 'x')):
            axes[1].scatter(*coords[mask].T, s=17, alpha=.65, c=color, marker=marker, label=label)
        axes[1].set_title('Failed pairs only: lines connect the two candidates')
        for ax in axes:
            low, high = coords.min(axis=0), coords.max(axis=0)
            padding = np.maximum((high - low) * .05, 1e-6)
            ax.set_xlim(low[0] - padding[0], high[0] + padding[0])
            ax.set_ylim(low[1] - padding[1], high[1] + padding[1])
            ax.legend(fontsize=8)
            ax.set_xlabel('PC1' if method == 'pca' else 't-SNE 1')
            ax.set_ylabel('PC2' if method == 'pca' else 't-SNE 2')
        fig.suptitle(f'Guided model errors | {method.upper()} | {summary["failures"]}/{len(pairs)} failed pairs', fontsize=14)
        fig.savefig(out / f'{args.stem}_{method}_errors.png', dpi=160)
        fig.savefig(out / f'{args.stem}_{method}_errors.svg')
        plt.close(fig)
    (out / f'{args.stem}_errors_summary.json').write_text(json.dumps(summary, indent=2))
    # A chart contributes several correlated pairs; these are descriptive counts.
    report = ['# Guided model: failure analysis', '',
              f'Same {len({p["image"] for p in pairs})}-chart test sample and checkpoint as the original projection. {summary["successes"]} correct decisions and {summary["failures"]} failures out of {len(pairs)} pairs.', '',
              'Each failed pair has two points: the correct statement rejected and the incorrect statement chosen. Coordinates are reused from the original projections; neither projection is refitted on failures. Lines connect candidates of the same pair.', '',
              'The final decision uses both the guided visual vector shown here and the text embedding. Nearby points can therefore receive different scores. Scores and margins are not calibrated probabilities. Pairs within charts are correlated; counts and similarities are descriptive, without uncertainty estimates.', '']
    report += ['## Similarity in the original 512 dimensions', '',
               '| Pair outcome | Mean cosine similarity | Median |', '|---|---:|---:|']
    for outcome, stats in summary['guided_pair_cosine_similarity'].items():
        report.append(f'| {outcome} | {stats["mean"]:.6f} | {stats["median"]:.6f} |')
    report += ['', 'A value near 1 means the guided vectors are very similar. This measures the pooled visual vectors, not the concatenated visual/text input to the scorer. It does not identify a causal explanation for errors.', '']
    for field in ('source', 'category'):
        report += [f'## By {field}', '', '| Group | Pairs | Failures | Accuracy |', '|---|---:|---:|---:|']
        for label, stats in summary[field].items():
            report.append(f'| {label} | {stats["pairs"]} | {stats["failures"]} | {stats["accuracy"]:.1%} |')
        report.append('')
    report += ['## Examples with the largest incorrect score advantage', '',
               'These are selected by score margin, not a calibrated confidence measure.', '']
    for p in sorted(pairs, key=lambda p: p['margin'])[:8]:
        report += [f'### Pair {p["pair_id"]} ({p["source"]}, {p["category"]})', '',
                   f'Chart: `{p["image"]}`', '', f'- Correct, rejected: {p["correct_statement"]}',
                   f'- Incorrect, chosen: {p["incorrect_statement"]}', f'- Correct minus incorrect score: {p["margin"]:.3f}', '']
    (out / f'{args.stem}_errors.md').write_text('\n'.join(report))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
