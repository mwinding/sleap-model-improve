#!/usr/bin/env python3
"""Score bottom-up sweep models on the test video: larvae missed (crowded / all others / total) and precision,
counting complete skeletons and skeletons with >= 4 nodes, as predicted and after redrawing every
larva with the body pose model (redrawn_<name>.slp from body_points_for_redraw.py + sleap predict).

Writes <predictions-dir>/../sweep_scores.csv and prints a markdown table.
"""
from __future__ import annotations
import argparse
import csv
from pathlib import Path

from assess_centroids import ROOT
from plot_backbones import missed
from plot_pipelines import load_preds
from tune_merge import test_truth


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions-dir', type=Path, default=ROOT / 'outputs/model_assessment/bottomup_sweep/predictions')
    parser.add_argument('--models', nargs='+', required=True, help='label=prediction name (without .slp)')
    args = parser.parse_args()
    truth = test_truth()
    rows = []
    for item in args.models:
        label, name = item.split('=', 1)
        for kind, path in (('as predicted', args.predictions_dir / f'{name}.slp'),
                           ('redrawn with body pose model', args.predictions_dir / f'redrawn_{name}.slp')):
            if not path.exists():
                continue
            preds = load_preds(path)
            for min_nodes in (5, 4):
                s = missed(truth, preds, min_nodes)
                rows.append({'model': label, 'prediction': name, 'kind': kind, 'min_nodes': min_nodes, **s})
    out = args.predictions_dir.parent / 'sweep_scores.csv'
    with open(out, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print('| Model | Skeletons | Counting | Crowded missed (of 155) | Others missed (of 213) | Total missed (of 368) | Precision |')
    print('|---|---|---|---|---|---|---|')
    for r in rows:
        print(f"| {r['model']} | {r['kind']} | {'complete' if r['min_nodes'] == 5 else '>= 4 nodes'} | "
              f"{r['crowded']:.0f} ({100 - 100 * r['crowded'] / 155:.0f}%) | {r['others']:.0f} ({100 - 100 * r['others'] / 213:.0f}%) | "
              f"{r['total']:.0f} ({100 - 100 * r['total'] / 368:.0f}%) | {r['precision']:.1f} |")
    print(f'\nWrote {out}')


if __name__ == '__main__':
    main()
