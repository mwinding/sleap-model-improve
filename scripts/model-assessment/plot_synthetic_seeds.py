#!/usr/bin/env python3
"""Synthetic training data for the body centroid detector (full resolution, body anchor, sigma 2.5), seed 42 and
seed 7 side by side: crowded / other / overall recall and precision on the held-out benchmark.

The crop-based sets (two-animal and parallel crops) were upscaled about 5.7x by sleap-nn's size matcher during
training, so they test that setup rather than properly scaled crops.
"""
from __future__ import annotations
import argparse
from pathlib import Path

from assess_centroids import ROOT
from plot_anchors import paired_bars
from plot_assessment import read_csv

A = ROOT / 'outputs/model_assessment'
ROWS = [  # label, (folder, model) seed 42, (folder, model) seed 7
    ('Real data only', ('holdout', 'centroid_fullres_body_holdout'), ('anchors_body', 'centroid_fullres_body_holdout_seed7')),
    ('+ full-frame', ('holdout', 'centroid_body_synth_fullframe_darkoverlap_v1_holdout'),
     ('anchors_body', 'centroid_body_synth_fullframe_darkoverlap_v1_holdout_seed7')),
    ('+ hard crossings', ('holdout', 'centroid_body_synth_hard_crossings_v1_holdout'),
     ('seed7_singles', 'centroid_body_synth_hard_crossings_v1_holdout_seed7')),
    ('+ two-animal crops (upscaled)', ('holdout', 'centroid_test_with_synthetic_holdout'),
     ('seed7_singles', 'centroid_test_with_synthetic_holdout_seed7')),
    ('+ parallel crops 1k (upscaled)', ('holdout_round2', 'centroid_synth_crops_parallel1k_holdout'),
     ('seed7_singles', 'centroid_synth_crops_parallel1k_holdout_seed7')),
    ('+ parallel crops 3k (upscaled)', ('holdout_round2', 'centroid_synth_crops_parallel3k_holdout'),
     ('seed7_singles', 'centroid_synth_crops_parallel3k_holdout_seed7')),
]
PANELS = [('crowded', 'Crowded recall', 'Recall (%)'), ('isolated', 'Other larvae recall', 'Recall (%)'),
          ('all', 'Overall recall', 'Recall (%)'), ('precision', 'Precision', 'Precision (%)')]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=A / 'plots_synthetic/synthetic_both_seeds')
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cache = {}

    def row(folder, model):
        if folder not in cache:
            cache[folder] = {(r['model'], r['subset']): r for r in read_csv(A / folder / 'summary.csv') if r['threshold'] == '0.2'}
        s = cache[folder]
        out = {k: float(s[model, k]['recall_pct']) for k in ('crowded', 'isolated', 'all')}
        out['precision'] = float(s[model, 'all']['precision_pct'])
        return out

    scores = [[row(*seed42) for _, seed42, _ in ROWS], [row(*seed7) for _, _, seed7 in ROWS]]
    values = {k: [[s[k] for s in seed] for seed in scores] for k, _, _ in PANELS}
    paired_bars(args.output, [label for label, _, _ in ROWS], PANELS, values, {k: (100, True) for k, _, _ in PANELS})


if __name__ == '__main__':
    main()
