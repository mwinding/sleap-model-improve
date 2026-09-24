#!/usr/bin/env python3
"""Plot each model's original benchmark score next to its held-out retrain."""
from __future__ import annotations
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from assess_centroids import ROOT
from plot_assessment import GROUPS, format_percent_axis, gradient_colors, prism_style, read_csv, save_plot

LABELS = {
    'centroid_baseline': 'Baseline', 'centroid_fullres_sigma5': 'Full res, σ5',
    'centroid_halfres_sigma2p5': 'Half res, σ2.5', 'centroid_halfres_body': 'Half res + body',
    'centroid_fullres_body': 'Full res + body', 'centroid_fullres_body_sigma2': 'σ = 2',
    'centroid_fullres_body_sigma3p5': 'σ = 3.5', 'centroid_fullres_body_filters32': '32 filters',
    'centroid_fullres_sigma2p5': 'No body anchor', 'centroid_test_with_synthetic': 'Two-animal crops',
    'centroid_body_synth_hard_crossings_v1': 'Hard-crossing scenes',
    'centroid_body_synth_fullframe_darkoverlap_v1': 'Full-frame dark overlap',
}
PANELS = [('all', 'recall_pct', 'Overall recall', 'Recall (%)'),
          ('hard', 'recall_pct', 'Hard-frame recall', 'Recall (%)'),
          ('all', 'precision_pct', 'Precision', 'Precision (%)')]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before', type=Path, default=ROOT/'outputs/model_assessment',
                        help='Assessment of the original models (benchmark frames were in training)')
    parser.add_argument('--after', type=Path, default=ROOT/'outputs/model_assessment/holdout',
                        help='Assessment of the held-out retrains')
    parser.add_argument('--suffix', default='_holdout', help='Suffix that maps an original model name to its retrain')
    parser.add_argument('--output', type=Path, default=None, help='Default: <after>/plots/holdout_vs_original')
    args = parser.parse_args()
    before = {(r['model'], r['subset']): r for r in read_csv(args.before/'summary.csv')}
    after = {(r['model'], r['subset']): r for r in read_csv(args.after/'summary.csv')}

    # One row per model, in experimental order; only models present in both assessments.
    models = []
    for group in GROUPS.values():
        for model, _ in group:
            if model not in models and (model, 'all') in before and (model+args.suffix, 'all') in after:
                models.append(model)
    if not models:
        parser.error('No model appears in both assessments')
    labels = [LABELS.get(m, m) for m in models]

    prism_style()
    output = args.output or args.after/'plots'/'holdout_vs_original'
    output.parent.mkdir(parents=True, exist_ok=True)
    y = np.arange(len(models))
    light, dark = gradient_colors(4)[0], gradient_colors(4)[3]
    fig, axes = plt.subplots(1, 3, figsize=(19, max(6, len(models)*.62)), sharey=True)
    rows = []
    for ax, (subset, key, title, xlabel) in zip(axes, PANELS):
        old = np.array([float(before[m, subset][key] or 'nan') for m in models])
        new = np.array([float(after[m+args.suffix, subset][key] or 'nan') for m in models])
        b1 = ax.barh(y-.19, old, .36, color=light, edgecolor='black', linewidth=1.3, label='Original (test frames in training)')
        b2 = ax.barh(y+.19, new, .36, color=dark, edgecolor='black', linewidth=1.3, label='Held-out retrain')
        ax.bar_label(b1, fmt='%.1f', fontsize=10, padding=4)
        ax.bar_label(b2, fmt='%.1f', fontsize=10, padding=4, fontweight='bold')
        ax.set_yticks(y, labels, fontsize=12)
        ax.tick_params(axis='y', labelleft=True)
        ax.invert_yaxis()
        format_percent_axis(ax, horizontal=True)
        ax.set_xlabel(xlabel); ax.set_title(title)
        for m, o, n in zip(models, old, new):
            rows.append({'model': m, 'panel': title, 'original': round(o, 2), 'holdout': round(n, 2), 'change': round(n-o, 2)})
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, loc='lower center', ncol=2, fontsize=12, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(w_pad=2.5, rect=(0, 0.05, 1, 1))
    save_plot(fig, output)
    with output.with_suffix('.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    print(f'Wrote {output.with_suffix(".png")} and {output.with_suffix(".csv")}')


if __name__ == '__main__':
    main()
