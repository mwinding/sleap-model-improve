#!/usr/bin/env python3
"""Compare held-out experiments side by side: overall, hard-frame and crowded recall, and precision.

Models can come from different assessment directories (e.g. mouthhooks-anchored models are
scored in their own directory against the mouthhooks node). Each entry names the directory,
the model, a short label and the experiment group it belongs to.
"""
from __future__ import annotations
import argparse
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from assess_centroids import ROOT
from plot_assessment import format_percent_axis, gradient_colors, prism_style, read_csv, save_plot

A = 'outputs/model_assessment'
# (assessment dir, model, label, group, target node)
DEFAULT = [
    (f'{A}/holdout', 'centroid_fullres_body_holdout', 'Real data only', 'Reference', 'body'),
    (f'{A}/holdout', 'centroid_test_with_synthetic_holdout', 'Two-animal crops', 'Synthetic data', 'body'),
    (f'{A}/holdout', 'centroid_body_synth_hard_crossings_v1_holdout', 'Hard-crossing scenes', 'Synthetic data', 'body'),
    (f'{A}/holdout', 'centroid_body_synth_fullframe_darkoverlap_v1_holdout', 'Full-frame', 'Synthetic data', 'body'),
    (f'{A}/holdout_round2', 'centroid_synth_crops_parallel1k_holdout', 'Parallel crops 1k', 'Synthetic data', 'body'),
    (f'{A}/holdout_round2', 'centroid_synth_crops_parallel3k_holdout', 'Parallel crops 3k', 'Synthetic data', 'body'),
    (f'{A}/holdout_round2', 'centroid_synth_fullframe_stride1_sigma2_holdout', 'Stride 1, σ2', 'Full-frame + one change', 'body'),
    (f'{A}/holdout_mouthhooks', 'centroid_synth_fullframe_mouthhooks_holdout', 'Mouthhooks anchor', 'Full-frame + one change', 'mouthhooks'),
]
PANELS = [('all', 'recall_pct', 'Overall recall', 'Recall (%)'),
          ('hard', 'recall_pct', 'Hard-frame recall', 'Recall (%)'),
          ('crowded', 'recall_pct', 'Crowded recall', 'Recall (%)'),
          ('all', 'precision_pct', 'Precision', 'Precision (%)')]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / A / 'holdout_experiments')
    args = parser.parse_args()

    summaries = {}
    rows = []
    for folder, model, label, group, node in DEFAULT:
        if folder not in summaries:
            summaries[folder] = {(r['model'], r['subset']): r for r in read_csv(ROOT / folder / 'summary.csv')}
        if (model, 'all') not in summaries[folder]:
            print(f'Skipping {model}: not in {folder}')
            continue
        rows.append({'lookup': summaries[folder], 'model': model, 'label': label, 'group': group, 'node': node})

    # Lay out groups top to bottom with a header row each, like the all-models plot.
    y, headers, position, previous = [], [], 0.0, None
    for row in rows:
        if row['group'] != previous:
            if previous is not None:
                position += 0.6
            headers.append((position, row['group']))
            position += 1.0
            previous = row['group']
        row['y'] = position
        y.append(position)
        position += 1.0

    prism_style()
    colors = gradient_colors(len(rows))
    height = max(6.5, position * 0.46)
    fig, axes = plt.subplots(1, len(PANELS), figsize=(22, height), sharey=True)
    for ax, (subset, key, title, xlabel) in zip(axes, PANELS):
        values = [float(r['lookup'].get((r['model'], subset), {}).get(key) or 'nan') for r in rows]
        bars = ax.barh(y, values, 0.68, color=colors, edgecolor='black', linewidth=1.3, zorder=3)
        ax.bar_label(bars, fmt='%.1f', fontsize=11, padding=4)
        ax.set_yticks(y, [textwrap.fill(r['label'], 22) for r in rows], fontsize=12)
        ax.tick_params(axis='y', labelleft=True)
        ax.set_ylim(position - 0.3, -0.8)
        format_percent_axis(ax, horizontal=True)
        ax.set_xlabel(xlabel)
        ax.set_title(title)
        for header_y, header in headers:
            ax.text(0, header_y, header, ha='left', va='center', fontsize=12, fontweight='bold',
                    bbox={'facecolor': 'white', 'edgecolor': 'none', 'pad': 2}, zorder=5)
        for tick, r in zip(ax.get_yticklabels(), rows):
            if r['group'] == 'Reference':
                tick.set_fontweight('bold')
    fig.tight_layout(w_pad=2.2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_plot(fig, args.output)
    print(f'Wrote {args.output.with_suffix(".png")}')


if __name__ == '__main__':
    main()
