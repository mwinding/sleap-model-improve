#!/usr/bin/env python3
"""Anchor comparison plots, both seeds side by side.

1. anchors_comparison: centroid models on each of the five anchor nodes (real data only).
2. anchor_combinations: detection when several anchor models are combined (a larva counts
   if any member found it), with the summed false positives as an upper bound.
3. anchor_synthetic_grid: body vs mouthhooks anchor, real data only vs + full-frame synthetic.

Recall uses one body-based crowded definition for every model, so all bars cover the same
larvae. Needs the per-anchor assessments in outputs/model_assessment/anchors_<node>.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from assess_centroids import ROOT
from plot_assessment import format_percent_axis, gradient_colors, prism_style, read_csv, save_plot
from union_anchors import combined_recall, common_subsets, load_hits

A = ROOT / 'outputs/model_assessment'
SEEDS = [('', 'Seed 42'), ('_seed7', 'Seed 7')]
ANCHORS = [('head', 'Head'), ('mouthhooks', 'Mouthhooks'), ('body', 'Body'), ('tail', 'Tail'), ('spiracle', 'Spiracle')]
REAL = 'centroid_fullres_{node}_holdout{seed}'
COMBOS = [(('mouthhooks',), 'Mouthhooks'), (('head',), 'Head'), (('body',), 'Body'),
          (('head', 'body'), 'Head + body'), (('mouthhooks', 'body'), 'Mouthhooks + body'),
          (('head', 'body', 'tail'), 'Head + body + tail'), (('mouthhooks', 'body', 'tail'), 'Mouthhooks + body + tail'),
          (tuple(n for n, _ in ANCHORS), 'All five')]
GRID = [('body', 'centroid_fullres_body_holdout{seed}', 'Body, real only'),
        ('body', 'centroid_body_synth_fullframe_darkoverlap_v1_holdout{seed}', 'Body + synthetic'),
        ('mouthhooks', 'centroid_fullres_mouthhooks_holdout{seed}', 'Mouthhooks, real only'),
        ('mouthhooks', 'centroid_synth_fullframe_mouthhooks_holdout{seed}', 'Mouthhooks + synthetic')]


def paired_bars(output, labels, panels, values, limits):
    """values[panel][seed] -> list aligned with labels; limits[panel] -> (xmax, is_percent)."""
    prism_style()
    colors = [gradient_colors(4)[1], gradient_colors(4)[3]]
    y = np.arange(len(labels))
    fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), max(4.5, len(labels) * 0.72)), sharey=True)
    for ax, (key, title, xlabel) in zip(axes, panels):
        for s, ((_, seed_label), color) in enumerate(zip(SEEDS, colors)):
            v = values[key][s]
            bars = ax.barh(y + (s - 0.5) * 0.38, v, 0.36, color=color, edgecolor='black', linewidth=1.1,
                           label=seed_label, zorder=3)
            ax.bar_label(bars, fmt='%.1f', fontsize=10, padding=3)
        ax.set_yticks(y, labels, fontsize=12)
        ax.tick_params(axis='y', labelleft=True)
        if limits[key][1]:
            format_percent_axis(ax, horizontal=True)
        else:
            ax.set_xlim(0, limits[key][0] * 1.15)
            ax.spines['bottom'].set_bounds(0, limits[key][0])
        ax.set_xlabel(xlabel)
        ax.set_title(title)
    axes[0].invert_yaxis()  # shared y-axis: flip once, top-to-bottom in list order
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, loc='lower center', ncol=2, fontsize=12, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(w_pad=2.2, rect=(0, 0.05, 1, 1))
    save_plot(fig, output)
    print(f'Wrote {output.with_suffix(".png")}')


def model_metrics(node, model, crowded, hard):
    hits, fp, meta = load_hits(A / f'anchors_{node}', model)
    recall, missed = combined_recall([hits], crowded, hard, meta['repeats'])
    summary = {(r['model'], r['subset']): r for r in read_csv(A / f'anchors_{node}' / 'summary.csv')}
    return {**recall, 'precision': float(summary[model, 'all']['precision_pct']), 'missed': missed}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=A / 'plots_anchors')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _, _, meta = load_hits(A / 'anchors_body', 'centroid_fullres_body_holdout')
    crowded, hard = common_subsets(meta)
    frames = len(hard) * meta['repeats']
    recall_panels = [('all', 'Overall recall', 'Recall (%)'), ('hard', 'Hard-frame recall', 'Recall (%)'),
                     ('crowded', 'Crowded recall', 'Recall (%)'), ('precision', 'Precision', 'Precision (%)')]
    percent = {k: (100, True) for k, _, _ in recall_panels}

    # 1. Single anchors, real data only
    rows = {seed: [model_metrics(n, REAL.format(node=n, seed=seed), crowded, hard) for n, _ in ANCHORS] for seed, _ in SEEDS}
    values = {k: [[r[k] for r in rows[seed]] for seed, _ in SEEDS] for k, _, _ in recall_panels}
    paired_bars(args.output_dir / 'anchors_comparison', [l for _, l in ANCHORS], recall_panels, values, percent)

    # 2. Combinations of anchors, real data only
    combo_vals = {k: [[], []] for k in ('all', 'crowded', 'missed', 'fp')}
    for s, (seed, _) in enumerate(SEEDS):
        loaded = {n: load_hits(A / f'anchors_{n}', REAL.format(node=n, seed=seed)) for n, _ in ANCHORS}
        for combo, _ in COMBOS:
            recall, missed = combined_recall([loaded[n][0] for n in combo], crowded, hard, meta['repeats'])
            combo_vals['all'][s].append(recall['all'])
            combo_vals['crowded'][s].append(recall['crowded'])
            combo_vals['missed'][s].append(missed)
            combo_vals['fp'][s].append(sum(loaded[n][1] for n in combo) / frames)
    combo_panels = [('all', 'Overall recall', 'Recall (%)'), ('crowded', 'Crowded recall', 'Recall (%)'),
                    ('missed', 'Missed larvae', 'Larvae (of 368)'), ('fp', 'False positives', 'Per frame (upper bound)')]
    limits = {'all': (100, True), 'crowded': (100, True),
              'missed': (max(max(v) for v in combo_vals['missed']), False),
              'fp': (max(max(v) for v in combo_vals['fp']), False)}
    paired_bars(args.output_dir / 'anchor_combinations', [l for _, l in COMBOS], combo_panels, combo_vals, limits)

    # 3. Anchor x synthetic data grid
    grid_rows = {seed: [model_metrics(n, m.format(seed=seed), crowded, hard) for n, m, _ in GRID] for seed, _ in SEEDS}
    values = {k: [[r[k] for r in grid_rows[seed]] for seed, _ in SEEDS] for k, _, _ in recall_panels}
    paired_bars(args.output_dir / 'anchor_synthetic_grid', [l for _, _, l in GRID], recall_panels, values, percent)


if __name__ == '__main__':
    main()
