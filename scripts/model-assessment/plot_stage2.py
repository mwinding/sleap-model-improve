#!/usr/bin/env python3
"""Plots for the pose-model (stage 2) evaluation on crops centred at the labelled anchor.

1. stage2_summary: wrong-larva rate, skeletons with all nodes within 10 px, nodes within
   5 px and complete skeletons, for all larvae and crowded larvae.
2. stage2_nodes: share of each node within 5 px, per model (all and crowded larvae).
3. stage2_examples: a random (fixed-seed) sample of crowded larvae, each drawn by every
   model next to its labelled skeleton.
Reads outputs/model_assessment/stage2 (from assess_stage2.py).
"""
from __future__ import annotations
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sleap_io as sio

from assess_centroids import ROOT
from assess_stage2 import frame_key
from plot_assessment import format_percent_axis, gradient_colors, prism_style, read_csv, save_plot

S = ROOT / 'outputs/model_assessment/stage2'
LABELS = {'centered_instance_body_holdout': 'Body', 'centered_instance_head_holdout': 'Head',
          'centered_instance_mouthhooks_holdout': 'Mouthhooks', 'centered_instance_tail_holdout': 'Tail'}
ORDER = ['centered_instance_body_holdout', 'centered_instance_head_holdout',
         'centered_instance_mouthhooks_holdout', 'centered_instance_tail_holdout']


def summary_plot(summary, models, output):
    panels = [('wrong_animal_pct', 'Wrong larva', 'Larvae (%)'),
              ('all_nodes_within_10px_pct', 'All nodes within 10 px', 'Larvae (%)'),
              ('pck5_pct', 'Nodes within 5 px', 'Nodes (%)'),
              ('complete_pct', 'Complete skeletons', 'Larvae (%)')]
    subsets = [('all', 'All larvae'), ('crowded', 'Crowded larvae')]
    colors = [gradient_colors(4)[1], gradient_colors(4)[3]]
    y = np.arange(len(models))
    fig, axes = plt.subplots(1, len(panels), figsize=(20, max(3.8, 0.9 * len(models) + 1.5)), sharey=True)
    for ax, (key, title, xlabel) in zip(axes, panels):
        for s, ((subset, name), color) in enumerate(zip(subsets, colors)):
            v = [float(summary[m, subset][key]) for m in models]
            bars = ax.barh(y + (s - 0.5) * 0.38, v, 0.36, color=color, edgecolor='black', linewidth=1.1, label=name, zorder=3)
            ax.bar_label(bars, fmt='%.1f', fontsize=10, padding=3)
        ax.set_yticks(y, [LABELS.get(m, m) for m in models], fontsize=12)
        ax.tick_params(axis='y', labelleft=True)
        format_percent_axis(ax, horizontal=True)
        ax.set_xlabel(xlabel)
        ax.set_title(title)
    axes[0].invert_yaxis()
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, loc='lower center', ncol=2, fontsize=12, bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(w_pad=2.2, rect=(0, 0.07, 1, 1))
    save_plot(fig, output)


def nodes_plot(nodes, models, node_names, output):
    colors = gradient_colors(len(models) + 1)[1:]
    x = np.arange(len(node_names))
    width = 0.8 / len(models)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), sharey=True)
    for ax, (subset, title) in zip(axes, [('all', 'All larvae'), ('crowded', 'Crowded larvae')]):
        for k, (m, color) in enumerate(zip(models, colors)):
            # The crop is centred on the model's own anchor, so that node is trivially right: leave it out.
            v = [np.nan if n == LABELS[m].lower() else float(nodes[m, subset, n]['pck5_pct']) for n in node_names]
            bars = ax.bar(x + (k - (len(models) - 1) / 2) * width, v, width * 0.92, color=color,
                          edgecolor='black', linewidth=1.0, label=LABELS.get(m, m), zorder=3)
            ax.bar_label(bars, labels=['' if np.isnan(a) else f'{a:.0f}' for a in v], fontsize=9, padding=2)
        ax.set_xticks(x, [n.capitalize() for n in node_names], fontsize=12)
        format_percent_axis(ax)
        ax.tick_params(axis='y', labelleft=True)
        ax.set_ylabel('Within 5 px (%)')
        ax.set_title(title)
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, loc='lower center', ncol=len(models), fontsize=12, bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(w_pad=3, rect=(0, 0.07, 1, 1))
    save_plot(fig, output)


CORRECT, WRONG, TRUTH = '#2ecc40', '#ff4136', 'white'


def draw(ax, image, skeleton, view_centre, window, colour, reference=None, anchor=None):
    """One crop: optional thin reference skeleton, the skeleton itself, and the crop centre as a dot."""
    h, w = image.shape[:2]
    x0 = int(np.clip(view_centre[0] - window / 2, 0, w - window))
    y0 = int(np.clip(view_centre[1] - window / 2, 0, h - window))
    crop = image[y0:y0 + window, x0:x0 + window]
    ax.imshow(crop, cmap='gray' if crop.ndim == 2 else None, vmin=0, vmax=255)
    if reference is not None:
        ok = np.isfinite(reference).all(axis=1)
        ax.plot(reference[ok, 0] - x0, reference[ok, 1] - y0, color=TRUTH, linewidth=1.2, zorder=2)
    ok = np.isfinite(skeleton).all(axis=1)
    ax.plot(skeleton[ok, 0] - x0, skeleton[ok, 1] - y0, color=colour, linewidth=3, solid_capstyle='round', zorder=3)
    if anchor is not None:
        ax.plot(anchor[0] - x0, anchor[1] - y0, 'o', color='white', markeredgecolor='black', markeredgewidth=1.2,
                markersize=8, zorder=4)
    ax.set_xlim(0, window); ax.set_ylim(window, 0)   # points outside the tile must not widen it
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def examples_plot(per_animal, models, ground_truth, predictions_dir, n_examples, seed, output, window=180, max_error=5.0):
    gt_labels = sio.load_slp(str(ground_truth))
    names = gt_labels.skeletons[0].node_names
    frames = {frame_key(f): f for f in gt_labels}
    preds = {m: {frame_key(f): np.stack([i.numpy() for i in f.instances])
                 for f in sio.load_slp(str(predictions_dir / f'{m}.slp'), open_videos=False)} for m in models}
    first = [r for r in per_animal if r['model'] == models[0] and r['crowded'] == 'True']
    rng = np.random.default_rng(seed)
    chosen = [first[i] for i in sorted(rng.choice(len(first), size=min(n_examples, len(first)), replace=False))]
    lookup = {(r['model'], r['video'], r['frame_idx'], r['gt_index']): r for r in per_animal}
    columns = ['Labelled'] + [LABELS.get(m, m) for m in models]
    fig, axes = plt.subplots(len(chosen), len(columns), figsize=(2.6 * len(columns), 2.6 * len(chosen)))
    axes = np.atleast_2d(axes)
    for row, r in enumerate(chosen):
        key = (r['video'], int(r['frame_idx']))
        frame = frames[key]
        image = np.asarray(frame.image)
        image = image[..., 0] if image.ndim == 3 and image.shape[-1] == 1 else image
        i = int(r['gt_index'])
        gt = frame.instances[i].numpy()
        centre = gt[names.index('body')]
        draw(axes[row, 0], image, gt, centre, window, TRUTH)
        for col, m in enumerate(models, start=1):
            rec = lookup[m, r['video'], r['frame_idx'], r['gt_index']]
            complete = all(rec[f'err_{n}'] for n in names)
            correct = rec['wrong_animal'] != 'True' and complete and float(rec['mean_error_px']) <= max_error
            draw(axes[row, col], image, preds[m][key][i], centre, window, CORRECT if correct else WRONG,
                 reference=gt, anchor=gt[names.index(LABELS[m].lower())])
    for ax, name in zip(axes[0], columns):
        ax.set_title(name, fontsize=14, fontweight='bold', pad=8)
    fig.tight_layout(h_pad=0.3, w_pad=0.3)
    save_plot(fig, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage2', type=Path, default=S)
    parser.add_argument('--examples', type=int, default=8)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    rows = read_csv(args.stage2 / 'stage2_summary.csv')
    models = [m for m in ORDER if any(r['model'] == m for r in rows)]
    summary = {(r['model'], r['subset']): r for r in rows}
    nodes = {(r['model'], r['subset'], r['node']): r for r in read_csv(args.stage2 / 'stage2_nodes.csv')}
    node_names = list(dict.fromkeys(r['node'] for r in read_csv(args.stage2 / 'stage2_nodes.csv')))
    meta = __import__('json').loads((args.stage2 / 'stage2_assessment.json').read_text())
    out = args.stage2 / 'plots'
    out.mkdir(parents=True, exist_ok=True)
    prism_style()
    summary_plot(summary, models, out / 'stage2_summary')
    nodes_plot(nodes, models, node_names, out / 'stage2_nodes')
    examples_plot(read_csv(args.stage2 / 'stage2_per_animal.csv'), models, Path(meta['ground_truth']),
                  args.stage2 / 'predictions', args.examples, args.seed, out / 'stage2_examples')
    print(f'Wrote plots to {out}')


if __name__ == '__main__':
    main()
