#!/usr/bin/env python3
"""Compare full-pose pipelines on the test frames: summary bars and crowded-region examples.

1. pipeline_comparison: recall, precision, crowded recall and nodes within 5 px for a list of
   prediction files (e.g. body pipeline alone vs merged pipelines), scored as in assess_poses.
2. pipeline_examples: for a fixed set of benchmark images, the densest labelled region, with
   every labelled larva drawn thin in white and each pipeline's skeletons on top: green when
   matched to a labelled larva, red when not (false positive). A labelled larva with nothing
   on top was missed.

  python scripts/model-assessment/plot_pipelines.py \
     --entry "Body alone" outputs/model_assessment/topdown/predictions/topdown_body.slp \
     --entry "Merged" outputs/model_assessment/topdown/predictions/merged_chosen.slp
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sleap_io as sio

from assess_centroids import ROOT, decode_source, load_benchmark, match_distances
from assess_poses import mean_node_distance
from plot_assessment import format_percent_axis, gradient_colors, prism_style, save_plot
from tune_merge import score, test_truth

GT = ROOT / 'data/combined_ground_truth_cleaned_visibility-fixed.pkg.slp'
CORRECT, WRONG, TRUTH = '#2ecc40', '#ff4136', 'white'


def load_preds(path):
    return {int(f.frame_idx): np.stack([i.numpy() for i in f.instances])
            for f in sio.load_slp(str(path), open_videos=False) if len(f.instances)}


def comparison_plot(entries, truth, output):
    panels = [('recall', 'Recall', 'Larvae (%)'), ('precision', 'Precision', 'Skeletons (%)'),
              ('crowded_recall', 'Crowded recall', 'Larvae (%)'), ('pck5', 'Nodes within 5 px', 'Nodes (%)')]
    results = [score(truth, load_preds(path)) for _, path in entries]
    colors = gradient_colors(len(entries) + 1)[1:]
    y = np.arange(len(entries))
    fig, axes = plt.subplots(1, len(panels), figsize=(20, max(3.6, 0.7 * len(entries) + 1.6)), sharey=True)
    for ax, (key, title, xlabel) in zip(axes, panels):
        bars = ax.barh(y, [r[key] for r in results], 0.64, color=colors, edgecolor='black', linewidth=1.2, zorder=3)
        ax.bar_label(bars, fmt='%.1f', fontsize=11, padding=3)
        ax.set_yticks(y, [name for name, _ in entries], fontsize=12)
        ax.tick_params(axis='y', labelleft=True)
        format_percent_axis(ax, horizontal=True)
        ax.set_xlabel(xlabel)
        ax.set_title(title)
    axes[0].invert_yaxis()
    fig.tight_layout(w_pad=2.2)
    save_plot(fig, output)
    return results


def densest_window(points, window):
    centres = points[:, 2]                                   # body points
    counts = [(np.linalg.norm(centres - c, axis=1) < window / 2).sum() for c in centres]
    return centres[int(np.argmax(counts))]


def draw_tile(ax, image, window, centre, gt, pred=None):
    h, w = image.shape[:2]
    x0 = int(np.clip(centre[0] - window / 2, 0, w - window)); y0 = int(np.clip(centre[1] - window / 2, 0, h - window))
    ax.imshow(image[y0:y0 + window, x0:x0 + window], vmin=0, vmax=255)
    width = 2.2 if pred is None else 1.1
    for g in gt:
        ok = np.isfinite(g).all(axis=1)
        ax.plot(g[ok, 0] - x0, g[ok, 1] - y0, color=TRUTH, linewidth=width, zorder=2)
    if pred is not None:
        matches = match_distances(mean_node_distance(gt, pred), 15.0) if len(pred) else []
        matched = {j for _, j, _ in matches}
        for j, p in enumerate(pred):
            ok = np.isfinite(p).all(axis=1)
            ax.plot(p[ok, 0] - x0, p[ok, 1] - y0, color=CORRECT if j in matched else WRONG, linewidth=2.6,
                    solid_capstyle='round', zorder=3)
    ax.set_xlim(0, window); ax.set_ylim(window, 0)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def examples_plot(entries, orders, output, window=320):
    rows = {r['order']: r for r in load_benchmark(Path(__file__).with_name('comparison_manifest.csv'), GT, set(), 'body')}
    labels = sio.load_slp(str(GT), open_videos=False)
    lookup = {(labels.videos.index(f.video), int(f.frame_idx)): f for f in labels}
    preds = [load_preds(path) for _, path in entries]
    columns = ['Labelled'] + [name for name, _ in entries]
    fig, axes = plt.subplots(len(orders), len(columns), figsize=(3.4 * len(columns), 3.4 * len(orders)))
    axes = np.atleast_2d(axes)
    with h5py.File(GT) as handle:
        for r, order in enumerate(orders):
            row = rows[order]
            image = decode_source(handle, row['video_id'], row['frame_idx'])[..., ::-1]
            gt = np.stack([i.numpy() for i in lookup[row['video_id'], row['frame_idx']].instances])
            centre = densest_window(gt, window)
            index = (order - 1) * 10                               # repeat 0 of this image
            draw_tile(axes[r, 0], image, window, centre, gt)
            for c, p in enumerate(preds, start=1):
                draw_tile(axes[r, c], image, window, centre, gt, p.get(index, np.empty((0, 5, 2))))
    for ax, name in zip(axes[0], columns):
        ax.set_title(name, fontsize=14, fontweight='bold', pad=8)
    fig.tight_layout(h_pad=0.3, w_pad=0.3)
    save_plot(fig, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--entry', nargs=2, action='append', required=True, metavar=('LABEL', 'SLP'))
    parser.add_argument('--example-orders', type=int, nargs='+', default=[1, 2, 3, 4, 8, 13],
                        help='Benchmark images for the examples (default: the 4 hard frames and 2 others)')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'outputs/model_assessment/topdown/plots')
    parser.add_argument('--examples-only', action='store_true')
    args = parser.parse_args()
    entries = [(name, Path(path)) for name, path in args.entry]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prism_style()
    if not args.examples_only:
        results = comparison_plot(entries, test_truth(), args.output_dir / 'pipeline_comparison')
        with (args.output_dir / 'pipeline_comparison.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=['pipeline', *results[0]])
            writer.writeheader()
            writer.writerows({'pipeline': name, **r} for (name, _), r in zip(entries, results))
    examples_plot(entries, args.example_orders, args.output_dir / 'pipeline_examples')
    print(f'Wrote plots to {args.output_dir}')


if __name__ == '__main__':
    main()
