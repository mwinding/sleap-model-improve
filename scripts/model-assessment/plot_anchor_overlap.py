#!/usr/bin/env python3
"""Plot how often each skeleton node coincides with the same node of another larva.

For each anchor choice, the curve is the percentage of larvae whose anchor point
lies within d px of another larva's same anchor point; two such points fuse into a
single confidence-map peak once d is below roughly twice the effective sigma.
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
from plot_assessment import format_percent_axis, gradient_colors, prism_style, save_plot


def nearest_same_node(frame, node):
    points = np.array([inst.numpy()[node] for inst in frame.instances])
    ok = np.isfinite(points).all(axis=1)
    dist = np.linalg.norm(points[:, None] - points[None], axis=2)
    np.fill_diagonal(dist, np.inf)
    dist[~ok] = np.inf
    dist[:, ~ok] = np.inf
    return dist.min(axis=1)[ok]


def spread_labels(values, min_gap):
    """Shift label positions apart so none are closer than min_gap, keeping their order."""
    order = np.argsort(values)
    placed = []
    for index in order:
        y = values[index]
        if placed and y - placed[-1][1] < min_gap:
            y = placed[-1][1] + min_gap
        placed.append((index, y))
    # If the stack drifted upward, share the excess between the two ends.
    shift = (placed[-1][1] - values[order[-1]]) / 2
    return [y - shift for _, y in sorted(placed)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ground-truth', type=Path, default=ROOT / 'data/combined_ground_truth_cleaned_visibility-fixed.pkg.slp')
    parser.add_argument('--manifest', type=Path, default=Path(__file__).with_name('comparison_manifest.csv'))
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/model_assessment/anchor_overlap')
    parser.add_argument('--max-distance', type=float, default=40)
    parser.add_argument('--fusion-distance', type=float, default=10, help='~2 x effective sigma of the current models')
    args = parser.parse_args()

    labels = sio.load_slp(str(args.ground_truth), open_videos=False)
    nodes = [n.name for n in labels.skeletons[0].nodes]
    bench = {(int(r['video_id']), int(r['frame_idx'])) for r in csv.DictReader(args.manifest.open())}
    subsets = {'Benchmark frames': [f for f in labels if (labels.videos.index(f.video), int(f.frame_idx)) in bench],
               'All labeled frames': list(labels)}
    distances = np.arange(0, args.max_distance + 1)

    prism_style()
    colors = gradient_colors(len(nodes))
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), sharey=True)
    rows = []
    for ax, (title, frames) in zip(axes, subsets.items()):
        n_larvae = sum(len(f.instances) for f in frames)
        ends = []
        for k, (node, color) in enumerate(zip(nodes, colors)):
            nearest = np.concatenate([nearest_same_node(f, k) for f in frames])
            curve = [100 * (nearest <= d).mean() for d in distances]
            ax.plot(distances, curve, color=color, zorder=3)
            ends.append((curve[-1], node, color))
            rows.extend({'subset': title, 'anchor': node, 'distance_px': int(d), 'percent_within': round(c, 2)}
                        for d, c in zip(distances, curve))
        ax.axvline(args.fusion_distance, color='0.35', linestyle=':', linewidth=1.6, zorder=2)
        ax.text(args.fusion_distance + 0.8, 79, f'peaks fuse\n(< 2σ = {args.fusion_distance:g} px)', fontsize=10, color='0.3', va='top')
        ax.set_title(f'{title} ({n_larvae} larvae)')
        ax.set_xlabel('Distance to nearest same node of another larva (px)')
        for y_label, (_, node, color) in zip(spread_labels([e[0] for e in ends], 4.0), ends):
            ax.text(args.max_distance + 0.8, y_label, node, color=color, fontsize=12, va='center', ha='left')
        ax.set_xlim(0, args.max_distance + 8)
        ax.set_xticks(np.arange(0, args.max_distance + 1, 10))
        ax.set_ylim(0, 80)
        ax.set_yticks(np.arange(0, 81, 20))
        ax.spines['left'].set_bounds(0, 80)
        ax.spines['bottom'].set_bounds(0, args.max_distance)
        ax.tick_params(axis='y', labelleft=True)
        ax.grid(False)
    axes[0].set_ylabel('Larvae with a neighbour within distance (%)')
    fig.tight_layout(w_pad=3)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_plot(fig, args.output)
    with args.output.with_suffix('.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    print(f'Wrote {args.output.with_suffix(".png")} and {args.output.with_suffix(".csv")}')


if __name__ == '__main__':
    main()
