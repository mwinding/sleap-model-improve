#!/usr/bin/env python3
"""How crowded are the larvae? Distance from each larva's body point to the nearest other larva's skeleton
(the polyline through its labelled nodes). A larva counts as crowded at <= 15 px, the matching tolerance.

1. crowding_distribution: cumulative distribution of that distance for the training labels and the
   benchmark frames, and the share of crowded larvae in each training video.
2. crowding_examples: training larvae at increasing distances from their nearest neighbour.
Writes outputs/model_assessment/crowding/ (plots and crowding_by_video.csv).
"""
from __future__ import annotations
import argparse
import csv
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sleap_io as sio

from assess_centroids import ROOT, nearest_skeleton
from plot_anchor_overlap import spread_labels
from plot_assessment import gradient_colors, prism_style, save_plot

TRAIN = Path('/Volumes/lab-windingm/home/shared/sleap/groundtruth/combined/combined_ground_truth_train.pkg.slp')
BENCHMARK = ROOT / 'outputs/benchmark/benchmark_frames.pkg.slp'
GROUND_TRUTH = ROOT / 'data/combined_ground_truth_cleaned_visibility-fixed.pkg.slp'   # original video names
CROWDED_PX = 15.0
BINS = [(0, 5), (5, 10), (10, 15), (15, 20), (20, 30)]


def short_name(filename):
    date = re.search(r'(\d{4})[-_](\d{2})[-_](\d{2})', filename)
    camera = re.search(r'(SV\d+|DLC\d+)', filename)
    return f"{camera.group(1) if camera else 'video'} ({'-'.join(date.groups()) if date else '?'})"


def original_name(video):
    """File name of the video the frames came from (embedded videos keep it as source_video)."""
    return (video.source_video if video.source_video is not None else video).filename.split('/')[-1]


def larvae(labels):
    """One record per user-labelled larva, with the distance to the nearest other larva's skeleton."""
    body = labels.skeletons[0].node_names.index('body')
    records = []
    for frame in labels.labeled_frames:
        instances = frame.user_instances
        if not instances:
            continue
        points = np.stack([i.numpy() for i in instances])
        dist, owner, closest = nearest_skeleton(instances, points[:, body])
        for k in range(len(instances)):
            records.append({'frame': frame, 'video': labels.videos.index(frame.video), 'points': points, 'index': k,
                            'dist': dist[k], 'owner': owner[k], 'closest': closest[k], 'body': body})
    return records


def distribution_plot(train, bench, names, output):
    colors = gradient_colors(4)
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(15, 5.2), gridspec_kw={'width_ratios': [1.25, 1]})
    x = np.linspace(0, 60, 601)
    ends = []
    for records, label, color in ((train, 'Training labels', colors[3]), (bench, 'Benchmark', colors[1])):
        d = np.array([r['dist'] for r in records])
        y = 100 * (d[:, None] <= x[None, :]).mean(axis=0)
        ax.plot(x, y, color=color, linewidth=2.5)
        at = 100 * float((d <= CROWDED_PX).mean())
        ax.annotate(f'{at:.0f}%', (CROWDED_PX, at), xytext=(-8, 6), textcoords='offset points', ha='right',
                    color=color, fontsize=12, fontweight='bold')
        ends.append((label, color, y[-1], len(d)))
    for (label, color, _, n), yl in zip(ends, spread_labels(np.array([e[2] for e in ends]), 7)):
        ax.text(x[-1] + 1, yl, f'{label}, n = {n}', color=color, fontsize=12, va='center')
    ax.axvline(CROWDED_PX, color='0.4', linestyle='--', linewidth=1.2, zorder=0)
    ax.text(CROWDED_PX + 1, 4, '15 px: crowded', color='0.3', fontsize=11)
    ax.set_xlim(0, 60)
    ax.set_ylim(0, 100)
    ax.set_xlabel('Distance to nearest other larva (px)')
    ax.set_ylabel('Larvae (%)')
    ax.set_title('Body point to nearest other larva')

    per_video = {}
    for r in train:
        n, c = per_video.get(r['video'], (0, 0))
        per_video[r['video']] = (n + 1, c + (r['dist'] <= CROWDED_PX))
    order = sorted(per_video, key=lambda v: per_video[v][1] / per_video[v][0], reverse=True)
    pct = [100 * per_video[v][1] / per_video[v][0] for v in order]
    y = np.arange(len(order))
    bars = bx.barh(y, pct, 0.7, color=colors[3], edgecolor='black', linewidth=1.1, zorder=3)
    bx.bar_label(bars, labels=[f'{p:.0f}% of {per_video[v][0]}' for p, v in zip(pct, order)], fontsize=10, padding=3)
    overall = 100 * float(np.mean([r['dist'] <= CROWDED_PX for r in train]))
    bx.set_yticks(y, [names[v] for v in order], fontsize=11)
    bx.invert_yaxis()
    bx.set_xlim(0, 100)
    bx.set_xlabel('Larvae (%)')
    bx.set_title(f'Crowded larvae per training video (all: {overall:.0f}%)')
    fig.tight_layout(w_pad=4)
    save_plot(fig, output)
    return per_video, order


def examples_plot(train, output, per_bin=3, window=150, seed=0):
    rng = np.random.default_rng(seed)
    chosen = []
    for lo, hi in BINS:
        pool = [r for r in train if lo <= r['dist'] < hi]
        picks, seen = [], set()
        for i in rng.permutation(len(pool)):          # prefer different videos within a bin
            if pool[i]['video'] not in seen or len(seen) >= per_bin:
                picks.append(pool[i]); seen.add(pool[i]['video'])
            if len(picks) == per_bin:
                break
        chosen.append(picks)
    fig, axes = plt.subplots(per_bin, len(BINS), figsize=(2.9 * len(BINS), 2.9 * per_bin))
    for col, ((lo, hi), picks) in enumerate(zip(BINS, chosen)):
        for row, r in enumerate(picks):
            ax = axes[row, col]
            image = np.asarray(r['frame'].image)
            image = image[..., 0] if image.ndim == 3 and image.shape[-1] == 1 else image
            own, other = r['points'][r['index']], r['points'][r['owner']]
            centre = (own[r['body']] + r['closest']) / 2
            h, w = image.shape[:2]
            x0 = int(np.clip(centre[0] - window / 2, 0, w - window))
            y0 = int(np.clip(centre[1] - window / 2, 0, h - window))
            ax.imshow(image[y0:y0 + window, x0:x0 + window], cmap='gray' if image.ndim == 2 else None, vmin=0, vmax=255)
            for pts, color in ((other, '#ff4136'), (own, 'white')):
                ok = np.isfinite(pts).all(axis=1)
                ax.plot(pts[ok, 0] - x0, pts[ok, 1] - y0, color=color, linewidth=2.5, solid_capstyle='round', zorder=3)
            b, c = own[r['body']], r['closest']
            ax.plot([b[0] - x0, c[0] - x0], [b[1] - y0, c[1] - y0], color='#ffdc00', linewidth=2, zorder=4)
            ax.plot(b[0] - x0, b[1] - y0, 'o', color='white', markeredgecolor='black', markeredgewidth=1.2,
                    markersize=8, zorder=5)
            ax.text(4, window - 6, f"{r['dist']:.1f} px", color='white', fontsize=12, fontweight='bold', va='bottom',
                    bbox={'facecolor': 'black', 'alpha': 0.55, 'pad': 2, 'linewidth': 0})
            ax.set_xlim(0, window); ax.set_ylim(window, 0)
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
        state = 'crowded' if hi <= CROWDED_PX else 'not crowded'
        axes[0, col].set_title(f'{lo}–{hi} px\n{state}', fontsize=14, fontweight='bold', pad=8)
    fig.tight_layout(h_pad=0.4, w_pad=0.4)
    save_plot(fig, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', type=Path, default=TRAIN)
    parser.add_argument('--benchmark', type=Path, default=BENCHMARK)
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/model_assessment/crowding')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    gt = sio.load_slp(str(GROUND_TRUTH), open_videos=False)
    names = {i: short_name(original_name(v)) for i, v in enumerate(gt.videos)}
    train = larvae(sio.load_slp(str(args.train)))
    bench = larvae(sio.load_slp(str(args.benchmark), open_videos=False))
    prism_style()
    per_video, order = distribution_plot(train, bench, names, args.output / 'crowding_distribution')
    with open(args.output / 'crowding_by_video.csv', 'w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['video_index', 'video', 'larvae', 'crowded', 'crowded_pct'])
        for v in order:
            n, c = per_video[v]
            writer.writerow([v, original_name(gt.videos[v]), n, c, f'{100 * c / n:.1f}'])
    examples_plot(train, args.output / 'crowding_examples')
    print(f'Wrote plots and crowding_by_video.csv to {args.output}')


if __name__ == '__main__':
    main()
