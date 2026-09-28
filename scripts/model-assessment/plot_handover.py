#!/usr/bin/env python3
"""Where do end-anchored pipelines put a larva's body point? ("handover" to the body pose model)

In the two-step merge, a head/mouthhooks/tail detection is turned into a skeleton by that
anchor's pose model, and only the skeleton's body point is kept and handed to the body pose
model. This checks whether that body point lands on the detected larva, on a neighbour, on
no larva, or is missing:
  1. real pipeline on the test video: each detection is assigned to the larva whose anchor it
     sits on (within 15 px); the body point is assigned to the nearest labelled body (15 px);
  2. crops centred on each labelled larva's anchor (stage-2 predictions), which isolates the
     pose models from detection.
Outputs body_point_handover.png (summary), body_point_handover_examples.png (random handover
cases on crowded larvae) and body_point_handover.csv.
"""
from __future__ import annotations
import argparse
import csv
import warnings
from pathlib import Path

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sleap_io as sio

from assess_centroids import ROOT, crowded_flags, decode_source, load_benchmark
from assess_stage2 import frame_key, pair_with_larvae
from plot_assessment import format_percent_axis, prism_style, save_plot
from tune_merge import test_truth

A = ROOT / 'outputs/model_assessment'
GT = ROOT / 'data/combined_ground_truth_cleaned_visibility-fixed.pkg.slp'
NODES = ['head', 'mouthhooks', 'body', 'tail', 'spiracle']
BODY = NODES.index('body')
ANCHORS = ['head', 'mouthhooks', 'tail']
OUTCOMES = [('own', 'Own larva', '#2e8b57'), ('neighbour', 'A neighbour', '#e0453a'),
            ('nothing', 'No larva', '#b0b0b0'), ('missing', 'No body point', '#dcdcdc')]
SUBSETS = [('crowded', 'Crowded'), ('others', 'All others'), ('total', 'Total')]
TOL = 15.0


def outcome(body_point, gt, own):
    if not np.isfinite(body_point).all():
        return 'missing', None
    d = np.linalg.norm(gt[:, BODY] - body_point, axis=1)
    j = int(np.argmin(d))
    if d[j] > TOL:
        return 'nothing', None
    return ('own' if j == own else 'neighbour'), j


def empty_counts():
    return {s: {o: 0 for o, _, _ in OUTCOMES} for s, _ in SUBSETS}


def pipeline_cases(folder):
    """Real pipeline: every detection on a labelled larva, with its skeleton and outcome (repeat 0)."""
    truth = test_truth()
    counts, cases = {a: empty_counts() for a in ANCHORS}, []
    for a in ANCHORS:
        k = NODES.index(a)
        crops = {int(f.frame_idx): np.stack([i.numpy() for i in f.instances])
                 for f in sio.load_slp(str(folder / f'crops_{a}.slp'), open_videos=False)}
        known = {int(f.frame_idx): np.stack([i.numpy() for i in f.instances])
                 for f in sio.load_slp(str(folder / f'known_{a}.slp'), open_videos=False) if len(f.instances)}
        redrawn = {int(f.frame_idx): np.stack([i.numpy() for i in f.instances])
                   for f in sio.load_slp(str(folder / f'redrawn_{a}.slp'), open_videos=False) if len(f.instances)}
        for frame, centres in crops.items():
            if frame % 10:
                continue
            gt, crowded = truth[frame]
            skel = known.get(frame)
            aligned = pair_with_larvae(centres, skel, frame, a) if skel is not None else np.full_like(centres, np.nan)
            for det, s in zip(centres[:, k], aligned):
                d = np.linalg.norm(gt[:, k] - det, axis=1)
                own = int(np.argmin(d))
                if d[own] > TOL:
                    continue
                o, neighbour = outcome(s[BODY], gt, own)
                for sub in ('crowded' if crowded[own] else 'others', 'total'):
                    counts[a][sub][o] += 1
                redraw = None
                if o != 'missing' and frame in redrawn:
                    r = redrawn[frame]
                    dd = np.linalg.norm(r[:, BODY] - s[BODY], axis=1)
                    if np.nanmin(dd) < 3:
                        redraw = r[int(np.nanargmin(dd))]
                cases.append({'anchor': a, 'frame': frame, 'order': frame // 10 + 1, 'own': own, 'neighbour': neighbour,
                              'crowded': bool(crowded[own]), 'outcome': o, 'detection': det, 'skeleton': s,
                              'redrawn': redraw})
    return counts, cases


def stage2_counts():
    labels = sio.load_slp(str(A / '../benchmark/benchmark_frames.pkg.slp'), open_videos=False)
    truth = {}
    for f in labels:
        gt = np.stack([i.numpy() for i in f.instances])
        truth[frame_key(f)] = (gt, crowded_flags(list(f.instances), gt[:, BODY], 15.0))
    counts = {a: empty_counts() for a in ANCHORS}
    for a in ANCHORS:
        for f in sio.load_slp(str(A / f'stage2/predictions/centered_instance_{a}_holdout.slp'), open_videos=False):
            gt, crowded = truth[frame_key(f)]
            pred = pair_with_larvae(gt, np.stack([i.numpy() for i in f.instances]), frame_key(f), a)
            for i in range(len(gt)):
                o, _ = outcome(pred[i, BODY], gt, i)
                for sub in ('crowded' if crowded[i] else 'others', 'total'):
                    counts[a][sub][o] += 1
    return counts


OWNERSHIP_ANCHORS = ['body', 'head', 'mouthhooks', 'tail']
OWNERSHIP = [('own', 'Detected larva', '#2e8b57'), ('neighbour', 'A neighbour', '#e0453a'),
             ('nothing', 'No larva', '#b0b0b0'), ('missing', 'No skeleton', '#dcdcdc')]


def skeleton_owner(skeleton, gt, own):
    """Whose skeleton is it: the detected larva, a neighbour, no larva (>15 px mean from all), or missing."""
    if not np.isfinite(skeleton).any():
        return 'missing'
    d = np.nanmean(np.linalg.norm(gt - skeleton[None], axis=2), axis=1)
    j = int(np.nanargmin(d))
    if d[j] > TOL:
        return 'nothing'
    return 'own' if j == own else 'neighbour'


def ownership_counts(folder):
    """Real pipeline and crops at the labelled anchor, for all four pipelines including body."""
    truth = test_truth()
    real = {a: {s: {o: 0 for o, _, _ in OWNERSHIP} for s, _ in SUBSETS} for a in OWNERSHIP_ANCHORS}
    for a in OWNERSHIP_ANCHORS:
        k = NODES.index(a)
        crops = {int(f.frame_idx): np.stack([i.numpy() for i in f.instances])
                 for f in sio.load_slp(str(folder / f'crops_{a}.slp'), open_videos=False)}
        known = {int(f.frame_idx): np.stack([i.numpy() for i in f.instances])
                 for f in sio.load_slp(str(folder / f'known_{a}.slp'), open_videos=False) if len(f.instances)}
        for frame, centres in crops.items():
            if frame % 10:
                continue
            gt, crowded = truth[frame]
            skel = known.get(frame)
            aligned = pair_with_larvae(centres, skel, frame, a) if skel is not None else np.full_like(centres, np.nan)
            for det, s in zip(centres[:, k], aligned):
                d = np.linalg.norm(gt[:, k] - det, axis=1)
                own = int(np.argmin(d))
                if d[own] > TOL:
                    continue
                o = skeleton_owner(s, gt, own)
                for sub in ('crowded' if crowded[own] else 'others', 'total'):
                    real[a][sub][o] += 1
    labels = sio.load_slp(str(A / '../benchmark/benchmark_frames.pkg.slp'), open_videos=False)
    truth2 = {}
    for f in labels:
        gt = np.stack([i.numpy() for i in f.instances])
        truth2[frame_key(f)] = (gt, crowded_flags(list(f.instances), gt[:, BODY], 15.0))
    crops = {a: {s: {o: 0 for o, _, _ in OWNERSHIP} for s, _ in SUBSETS} for a in OWNERSHIP_ANCHORS}
    for a in OWNERSHIP_ANCHORS:
        for f in sio.load_slp(str(A / f'stage2/predictions/centered_instance_{a}_holdout.slp'), open_videos=False):
            gt, crowded = truth2[frame_key(f)]
            pred = pair_with_larvae(gt, np.stack([i.numpy() for i in f.instances]), frame_key(f), a)
            for i in range(len(gt)):
                o = skeleton_owner(pred[i], gt, i)
                for sub in ('crowded' if crowded[i] else 'others', 'total'):
                    crops[a][sub][o] += 1
    return real, crops


def summary_plot(panels, output, anchors=ANCHORS, outcomes=OUTCOMES, legend_title='Body point lands on',
                 xlabels=('Detections (%)', 'Larvae (%)')):
    fig, axes = plt.subplots(1, len(panels), figsize=(8.2 * len(panels), 1.6 + 0.62 * 3 * len(anchors)), sharey=True)
    rows = [(a, s) for a in anchors for s, _ in SUBSETS]
    y, pos = [], 0.0
    for r, (a, s) in enumerate(rows):
        if r and s == 'crowded':
            pos += 0.6
        y.append(pos)
        pos += 1.0
    labels = [f"{a.capitalize()}: {dict(SUBSETS)[s].lower()}" for a, s in rows]
    for ax, (title, counts), xlabel in zip(axes, panels, xlabels):
        left = np.zeros(len(rows))
        for o, name, color in outcomes:
            vals = np.array([100 * counts[a][s][o] / max(sum(counts[a][s].values()), 1) for a, s in rows])
            bars = ax.barh(y, vals, 0.72, left=left, color=color, edgecolor='white', linewidth=1.2, label=name, zorder=3)
            for bar, v, l in zip(bars, vals, left):
                if v >= 6:
                    ax.text(l + v / 2, bar.get_y() + bar.get_height() / 2, f'{v:.0f}', ha='center', va='center',
                            fontsize=11, color='white' if o in ('own', 'neighbour') else 'black', zorder=4)
            left += vals
        ax.set_yticks(y, labels, fontsize=12)
        ax.tick_params(axis='y', labelleft=True)
        ax.set_ylim(pos - 0.2, -0.8)
        format_percent_axis(ax, horizontal=True)
        ax.set_xlabel(xlabel)
        ax.set_title(title)
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, loc='lower center', ncol=len(outcomes), fontsize=12, bbox_to_anchor=(0.5, -0.02),
               title=legend_title, title_fontsize=12)
    fig.tight_layout(w_pad=3, rect=(0, 0.07, 1, 1))
    save_plot(fig, output)


def draw(ax, image, gts, centre, window, own=None, skeleton=None, colour=None, dot=None, ring=None):
    h, w = image.shape[:2]
    x0 = int(np.clip(centre[0] - window / 2, 0, w - window)); y0 = int(np.clip(centre[1] - window / 2, 0, h - window))
    ax.imshow(image[y0:y0 + window, x0:x0 + window], vmin=0, vmax=255)
    for j, g in enumerate(gts):
        ok = np.isfinite(g).all(axis=1)
        thick = own is not None and j == own and skeleton is None
        ax.plot(g[ok, 0] - x0, g[ok, 1] - y0, color='white', linewidth=2.8 if thick else 1.0, zorder=2)
    if skeleton is not None:
        ok = np.isfinite(skeleton).all(axis=1)
        ax.plot(skeleton[ok, 0] - x0, skeleton[ok, 1] - y0, color=colour, linewidth=3, solid_capstyle='round', zorder=3)
    if ring is not None:
        ax.plot(ring[0] - x0, ring[1] - y0, 'o', markerfacecolor='none', markeredgecolor=colour, markeredgewidth=2.2,
                markersize=12, zorder=4)
    if dot is not None:
        ax.plot(dot[0] - x0, dot[1] - y0, 'o', color='white', markeredgecolor='black', markeredgewidth=1.2, markersize=8, zorder=5)
    ax.set_xlim(0, window); ax.set_ylim(window, 0)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def examples_plot(cases, n, seed, output, window=200):
    pool = [c for c in cases if c['crowded'] and c['outcome'] == 'neighbour' and c['redrawn'] is not None
            and c['anchor'] in ('mouthhooks', 'tail')]
    rng = np.random.default_rng(seed)
    chosen = [pool[i] for i in sorted(rng.choice(len(pool), size=min(n, len(pool)), replace=False))]
    rows = {r['order']: r for r in load_benchmark(Path(__file__).with_name('comparison_manifest.csv'), GT, set(), 'body')}
    truth = test_truth()
    red, orange = '#e0453a', '#ff9f1c'
    fig, axes = plt.subplots(len(chosen), 3, figsize=(10, 3.4 * len(chosen)))
    axes = np.atleast_2d(axes)
    with h5py.File(GT) as handle:
        for r, c in enumerate(chosen):
            row = rows[c['order']]
            image = decode_source(handle, row['video_id'], row['frame_idx'])[..., ::-1]
            gts = truth[c['frame']][0]
            centre = (gts[c['own'], BODY] + gts[c['neighbour'], BODY]) / 2
            draw(axes[r, 0], image, gts, centre, window, own=c['own'], dot=c['detection'])
            draw(axes[r, 1], image, gts, centre, window, skeleton=c['skeleton'], colour=red, dot=c['detection'],
                 ring=c['skeleton'][BODY])
            draw(axes[r, 2], image, gts, centre, window, skeleton=c['redrawn'], colour=orange, ring=c['skeleton'][BODY])
            axes[r, 0].set_ylabel(c['anchor'].capitalize(), fontsize=13, fontweight='bold')
    for ax, name in zip(axes[0], ['Detected larva', 'Its pose-model skeleton', 'Redrawn by body model']):
        ax.set_title(name, fontsize=14, fontweight='bold', pad=8)
    fig.tight_layout(h_pad=0.4, w_pad=0.3)
    save_plot(fig, output)
    return len(pool)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--pipeline-dir', type=Path, default=A / 'topdown/predictions')
    parser.add_argument('--output-dir', type=Path, default=A / 'topdown/plots')
    parser.add_argument('--examples', type=int, default=8)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    warnings.simplefilter('ignore', RuntimeWarning)
    prism_style()
    real, cases = pipeline_cases(args.pipeline_dir)
    crops = stage2_counts()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_plot([('Real pipeline, test video', real), ('Crops at the labelled anchor', crops)],
                 args.output_dir / 'body_point_handover')
    with (args.output_dir / 'body_point_handover.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['analysis', 'anchor', 'subset', 'n', *[o for o, _, _ in OUTCOMES]])
        for name, counts in (('real_pipeline', real), ('crops_at_anchor', crops)):
            for a in ANCHORS:
                for s, _ in SUBSETS:
                    writer.writerow([name, a, s, sum(counts[a][s].values()), *[counts[a][s][o] for o, _, _ in OUTCOMES]])
    real_own, crops_own = ownership_counts(args.pipeline_dir)
    summary_plot([('Real pipeline, test video', real_own), ('Crops at the labelled anchor', crops_own)],
                 args.output_dir / 'skeleton_ownership', anchors=OWNERSHIP_ANCHORS, outcomes=OWNERSHIP,
                 legend_title='Skeleton belongs to')
    with (args.output_dir / 'skeleton_ownership.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['analysis', 'anchor', 'subset', 'n', *[o for o, _, _ in OWNERSHIP]])
        for name, counts in (('real_pipeline', real_own), ('crops_at_anchor', crops_own)):
            for a in OWNERSHIP_ANCHORS:
                for sub, _ in SUBSETS:
                    writer.writerow([name, a, sub, sum(counts[a][sub].values()), *[counts[a][sub][o] for o, _, _ in OWNERSHIP]])
    pool = examples_plot(cases, args.examples, args.seed, args.output_dir / 'body_point_handover_examples')
    print(f'Wrote summary, examples (random {args.examples} of {pool} crowded handover cases) and CSV to {args.output_dir}')


if __name__ == '__main__':
    main()
