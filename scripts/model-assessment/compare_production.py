#!/usr/bin/env python3
"""Compare two tracking pipelines on one unlabelled recording with a known number of larvae.

For each pipeline (a folder with <video>.predictions.slp and <video>.tracks.slp, as written by
sleap-track_batch / sleap-nn-track_batch): skeletons per frame (all, and complete = all nodes present), the share of
frames with every larva found as a complete skeleton, and the number and length of tracks. Writes summary.csv,
larvae_per_frame.png (complete skeletons per frame over the recording, 1 s rolling median), per_frame.csv,
disagreement.png (the frames where the pipelines differ most, both pipelines' skeletons drawn) and, with --gifs,
gifs/*.gif that flip between the first two pipelines on more such frames, cropped to the larvae: --gifs frames where
the second draws more complete skeletons and --gifs-worse where the first does. Counts cannot tell larvae from false
positives (skeletons on walls or debris), so look at the frames.

  python scripts/model-assessment/compare_production.py \\
      --entry "Current (UNet)" /Volumes/lab-windingm/home/users/windinm/sleap-tf-track-test \\
      --entry "New (ConvNeXt)" /Volumes/lab-windingm/home/users/windinm/sleap-nn-track-test \\
      --video 2026-03-10_14-31-01_SV37 --larvae 20 \\
      --video-file /Volumes/lab-windingm/home/shared/sleap/groundtruth/combined/videos/2026-03-10_14-31-01_SV37.mp4
"""
from __future__ import annotations
import argparse
import csv
import re
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sleap_io as sio
from PIL import Image

from assess_poses import mean_node_distance
from plot_anchor_overlap import spread_labels
from plot_assessment import gradient_colors, prism_style, save_plot

ROOT = Path(__file__).resolve().parents[2]
EDGES = [(0, 1), (1, 2), (2, 3), (3, 4)]


def per_frame(labels, n_frames, dup_px=10.0):
    """Skeletons, complete skeletons and duplicate skeletons (another skeleton within dup_px mean node distance)
    per frame; 0 for frames without predictions."""
    total, complete, dups = (np.zeros(n_frames, int) for _ in range(3))
    for frame in labels.labeled_frames:
        points = [i.numpy() for i in frame.instances]
        total[frame.frame_idx] = len(points)
        complete[frame.frame_idx] = sum(bool(np.isfinite(p).all()) for p in points)
        if len(points) > 1:
            d = mean_node_distance(np.stack(points), np.stack(points))
            np.fill_diagonal(d, np.inf)
            dups[frame.frame_idx] = int((np.nanmin(d, axis=1) < dup_px).sum()) // 2
    return total, complete, dups


def track_lengths(labels):
    counts = {}
    for frame in labels.labeled_frames:
        for inst in frame.instances:
            if inst.track is not None:
                counts[inst.track.name] = counts.get(inst.track.name, 0) + 1
    return np.array(sorted(counts.values(), reverse=True))


def draw(image, labels_frame, color, width=2):
    for inst in labels_frame.instances if labels_frame is not None else []:
        p = inst.numpy()
        for a, b in EDGES:
            if np.isfinite(p[[a, b]]).all():
                cv2.line(image, tuple(np.round(p[a]).astype(int)), tuple(np.round(p[b]).astype(int)), color, width, cv2.LINE_AA)
    return image


def pick_frames(diff, n, fps, sign=1, exclude=()):
    """Frames with the largest sign * diff, at least a minute apart from each other and from exclude."""
    picks = []
    for idx in np.argsort(-sign * diff):
        if sign * diff[idx] <= 0 or len(picks) == n:
            break
        if all(abs(idx - p) > fps * 60 for p in list(picks) + list(exclude)):
            picks.append(int(idx))
    return picks


def write_gif(path, image, frames, labels, counts, colors, margin=60):
    """Flip between the pipelines' skeletons on one frame, cropped to the area with skeletons."""
    points = np.concatenate([np.concatenate([i.numpy() for i in f.instances]) for f in frames if f is not None and len(f.instances)])
    points = points[np.isfinite(points).all(axis=1)]
    x0, y0 = np.maximum(np.floor(points.min(axis=0)) - margin, 0).astype(int)
    x1, y1 = np.minimum(np.ceil(points.max(axis=0)) + margin, [image.shape[1], image.shape[0]]).astype(int)
    tiles = []
    for frame, label, count, color in zip(frames, labels, counts, colors):
        tile = draw(image.copy(), frame, color, width=3)[y0:y1, x0:x1].copy()
        text = f'{label}: {count} complete'
        (w, h), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        cv2.rectangle(tile, (8, 8), (20 + w, 20 + h + base), (255, 255, 255), -1)
        cv2.putText(tile, text, (14, 14 + h), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
        tiles.append(Image.fromarray(tile))
    tiles[0].save(path, save_all=True, append_images=tiles[1:], duration=1000, loop=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--entry', nargs=2, action='append', required=True, metavar=('LABEL', 'FOLDER'))
    parser.add_argument('--video', required=True, help='video name without .mp4')
    parser.add_argument('--video-file', type=Path, help='the .mp4 (default: <first folder>/<video>.mp4)')
    parser.add_argument('--larvae', type=int, required=True, help='number of larvae in the arena')
    parser.add_argument('--fps', type=float, default=5.0, help='capture rate (sideview: 5 Hz; the mp4s declare 25 fps)')
    parser.add_argument('--examples', type=int, default=3)
    parser.add_argument('--gifs', type=int, default=0, help='GIFs of frames where the second pipeline finds more')
    parser.add_argument('--gifs-worse', type=int, default=0, help='GIFs of frames where the first pipeline finds more')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/model_assessment/production_test')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    video = sio.load_video(str(args.video_file or Path(args.entry[0][1]) / f'{args.video}.mp4'))
    n_frames = len(video)
    counts, tracks, lengths = {}, {}, {}
    for label, folder in args.entry:
        folder = Path(folder)
        counts[label] = per_frame(sio.load_slp(str(folder / f'{args.video}.predictions.slp'), open_videos=False), n_frames)
        tracked = sio.load_slp(str(folder / f'{args.video}.tracks.slp'), open_videos=False)
        tracks[label] = {f.frame_idx: f for f in tracked.labeled_frames}
        lengths[label] = track_lengths(tracked)
    # Frames where no pipeline finds anything (e.g. a blank end of the recording) are left out
    used = np.any([total > 0 for total, _, _ in counts.values()], axis=0)
    n_frames = int(np.flatnonzero(used).max()) + 1
    used = used[:n_frames]
    rows, series = [], {}
    for label, (total, complete, dups) in counts.items():
        total, complete, dups = total[:n_frames][used], complete[:n_frames][used], dups[:n_frames][used]
        series[label] = counts[label][1][:n_frames]
        rows.append({'pipeline': label, 'frames': int(used.sum()),
                     'skeletons_per_frame': round(total.mean(), 2), 'complete_per_frame': round(complete.mean(), 2),
                     'frames_all_larvae_complete_pct': round(100 * np.mean(complete >= args.larvae), 1),
                     'frames_missing_2plus_pct': round(100 * np.mean(complete <= args.larvae - 2), 1),
                     'frames_more_than_larvae_pct': round(100 * np.mean(total > args.larvae), 1),
                     'duplicates_per_frame': round(dups.mean(), 3),
                     'tracks': len(lengths[label]),
                     'tracks_over_half_video': int(np.sum(lengths[label] >= n_frames / 2)),
                     'median_track_length_s': round(np.median(lengths[label]) / args.fps, 1) if len(lengths[label]) else 0.0})
    with (args.output / 'summary.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for r in rows:
        print(r)

    # Complete skeletons per frame over the recording
    prism_style()
    colors = dict(zip(series, gradient_colors(len(series) + 1)[1:]))
    fig, ax = plt.subplots(figsize=(12, 4.5))
    minutes = np.arange(n_frames) / args.fps / 60
    ends, lowest = [], args.larvae
    for label, complete in series.items():
        smooth = pd.Series(complete).rolling(int(args.fps), center=True, min_periods=1).median().to_numpy()
        ax.plot(minutes, smooth, color=colors[label], lw=1.6)
        ends.append((label, smooth[-int(args.fps * 30):].mean()))
        lowest = min(lowest, np.percentile(smooth[used], 0.5))
    ax.axhline(args.larvae, color='0.5', lw=1, ls='--')
    labels = sorted(ends, key=lambda e: e[1]) + [(f'{args.larvae} larvae', args.larvae)]
    for (label, y), y_text in zip(labels, spread_labels([y for _, y in labels], min_gap=0.7)):
        ax.text(minutes[-1] * 1.01, y_text, label, color=colors.get(label, '0.4'), fontsize=12, va='center',
                fontweight='bold' if label in colors else 'normal')
    ax.set_xlim(0, minutes[-1])
    ax.set_ylim(np.floor(lowest) - 1, args.larvae + 1.5)
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('Larvae per frame')
    ax.set_title('Complete skeletons per frame (1 s median)')
    save_plot(fig, args.output / 'larvae_per_frame')

    # Frames where the two pipelines differ most (first two entries)
    a, b = list(series)[:2]
    diff = pd.Series(series[b] - series[a]).rolling(int(args.fps), center=True, min_periods=1).mean().to_numpy()
    picks = pick_frames(np.abs(diff), args.examples, args.fps)
    tiles = []
    for idx in sorted(picks):
        image = np.asarray(video[idx])
        pair = []
        for label in (a, b):
            tile = draw(image.copy(), tracks[label].get(idx), (255, 60, 60) if label == a else (60, 120, 255))
            cv2.putText(tile, f'{label}, frame {idx}: {series[label][idx]} complete', (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1.4, (0, 0, 0), 3, cv2.LINE_AA)
            pair.append(tile)
        tiles.append(np.concatenate(pair, axis=1))
    fig = plt.figure(figsize=(16, 4.6 * len(tiles)))
    plt.imshow(np.concatenate(tiles, axis=0))
    plt.axis('off')
    save_plot(fig, args.output / 'disagreement')
    pd.DataFrame({f'{label} complete': counts[label][1][:n_frames] for label in counts}).rename_axis('frame').to_csv(
        args.output / 'per_frame.csv')

    if args.gifs or args.gifs_worse:
        (args.output / 'gifs').mkdir(exist_ok=True)
        better = pick_frames(diff, args.gifs, args.fps)
        worse = pick_frames(diff, args.gifs_worse, args.fps, sign=-1, exclude=better)
        slug = lambda label: re.sub(r'[^a-z0-9]+', '_', label.lower()).strip('_')
        for kind, frames in ((f'more_{slug(b)}', better), (f'more_{slug(a)}', worse)):
            for idx in frames:
                write_gif(args.output / 'gifs' / f'{kind}_frame{idx:05d}.gif', np.asarray(video[idx]),
                          [tracks[a].get(idx), tracks[b].get(idx)], [a, b], [series[a][idx], series[b][idx]],
                          [(220, 40, 40), (40, 100, 240)])
        print(f'Wrote {len(better) + len(worse)} GIFs')
    print(f'Wrote {args.output}')


if __name__ == '__main__':
    main()
