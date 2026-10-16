#!/usr/bin/env python3
"""Compare two tracking pipelines on one unlabelled recording with a known number of larvae.

For each pipeline (a folder with <video>.predictions.slp and <video>.tracks.slp, as written by
sleap-track_batch / sleap-nn-track_batch): skeletons per frame (all, and complete = all nodes present), the share of
frames with every larva found as a complete skeleton, and the number and length of tracks. Writes summary.csv,
larvae_per_frame.png (complete skeletons per frame over the recording, 1 s rolling median) and disagreement.png (the
frames where the pipelines differ most, both pipelines' skeletons drawn).

  python scripts/model-assessment/compare_production.py \\
      --entry "Current (UNet)" /Volumes/lab-windingm/home/users/windinm/sleap-tf-track-test \\
      --entry "New (ConvNeXt)" /Volumes/lab-windingm/home/users/windinm/sleap-nn-track-test \\
      --video 2026-03-10_14-31-01_SV37 --larvae 20 \\
      --video-file /Volumes/lab-windingm/home/shared/sleap/groundtruth/combined/videos/2026-03-10_14-31-01_SV37.mp4
"""
from __future__ import annotations
import argparse
import csv
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sleap_io as sio

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


def draw(image, labels_frame, color):
    for inst in labels_frame.instances if labels_frame is not None else []:
        p = inst.numpy()
        for a, b in EDGES:
            if np.isfinite(p[[a, b]]).all():
                cv2.line(image, tuple(np.round(p[a]).astype(int)), tuple(np.round(p[b]).astype(int)), color, 2, cv2.LINE_AA)
    return image


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--entry', nargs=2, action='append', required=True, metavar=('LABEL', 'FOLDER'))
    parser.add_argument('--video', required=True, help='video name without .mp4')
    parser.add_argument('--video-file', type=Path, help='the .mp4 (default: <first folder>/<video>.mp4)')
    parser.add_argument('--larvae', type=int, required=True, help='number of larvae in the arena')
    parser.add_argument('--fps', type=float, default=25.0)
    parser.add_argument('--examples', type=int, default=3)
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
    picks = []
    for idx in np.argsort(-np.abs(diff)):
        if all(abs(idx - p) > args.fps * 60 for p in picks):
            picks.append(int(idx))
        if len(picks) == args.examples:
            break
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
    print(f'Wrote {args.output}')


if __name__ == '__main__':
    main()
