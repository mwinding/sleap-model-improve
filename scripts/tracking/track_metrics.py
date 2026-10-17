#!/usr/bin/env python3
"""Label-free tracking metrics: breaks, persistence and suspected swaps, per tracker on the same frames.

Breaks are cheap to proofread (join two tracks) and swaps are expensive (find, cut, re-join), so they are counted
separately:
- tracks, track starts per minute: breaks;
- persistence: duration of the track holding a typical detection (detection-weighted mean track length), and the
  median track duration;
- suspected swaps: a track's body point moving more than --jump-px between consecutive appearances (at most
  --max-gap frames apart), per larva-hour. At 5 Hz a crawling larva moves a few px per frame, so large jumps are
  mostly identity swaps; swaps between touching larvae can move less and are missed.
The per-frame displacement percentiles of all tracks are printed to judge the threshold.

  python scripts/tracking/track_metrics.py --frames 0-999 --entry "Flow" flow.slp --entry "DREEM" dreem.slp \\
      --output outputs/dreem_pilot/track_metrics.csv
"""
from __future__ import annotations
import argparse
import csv
from pathlib import Path

import numpy as np
import sleap_io as sio


def body_tracks(path, first, last, node):
    """track name -> (frames, body xy) for detections between first and last."""
    labels = sio.load_slp(str(path), open_videos=False)
    k = labels.skeletons[0].node_names.index(node)
    tracks = {}
    for frame in labels.labeled_frames:
        if not first <= frame.frame_idx <= last:
            continue
        for inst in frame.instances:
            name = str(inst.track.name) if inst.track is not None else None
            if name is None:
                continue
            tracks.setdefault(name, []).append((frame.frame_idx, inst.numpy()[k]))
    return {n: (np.array([f for f, _ in v]), np.array([p for _, p in v])) for n, v in tracks.items()}


def steps(frames, xy, max_gap):
    """Displacements between consecutive appearances at most max_gap frames apart (NaN body points skipped)."""
    ok = np.isfinite(xy).all(axis=1)
    frames, xy = frames[ok], xy[ok]
    order = np.argsort(frames)
    frames, xy = frames[order], xy[order]
    gap = np.diff(frames)
    d = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    return d[(gap >= 1) & (gap <= max_gap)]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--entry', nargs=2, action='append', required=True, metavar=('LABEL', 'SLP'))
    parser.add_argument('--frames', required=True, help='first-last frame, e.g. 0-999')
    parser.add_argument('--hz', type=float, default=5.0, help='capture rate')
    parser.add_argument('--node', default='body')
    parser.add_argument('--jump-px', type=float, nargs='+', default=[20.0, 40.0])
    parser.add_argument('--max-gap', type=int, default=5, help='frames')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    first, last = map(int, args.frames.split('-'))
    minutes = (last - first + 1) / args.hz / 60

    rows = []
    for label, path in args.entry:
        tracks = body_tracks(path, first, last, args.node)
        lengths = np.array([len(f) for f, _ in tracks.values()])
        moves = np.concatenate([steps(f, xy, args.max_gap) for f, xy in tracks.values()] or [np.empty(0)])
        larva_hours = lengths.sum() / args.hz / 3600
        row = {'tracker': label, 'tracks': len(tracks), 'track_starts_per_min': round(len(tracks) / minutes, 1),
               'persistence_s': round((lengths ** 2).sum() / lengths.sum() / args.hz, 1),
               'median_track_s': round(np.median(lengths) / args.hz, 1)}
        for px in args.jump_px:
            row[f'jumps_over_{px:g}px_per_larva_hour'] = round(int((moves > px).sum()) / larva_hours, 1)
        rows.append(row)
        print(f"{label}: per-frame body displacement px, p50/p90/p99/p99.9: {np.percentile(moves, [50, 90, 99, 99.9]).round(1)}")
    for row in rows:
        print(row)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == '__main__':
    main()
