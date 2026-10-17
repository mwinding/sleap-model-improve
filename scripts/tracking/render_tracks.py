#!/usr/bin/env python3
"""Render tracked skeletons on a video, each larva coloured and numbered by its track, one panel per tracker.

A larva that changes colour/number has been given a new track (a break) or another larva's track (a swap).

  python scripts/tracking/render_tracks.py --video clip.mp4 --frames 0-999 --fps 10 --scale 0.5 \\
      --entry "Flow tracker" flow.tracks.slp --entry "DREEM" dreem.slp --output flow_vs_dreem.mp4
"""
from __future__ import annotations
import argparse
import colorsys
import subprocess
from pathlib import Path

import cv2
import numpy as np
import sleap_io as sio

EDGES = [(0, 1), (1, 2), (2, 3), (3, 4)]


def track_color(name, order):
    """Distinct colours for the first tracks (golden-ratio hue steps), stable per track name."""
    index = order.setdefault(name, len(order))
    r, g, b = colorsys.hsv_to_rgb((index * 0.618034) % 1.0, 0.85, 0.95)
    return int(r * 255), int(g * 255), int(b * 255)


def draw(image, frame, order, width, font_scale):
    for inst in frame.instances if frame is not None else []:
        name = str(inst.track.name) if inst.track is not None else '?'  # DREEM names tracks with ints
        color = track_color(name, order)
        p = inst.numpy()
        for a, b in EDGES:
            if np.isfinite(p[[a, b]]).all():
                cv2.line(image, tuple(np.round(p[a]).astype(int)), tuple(np.round(p[b]).astype(int)), color, width, cv2.LINE_AA)
        ok = np.isfinite(p).all(axis=1)
        if ok.any():
            x, y = np.round(p[ok][0]).astype(int)  # label at the front-most visible node
            label = name.replace('track_', '')
            cv2.putText(image, label, (x + 4, y - 4), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(image, label, (x + 4, y - 4), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2, cv2.LINE_AA)
    return image


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--video', type=Path, required=True)
    parser.add_argument('--entry', nargs=2, action='append', required=True, metavar=('LABEL', 'SLP'))
    parser.add_argument('--frames', default=None, help='first-last frame, e.g. 0-999 (default: all)')
    parser.add_argument('--fps', type=float, default=10.0, help='playback rate (recordings are captured at 5 Hz)')
    parser.add_argument('--capture-hz', type=float, default=5.0)
    parser.add_argument('--scale', type=float, default=1.0)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()

    video = sio.load_video(str(args.video))
    first, last = (map(int, args.frames.split('-')) if args.frames else (0, len(video) - 1))
    entries = []
    for label, path in args.entry:
        labels = sio.load_slp(path, open_videos=False)
        frames = {f.frame_idx: f for f in labels.labeled_frames if first <= f.frame_idx <= last}
        n_tracks = len({i.track.name for f in frames.values() for i in f.instances if i.track is not None})
        entries.append((f'{label}: {n_tracks} tracks', frames, {}))
    h, w = video.shape[1:3]
    ph, pw = int(round(h * args.scale)) // 2 * 2, int(round(w * args.scale)) // 2 * 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                               '-s', f'{pw * len(entries)}x{ph}', '-r', str(args.fps), '-i', '-',
                               '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '20', str(args.output)],
                              stdin=subprocess.PIPE)
    for idx in range(first, last + 1):
        image = np.asarray(video[idx])
        panels = []
        for title, frames, order in entries:
            panel = draw(image.copy(), frames.get(idx), order, width=max(2, int(3 / args.scale)), font_scale=0.6 / args.scale)
            panel = cv2.resize(panel, (pw, ph), interpolation=cv2.INTER_AREA)
            text = f'{title}   t = {(idx - first) / args.capture_hz:5.1f} s'
            cv2.rectangle(panel, (0, 0), (pw, 34), (255, 255, 255), -1)
            cv2.putText(panel, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
            panels.append(panel)
        ffmpeg.stdin.write(np.ascontiguousarray(np.concatenate(panels, axis=1)).tobytes())
    ffmpeg.stdin.close()
    ffmpeg.wait()
    print(f'Wrote {args.output} ({last - first + 1} frames at {args.fps:g} fps)')


if __name__ == '__main__':
    main()
