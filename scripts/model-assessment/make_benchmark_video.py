#!/usr/bin/env python3
"""Rebuild the 140-frame benchmark video from the embedded ground-truth frames: the manifest images in order, each
repeated --repeats times, written with OpenCV's mp4v encoder at 5 fps like comparison_frames_plain.mp4.

comparison_frames_plain.mp4 has red and blue swapped. --colour true writes the frames as the cameras recorded them, which
is how models see real videos; --colour swapped rebuilds the old video with the same encoder, as a control.
"""
from __future__ import annotations
import argparse
import csv
from pathlib import Path

import cv2
import h5py

from assess_centroids import ROOT, decode_source, load_benchmark, verify_video


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--colour', choices=['true', 'swapped'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--ground-truth', type=Path, default=ROOT/'data/combined_ground_truth_cleaned_visibility-fixed.pkg.slp')
    parser.add_argument('--manifest', type=Path, default=ROOT/'scripts/model-assessment/comparison_manifest.csv')
    parser.add_argument('--repeats', type=int, default=10)
    parser.add_argument('--fps', type=float, default=5.0)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f'{args.output} exists')
    rows = sorted(csv.DictReader(args.manifest.open()), key=lambda r: int(r['order']))
    with h5py.File(args.ground_truth) as handle:
        images = [decode_source(handle, int(r['video_id']), int(r['frame_idx'])) for r in rows]
    height, width = images[0].shape[:2]
    writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*'mp4v'), args.fps, (width, height))
    for image in images:
        frame = image if args.colour == 'true' else image[..., ::-1].copy()
        for _ in range(args.repeats):
            writer.write(frame)
    writer.release()
    benchmark = load_benchmark(args.manifest, args.ground_truth, set())
    mapping = verify_video(args.output, args.ground_truth, benchmark, args.repeats)
    if mapping[0]['colour'] != args.colour:
        raise ValueError(f"{args.output} reads back as {mapping[0]['colour']} colour")
    print(f"{args.output}: {len(mapping)} frames, {args.colour} colour, "
          f"thumbnail error {min(m['thumbnail_mae'] for m in mapping):.2f}-{max(m['thumbnail_mae'] for m in mapping):.2f}")


if __name__ == '__main__':
    main()
