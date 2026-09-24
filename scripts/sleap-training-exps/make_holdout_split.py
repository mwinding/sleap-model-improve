#!/usr/bin/env python3
"""Hold out benchmark frames (plus a temporal buffer) from the real training labels.

Writes a training SLP without the held-out frames and a JSON list of them. The
synthetic-data scripts read the same JSON so held-out frames are never used as
backgrounds or donor sources. Video indices are preserved, so donor records and
manifest entries still refer to the same videos.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import sleap_io


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/combined_ground_truth_cleaned_visibility-fixed.pkg.slp"))
    parser.add_argument("--manifest", type=Path, default=Path("scripts/model-assessment/comparison_manifest.csv"))
    parser.add_argument("--buffer", type=int, default=100,
                        help="Also hold out labeled frames within this many frames of a benchmark frame.")
    parser.add_argument("--train-output", type=Path, default=Path("data/combined_ground_truth_train.pkg.slp"))
    parser.add_argument("--holdout-output", type=Path, default=Path("data/holdout_frames.json"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.buffer < 0:
        raise ValueError("--buffer must be non-negative")
    with args.manifest.open(newline="") as handle:
        benchmark = [(int(row["video_id"]), int(row["frame_idx"]), int(row["order"]))
                     for row in csv.DictReader(handle)]
    labels = sleap_io.load_slp(str(args.source))
    frames = {(labels.videos.index(lf.video), int(lf.frame_idx)): lf for lf in labels}
    missing = [(video_id, frame_idx) for video_id, frame_idx, _ in benchmark if (video_id, frame_idx) not in frames]
    if missing:
        raise ValueError(f"Benchmark frames not found in {args.source}: {missing}")

    held_out = []
    for (video_id, frame_idx), lf in frames.items():
        nearby = [(abs(frame_idx - bench_frame), order) for bench_video, bench_frame, order in benchmark
                  if bench_video == video_id and abs(frame_idx - bench_frame) <= args.buffer]
        if nearby:
            distance, order = min(nearby)
            held_out.append({"video_id": video_id, "frame_idx": frame_idx,
                             "reason": "benchmark" if distance == 0 else "buffer",
                             "manifest_order": order, "frames_from_benchmark": distance,
                             "instances": len(lf.instances)})
    held_out.sort(key=lambda row: (row["video_id"], row["frame_idx"]))
    held_keys = {(row["video_id"], row["frame_idx"]) for row in held_out}
    kept = [lf for key, lf in frames.items() if key not in held_keys]

    args.train_output.parent.mkdir(parents=True, exist_ok=True)
    train = sleap_io.Labels(labeled_frames=kept, videos=list(labels.videos), skeletons=labels.skeletons)
    train.save(str(args.train_output), embed=True, verbose=False)
    check = sleap_io.load_slp(str(args.train_output), open_videos=False)
    if len(check.videos) != len(labels.videos) or len(check) != len(kept):
        raise RuntimeError("Saved training labels do not match the requested split")
    if any((check.videos.index(lf.video), int(lf.frame_idx)) in held_keys for lf in check):
        raise RuntimeError("Saved training labels still contain held-out frames")

    args.holdout_output.write_text(json.dumps({
        "source": str(args.source), "manifest": str(args.manifest), "buffer_frames": args.buffer,
        "train_labels": str(args.train_output), "frames": held_out,
    }, indent=2), encoding="utf-8")
    held_instances = sum(row["instances"] for row in held_out)
    total_instances = sum(len(lf.instances) for lf in frames.values())
    print(f"Held out {len(held_out)}/{len(frames)} frames ({held_instances}/{total_instances} instances): "
          f"{sum(row['reason'] == 'benchmark' for row in held_out)} benchmark, "
          f"{sum(row['reason'] == 'buffer' for row in held_out)} buffer")
    print(f"Wrote {args.train_output} and {args.holdout_output}")


if __name__ == "__main__":
    main()
