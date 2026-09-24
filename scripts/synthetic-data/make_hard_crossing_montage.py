#!/usr/bin/env python3
"""Create a before/after montage for hard full-frame crossings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import h5py
import numpy as np
import sleap_io


def source_frame(h5_file, video_id: int, frame_number: int) -> np.ndarray:
    numbers = np.asarray(h5_file[f"video{video_id}/frame_numbers"])
    matches = np.flatnonzero(numbers == frame_number)
    if len(matches) != 1:
        raise ValueError(f"Could not resolve video {video_id}, frame {frame_number}")
    encoded = np.asarray(h5_file[f"video{video_id}/video"][int(matches[0])], dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not decode video {video_id}, frame {frame_number}")
    return image


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/combined_ground_truth_cleaned_visibility-fixed.pkg.slp"))
    parser.add_argument("--hard", type=Path, default=Path("outputs/synthetic_data/crossings_hard.slp"))
    parser.add_argument("--metadata", type=Path, default=Path("outputs/synthetic_data/crossings_hard.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/synthetic_data/crossings_hard_before_after.png"))
    parser.add_argument("--count", type=int, default=12)
    return parser


def main():
    args = build_parser().parse_args()
    metadata = json.loads(args.metadata.read_text())
    labels = sleap_io.load_slp(str(args.hard), open_videos=True)
    records = metadata["records"][:args.count]
    tile_width, tile_height = 450, 275
    pair_width = tile_width * 2
    columns = 2
    rows = int(np.ceil(len(records) / columns))
    montage = np.full((rows * tile_height, columns * pair_width, 3), 255, dtype=np.uint8)

    with h5py.File(args.source, "r") as h5_file:
        for index, record in enumerate(records):
            before = source_frame(h5_file, record["source_video"], record["source_frame"])
            after = labels.videos[0].backend.get_frame(index)
            after = cv2.cvtColor(np.asarray(after), cv2.COLOR_RGB2BGR)
            before = cv2.resize(before, (tile_width, tile_height), interpolation=cv2.INTER_AREA)
            after = cv2.resize(after, (tile_width, tile_height), interpolation=cv2.INTER_AREA)
            cv2.putText(before, f"BEFORE  hard {record['hard_id']:03d}", (8, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(after, f"AFTER  {len(record['events'])} crossings", (8, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)
            pair = np.concatenate([before, after], axis=1)
            row, column = divmod(index, columns)
            y0, x0 = row * tile_height, column * pair_width
            montage[y0:y0 + tile_height, x0:x0 + pair_width] = pair
    cv2.imwrite(str(args.output), montage)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()