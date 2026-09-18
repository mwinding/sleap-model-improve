#!/usr/bin/env python3
"""Create a labeled donor review montage and editable review CSV."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, default=Path("outputs/donor_library"))
    parser.add_argument("--columns", type=int, default=8)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    metadata = json.loads((args.library / "donors.json").read_text(encoding="utf-8"))
    donors = metadata["donors"]
    crop_size = int(metadata["crop_size"])
    review_rows = []
    tiles = []
    cutout_tiles = []

    for donor in donors:
        image = cv2.imread(str(args.library / donor["image"]), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(args.library / donor["mask"]), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            raise RuntimeError(f"Could not read donor {donor['donor_id']}")
        overlay = np.zeros_like(image)
        overlay[:, :, 1] = mask
        tile = cv2.addWeighted(image, 0.78, overlay, 0.22, 0)
        cv2.rectangle(tile, (0, 0), (crop_size - 1, 21), (0, 0, 0), -1)
        cv2.putText(tile, f"donor {donor['donor_id']:03d}", (5, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(tile)

        alpha = cv2.GaussianBlur(mask, (0, 0), 1.2).astype(np.float32) / 255.0
        alpha = alpha[..., None]
        background = np.full_like(image, 235, dtype=np.float32)
        cutout = np.rint(image.astype(np.float32) * alpha + background * (1 - alpha)).astype(np.uint8)
        cv2.rectangle(cutout, (0, 0), (crop_size - 1, 21), (0, 0, 0), -1)
        cv2.putText(cutout, f"donor {donor['donor_id']:03d}", (5, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        cutout_tiles.append(cutout)
        review_rows.append({
            "donor_id": donor["donor_id"],
            "status": "pending",
            "reason": "",
        })

    rows = int(np.ceil(len(tiles) / args.columns))
    montage = np.full((rows * crop_size, args.columns * crop_size, 3), 255, dtype=np.uint8)
    for index, tile in enumerate(tiles):
        row, column = divmod(index, args.columns)
        montage[row * crop_size:(row + 1) * crop_size,
                column * crop_size:(column + 1) * crop_size] = tile
    cv2.imwrite(str(args.library / "review_montage.png"), montage)

    cutout_montage = np.full_like(montage, 235)
    for index, tile in enumerate(cutout_tiles):
        row, column = divmod(index, args.columns)
        cutout_montage[row * crop_size:(row + 1) * crop_size,
                       column * crop_size:(column + 1) * crop_size] = tile
    cv2.imwrite(str(args.library / "cutout_montage.png"), cutout_montage)

    with (args.library / "donor_review.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("donor_id", "status", "reason"))
        writer.writeheader()
        writer.writerows(review_rows)
    print(f"Wrote {args.library / 'review_montage.png'}")
    print(f"Wrote {args.library / 'cutout_montage.png'}")
    print(f"Wrote {args.library / 'donor_review.csv'}")


if __name__ == "__main__":
    main()