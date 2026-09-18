#!/usr/bin/env python3
"""Replace tube-prior donor masks with SAM2 point-prompt masks."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch
from PIL import Image, ImageDraw
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, default=Path("outputs/donor_library"))
    parser.add_argument("--output", type=Path, default=Path("outputs/donor_library_sam2"))
    parser.add_argument("--checkpoint", type=Path, default=Path("models/sam2/sam2.1_hiera_base_plus.pt"))
    parser.add_argument("--config", default="sam2_hiera_b+.yaml",
                        help="Hydra config name installed with SAM2.")
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--max-donors", type=int, default=None)
    return parser


def choose_mask(masks: np.ndarray, scores: np.ndarray, points: np.ndarray) -> tuple[np.ndarray, int, str]:
    height, width = masks.shape[-2:]
    point_xy = np.rint(points).astype(int)
    inside = np.zeros(len(masks), dtype=bool)
    for index, mask in enumerate(masks):
        valid = ((point_xy[:, 0] >= 0) & (point_xy[:, 0] < width) &
                 (point_xy[:, 1] >= 0) & (point_xy[:, 1] < height))
        inside[index] = bool(np.all(mask[point_xy[valid, 1], point_xy[valid, 0]])) if np.any(valid) else False

    area_fraction = masks.reshape(len(masks), -1).mean(axis=1)
    reasonable = (area_fraction >= 0.005) & (area_fraction <= 0.35)
    candidates = np.flatnonzero(inside & reasonable)
    if len(candidates):
        selected = candidates[np.argmax(scores[candidates])]
        return masks[selected], int(selected), "all_points_reasonable"
    candidates = np.flatnonzero(inside)
    if len(candidates):
        selected = candidates[np.argmax(scores[candidates])]
        return masks[selected], int(selected), "all_points"
    selected = int(np.argmax(scores))
    return masks[selected], selected, "highest_score_fallback"


def skeleton_prior(points: np.ndarray, body_length: float, shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    prior = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(prior)
    line_points = [tuple(np.rint(point).astype(int)) for point in points]
    line_width = max(6, int(round(body_length * 0.14)))
    draw.line(line_points, fill=255, width=line_width, joint="curve")
    prior = np.asarray(prior.resize((256, 256), Image.Resampling.BILINEAR), dtype=np.float32)
    return np.where(prior > 127, 10.0, -10.0)[None, ...]


def main() -> None:
    args = build_parser().parse_args()
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    metadata = json.loads((args.library / "donors.json").read_text(encoding="utf-8"))
    donors = metadata["donors"][:args.max_donors]
    args.output.mkdir(parents=True, exist_ok=True)
    mask_dir = args.output / "masks"
    mask_dir.mkdir(exist_ok=True)
    image_dir = args.output / "images"
    image_dir.mkdir(exist_ok=True)

    print(f"Loading SAM2 on {args.device}...")
    model = build_sam2(args.config, str(args.checkpoint), device=args.device)
    predictor = SAM2ImagePredictor(model)
    updated = []

    for number, donor in enumerate(donors, start=1):
        source_image = args.library / donor["image"]
        try:
            image_rgb = np.array(Image.open(source_image).convert("RGB"), copy=True)
        except OSError as error:
            raise RuntimeError(f"Could not read {source_image}") from error
        if image_rgb.ndim != 3:
            raise RuntimeError(f"Could not read {source_image}")
        points = np.asarray(donor["points_xy"], dtype=np.float32)
        mask_input = skeleton_prior(points, donor["body_length"], image_rgb.shape[:2])
        predictor.set_image(image_rgb)
        masks, scores, _ = predictor.predict(
            point_coords=points,
            point_labels=np.ones(len(points), dtype=np.int32),
            mask_input=mask_input,
            multimask_output=True,
        )
        mask, selected, selection = choose_mask(masks, scores, points)
        mask = (mask.astype(np.uint8) * 255)
        mask_path = mask_dir / Path(donor["mask"]).name
        Image.fromarray(mask).save(mask_path)
        image_path = image_dir / Path(donor["image"]).name
        if not image_path.exists():
            shutil.copy2(source_image, image_path)
        updated_donor = dict(donor)
        updated_donor["image"] = str(image_path.relative_to(args.output))
        updated_donor["mask"] = str(mask_path.relative_to(args.output))
        updated_donor["sam2_score"] = float(scores[selected])
        updated_donor["sam2_mask_index"] = selected
        updated_donor["sam2_selection"] = selection
        updated.append(updated_donor)
        if number % 25 == 0 or number == len(donors):
            print(f"Segmented {number}/{len(donors)} donors")

    (args.output / "donors.json").write_text(
        json.dumps({**metadata, "segmentation": "sam2_point_prompt", "donors": updated}, indent=2),
        encoding="utf-8",
    )
    shutil.copy2(args.library / "donors.slp", args.output / "donors.slp")
    print(f"Wrote SAM2 donor masks to {args.output}")


if __name__ == "__main__":
    main()