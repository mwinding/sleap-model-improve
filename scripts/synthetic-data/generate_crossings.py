#!/usr/bin/env python3
"""Generate synthetic crossing crops from a donor library."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import sleap_io


def transform_donor(image: np.ndarray, mask: np.ndarray, points: np.ndarray, angle: float,
                    scale: float, center: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = image.shape[:2]
    source_center = np.array([width / 2, height / 2], dtype=np.float32)
    matrix = cv2.getRotationMatrix2D(tuple(source_center), angle, scale)
    matrix[:, 2] += center - source_center
    transformed_image = cv2.warpAffine(image, matrix, (width, height), borderMode=cv2.BORDER_REFLECT)
    transformed_mask = cv2.warpAffine(mask, matrix, (width, height), borderMode=cv2.BORDER_CONSTANT)
    transformed_points = np.c_[points, np.ones(len(points))] @ matrix.T
    return transformed_image, transformed_mask, transformed_points


def composite(base: np.ndarray, donor: np.ndarray, mask: np.ndarray, mode: str,
              alpha_strength: float, target_mask: np.ndarray | None = None) -> np.ndarray:
    alpha = cv2.GaussianBlur(mask, (0, 0), 1.2).astype(np.float32) / 255.0
    strength = 1.0 if mode == "hard" else alpha_strength
    alpha = (alpha * strength)[..., None]
    base_float = base.astype(np.float32)
    donor_float = donor.astype(np.float32)
    if mode == "hard":
        blended = donor_float
    elif mode == "alpha":
        blended = base_float * (1 - alpha) + donor_float * alpha
    elif mode == "darken":
        blended = np.minimum(base_float, donor_float)
    elif mode == "darken_overlap":
        if target_mask is None:
            raise ValueError("Darken-overlap mode requires a target mask")
        donor_alpha = cv2.GaussianBlur(mask, (0, 0), 1.2).astype(np.float32) / 255.0
        target_alpha = cv2.GaussianBlur(target_mask, (0, 0), 1.2).astype(np.float32) / 255.0
        target_alpha = target_alpha[..., None]
        donor_alpha = donor_alpha[..., None]
        overlap = donor_alpha * target_alpha
        donor_only = donor_alpha * (1 - target_alpha)
        blended = base_float * (1 - donor_only - overlap)
        blended += donor_float * donor_only
        blended += np.minimum(base_float, donor_float) * overlap
        return np.rint(blended).clip(0, 255).astype(np.uint8)
    elif mode == "density":
        darkness = (255 - base_float) + (255 - donor_float) * alpha_strength
        density_blended = 255 - np.clip(darkness, 0, 255)
        blended = base_float * (1 - alpha) + density_blended * alpha
    else:
        raise ValueError(f"Unknown compositing mode: {mode}")
    if mode in {"hard", "darken"}:
        return np.rint(base_float * (1 - alpha) + blended * alpha).clip(0, 255).astype(np.uint8)
    return np.rint(blended).clip(0, 255).astype(np.uint8)


def appearance(image: np.ndarray, mask: np.ndarray) -> dict:
    """Estimate local illumination and body contrast, avoiding distant debris."""
    body = mask > 127
    if not np.any(body):
        raise ValueError("Cannot measure an empty donor mask")
    expanded = cv2.dilate(body.astype(np.uint8), np.ones((17, 17), np.uint8)) > 0
    ring = expanded & ~body
    if not np.any(ring):
        raise ValueError("Cannot measure background around donor")
    background = np.percentile(image[ring], 65, axis=0).astype(np.float32)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    background_gray = float(background @ np.array([0.114, 0.587, 0.299]))
    contrast = max(5.0, background_gray - float(np.median(gray[body])))
    return {"background": background.tolist(), "contrast": contrast}


def appearance_distance(target: dict, donor: dict) -> float:
    background_delta = np.max(np.abs(np.array(target["background"]) - donor["background"]))
    contrast_delta = abs(np.log(target["contrast"] / donor["contrast"]))
    return float(max(background_delta / 18.0, contrast_delta / np.log(1.5)))


def match_appearance(image: np.ndarray, target: dict, donor: dict) -> tuple[np.ndarray, float]:
    # Partial, bounded correction preserves natural differences and image texture.
    gain = float(np.clip((target["contrast"] / donor["contrast"]) ** 0.65, 0.8, 1.25))
    source_bg = np.asarray(donor["background"], dtype=np.float32)
    target_bg = np.asarray(target["background"], dtype=np.float32)
    background_shift = np.clip((target_bg - source_bg) * 0.8, -20, 20)
    corrected = source_bg + background_shift + (image.astype(np.float32) - source_bg) * gain
    return np.rint(corrected).clip(0, 255).astype(np.uint8), gain


def approved_donors(library: Path, donors: list[dict]) -> list[dict]:
    """Require a completed binary review and preserve original donor IDs."""
    review_path = library / "donor_review.csv"
    statuses = {}
    known_ids = {donor["donor_id"] for donor in donors}
    with review_path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            donor_id = int(row["donor_id"])
            status = row["status"].strip()
            if donor_id not in known_ids or donor_id in statuses:
                raise ValueError(f"Unknown or duplicate donor {donor_id} in {review_path}")
            if status not in {"0", "1"}:
                raise ValueError(f"Donor {donor_id} must have review status 0 or 1, got {status!r}")
            statuses[donor_id] = status
    missing = known_ids - statuses.keys()
    if missing:
        raise ValueError(f"Missing donor reviews in {review_path}: {sorted(missing)}")
    return [donor for donor in donors if statuses[donor["donor_id"]] == "1"]


def straightness(points: np.ndarray) -> float:
    """Head-spiracle chord length divided by skeleton path length (1 = straight)."""
    path = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
    return float(np.linalg.norm(points[-1] - points[0])) / path if path else 0.0


def load_holdout(path: Path | None) -> set[tuple[int, int]]:
    """Return (video_id, frame_idx) pairs that must not feed synthetic data."""
    if path is None:
        return set()
    frames = json.loads(path.read_text(encoding="utf-8"))["frames"]
    return {(int(row["video_id"]), int(row["frame_idx"])) for row in frames}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, default=Path("outputs/donor_library_sam2"),
                        help="Donor library containing a completed 0/1 donor_review.csv.")
    parser.add_argument("--output", type=Path, default=Path("outputs/synthetic_data"))
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=165)
    parser.add_argument("--qc-tiles", type=int, default=64)
    parser.add_argument("--mode", choices=("hard", "alpha", "darken", "darken_overlap", "density"), default="alpha")
    parser.add_argument("--alpha", type=float, default=0.65,
                        help="Donor opacity/strength for alpha and density modes.")
    parser.add_argument("--no-appearance-matching", action="store_true",
                        help="Disable compatible pairing and contrast correction for comparison.")
    parser.add_argument("--holdout", type=Path, default=Path("data/holdout_frames.json"),
                        help="Held-out frames from make_holdout_split.py; their donors are skipped.")
    parser.add_argument("--no-holdout", action="store_true", help="Use donors from every source frame.")
    parser.add_argument("--parallel-probability", type=float, default=0.0,
                        help="Probability of a side-by-side placement instead of a centred crossing.")
    parser.add_argument("--parallel-jitter", type=float, default=10.0,
                        help="Standard deviation in degrees of the donor heading around the target heading.")
    parser.add_argument("--parallel-offset", type=float, nargs=2, default=(6.0, 25.0), metavar=("MIN", "MAX"),
                        help="Sideways distance in px between the two body points for parallel placements.")
    parser.add_argument("--parallel-min-straightness", type=float, default=0.9,
                        help="Minimum head-spiracle chord / skeleton length for donors in parallel placements.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    holdout = load_holdout(None if args.no_holdout else args.holdout)
    metadata = json.loads((args.library / "donors.json").read_text(encoding="utf-8"))
    donors = approved_donors(args.library, metadata["donors"])
    approved_count = len(donors)
    donors = [donor for donor in donors if (donor["source_video"], donor["source_frame"]) not in holdout]
    print(f"Using {len(donors)}/{len(metadata['donors'])} approved donors from {args.library} "
          f"({approved_count - len(donors)} skipped from held-out frames)")
    crop_size = int(metadata["crop_size"])
    if len(donors) < 2:
        raise RuntimeError("The donor library must contain at least two approved donors")
    if not 0 < args.alpha <= 1:
        raise ValueError("--alpha must be greater than 0 and no greater than 1")
    if not 0 <= args.parallel_probability <= 1:
        raise ValueError("--parallel-probability must be between 0 and 1")
    body = metadata["nodes"].index("body")

    rng = np.random.default_rng(args.seed)
    image_dir = args.library / "images"
    mask_dir = args.library / "masks"
    output_images = args.output / "images"
    output_images.mkdir(parents=True, exist_ok=True)
    records = []
    qc_tiles = []
    comparison_tiles = []
    images, masks, appearances = [], [], []
    for donor in donors:
        image = cv2.imread(str(args.library / donor["image"]), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(args.library / donor["mask"]), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            raise ValueError(f"Cannot read donor {donor['donor_id']}")
        images.append(image)
        masks.append(mask)
        appearances.append(appearance(image, mask))
    distances = np.array([[appearance_distance(a, b) for b in appearances] for a in appearances])
    np.fill_diagonal(distances, np.inf)
    skeleton = sleap_io.load_slp(str(args.library / "donors.slp"), open_videos=False).skeletons[0]

    # Parallel placements use only fairly straight larvae: two bent larvae with
    # aligned head-spiracle axes still look like a crossing.
    straight = np.array([straightness(np.asarray(d["points_xy"])) >= args.parallel_min_straightness for d in donors])
    if args.parallel_probability > 0 and straight.sum() < 2:
        raise RuntimeError("Fewer than two donors meet --parallel-min-straightness")

    for crossing_id in range(args.count):
        # Only draw from the RNG when enabled, so default runs reproduce earlier sets.
        parallel = args.parallel_probability > 0 and rng.random() < args.parallel_probability
        if parallel:
            target_id, occluder_id = rng.choice(np.flatnonzero(straight), size=2, replace=False)
        else:
            target_id, occluder_id = rng.choice(len(donors), size=2, replace=False)
        if not args.no_appearance_matching:
            compatible = np.flatnonzero(distances[target_id] <= 1.0)
            if parallel:
                compatible = compatible[straight[compatible]]
            if not len(compatible):
                ranked = np.argsort(distances[target_id])
                if parallel:
                    ranked = ranked[straight[ranked]]
                compatible = ranked[:min(5, len(donors) - 1)]
            occluder_id = int(rng.choice(compatible))
        target = donors[int(target_id)]
        occluder = donors[int(occluder_id)]
        target_image, target_mask = images[target_id], masks[target_id]
        occluder_image, occluder_mask = images[occluder_id], masks[occluder_id]
        original_occluder = occluder_image
        gain = 1.0
        if not args.no_appearance_matching:
            occluder_image, gain = match_appearance(
                occluder_image, appearances[target_id], appearances[occluder_id])
        corrected_appearance = appearance(occluder_image, occluder_mask)
        residual = appearance_distance(appearances[target_id], corrected_appearance)
        review_reasons = []
        if distances[target_id, occluder_id] > 1.0:
            review_reasons.append("no_compatible_pair")
        if residual > 1.0:
            review_reasons.append("residual_appearance_mismatch")
        target_points = np.asarray(target["points_xy"], dtype=np.float32)
        occluder_points = np.asarray(occluder["points_xy"], dtype=np.float32)

        target_axis = target_points[-1] - target_points[0]
        axis_length = max(float(np.linalg.norm(target_axis)), 1.0)
        axis = target_axis / axis_length
        if parallel:
            # Side by side: donor heading near the target's, body points a few px apart.
            target_angle = np.arctan2(axis[1], axis[0])
            donor_axis = occluder_points[-1] - occluder_points[0]
            donor_angle = np.arctan2(donor_axis[1], donor_axis[0])
            desired_angle = target_angle + rng.normal(0, np.deg2rad(args.parallel_jitter))
            angle = float(np.rad2deg(donor_angle - desired_angle))
            scale = float(rng.uniform(0.9, 1.1))
            offset = float(rng.uniform(*args.parallel_offset))
            side = np.array([-axis[1], axis[0]]) * rng.choice([-1, 1])
            desired_body = target_points[body] + side * offset + axis * rng.uniform(-8, 8)
            # Place once to find where the donor's body point lands, then shift it onto the goal.
            _, _, trial = transform_donor(occluder_image, occluder_mask, occluder_points, angle, scale, target_points[body])
            cross_point = target_points[body] + (desired_body - trial[body])
        else:
            cross_point = target_points[0] + axis * rng.uniform(0.35, 0.7) * axis_length
            angle = float(rng.uniform(25, 155) * rng.choice([-1, 1]))
            scale = float(rng.uniform(0.9, 1.1))
        occluder_image, occluder_mask, transformed_points = transform_donor(
            occluder_image, occluder_mask, occluder_points, angle, scale, cross_point
        )
        output = composite(target_image, occluder_image, occluder_mask, args.mode, args.alpha, target_mask)
        if len(comparison_tiles) < 16:
            raw_image, raw_mask, _ = transform_donor(
                original_occluder, masks[occluder_id], occluder_points, angle, scale, cross_point)
            raw = composite(target_image, raw_image, raw_mask, args.mode, args.alpha, target_mask)
            pair = np.concatenate([raw, output], axis=1)
            cv2.putText(pair, f"{crossing_id:05d} original | matched", (5, 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
            comparison_tiles.append(pair)
        # Larvae are transparent: overlap does not hide either animal's keypoints.
        target_visible = np.ones(len(target_points), dtype=bool)

        image_path = output_images / f"crossing_{crossing_id:05d}.png"
        cv2.imwrite(str(image_path), output)
        records.append({
            "crossing_id": crossing_id,
            "target_donor": int(target["donor_id"]),
            "occluder_donor": int(occluder["donor_id"]),
            "appearance_distance_before": float(distances[target_id, occluder_id]),
            "appearance_distance_after": residual,
            "contrast_gain": gain,
            "review_reasons": review_reasons,
            "angle": angle,
            "scale": scale,
            "placement": "parallel" if parallel else "crossing",
            "body_separation_px": float(np.linalg.norm(transformed_points[body] - target_points[body])),
            "points_xy": target_points.tolist(),
            "visible": target_visible.tolist(),
            "occluder_points_xy": transformed_points.tolist(),
            "occluder_visible": [True] * len(transformed_points),
            "compositing_mode": args.mode,
            "compositing_alpha": args.alpha,
            "image": str(image_path.relative_to(args.output)),
        })
        if len(qc_tiles) < args.qc_tiles:
            tile = output.copy()
            for point_index, (point, visible) in enumerate(zip(target_points, target_visible), start=1):
                color = (0, 180, 0) if visible else (0, 0, 255)
                cv2.circle(tile, tuple(np.rint(point).astype(int)), 3, color, -1)
                cv2.putText(tile, f"T{point_index}", tuple(np.rint(point).astype(int) + [4, -4]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.28, color, 1, cv2.LINE_AA)
            for point_index, point in enumerate(transformed_points, start=1):
                location = tuple(np.rint(point).astype(int))
                cv2.circle(tile, location, 3, (255, 120, 0), -1)
                cv2.putText(tile, f"O{point_index}", (location[0] + 4, location[1] + 9),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.28, (255, 120, 0), 1, cv2.LINE_AA)
            cv2.putText(tile, f"{crossing_id:05d}", (5, 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
            qc_tiles.append(tile)

    with (args.output / "crossings.json").open("w", encoding="utf-8") as handle:
        json.dump({"nodes": metadata["nodes"], "crop_size": crop_size, "seed": args.seed,
               "compositing_mode": args.mode, "compositing_alpha": args.alpha,
               "appearance_matching": not args.no_appearance_matching,
               "holdout": None if args.no_holdout else str(args.holdout),
               "parallel_probability": args.parallel_probability, "parallel_jitter": args.parallel_jitter,
               "parallel_offset": list(args.parallel_offset),
               "parallel_min_straightness": args.parallel_min_straightness,
               "crossings": records}, handle, indent=2)

    with (args.output / "appearance_review.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["crossing_id", "target_donor", "occluder_donor", "appearance_distance_before",
                  "appearance_distance_after", "contrast_gain", "review_reasons", "image"]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            if record["review_reasons"]:
                writer.writerow({**record, "review_reasons": ";".join(record["review_reasons"])})
    comparison = np.full((int(np.ceil(len(comparison_tiles) / 2)) * crop_size,
                          4 * crop_size, 3), 255, dtype=np.uint8)
    for index, pair in enumerate(comparison_tiles):
        row, col = divmod(index, 2)
        comparison[row * crop_size:(row + 1) * crop_size,
                   col * crop_size * 2:(col + 1) * crop_size * 2] = pair
    cv2.imwrite(str(args.output / "appearance_comparison.png"), comparison)
    print(f"Flagged {sum(bool(r['review_reasons']) for r in records)} crossings for appearance review")

    video_paths = [str((args.output / record["image"]).resolve()) for record in records]
    video = sleap_io.Video(filename=video_paths)
    labeled_frames = []
    for index, record in enumerate(records):
        instances = []
        for key in ("points_xy", "occluder_points_xy"):
            instance = sleap_io.Instance.from_numpy(np.asarray(record[key], dtype=np.float64), skeleton)
            instance.points["visible"] = True
            instance.points["complete"] = True
            instances.append(instance)
        labeled_frames.append(sleap_io.LabeledFrame(video, index, instances))
    labels = sleap_io.Labels(labeled_frames=labeled_frames, videos=[video], skeletons=[skeleton])
    labels.save(str(args.output / "crossings.slp"), embed=True, verbose=False)

    columns = 8
    rows = int(np.ceil(len(qc_tiles) / columns))
    montage = np.full((rows * crop_size, columns * crop_size, 3), 255, dtype=np.uint8)
    for index, tile in enumerate(qc_tiles):
        row, column = divmod(index, columns)
        montage[row * crop_size:(row + 1) * crop_size, column * crop_size:(column + 1) * crop_size] = tile
    cv2.imwrite(str(args.output / "qc_montage.png"), montage)
    print(f"Generated {len(records)} synthetic crossings in {args.output}")


if __name__ == "__main__":
    main()
