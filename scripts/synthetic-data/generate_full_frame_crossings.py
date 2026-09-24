#!/usr/bin/env python3
"""Create full-frame synthetic crossings on top of real labeled frames."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import cv2
import h5py
import numpy as np
import sleap_io


NODE_ORDER = ("head", "mouthhooks", "body", "tail", "spiracle")


def decode_frame(h5_file: h5py.File, video_id: int, source_frame: int) -> np.ndarray:
    frame_numbers = np.asarray(h5_file[f"video{video_id}/frame_numbers"])
    matches = np.flatnonzero(frame_numbers == source_frame)
    if len(matches) != 1:
        raise ValueError(f"Could not resolve video {video_id}, frame {source_frame}")
    encoded = np.asarray(h5_file[f"video{video_id}/video"][int(matches[0])], dtype=np.uint8)
    frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError(f"Could not decode video {video_id}, frame {source_frame}")
    return frame


def skeleton_mask(points: np.ndarray, body_length: float, shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    visible = np.isfinite(points).all(axis=1)
    polyline = np.rint(points[visible]).astype(np.int32)
    if len(polyline) >= 2:
        width = max(5, int(round(body_length * 0.14)))
        cv2.polylines(mask, [polyline], False, 255, width)
        mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
    return mask


def transform_donor(image: np.ndarray, mask: np.ndarray, points: np.ndarray, angle: float,
                    scale: float, center: np.ndarray, output_shape: tuple[int, int]):
    height, width = image.shape[:2]
    source_center = np.array([width / 2, height / 2], dtype=np.float32)
    matrix = cv2.getRotationMatrix2D(tuple(source_center), angle, scale)
    matrix[:, 2] += center - source_center
    output_height, output_width = output_shape
    transformed_image = cv2.warpAffine(image, matrix, (output_width, output_height),
                                       borderMode=cv2.BORDER_REFLECT)
    transformed_mask = cv2.warpAffine(mask, matrix, (output_width, output_height),
                                      borderMode=cv2.BORDER_CONSTANT)
    transformed_points = np.c_[points, np.ones(len(points))] @ matrix.T
    return transformed_image, transformed_mask, transformed_points


def darken_overlap(base: np.ndarray, donor: np.ndarray, donor_mask: np.ndarray,
                   target_mask: np.ndarray) -> np.ndarray:
    donor_alpha = cv2.GaussianBlur(donor_mask, (0, 0), 1.2).astype(np.float32) / 255.0
    target_alpha = cv2.GaussianBlur(target_mask, (0, 0), 1.2).astype(np.float32) / 255.0
    donor_alpha = donor_alpha[..., None]
    target_alpha = target_alpha[..., None]
    overlap = donor_alpha * target_alpha
    donor_only = donor_alpha * (1 - target_alpha)
    base_float = base.astype(np.float32)
    donor_float = donor.astype(np.float32)
    output = base_float * (1 - donor_only - overlap)
    output += donor_float * donor_only
    output += np.minimum(base_float, donor_float) * overlap
    return np.rint(output).clip(0, 255).astype(np.uint8)


def appearance(image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, float]:
    body = mask > 127
    expanded = cv2.dilate(body.astype(np.uint8), np.ones((17, 17), np.uint8)) > 0
    ring = expanded & ~body
    if not np.any(body) or not np.any(ring):
        return np.array([200, 200, 200], dtype=np.float32), 30.0
    background = np.percentile(image[ring], 65, axis=0).astype(np.float32)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    background_gray = float(background @ np.array([0.114, 0.587, 0.299]))
    contrast = max(5.0, background_gray - float(np.median(gray[body])))
    return background, contrast


def match_appearance(image: np.ndarray, donor_mask: np.ndarray,
                     target_image: np.ndarray, target_mask: np.ndarray) -> np.ndarray:
    source_background, source_contrast = appearance(image, donor_mask)
    target_background, target_contrast = appearance(target_image, target_mask)
    gain = float(np.clip((target_contrast / source_contrast) ** 0.65, 0.8, 1.25))
    shift = np.clip((target_background - source_background) * 0.8, -20, 20)
    corrected = source_background + shift + (image.astype(np.float32) - source_background) * gain
    return np.rint(corrected).clip(0, 255).astype(np.uint8)


def load_holdout(path: Path | None) -> set[tuple[int, int]]:
    """Return (video_id, frame_idx) pairs that must not feed synthetic data."""
    if path is None:
        return set()
    frames = json.loads(path.read_text(encoding="utf-8"))["frames"]
    return {(int(row["video_id"]), int(row["frame_idx"])) for row in frames}


def approved_donors(library: Path, holdout: set[tuple[int, int]]) -> list[dict]:
    metadata = json.loads((library / "donors.json").read_text(encoding="utf-8"))
    donors = metadata["donors"]
    statuses = {}
    with (library / "donor_review.csv").open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            statuses[int(row["donor_id"])] = row["status"].strip()
    approved = [donor for donor in donors if statuses.get(donor["donor_id"]) == "1"
                and (donor["source_video"], donor["source_frame"]) not in holdout]
    if len(approved) < 2:
        raise ValueError("At least two approved donors are required")
    return approved


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/combined_ground_truth_cleaned_visibility-fixed.pkg.slp"))
    parser.add_argument("--library", type=Path, default=Path("outputs/donor_library_sam2"))
    parser.add_argument("--output", type=Path, default=Path("outputs/synthetic_data"))
    parser.add_argument("--slp-name", default="crossings_full-frame.slp")
    parser.add_argument("--variants-per-frame", type=int, default=2)
    parser.add_argument("--max-source-frames", type=int, default=None)
    parser.add_argument("--two-crossing-probability", type=float, default=0.25)
    parser.add_argument("--aligned-probability", type=float, default=0.7,
                        help="Probability that a donor is oriented near the target axis.")
    parser.add_argument("--alignment-jitter", type=float, default=25.0,
                        help="Standard deviation in degrees for aligned placements.")
    parser.add_argument("--seed", type=int, default=165)
    parser.add_argument("--max-attempts", type=int, default=30)
    parser.add_argument("--holdout", type=Path, default=Path("data/holdout_frames.json"),
                        help="Held-out frames from make_holdout_split.py; skipped as backgrounds and donors.")
    parser.add_argument("--no-holdout", action="store_true", help="Use every source frame.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.variants_per_frame < 1:
        raise ValueError("--variants-per-frame must be at least 1")
    if not 0 <= args.two_crossing_probability <= 1:
        raise ValueError("--two-crossing-probability must be between 0 and 1")
    if not 0 <= args.aligned_probability <= 1:
        raise ValueError("--aligned-probability must be between 0 and 1")
    holdout = load_holdout(None if args.no_holdout else args.holdout)
    source = sleap_io.load_slp(str(args.source), open_videos=False)
    donors = approved_donors(args.library, holdout)
    source_skeleton = source.skeletons[0]
    donor_skeleton = sleap_io.load_slp(str(args.library / "donors.slp"), open_videos=False).skeletons[0]
    if [node.name for node in source_skeleton.nodes] != list(NODE_ORDER):
        raise ValueError("Source skeleton node order does not match donor node order")
    donor_images = [cv2.imread(str(args.library / donor["image"]), cv2.IMREAD_COLOR) for donor in donors]
    donor_masks = [cv2.imread(str(args.library / donor["mask"]), cv2.IMREAD_GRAYSCALE) for donor in donors]
    rng = np.random.default_rng(args.seed)
    output_images = args.output / ".full_frame_images"
    output_images.mkdir(parents=True, exist_ok=True)
    records = []
    image_paths = []
    labeled_frames = []

    source_frames = [lf for lf in source
                     if (source.videos.index(lf.video), int(lf.frame_idx)) not in holdout]
    print(f"Using {len(source_frames)}/{len(source)} source frames and {len(donors)} donors after holdout")
    if args.max_source_frames is not None:
        source_frames = source_frames[:args.max_source_frames]

    with h5py.File(args.source, "r") as h5_file:
        crossing_id = 0
        for labeled_frame in source_frames:
            for variant_index in range(args.variants_per_frame):
                for _ in range(args.max_attempts):
                    video_id = source.videos.index(labeled_frame.video)
                    frame = decode_frame(h5_file, video_id, int(labeled_frame.frame_idx))
                    instances = labeled_frame.instances
                    target_count = 2 if rng.random() < args.two_crossing_probability else 1
                    if len(instances) < target_count:
                        continue
                    target_indices = rng.choice(len(instances), size=target_count, replace=False)
                    target_indices = [int(index) for index in np.atleast_1d(target_indices)]
                    output = frame.copy()
                    output_instances = [sleap_io.Instance.from_numpy(np.asarray(instance.numpy()), source_skeleton)
                                        for instance in instances]
                    placed_events = []
                    used_target_indices = set()
                    placed_masks = []
                    success = True
                    for target_index in target_indices:
                        if target_index in used_target_indices:
                            success = False
                            break
                        used_target_indices.add(target_index)
                        target_points = np.asarray(instances[target_index].numpy(), dtype=np.float32)
                        if not np.isfinite(target_points).all():
                            success = False
                            break
                        target_length = float(np.linalg.norm(target_points[0] - target_points[-1]))
                        donor_index = int(rng.integers(len(donors)))
                        donor = donors[donor_index]
                        donor_points = np.asarray(donor["points_xy"], dtype=np.float32)
                        axis = target_points[-1] - target_points[0]
                        axis_length = max(float(np.linalg.norm(axis)), 1.0)
                        axis /= axis_length
                        cross_point = target_points[0] + axis * rng.uniform(0.35, 0.7) * axis_length
                        target_angle = np.arctan2(axis[1], axis[0])
                        donor_axis = donor_points[-1] - donor_points[0]
                        donor_angle = np.arctan2(donor_axis[1], donor_axis[0])
                        if rng.random() < args.aligned_probability:
                            desired_angle = target_angle + rng.normal(0, np.deg2rad(args.alignment_jitter))
                            angle = float(np.rad2deg(donor_angle - desired_angle))
                            orientation_mode = "aligned"
                        else:
                            angle = float(rng.uniform(25, 155) * rng.choice([-1, 1]))
                            orientation_mode = "free"
                        scale = float(rng.uniform(0.9, 1.1))
                        donor_image, donor_mask, transformed_points = transform_donor(
                            donor_images[donor_index], donor_masks[donor_index], donor_points,
                            angle, scale, cross_point, frame.shape[:2])
                        other_points = [np.asarray(instance.numpy(), dtype=np.float32)
                                        for index, instance in enumerate(instances) if index != target_index]
                        other_points.extend(event["points"] for event in placed_events)
                        for points in other_points:
                            finite = np.isfinite(points).all(axis=1)
                            xy = np.rint(points[finite]).astype(int)
                            valid = ((xy[:, 0] >= 0) & (xy[:, 0] < frame.shape[1]) &
                                     (xy[:, 1] >= 0) & (xy[:, 1] < frame.shape[0]))
                            if np.any(donor_mask[xy[valid, 1], xy[valid, 0]] > 127):
                                success = False
                                break
                        if not success:
                            break
                        target_mask = skeleton_mask(target_points, target_length, frame.shape[:2])
                        donor_image = match_appearance(
                            donor_image, donor_mask, output, target_mask)
                        output = darken_overlap(output, donor_image, donor_mask, target_mask)
                        output_instances.append(sleap_io.Instance.from_numpy(transformed_points, donor_skeleton))
                        placed_masks.append(donor_mask)
                        placed_events.append({"target_index": target_index, "donor": donor,
                                              "points": transformed_points, "angle": angle, "scale": scale,
                                              "orientation_mode": orientation_mode})
                    if success and len(placed_events) == target_count:
                        break
                else:
                    raise RuntimeError(f"Could not place variant {variant_index} for source frame {labeled_frame.frame_idx}")

                image_path = output_images / f"crossing_{crossing_id:05d}.png"
                cv2.imwrite(str(image_path), output)
                image_paths.append(str(image_path.resolve()))
                labeled_frames.append((int(labeled_frame.frame_idx), output_instances))
                records.append({
                    "crossing_id": crossing_id,
                    "variant_index": variant_index,
                    "source_video": video_id,
                    "source_frame": int(labeled_frame.frame_idx),
                    "events": [{"target_instance": event["target_index"],
                                "donor_id": int(event["donor"]["donor_id"]),
                                "angle": event["angle"], "scale": event["scale"],
                                "orientation_mode": event["orientation_mode"]}
                               for event in placed_events],
                    "image": str(image_path.relative_to(args.output)),
                })
                crossing_id += 1

    video = sleap_io.Video(filename=image_paths)
    frames = [sleap_io.LabeledFrame(video, index, instances) for index, (_, instances) in enumerate(labeled_frames)]
    labels = sleap_io.Labels(labeled_frames=frames, videos=[video], skeletons=[source_skeleton])
    labels.save(str(args.output / args.slp_name), embed=True, verbose=False)
    (args.output / "crossings.json").write_text(json.dumps({
        "source": str(args.source), "donor_library": str(args.library),
        "seed": args.seed, "count": len(records), "variants_per_frame": args.variants_per_frame,
        "two_crossing_probability": args.two_crossing_probability,
        "aligned_probability": args.aligned_probability,
        "alignment_jitter": args.alignment_jitter, "mode": "darken_overlap",
        "holdout": None if args.no_holdout else str(args.holdout),
        "crossings": records,
    }, indent=2), encoding="utf-8")
    shutil.rmtree(output_images)
    print(f"Generated {len(records)} full-frame crossings in {args.output / args.slp_name}")


if __name__ == "__main__":
    main()