#!/usr/bin/env python3
"""Generate deliberately hard full-frame scenes with five crossing pairs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import h5py
import numpy as np
import sleap_io


NODE_ORDER = ("head", "mouthhooks", "body", "tail", "spiracle")


def decode_frame(h5_file, video_id: int, source_frame: int) -> np.ndarray:
    numbers = np.asarray(h5_file[f"video{video_id}/frame_numbers"])
    matches = np.flatnonzero(numbers == source_frame)
    if len(matches) != 1:
        raise ValueError(f"Could not resolve video {video_id}, frame {source_frame}")
    encoded = np.asarray(h5_file[f"video{video_id}/video"][int(matches[0])], dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not decode video {video_id}, frame {source_frame}")
    return image


def median_background(h5_file, video_id: int) -> np.ndarray:
    dataset = h5_file[f"video{video_id}/video"]
    frames = []
    for encoded in dataset:
        image = cv2.imdecode(np.asarray(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is not None:
            frames.append(image)
    if not frames:
        raise ValueError(f"Could not decode any frames for video {video_id}")
    return np.median(np.stack(frames), axis=0).astype(np.uint8)


def approved_donors(library: Path) -> list[dict]:
    metadata = json.loads((library / "donors.json").read_text())
    with (library / "donor_review.csv").open(newline="", encoding="utf-8-sig") as handle:
        statuses = {int(row["donor_id"]): row["status"].strip() for row in csv.DictReader(handle)}
    donors = [donor for donor in metadata["donors"] if statuses.get(donor["donor_id"]) == "1"]
    if len(donors) < 10:
        raise ValueError("At least 10 approved donors are required")
    return donors


def separated_subset(instances, minimum_distance: float, count: int) -> list[int]:
    centers = []
    for instance in instances:
        points = np.asarray(instance.numpy(), dtype=np.float32)
        if np.isfinite(points).all():
            centers.append(points.mean(axis=0))
        else:
            centers.append(None)
    selected = []
    for index in np.argsort([sum(center is not None for center in centers)] * len(centers)):
        center = centers[int(index)]
        if center is None:
            continue
        if all(np.linalg.norm(center - centers[other]) >= minimum_distance for other in selected):
            selected.append(int(index))
            if len(selected) == count:
                break
    return selected


def skeleton_mask(points: np.ndarray, shape: tuple[int, int], width: int = 14) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    if np.isfinite(points).all():
        cv2.polylines(mask, [np.rint(points).astype(np.int32)], False, 255, width)
    return cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)


def transform(image, mask, points, angle, center, output_shape):
    h, w = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    matrix[:, 2] += center - np.array([w / 2, h / 2])
    oh, ow = output_shape
    transformed_image = cv2.warpAffine(image, matrix, (ow, oh), borderMode=cv2.BORDER_REFLECT)
    transformed_mask = cv2.warpAffine(mask, matrix, (ow, oh), borderMode=cv2.BORDER_CONSTANT)
    transformed_points = np.c_[points, np.ones(len(points))] @ matrix.T
    return transformed_image, transformed_mask, transformed_points


def darken_overlap(base, donor, donor_mask, target_mask):
    donor_alpha = cv2.GaussianBlur(donor_mask, (0, 0), 1.2).astype(np.float32) / 255
    target_alpha = cv2.GaussianBlur(target_mask, (0, 0), 1.2).astype(np.float32) / 255
    donor_alpha, target_alpha = donor_alpha[..., None], target_alpha[..., None]
    overlap = donor_alpha * target_alpha
    donor_only = donor_alpha * (1 - target_alpha)
    base = base.astype(np.float32)
    donor = donor.astype(np.float32)
    output = base * (1 - donor_only - overlap) + donor * donor_only
    output += np.minimum(base, donor) * overlap
    return np.rint(output).clip(0, 255).astype(np.uint8)


def appearance(image, mask):
    body = mask > 127
    expanded = cv2.dilate(body.astype(np.uint8), np.ones((17, 17), np.uint8)) > 0
    ring = expanded & ~body
    if not np.any(body) or not np.any(ring):
        return np.array([200, 200, 200], dtype=np.float32), 30.0
    background = np.percentile(image[ring], 65, axis=0).astype(np.float32)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    background_gray = float(background @ np.array([0.114, 0.587, 0.299]))
    return background, max(5.0, background_gray - float(np.median(gray[body])))


def match_appearance(image, mask, target_image, target_mask):
    source_background, source_contrast = appearance(image, mask)
    target_background, target_contrast = appearance(target_image, target_mask)
    gain = float(np.clip((target_contrast / source_contrast) ** 0.65, 0.8, 1.25))
    shift = np.clip((target_background - source_background) * 0.8, -20, 20)
    corrected = source_background + shift + (image.astype(np.float32) - source_background) * gain
    return np.rint(corrected).clip(0, 255).astype(np.uint8)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/combined_ground_truth_cleaned_visibility-fixed.pkg.slp"))
    parser.add_argument("--library", type=Path, default=Path("outputs/donor_library_sam2"))
    parser.add_argument("--output", type=Path, default=Path("outputs/synthetic_data"))
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--min-separation", type=float, default=90)
    parser.add_argument("--max-added-donors", type=int, default=10,
                        help="Maximum one-donor crossings to add per frame.")
    parser.add_argument("--seed", type=int, default=165)
    return parser


def main():
    args = build_parser().parse_args()
    source = sleap_io.load_slp(str(args.source), open_videos=False)
    donors = approved_donors(args.library)
    skeleton = source.skeletons[0]
    donor_images = [cv2.imread(str(args.library / donor["image"])) for donor in donors]
    donor_masks = [cv2.imread(str(args.library / donor["mask"]), cv2.IMREAD_GRAYSCALE) for donor in donors]
    candidates = []
    if args.max_added_donors < 1:
        raise ValueError("--max-added-donors must be at least 1")
    for labeled in source:
        selected = separated_subset(labeled.instances, args.min_separation, args.max_added_donors)
        if len(selected) >= min(5, args.max_added_donors):
            candidates.append(labeled)
    if not candidates:
        raise RuntimeError("No source frames contain enough separated larvae at the requested separation")
    rng = np.random.default_rng(args.seed)
    output_images = args.output / ".hard_crossing_images"
    output_images.mkdir(parents=True, exist_ok=True)
    records, image_paths, labeled_frames = [], [], []
    with h5py.File(args.source, "r") as h5_file:
        for hard_id in range(args.count):
            labeled = candidates[hard_id % len(candidates)]
            video_id = source.videos.index(labeled.video)
            output = decode_frame(h5_file, video_id, int(labeled.frame_idx))
            output_instances = [sleap_io.Instance.from_numpy(np.asarray(instance.numpy()), skeleton)
                                for instance in labeled.instances]
            events = []
            selected = separated_subset(labeled.instances, args.min_separation, args.max_added_donors)
            anchors = [(np.asarray(labeled.instances[index].numpy(), dtype=np.float32).mean(axis=0),
                        np.asarray(labeled.instances[index].numpy(), dtype=np.float32))
                       for index in selected[:args.max_added_donors]]
            for pair_index, (center, target_points) in enumerate(anchors):
                donor_index = int(rng.integers(len(donors)))
                donor = donors[donor_index]
                donor_points = np.asarray(donor["points_xy"], dtype=np.float32)
                target_axis = target_points[-1] - target_points[0]
                target_angle = float(np.rad2deg(np.arctan2(target_axis[1], target_axis[0])))
                donor_axis = donor_points[-1] - donor_points[0]
                donor_angle = float(np.rad2deg(np.arctan2(donor_axis[1], donor_axis[0])))
                angle = donor_angle - target_angle + float(rng.normal(0, 25))
                donor_image, donor_mask, transformed_points = transform(
                    donor_images[donor_index], donor_masks[donor_index], donor_points,
                    angle, np.asarray(center, dtype=np.float32), output.shape[:2])
                target_mask = skeleton_mask(target_points, output.shape[:2])
                donor_image = match_appearance(donor_image, donor_mask, output, target_mask)
                output = darken_overlap(output, donor_image, donor_mask, target_mask)
                output_instances.append(sleap_io.Instance.from_numpy(transformed_points, skeleton))
                events.append({"pair": pair_index, "target_instance": int(selected[pair_index]),
                               "donor_id": int(donor["donor_id"]), "center": [float(value) for value in center],
                               "relative_angle": angle})
            image_path = output_images / f"hard_{hard_id:05d}.png"
            cv2.imwrite(str(image_path), output)
            image_paths.append(str(image_path.resolve()))
            labeled_frames.append(output_instances)
            records.append({"hard_id": hard_id, "source_video": video_id, "source_frame": int(labeled.frame_idx), "background_instances_preserved": len(labeled.instances), "events": events, "image": str(image_path.relative_to(args.output))})
    video = sleap_io.Video(filename=image_paths)
    frames = [sleap_io.LabeledFrame(video, i, instances) for i, instances in enumerate(labeled_frames)]
    sleap_io.Labels(labeled_frames=frames, videos=[video], skeletons=[skeleton]).save(str(args.output / "crossings_hard.slp"), embed=True, verbose=False)
    (args.output / "crossings_hard.json").write_text(json.dumps({
        "source": str(args.source), "count": len(records),
        "min_separation": args.min_separation,
        "max_added_donors": args.max_added_donors,
        "background_instances_preserved": True, "records": records,
    }, indent=2))
    print(f"Generated {len(records)} hard crossing frames from {len(candidates)} eligible source frames")


if __name__ == "__main__":
    main()