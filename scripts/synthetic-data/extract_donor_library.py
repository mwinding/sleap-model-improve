#!/usr/bin/env python3
"""Extract isolated larva crops and approximate masks from a packaged SLP.

The output is intended as the first stage of the synthetic-crossing pipeline:
real donor crops retain the imaging texture of the source data, while the
stored local keypoints make them ready for later compositing.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import h5py
import numpy as np
import sleap_io


NODE_ORDER = ("head", "mouthhooks", "body", "tail", "spiracle")


@dataclass
class DonorRecord:
    donor_id: int
    source_video: int
    source_frame: int
    source_instance: int
    body_length: float
    nearest_neighbor_distance: float
    visible: list[bool]
    points_xy: list[list[float]]
    crop_xyxy: list[int]
    image: str
    mask: str


def decode_frame(h5_file: h5py.File, video_id: int, frame_idx: int) -> np.ndarray:
    frame_numbers = np.asarray(h5_file[f"video{video_id}/frame_numbers"])
    local_matches = np.flatnonzero(frame_numbers == frame_idx)
    if len(local_matches) != 1:
        raise RuntimeError(f"Could not resolve video {video_id}, source frame {frame_idx}")
    local_idx = int(local_matches[0])
    encoded = np.asarray(h5_file[f"video{video_id}/video"][local_idx], dtype=np.uint8)
    frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError(f"Could not decode video {video_id}, frame {frame_idx}")
    return frame


def instance_points(instance: sleap_io.Instance) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(instance.numpy(), dtype=np.float64)
    visible = np.asarray(instance.points["visible"], dtype=bool)
    return points, visible


def body_length(points: np.ndarray, visible: np.ndarray) -> float:
    head = NODE_ORDER.index("head")
    spiracle = NODE_ORDER.index("spiracle")
    if not visible[head] or not visible[spiracle]:
        return float("nan")
    return float(np.linalg.norm(points[head] - points[spiracle]))


def make_mask(crop: np.ndarray, points: np.ndarray, visible: np.ndarray, length: float) -> np.ndarray:
    """Make a dark-pixel mask constrained by the labeled larval skeleton."""
    prior = np.zeros(crop.shape[:2], dtype=np.uint8)
    skeleton_points = np.rint(points[visible]).astype(np.int32)
    if len(skeleton_points) >= 2:
        radius = max(3, int(round(length * 0.06)))
        cv2.polylines(prior, [skeleton_points], False, 255, thickness=radius * 2)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        prior = cv2.dilate(prior, kernel, iterations=1)

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    dark_pixels = cv2.adaptiveThreshold(
        255 - gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        5,
    )
    mask = cv2.bitwise_and(dark_pixels, prior)
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, close_kernel, iterations=1)

    if cv2.countNonZero(mask) < max(20, int(length * 2)):
        mask = prior
    return mask


def make_qc_tile(crop: np.ndarray, mask: np.ndarray, points: np.ndarray) -> np.ndarray:
    tile = crop.copy()
    overlay = np.zeros_like(tile)
    overlay[:, :, 1] = mask
    tile = cv2.addWeighted(tile, 0.78, overlay, 0.22, 0)
    for x, y in np.rint(points).astype(int):
        cv2.circle(tile, (x, y), 3, (0, 0, 255), -1)
    return tile


def load_candidates(slp_path: Path, crop_size: int) -> tuple[list[dict], list[str]]:
    labels = sleap_io.load_slp(str(slp_path), open_videos=False)
    skeleton = labels.skeletons[0]
    node_names = [node.name for node in skeleton.nodes]
    if tuple(node_names) != NODE_ORDER:
        raise ValueError(f"Unexpected node order: {node_names}; expected {list(NODE_ORDER)}")

    candidates = []
    half = crop_size / 2
    for lf in labels:
        video_id = labels.videos.index(lf.video)
        for instance_number, instance in enumerate(lf.instances):
            points, visible = instance_points(instance)
            length = body_length(points, visible)
            if not np.isfinite(length):
                continue
            center = points[visible].mean(axis=0)
            centers = []
            for other in lf.instances:
                if other is instance:
                    continue
                other_points, other_visible = instance_points(other)
                centers.append(other_points[other_visible].mean(axis=0))
            nearest = min((float(np.linalg.norm(center - other)) for other in centers), default=float("inf"))
            candidates.append({
                "video_id": video_id,
                "frame_idx": int(lf.frame_idx),
                "instance_number": instance_number,
                "points": points,
                "visible": visible,
                "body_length": length,
                "nearest": nearest,
                "center": center,
                "frame_shape": (1100, 1800),
                "half": half,
            })
    return candidates, node_names


def select_candidates(candidates: list[dict], crop_size: int, min_body: float, max_body: float,
                      isolation_ratio: float) -> list[dict]:
    selected = []
    half = crop_size / 2
    for candidate in candidates:
        x, y = candidate["center"]
        length = candidate["body_length"]
        if not (min_body <= length <= max_body):
            continue
        if candidate["nearest"] <= isolation_ratio * length:
            continue
        width, height = candidate["frame_shape"][1], candidate["frame_shape"][0]
        if not (half <= x < width - half and half <= y < height - half):
            continue
        selected.append(candidate)
    return selected


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slp", type=Path, default=Path("data/combined_ground_truth_cleaned_visibility-fixed.pkg.slp"))
    parser.add_argument("--output", type=Path, default=Path("outputs/donor_library"))
    parser.add_argument("--crop-size", type=int, default=192)
    parser.add_argument("--min-body-length", type=float, default=50)
    parser.add_argument("--max-body-length", type=float, default=100)
    parser.add_argument("--isolation-ratio", type=float, default=1.5)
    parser.add_argument("--max-donors", type=int, default=None)
    parser.add_argument("--qc-tiles", type=int, default=64)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.crop_size % 2:
        raise ValueError("--crop-size must be even")

    candidates, node_names = load_candidates(args.slp, args.crop_size)
    selected = select_candidates(
        candidates,
        args.crop_size,
        args.min_body_length,
        args.max_body_length,
        args.isolation_ratio,
    )
    if args.max_donors is not None:
        selected = selected[:args.max_donors]
    if not selected:
        raise RuntimeError("No donor candidates matched the selection criteria")

    image_dir = args.output / "images"
    mask_dir = args.output / "masks"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    records: list[DonorRecord] = []
    qc_tiles = []
    slp_frames = []
    slp_images = []
    source_labels = sleap_io.load_slp(str(args.slp), open_videos=False)
    skeleton = source_labels.skeletons[0]

    with h5py.File(args.slp, "r") as h5_file:
        for donor_id, candidate in enumerate(selected):
            frame = decode_frame(h5_file, candidate["video_id"], candidate["frame_idx"])
            cx, cy = np.rint(candidate["center"]).astype(int)
            half = args.crop_size // 2
            x0, y0 = cx - half, cy - half
            crop = frame[y0:cy + half, x0:cx + half]
            local_points = candidate["points"] - np.array([x0, y0])
            mask = make_mask(crop, local_points, candidate["visible"], candidate["body_length"])

            image_path = image_dir / f"donor_{donor_id:05d}.png"
            mask_path = mask_dir / f"donor_{donor_id:05d}.png"
            cv2.imwrite(str(image_path), crop)
            cv2.imwrite(str(mask_path), mask)
            slp_images.append(str(image_path.resolve()))

            instance = sleap_io.Instance.from_numpy(local_points, skeleton)
            instance.points["visible"] = candidate["visible"]
            instance.points["complete"] = candidate["visible"]
            records.append(DonorRecord(
                donor_id=donor_id,
                source_video=candidate["video_id"],
                source_frame=candidate["frame_idx"],
                source_instance=candidate["instance_number"],
                body_length=candidate["body_length"],
                nearest_neighbor_distance=candidate["nearest"],
                visible=candidate["visible"].tolist(),
                points_xy=local_points.tolist(),
                crop_xyxy=[int(x0), int(y0), int(x0 + args.crop_size), int(y0 + args.crop_size)],
                image=str(image_path.relative_to(args.output)),
                mask=str(mask_path.relative_to(args.output)),
            ))
            if len(qc_tiles) < args.qc_tiles:
                qc_tiles.append(make_qc_tile(crop, mask, local_points))

    video = sleap_io.Video(filename=slp_images)
    slp_frames = [sleap_io.LabeledFrame(video, index, [instance])
                  for index, instance in enumerate(
                      [sleap_io.Instance.from_numpy(record.points_xy, skeleton) for record in records]
                  )]
    for labeled_frame, record in zip(slp_frames, records):
        labeled_frame.instances[0].points["visible"] = np.asarray(record.visible, dtype=bool)
        labeled_frame.instances[0].points["complete"] = np.asarray(record.visible, dtype=bool)
    labels = sleap_io.Labels(labeled_frames=slp_frames, videos=[video], skeletons=[skeleton])
    labels.save(str(args.output / "donors.slp"), embed=True, verbose=False)

    with (args.output / "donors.json").open("w", encoding="utf-8") as handle:
        json.dump({"nodes": node_names, "crop_size": args.crop_size, "donors": [asdict(record) for record in records]}, handle, indent=2)

    columns = 8
    rows = int(np.ceil(len(qc_tiles) / columns))
    qc = np.full((rows * args.crop_size, columns * args.crop_size, 3), 255, dtype=np.uint8)
    for index, tile in enumerate(qc_tiles):
        row, column = divmod(index, columns)
        qc[row * args.crop_size:(row + 1) * args.crop_size, column * args.crop_size:(column + 1) * args.crop_size] = tile
    cv2.imwrite(str(args.output / "qc_montage.png"), qc)

    print(f"Selected {len(selected)} donors from {len(candidates)} labeled instances")
    print(f"Wrote images, masks, donors.json, donors.slp, and qc_montage.png to {args.output}")


if __name__ == "__main__":
    main()