#!/usr/bin/env python3
"""Characterise missed larvae (false negatives) from a centroid assessment.

For each missed GT body point: distance to the nearest other larva's skeleton,
whether an unmatched prediction sits nearby, distance to the frame edge, local
contrast, and how many of the repeats missed it. Writes a CSV, a category
breakdown and a montage of crops for visual review.
"""
from __future__ import annotations
import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import h5py
import numpy as np
import sleap_io as sio

from assess_centroids import ROOT, decode_source

OVERLAP_PX = 15      # nearest other skeleton within this: bodies touch/overlap
CLOSE_PX = 45        # within this: a neighbour is close but not touching
NEAR_MISS_PX = 40    # unmatched prediction within this of the missed GT
EDGE_PX = 60


def segment_distance(point, a, b):
    ab = b - a
    t = 0.0 if not ab.any() else float(np.clip(np.dot(point - a, ab) / np.dot(ab, ab), 0, 1))
    return float(np.linalg.norm(point - (a + t * ab)))


def skeleton_distance(point, instance):
    pts = instance.numpy()
    pts = pts[np.isfinite(pts).all(axis=1)]
    if len(pts) == 0:
        return np.inf
    if len(pts) == 1:
        return float(np.linalg.norm(point - pts[0]))
    return min(segment_distance(point, pts[i], pts[i + 1]) for i in range(len(pts) - 1))


def local_contrast(gray, xy, radius=6, ring=30):
    x, y = int(round(xy[0])), int(round(xy[1]))
    h, w = gray.shape
    y0, y1, x0, x1 = max(0, y - ring), min(h, y + ring + 1), max(0, x - ring), min(w, x + ring + 1)
    patch = gray[y0:y1, x0:x1].astype(np.float32)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    dist = np.hypot(xx - x, yy - y)
    body = patch[dist <= radius]
    background = patch[dist > ring * 0.6]
    if not len(body) or not len(background):
        return float('nan')
    return float(np.percentile(background, 65) - np.median(body))


def categorise(row):
    if row['nearest_skeleton_px'] <= OVERLAP_PX:
        return 'overlapping'
    if row['nearest_skeleton_px'] <= CLOSE_PX:
        return 'close_neighbour'
    if row['edge_px'] <= EDGE_PX:
        return 'frame_edge'
    return 'isolated'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assessment', type=Path, default=ROOT / 'outputs/model_assessment')
    parser.add_argument('--models', nargs='+', default=['centroid_fullres_body', 'centroid_body_synth_fullframe_darkoverlap_v1'])
    parser.add_argument('--output', type=Path, default=None, help='Default: <assessment>/misses')
    parser.add_argument('--crop', type=int, default=160)
    parser.add_argument('--columns', type=int, default=8)
    args = parser.parse_args()
    output = args.output or args.assessment / 'misses'
    output.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((args.assessment / 'assessment.json').read_text())
    repeats = metadata['repeats']
    manifest = {int(r['order']): r for r in csv.DictReader(Path(metadata['manifest']).open())}
    labels = sio.load_slp(metadata['ground_truth'], open_videos=False)
    frames = {(labels.videos.index(f.video), int(f.frame_idx)): f for f in labels}
    with (args.assessment / 'matches.csv').open() as handle:
        matches = list(csv.DictReader(handle))

    half = args.crop // 2
    with h5py.File(metadata['ground_truth']) as h5:
        images = {}
        for order, row in manifest.items():
            image = decode_source(h5, int(row['video_id']), int(row['frame_idx']))
            images[order] = (image, cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))

    for model in args.models:
        rows = [r for r in matches if r['model'] == model]
        if not rows:
            print(f'{model}: no rows in matches.csv'); continue
        # How many repeats missed each GT; repeat 0 supplies the geometry.
        miss_count = Counter((int(r['order']), int(r['gt_index'])) for r in rows if r['status'] == 'FN')
        gt_count = len({(int(r['order']), r['gt_index']) for r in rows if r['gt_index']})
        results = []
        for r in rows:
            if int(r['repeat']) != 0 or r['status'] != 'FN':
                continue
            order = int(r['order'])
            entry = manifest[order]
            frame = frames[int(entry['video_id']), int(entry['frame_idx'])]
            gt_index = int(r['gt_index'])
            point = np.array([float(r['gt_x']), float(r['gt_y'])])
            others = [inst for i, inst in enumerate(frame.instances) if i != gt_index]
            nearest = min(skeleton_distance(point, inst) for inst in others) if others else np.inf
            unmatched = [(float(p['pred_x']), float(p['pred_y']), float(p['score'])) for p in rows
                         if p['status'] == 'FP' and int(p['order']) == order and int(p['repeat']) == 0]
            near = min((np.hypot(px - point[0], py - point[1]), s) for px, py, s in unmatched) if unmatched else (np.inf, np.nan)
            image, gray = images[order]
            h, w = gray.shape
            row = {'model': model, 'order': order, 'category': entry['category'], 'gt_index': gt_index,
                   'x': point[0], 'y': point[1], 'nearest_skeleton_px': round(nearest, 1),
                   'nearest_unmatched_pred_px': round(near[0], 1) if np.isfinite(near[0]) else '',
                   'nearest_unmatched_pred_score': round(near[1], 3) if np.isfinite(near[1]) else '',
                   'edge_px': round(min(point[0], point[1], w - point[0], h - point[1]), 1),
                   'contrast': round(local_contrast(gray, point), 1),
                   'missed_in_repeats': miss_count[order, gt_index]}
            row['miss_type'] = categorise(row)
            row['near_miss'] = np.isfinite(near[0]) and near[0] <= NEAR_MISS_PX
            results.append(row)
        results.sort(key=lambda r: (r['miss_type'], r['order'], r['gt_index']))
        with (output / f'{model}_misses.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(results[0]))
            writer.writeheader(); writer.writerows(results)

        # Breakdown
        total = len(results)
        print(f'\n== {model}: {total} misses out of {gt_count} larvae (repeat 0)')
        by_type = Counter(r['miss_type'] for r in results)
        for kind in ('overlapping', 'close_neighbour', 'frame_edge', 'isolated'):
            n = by_type.get(kind, 0)
            near = sum(r['near_miss'] for r in results if r['miss_type'] == kind)
            print(f'   {kind:16s} {n:3d} ({100*n/total:4.0f}%)   with an unmatched prediction within {NEAR_MISS_PX} px: {near}')
        consistent = sum(r['missed_in_repeats'] == repeats for r in results)
        print(f'   missed in all {repeats} repeats: {consistent}/{total}; hard-subset misses: '
              f"{sum(r['category']=='hard_early' for r in results)}")
        contrast_all = [local_contrast(images[int(r['order'])][1], (float(r['gt_x']), float(r['gt_y'])))
                        for r in rows if int(r['repeat']) == 0 and r['gt_x']]
        print(f"   median contrast: misses {np.nanmedian([r['contrast'] for r in results]):.0f} vs all larvae {np.nanmedian(contrast_all):.0f}")
        per_image = Counter(r['order'] for r in results)
        print('   misses per image:', dict(sorted(per_image.items())))

        # Montage of crops, grouped by miss type
        tiles = []
        for r in results:
            image = images[r['order']][0]
            h, w = image.shape[:2]
            cx, cy = int(round(r['x'])), int(round(r['y']))
            x0, y0 = int(np.clip(cx - half, 0, w - args.crop)), int(np.clip(cy - half, 0, h - args.crop))
            tile = image[y0:y0 + args.crop, x0:x0 + args.crop].copy()
            for p in rows:
                if int(p['order']) != r['order'] or int(p['repeat']) != 0:
                    continue
                if p['pred_x']:
                    px, py = int(round(float(p['pred_x']))) - x0, int(round(float(p['pred_y']))) - y0
                    if 0 <= px < args.crop and 0 <= py < args.crop:
                        cv2.circle(tile, (px, py), 6, (255, 255, 0), 1, cv2.LINE_AA)
                if p['gt_x']:
                    gx, gy = int(round(float(p['gt_x']))) - x0, int(round(float(p['gt_y']))) - y0
                    if 0 <= gx < args.crop and 0 <= gy < args.crop:
                        colour = (0, 190, 0) if p['status'] == 'TP' else (0, 0, 255)
                        cv2.drawMarker(tile, (gx, gy), colour, cv2.MARKER_TILTED_CROSS, 9, 1, cv2.LINE_AA)
            cv2.circle(tile, (cx - x0, cy - y0), 12, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.rectangle(tile, (0, 0), (args.crop - 1, 15), (0, 0, 0), -1)
            cv2.putText(tile, f"{r['miss_type'][:9]} im{r['order']} {r['missed_in_repeats']}/{repeats}", (3, 11),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 255, 255), 1, cv2.LINE_AA)
            tiles.append(tile)
        rows_n = int(np.ceil(len(tiles) / args.columns))
        montage = np.full((rows_n * args.crop, args.columns * args.crop, 3), 255, np.uint8)
        for i, tile in enumerate(tiles):
            rr, cc = divmod(i, args.columns)
            montage[rr * args.crop:(rr + 1) * args.crop, cc * args.crop:(cc + 1) * args.crop] = tile
        cv2.imwrite(str(output / f'{model}_misses.png'), montage)
        print(f'   wrote {output / f"{model}_misses.csv"} and {output / f"{model}_misses.png"}')
    (output / 'README.txt').write_text(
        'Crops centred on each missed GT body point (red ring). Red crosses: missed GT; green crosses: detected GT; '
        'cyan circles: predictions (repeat 0). Label: miss type, manifest image, repeats in which it was missed.\n'
        f'overlapping: nearest other skeleton <= {OVERLAP_PX} px; close_neighbour: <= {CLOSE_PX} px; '
        f'frame_edge: <= {EDGE_PX} px from the border; isolated: none of these.\n')


if __name__ == '__main__':
    main()
