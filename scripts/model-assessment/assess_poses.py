#!/usr/bin/env python3
"""Evaluate full-pose predictions (bottom-up or top-down) on the repeated-image benchmark.

Predicted skeletons are matched one-to-one to labelled larvae by mean distance over the
nodes present in both (maximum matches within --tolerance, then minimum total distance).
Reports instance recall/precision, per-node error and the share of nodes within 5 / 10 px,
overall and for crowded vs isolated larvae (crowded: body point within --crowded-px of
another larva's skeleton, the same definition as assess_centroids.py).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import sleap_io as sio

from assess_centroids import MOUNT, ROOT, load_benchmark, match_distances, verify_video, write_csv


def mean_node_distance(gt, pred):
    """GT x prediction matrix of mean distance over nodes present in both (inf if none)."""
    diff = np.linalg.norm(gt[:, None] - pred[None], axis=3)          # (n_gt, n_pred, n_nodes)
    valid = np.isfinite(diff)
    counts = valid.sum(axis=2)
    total = np.where(valid, diff, 0).sum(axis=2)
    return np.where(counts > 0, total / np.maximum(counts, 1), np.inf)


def load_pose_predictions(path, expected, node_names):
    labels = sio.load_slp(str(path), open_videos=False)
    if len(labels.videos) != 1 or labels.videos[0].shape[0] != expected:
        raise ValueError(f'{path}: expected one comparison video with {expected} frames')
    names = labels.skeletons[0].node_names
    if names != node_names:
        raise ValueError(f'{path}: skeleton {names} does not match ground truth {node_names}')
    poses = {i: np.empty((0, len(names), 2)) for i in range(expected)}
    for frame in labels:
        instances = [i for i in frame.instances if isinstance(i, sio.PredictedInstance)]
        if instances:
            poses[int(frame.frame_idx)] = np.stack([i.numpy() for i in instances])
    return poses


def gt_poses(ground_truth, benchmark):
    labels = sio.load_slp(str(ground_truth), open_videos=False)
    lookup = {(labels.videos.index(f.video), int(f.frame_idx)): f for f in labels}
    out = {}
    for row in benchmark:
        frame = lookup[row['video_id'], row['frame_idx']]
        out[row['order']] = np.stack([i.numpy() for i in frame.instances if not isinstance(i, sio.PredictedInstance)])
    return out, labels.skeletons[0].node_names


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ground-truth', type=Path, default=ROOT/'data/combined_ground_truth_cleaned_visibility-fixed.pkg.slp')
    parser.add_argument('--manifest', type=Path, default=Path(__file__).with_name('comparison_manifest.csv'))
    parser.add_argument('--predictions-dir', type=Path, default=MOUNT/'predictions')
    parser.add_argument('--comparison-video', type=Path, default=MOUNT/'comparison_frames_plain.mp4')
    parser.add_argument('--models', nargs='+', required=True, help='Prediction filenames without .slp')
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/model_assessment/poses')
    parser.add_argument('--tolerance', type=float, default=15.0, help='Max mean node distance for a skeleton match')
    parser.add_argument('--crowded-px', type=float, default=15.0)
    parser.add_argument('--repeats', type=int, default=10)
    args = parser.parse_args()

    benchmark = load_benchmark(args.manifest, args.ground_truth, set(), 'body', args.crowded_px)
    verify_video(args.comparison_video, args.ground_truth, benchmark, args.repeats)
    truth, node_names = gt_poses(args.ground_truth, benchmark)
    args.output.mkdir(parents=True, exist_ok=True)
    summary, per_node, per_animal = [], [], []
    for model in args.models:
        poses = load_pose_predictions(args.predictions_dir/f'{model}.slp', len(benchmark)*args.repeats, node_names)
        tp = {'all': 0, 'crowded': 0, 'isolated': 0}
        gt_total = {'all': 0, 'crowded': 0, 'isolated': 0}
        n_pred = 0
        errors = {s: [] for s in ('all', 'crowded', 'isolated')}      # (n_matched, n_nodes) arrays
        for row in benchmark:
            gt = truth[row['order']]
            for repeat in range(args.repeats):
                index = (row['order'] - 1) * args.repeats + repeat
                pred = poses[index]
                n_pred += len(pred)
                matches = match_distances(mean_node_distance(gt, pred), args.tolerance) if len(pred) else []
                matched = {i: j for i, j, _ in matches}
                for i in range(len(gt)):
                    subset = 'crowded' if row['crowded'][i] else 'isolated'
                    for s in ('all', subset):
                        gt_total[s] += 1
                    j = matched.get(i)
                    if repeat == 0:
                        per_animal.append({'model': model, 'order': row['order'], 'gt_index': i, 'crowded': bool(row['crowded'][i]),
                                           'matched': j is not None,
                                           **{f'err_{n}': (float(np.linalg.norm(gt[i, k] - pred[j, k])) if j is not None else None)
                                              for k, n in enumerate(node_names)}})
                    if j is None:
                        continue
                    err = np.linalg.norm(gt[i] - pred[j], axis=1)   # NaN where the prediction lacks a node
                    for s in ('all', subset):
                        tp[s] += 1
                        errors[s].append(err)
        for s in ('all', 'crowded', 'isolated'):
            e = np.array(errors[s]).reshape(-1, len(node_names))
            summary.append({'model': model, 'subset': s, 'larvae': gt_total[s] / args.repeats,
                            'instance_recall_pct': 100 * tp[s] / gt_total[s] if gt_total[s] else None,
                            'instance_precision_pct': (100 * tp['all'] / n_pred if n_pred else None) if s == 'all' else None,
                            'complete_pct': 100 * float(np.isfinite(e).all(axis=1).mean()) if len(e) else None,
                            'median_node_error_px': float(np.nanmedian(e)) if len(e) else None,
                            'pck5_pct': 100 * float((np.nan_to_num(e, nan=np.inf) <= 5).mean()) if len(e) else None,
                            'pck10_pct': 100 * float((np.nan_to_num(e, nan=np.inf) <= 10).mean()) if len(e) else None})
            for k, node in enumerate(node_names):
                col = e[:, k] if len(e) else np.array([])
                per_node.append({'model': model, 'subset': s, 'node': node,
                                 'missing_pct': 100 * float(np.isnan(col).mean()) if len(col) else None,
                                 'median_error_px': float(np.nanmedian(col)) if np.isfinite(col).any() else None,
                                 'p90_error_px': float(np.nanpercentile(col, 90)) if np.isfinite(col).any() else None,
                                 'pck5_pct': 100 * float((np.nan_to_num(col, nan=np.inf) <= 5).mean()) if len(col) else None,
                                 'pck10_pct': 100 * float((np.nan_to_num(col, nan=np.inf) <= 10).mean()) if len(col) else None})
        overall = next(r for r in summary if r['model'] == model and r['subset'] == 'all')
        print(f"{model}: instance recall {overall['instance_recall_pct']:.1f}%, precision {overall['instance_precision_pct']:.1f}%, "
              f"median node error {overall['median_node_error_px']:.2f} px, PCK@5 {overall['pck5_pct']:.1f}%", flush=True)
    write_csv(args.output/'pose_summary.csv', summary)
    write_csv(args.output/'pose_nodes.csv', per_node)
    write_csv(args.output/'pose_per_animal.csv', per_animal)
    (args.output/'pose_assessment.json').write_text(json.dumps({
        'models': args.models, 'tolerance_px': args.tolerance, 'crowded_px': args.crowded_px, 'repeats': args.repeats,
        'matching': 'one-to-one skeletons, max matches with mean node distance <= tolerance, then min total distance',
        'node_error': 'Euclidean distance per node for matched skeletons; a node missing from the prediction counts as a miss in PCK and in missing_pct'},
        indent=2))
    print(f'Wrote pose assessment to {args.output}')


if __name__ == '__main__':
    main()
