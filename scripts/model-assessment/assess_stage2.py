#!/usr/bin/env python3
"""Evaluate centered-instance (pose) models on crops centred at the labelled anchor.

Run each model on the benchmark frames with ground-truth centroids first, e.g.
  sleap predict -i outputs/benchmark/benchmark_frames.pkg.slp -m <centered_instance_model_dir> \
      -o outputs/model_assessment/stage2/predictions/<model>.slp
sleap-nn then crops every labelled larva around its anchor node and returns one skeleton
per larva, in label order. This isolates pose quality from detection: each prediction is
compared with its own larva. A prediction is "wrong animal" when its skeleton is closer
(mean node distance) to another larva than to its own. Crowded = body point within
--crowded-px of another larva's skeleton, as in assess_centroids.py.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import sleap_io as sio

from assess_centroids import ROOT, crowded_flags, write_csv
from assess_poses import mean_node_distance


def frame_key(frame):
    video = frame.video.source_video or frame.video
    return str(video.filename), int(frame.frame_idx)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--ground-truth', type=Path, default=ROOT/'outputs/benchmark/benchmark_frames.pkg.slp')
    parser.add_argument('--predictions-dir', type=Path, default=ROOT/'outputs/model_assessment/stage2/predictions')
    parser.add_argument('--models', nargs='+', required=True)
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/model_assessment/stage2')
    parser.add_argument('--crowded-px', type=float, default=15.0)
    args = parser.parse_args()

    gt_labels = sio.load_slp(str(args.ground_truth), open_videos=False)
    names = gt_labels.skeletons[0].node_names
    body = names.index('body')
    truth = {}
    for frame in gt_labels:
        instances = [i for i in frame.instances if not isinstance(i, sio.PredictedInstance)]
        poses = np.stack([i.numpy() for i in instances])
        truth[frame_key(frame)] = (poses, crowded_flags(instances, poses[:, body], args.crowded_px))

    summary, per_node, per_animal = [], [], []
    for model in args.models:
        pred_labels = sio.load_slp(str(args.predictions_dir/f'{model}.slp'), open_videos=False)
        if pred_labels.skeletons[0].node_names != names:
            raise ValueError(f'{model}: skeleton does not match ground truth')
        errors = {s: [] for s in ('all', 'crowded', 'isolated')}
        wrong = {s: 0 for s in errors}
        count = {s: 0 for s in errors}
        seen = set()
        for frame in pred_labels:
            key = frame_key(frame)
            gt, crowded = truth[key]
            pred = np.stack([i.numpy() for i in frame.instances])
            if len(pred) != len(gt):
                raise ValueError(f'{model}: {key} has {len(pred)} predictions for {len(gt)} larvae')
            seen.add(key)
            dist = mean_node_distance(gt, pred)          # rows GT, columns predictions
            for i in range(len(gt)):
                err = np.linalg.norm(gt[i] - pred[i], axis=1)
                own = dist[i, i]
                others = np.delete(dist[:, i], i)
                is_wrong = bool(len(others) and np.nanmin(others) < own)
                subset = 'crowded' if crowded[i] else 'isolated'
                for s in ('all', subset):
                    errors[s].append(err)
                    wrong[s] += is_wrong
                    count[s] += 1
                per_animal.append({'model': model, 'video': key[0], 'frame_idx': key[1], 'gt_index': i,
                                   'crowded': bool(crowded[i]), 'wrong_animal': is_wrong,
                                   'mean_error_px': float(own) if np.isfinite(own) else None,
                                   **{f'err_{n}': (float(e) if np.isfinite(e) else None) for n, e in zip(names, err)}})
        if seen != set(truth):
            raise ValueError(f'{model}: predictions cover {len(seen)} of {len(truth)} benchmark frames')
        for s in ('all', 'crowded', 'isolated'):
            e = np.array(errors[s])
            hit5 = np.nan_to_num(e, nan=np.inf) <= 5
            hit10 = np.nan_to_num(e, nan=np.inf) <= 10
            summary.append({'model': model, 'subset': s, 'larvae': count[s],
                            'wrong_animal_pct': 100 * wrong[s] / count[s],
                            'complete_pct': 100 * float(np.isfinite(e).all(axis=1).mean()),
                            'median_node_error_px': float(np.nanmedian(e)),
                            'pck5_pct': 100 * float(hit5.mean()), 'pck10_pct': 100 * float(hit10.mean()),
                            'all_nodes_within_10px_pct': 100 * float(hit10.all(axis=1).mean())})
            for k, node in enumerate(names):
                col = e[:, k]
                per_node.append({'model': model, 'subset': s, 'node': node,
                                 'missing_pct': 100 * float(np.isnan(col).mean()),
                                 'median_error_px': float(np.nanmedian(col)) if np.isfinite(col).any() else None,
                                 'p90_error_px': float(np.nanpercentile(col, 90)) if np.isfinite(col).any() else None,
                                 'pck5_pct': 100 * float((np.nan_to_num(col, nan=np.inf) <= 5).mean()),
                                 'pck10_pct': 100 * float((np.nan_to_num(col, nan=np.inf) <= 10).mean())})
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output/'stage2_summary.csv', summary)
    write_csv(args.output/'stage2_nodes.csv', per_node)
    write_csv(args.output/'stage2_per_animal.csv', per_animal)
    (args.output/'stage2_assessment.json').write_text(json.dumps(
        {'models': args.models, 'ground_truth': str(args.ground_truth), 'crowded_px': args.crowded_px,
         'crops': 'centred on each labelled larva at the model anchor node (sleap-nn ground-truth centroids)'}, indent=2))
    print(f"{'model':40s} {'subset':9s} {'wrong animal':>12s} {'all 5 within 10px':>18s} {'median err':>10s} {'PCK@5':>6s} {'complete':>8s}")
    for r in summary:
        print(f"{r['model']:40s} {r['subset']:9s} {r['wrong_animal_pct']:11.1f}% {r['all_nodes_within_10px_pct']:17.1f}% "
              f"{r['median_node_error_px']:9.2f}px {r['pck5_pct']:5.1f}% {r['complete_pct']:7.1f}%")
    print(f'Wrote {args.output}')


if __name__ == '__main__':
    main()
