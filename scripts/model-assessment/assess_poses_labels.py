#!/usr/bin/env python3
"""Evaluate full-pose predictions made directly on a labels file (e.g. held-back tuning frames).

Like assess_poses.py, but ground truth and predictions are matched per labelled frame
(same video and frame index) instead of via the repeated benchmark video. Skeletons are
matched one-to-one by mean node distance within --tolerance. Reports instance recall and
precision, F1, crowded/isolated recall and node accuracy.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import sleap_io as sio

from assess_centroids import crowded_flags, match_distances, write_csv
from assess_poses import mean_node_distance


def frame_key(frame):
    video = frame.video
    while video.source_video is not None:
        video = video.source_video
    return str(video.filename), int(frame.frame_idx)


def load_truth(path, crowded_px=15.0):
    labels = sio.load_slp(str(path), open_videos=False)
    body = labels.skeletons[0].node_names.index('body')
    truth = {}
    for frame in labels:
        instances = [i for i in frame.instances if not isinstance(i, sio.PredictedInstance)]
        poses = np.stack([i.numpy() for i in instances])
        truth[frame_key(frame)] = (poses, crowded_flags(instances, poses[:, body], crowded_px))
    return truth


def evaluate(truth, predictions_path, tolerance=15.0):
    labels = sio.load_slp(str(predictions_path), open_videos=False)
    preds = {frame_key(f): np.stack([i.numpy() for i in f.instances]) for f in labels if len(f.instances)}
    unknown = set(preds) - set(truth)
    if unknown:
        raise ValueError(f'{predictions_path}: predictions on frames without labels: {sorted(unknown)[:3]}')
    tp = {'all': 0, 'crowded': 0, 'isolated': 0}
    n = {'all': 0, 'crowded': 0, 'isolated': 0}
    n_pred, errors = 0, []
    for key, (gt, crowded) in truth.items():
        pred = preds.get(key, np.empty((0,) + gt.shape[1:]))
        n_pred += len(pred)
        matches = match_distances(mean_node_distance(gt, pred), tolerance) if len(pred) else []
        matched = {i for i, _, _ in matches}
        for i in range(len(gt)):
            for s in ('all', 'crowded' if crowded[i] else 'isolated'):
                n[s] += 1
                tp[s] += i in matched
        errors += [np.linalg.norm(gt[i] - pred[j], axis=1) for i, j, _ in matches]
    e = np.array(errors).reshape(-1, truth[next(iter(truth))][0].shape[1])
    recall = 100 * tp['all'] / n['all']
    precision = 100 * tp['all'] / n_pred if n_pred else float('nan')
    return {'recall_pct': recall, 'precision_pct': precision,
            'f1_pct': 2 * recall * precision / (recall + precision) if recall + precision else 0.0,
            'crowded_recall_pct': 100 * tp['crowded'] / n['crowded'] if n['crowded'] else float('nan'),
            'isolated_recall_pct': 100 * tp['isolated'] / n['isolated'] if n['isolated'] else float('nan'),
            'pck5_pct': 100 * float((np.nan_to_num(e, nan=np.inf) <= 5).mean()) if len(e) else float('nan'),
            'median_node_error_px': float(np.nanmedian(e)) if len(e) else float('nan'),
            'larvae': n['all'], 'crowded_larvae': n['crowded'], 'predictions': n_pred}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ground-truth', type=Path, required=True)
    parser.add_argument('--predictions', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, default=None, help='Optional CSV of results')
    parser.add_argument('--tolerance', type=float, default=15.0)
    args = parser.parse_args()
    truth = load_truth(args.ground_truth)
    rows = []
    print(f"{'predictions':44s} {'recall':>6s} {'prec':>6s} {'F1':>5s} {'crowd':>6s} {'PCK5':>5s}")
    for path in args.predictions:
        r = {'predictions': path.stem, **evaluate(truth, path, args.tolerance)}
        rows.append(r)
        print(f"{path.stem:44s} {r['recall_pct']:6.1f} {r['precision_pct']:6.1f} {r['f1_pct']:5.1f} "
              f"{r['crowded_recall_pct']:6.1f} {r['pck5_pct']:5.1f}")
    if args.output:
        write_csv(args.output, rows)


if __name__ == '__main__':
    main()
