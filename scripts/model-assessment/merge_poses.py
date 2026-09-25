#!/usr/bin/env python3
"""Merge full-pose predictions from several top-down pipelines (one per anchor) into one set.

Per frame, skeletons from all inputs are pooled and grouped greedily, most confident first:
a skeleton joins an existing group if its mean node distance to the group is within
--match-px and the group has no skeleton from the same input yet (one pipeline never
supplies two copies of the same larva). Each group is fused into one skeleton by averaging
every node weighted by its point score. The number of inputs supporting each fused skeleton
is written to a sidecar CSV and can be used as a filter (--min-support).

Example:
  python scripts/model-assessment/merge_poses.py \
    --input mouthhooks preds/topdown_mouthhooks.slp --input body preds/topdown_body.slp \
    --input tail preds/topdown_tail.slp --output preds/merged_mouthhooks_body_tail.slp
"""
from __future__ import annotations

import argparse
import csv
import warnings
from pathlib import Path

import numpy as np
import sleap_io as sio


def mean_distance(a, b):
    """Mean distance over nodes present in both skeletons (inf if none)."""
    d = np.linalg.norm(a - b, axis=1)
    return float(np.nanmean(d)) if np.isfinite(d).any() else np.inf


def merge_frame(skeletons, match_px=10.0):
    """skeletons: list of (source, points (n_nodes, 2), point_scores (n_nodes,)).

    Returns a list of (points, point_scores, sources) with one entry per merged larva.
    """
    order = sorted(range(len(skeletons)), key=lambda i: -np.nanmean(skeletons[i][2]))
    groups = []                                    # each: {'members': [index], 'sources': set, 'points': fused}
    for i in order:
        source, points, _ = skeletons[i]
        best, best_d = None, np.inf
        for g in groups:
            if source in g['sources']:
                continue
            d = mean_distance(points, g['points'])
            if d <= match_px and d < best_d:
                best, best_d = g, d
        if best is None:
            groups.append({'members': [i], 'sources': {source}, 'points': points.copy()})
        else:
            best['members'].append(i)
            best['sources'].add(source)
            best['points'] = fuse([skeletons[j] for j in best['members']])[0]
    out = []
    for g in groups:
        points, scores = fuse([skeletons[j] for j in g['members']])
        out.append((points, scores, sorted(g['sources'])))
    return out


def fuse(members):
    """Score-weighted mean position per node; score = max over members for that node."""
    points = np.stack([m[1] for m in members])          # (k, n_nodes, 2)
    scores = np.stack([m[2] for m in members]).astype(float)
    weights = np.where(np.isfinite(points).all(axis=2), np.nan_to_num(scores, nan=0.0), 0.0)
    total = weights.sum(axis=0)
    fused = np.where(total[:, None] > 0,
                     np.nansum(np.nan_to_num(points) * weights[..., None], axis=0) / np.maximum(total, 1e-12)[:, None],
                     np.nan)
    with warnings.catch_warnings():            # nodes missing from every member are expected
        warnings.simplefilter('ignore', RuntimeWarning)
        fused_scores = np.where(total > 0, np.nanmax(np.where(weights > 0, scores, np.nan), axis=0), np.nan)
    return fused, fused_scores


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input', nargs=2, action='append', required=True, metavar=('LABEL', 'SLP'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--match-px', type=float, default=10.0)
    parser.add_argument('--min-support', type=int, default=1)
    args = parser.parse_args()

    inputs = [(label, sio.load_slp(path, open_videos=False)) for label, path in args.input]
    skeleton = inputs[0][1].skeletons[0]
    if any(labels.skeletons[0].node_names != skeleton.node_names for _, labels in inputs):
        raise ValueError('All inputs must share a skeleton')
    video = inputs[0][1].videos[0]
    if any(len(labels.videos) != 1 or labels.videos[0].shape != video.shape for _, labels in inputs):
        raise ValueError('All inputs must be predictions on the same single video')

    by_frame = {}
    for label, labels in inputs:
        for frame in labels:
            for inst in frame.instances:
                by_frame.setdefault(int(frame.frame_idx), []).append((label, inst.numpy(), np.asarray(inst.points['score'], float)))
    frames, sidecar = [], []
    for frame_idx in sorted(by_frame):
        merged = [m for m in merge_frame(by_frame[frame_idx], args.match_px) if len(m[2]) >= args.min_support]
        instances = []
        for points, scores, sources in merged:
            instances.append(sio.PredictedInstance.from_numpy(points, skeleton, point_scores=np.nan_to_num(scores),
                                                              score=float(np.nanmean(scores))))
            sidecar.append({'frame_idx': frame_idx, 'instance': len(instances) - 1, 'support': len(sources),
                            'sources': '+'.join(sources)})
        frames.append(sio.LabeledFrame(video, frame_idx, instances))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sio.Labels(labeled_frames=frames, videos=[video], skeletons=[skeleton]).save(str(args.output))
    with args.output.with_suffix('.support.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['frame_idx', 'instance', 'support', 'sources'])
        writer.writeheader(); writer.writerows(sidecar)
    counts = np.bincount([r['support'] for r in sidecar], minlength=len(inputs) + 1)[1:]
    print(f'Wrote {args.output}: {len(sidecar)} skeletons over {len(frames)} frames; support ' +
          ', '.join(f'{k + 1}: {c}' for k, c in enumerate(counts)))


if __name__ == '__main__':
    main()
