#!/usr/bin/env python3
"""Evaluate centroid predictions against a fixed, repeated-image benchmark."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import cv2
import h5py
import numpy as np
import sleap_io as sio
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

ROOT = Path(__file__).resolve().parents[2]
MOUNT = Path('/Volumes/lab-windingm/home/shared/sleap/model-tests')
# Explicit historical model mapping; do not infer a target from arbitrary filenames.
HISTORICAL_MEAN_MODELS = {
    'centroid_baseline', 'centroid_fullres_sigma5',
    'centroid_halfres_sigma2p5', 'centroid_fullres_sigma2p5',
}
HISTORICAL_BODY_MODELS = {
    'centroid_halfres_body', 'centroid_fullres_body', 'centroid_fullres_body_filters32',
    'centroid_fullres_body_sigma2', 'centroid_fullres_body_sigma3p5',
    'centroid_test_with_synthetic', 'centroid_body_synth_hard_crossings_v1',
    'centroid_body_synth_fullframe_darkoverlap_v1',
}


def target_for_model(model, protocol):
    if protocol != 'historical':
        return protocol
    if model in HISTORICAL_MEAN_MODELS:
        return 'mean'
    if model in HISTORICAL_BODY_MODELS:
        return 'body'
    raise ValueError(f'No historical target mapping for {model}; use --target body or mean')


def match_points(gt, predictions, tolerance=15.0):
    """Maximum number of valid one-to-one matches, then minimum total distance.

    Dummy columns allow unmatched GT. Their cost exceeds the sum of all valid
    distances, so an invalid assignment can never displace a valid match.
    """
    gt = np.asarray(gt).reshape(-1, 2)
    predictions = np.asarray(predictions).reshape(-1, 2)
    if not len(gt) or not len(predictions):
        return []
    distances = cdist(gt, predictions)
    penalty = (min(len(gt), len(predictions)) + 1) * (tolerance + 1)
    cost = np.full((len(gt), len(predictions) + len(gt)), penalty)
    cost[:, :len(predictions)] = np.where(distances <= tolerance, distances, penalty * 2)
    rows, cols = linear_sum_assignment(cost)
    return [(int(i), int(j), float(distances[i, j])) for i, j in zip(rows, cols)
            if j < len(predictions) and distances[i, j] <= tolerance]


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def decode_source(handle, video_id, frame_idx):
    group = handle[f'video{video_id}']
    matches = np.flatnonzero(group['frame_numbers'][:] == frame_idx)
    if len(matches) != 1:
        raise ValueError(f'Cannot uniquely locate embedded video {video_id}, frame {frame_idx}')
    image = cv2.imdecode(np.asarray(group['video'][int(matches[0])], dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError('Cannot decode embedded source image')
    return image


def load_benchmark(manifest, ground_truth, excluded, target_node='body'):
    rows = sorted(csv.DictReader(Path(manifest).open()), key=lambda r: int(r['order']))
    if [int(r['order']) for r in rows] != list(range(1, len(rows) + 1)):
        raise ValueError('Manifest orders must be consecutive, starting at 1')
    if len({(r['video_id'], r['frame_idx']) for r in rows}) != len(rows):
        raise ValueError('Manifest contains duplicate source images')
    unknown_exclusions = excluded - {int(r['order']) for r in rows}
    if unknown_exclusions:
        raise ValueError(f'Unknown excluded image orders: {unknown_exclusions}')
    labels = sio.load_slp(str(ground_truth), open_videos=False)
    lookup = {(labels.videos.index(f.video), int(f.frame_idx)): f for f in labels}
    for row in rows:
        for key in ('order', 'video_id', 'frame_idx', 'instances'):
            row[key] = int(row[key])
        frame = lookup[row['video_id'], row['frame_idx']]
        points, means = [], []
        for instance in frame.instances:
            if isinstance(instance, sio.PredictedInstance):
                continue
            # Fixed anatomical coordinate (the model's anchor part), never a centroid fallback.
            node = instance.skeleton.node_names.index(target_node)
            point = instance.points['xy'][node]
            if not np.isfinite(point).all():
                raise ValueError(f'Missing {target_node} coordinate in benchmark image {row["order"]}')
            points.append(point)
            means.append(np.nanmean(instance.numpy(), axis=0))
        row['gt_mean'] = np.asarray(means).reshape(-1, 2)
        row['gt'] = np.asarray(points).reshape(-1, 2)
        if len(points) != row['instances']:
            raise ValueError(f'GT count differs from manifest for image {row["order"]}')
        row['included'] = row['order'] not in excluded
        row['subset'] = 'hard' if row['category'] == 'hard_early' else 'other'
    if not any(r['included'] for r in rows):
        raise ValueError('No benchmark images remain')
    return rows


def verify_video(video_path, gt_path, rows, repeats):
    """Verify every encoded frame against all manifest images, not just row order."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f'Cannot open comparison video: {video_path}')
    expected = len(rows) * repeats
    if int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) != expected:
        raise ValueError(f'Expected {expected} video frames')
    size = (180, 110)
    with h5py.File(gt_path) as handle:
        sources = [decode_source(handle, r['video_id'], r['frame_idx']) for r in rows]
    shapes = [im.shape[:2] for im in sources]
    thumbnails = np.array([cv2.resize(im, size).astype(np.float32) for im in sources])
    audit = []
    for frame_idx in range(expected):
        ok, image = cap.read()
        if not ok:
            raise ValueError(f'Cannot decode comparison frame {frame_idx}')
        expected_index = frame_idx // repeats
        if image.shape[:2] != shapes[expected_index]:
            raise ValueError('Comparison/source dimensions differ: coordinate scaling must be resolved')
        distances = np.mean(np.abs(thumbnails - cv2.resize(image, size).astype(np.float32)), axis=(1, 2, 3))
        nearest = int(np.argmin(distances))
        if nearest != expected_index or distances[nearest] > 12:
            raise ValueError(f'Comparison frame {frame_idx} does not match manifest image {expected_index + 1}')
        audit.append({'prediction_frame': frame_idx, 'order': expected_index + 1,
                      'source_video_id': rows[expected_index]['video_id'],
                      'source_frame': rows[expected_index]['frame_idx'],
                      'thumbnail_mae': float(distances[nearest])})
    cap.release()
    return audit


def load_predictions(path, expected):
    labels = sio.load_slp(str(path), open_videos=False)
    if len(labels.videos) != 1:
        raise ValueError(f'{path}: expected one comparison video')
    if labels.videos[0].shape[0] != expected:
        raise ValueError(f'{path}: wrong comparison video length')
    lookup = {}
    for frame in labels:
        index = int(frame.frame_idx)
        if index < 0 or index >= expected or index in lookup:
            raise ValueError(f'{path}: invalid/duplicate frame index {index}')
        points, scores = [], []
        for instance in frame.instances:
            if not isinstance(instance, sio.PredictedInstance):
                raise ValueError(f'{path}: contains user labels instead of predictions')
            if instance.skeleton.node_names != ['centroid']:
                raise ValueError(f'{path}: requires centroid-only prediction instances')
            point = instance.points['xy'][0]
            score = float(instance.points['score'][0])
            if not np.isfinite(point).all() or not np.isfinite(score):
                raise ValueError(f'{path}: invalid predicted coordinate/score')
            points.append(point)
            scores.append(score)
        lookup[index] = (np.asarray(points).reshape(-1, 2), np.asarray(scores))
    # SLP may omit frames with zero detections; retain these as all false negatives.
    missing = sorted(set(range(expected)) - lookup.keys())
    for index in missing:
        lookup[index] = (np.empty((0, 2)), np.empty(0))
    return lookup, missing


def metrics(rows, repeats=1):
    tp, fp, fn = (sum(r[key] for r in rows) for key in ('tp', 'fp', 'fn'))
    return {'tp_mean': tp / repeats, 'fp_mean': fp / repeats, 'fn_mean': fn / repeats,
            'recall_pct': 100 * tp / (tp + fn) if tp + fn else None,
            'precision_pct': 100 * tp / (tp + fp) if tp + fp else None,
            'f1_pct': 200 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
            'mean_error_px': sum(r['distance_sum'] for r in rows) / tp if tp else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ground-truth', type=Path, default=ROOT/'data/combined_ground_truth_cleaned_visibility-fixed.pkg.slp')
    parser.add_argument('--manifest', type=Path, default=Path(__file__).with_name('comparison_manifest.csv'))
    parser.add_argument('--predictions-dir', type=Path, default=MOUNT/'predictions')
    parser.add_argument('--comparison-video', type=Path, default=MOUNT/'comparison_frames_plain.mp4')
    parser.add_argument('--models', nargs='+', help='Prediction filenames without .slp; defaults to all .slp files')
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/model_assessment')
    parser.add_argument('--target', choices=['body', 'mean', 'historical'], default='body',
                        help='Fixed anatomical target, or explicit per-model legacy targets to reproduce old tables')
    parser.add_argument('--target-node', default='body',
                        help="Skeleton node used as the fixed target (the model's anchor part); default body")
    parser.add_argument('--threshold', type=float, default=0.2)
    parser.add_argument('--thresholds', type=float, nargs='+', default=[0.2, 0.3, 0.4, 0.5])
    parser.add_argument('--tolerance', type=float, default=15.0)
    parser.add_argument('--repeats', type=int, default=10)
    parser.add_argument('--exclude-orders', type=int, nargs='*', default=[])
    args = parser.parse_args()
    if args.repeats < 1 or args.tolerance <= 0:
        parser.error('Repeats and tolerance must be positive')
    thresholds = sorted(set([args.threshold, *args.thresholds]))
    if any(not 0 <= t <= 1 for t in thresholds):
        parser.error('Thresholds must be between 0 and 1')
    paths = ([args.predictions_dir/f'{name}.slp' for name in args.models] if args.models
             else sorted(args.predictions_dir.glob('*.slp')))
    if not paths or any(not p.is_file() for p in paths):
        parser.error('No predictions found or a requested model file is missing')
    args.output.mkdir(parents=True, exist_ok=True)
    benchmark = load_benchmark(args.manifest, args.ground_truth, set(args.exclude_orders), args.target_node)
    mapping = verify_video(args.comparison_video, args.ground_truth, benchmark, args.repeats)
    write_csv(args.output/'frame_mapping.csv', mapping)
    print(f'Verified {len(mapping)} comparison frames against {len(benchmark)} source images.', flush=True)
    summary, sweep, per_frame, per_image, details = [], [], [], [], []
    provenance = []
    for path in paths:
        target = target_for_model(path.stem, args.target)
        model_benchmark = [{**r, 'gt': r['gt_mean'] if target == 'mean' else r['gt']} for r in benchmark]
        predictions, missing = load_predictions(path, len(benchmark)*args.repeats)
        provenance.append({'model': path.stem, 'target': target, 'path': str(path.resolve()),
                           'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                           'omitted_prediction_frames_treated_as_empty': missing})
        for threshold in thresholds:
            frame_rows = []
            for row in model_benchmark:
                if not row['included']:
                    continue
                for repeat in range(args.repeats):
                    index = (row['order'] - 1) * args.repeats + repeat
                    points, scores = predictions[index]
                    selected = scores >= threshold
                    points, scores = points[selected], scores[selected]
                    matches = match_points(row['gt'], points, args.tolerance)
                    result = {'model': path.stem, 'target': target, 'order': row['order'], 'video_id': row['video_id'],
                              'source_frame': row['frame_idx'], 'subset': row['subset'], 'repeat': repeat,
                              'prediction_frame': index, 'threshold': threshold,
                              'gt_count': len(row['gt']), 'prediction_count': len(points),
                              'tp': len(matches), 'fp': len(points)-len(matches), 'fn': len(row['gt'])-len(matches),
                              'distance_sum': sum(m[2] for m in matches)}
                    frame_rows.append(result)
                    if threshold == args.threshold:
                        matched = {i: (j, distance) for i,j,distance in matches}
                        used = {j for _,j,_ in matches}
                        for i, xy in enumerate(row['gt']):
                            j, distance = matched.get(i, (None, None))
                            details.append({'model':path.stem, 'order':row['order'], 'repeat':repeat,
                                            'prediction_frame':index, 'status':'TP' if j is not None else 'FN',
                                            'gt_index':i, 'gt_x':float(xy[0]), 'gt_y':float(xy[1]),
                                            'pred_x':float(points[j,0]) if j is not None else None,
                                            'pred_y':float(points[j,1]) if j is not None else None,
                                            'score':float(scores[j]) if j is not None else None, 'distance_px':distance})
                        for j in set(range(len(points)))-used:
                            details.append({'model':path.stem, 'order':row['order'], 'repeat':repeat,
                                            'prediction_frame':index, 'status':'FP', 'gt_index':None,
                                            'gt_x':None, 'gt_y':None, 'pred_x':float(points[j,0]),
                                            'pred_y':float(points[j,1]), 'score':float(scores[j]), 'distance_px':None})
            for subset in ('all','hard','other'):
                selected_rows = [r for r in frame_rows if subset=='all' or r['subset']==subset]
                if not selected_rows:
                    continue
                result = {'model':path.stem, 'target': target, 'threshold':threshold, 'subset':subset,
                          'unique_images':len({r['order'] for r in selected_rows}),
                          'unique_gt':sum(r['gt_count'] for r in selected_rows)/args.repeats,
                          **metrics(selected_rows,args.repeats)}
                sweep.append(result)
                if threshold == args.threshold:
                    summary.append(result)
            if threshold == args.threshold:
                per_frame.extend(frame_rows)
                for row in model_benchmark:
                    selected_rows = [r for r in frame_rows if r['order']==row['order']]
                    if selected_rows:
                        per_image.append({'model':path.stem, 'target':target, 'order':row['order'], 'video_id':row['video_id'],
                                          'source_frame':row['frame_idx'], 'subset':row['subset'],
                                          'gt_count':len(row['gt']), **metrics(selected_rows,args.repeats),
                                          'tp_min':min(r['tp'] for r in selected_rows),
                                          'tp_max':max(r['tp'] for r in selected_rows)})
        overall = next(r for r in summary if r['model']==path.stem and r['subset']=='all')
        print(f"{path.stem}: recall {overall['recall_pct']:.2f}%, precision {(overall['precision_pct'] if overall['precision_pct'] is not None else float('nan')):.2f}%",flush=True)
    for name, rows in [('summary',summary),('threshold_sweep',sweep),('per_frame',per_frame),
                       ('per_image',per_image),('matches',details)]:
        write_csv(args.output/f'{name}.csv',rows)
    metadata = {'target_protocol':args.target, 'target_node':args.target_node, 'ground_truth':str(args.ground_truth.resolve()), 'manifest':str(args.manifest.resolve()),
                'ground_truth_sha256':hashlib.sha256(args.ground_truth.read_bytes()).hexdigest(),
                'manifest_sha256':hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
                'comparison_video':str(args.comparison_video.resolve()),
                'threshold':args.threshold,'thresholds':thresholds,'tolerance_px':args.tolerance,
                'repeats':args.repeats, 'excluded_orders':args.exclude_orders,
                'matching':'maximum cardinality within tolerance, then minimum total distance',
                'repeat_policy':'sum counts over repeats, divide counts by repeats; ratios use pooled counts',
                'hard_subset':'manifest category hard_early; historical frame-level selection, not per-animal classification',
                'leakage_status':'not audited; benchmark order 11 (video 10 frame 41972) was a suggested synthetic source',
                'threshold_limit':'post-filtering cannot recover predictions suppressed during inference',
                'models':provenance}
    (args.output/'assessment.json').write_text(json.dumps(metadata,indent=2))
    text = ['# Centroid assessment', '', f"{len({r['order'] for r in per_image})} unique images; {args.repeats} encoded repeats per image.",
            f'Target protocol: {args.target}. Matching tolerance: {args.tolerance:g} px. Score threshold: {args.threshold:g}.', '',
            '| Model | Recall (%) | Precision (%) | Hard-frame recall (%) | Other recall (%) | Mean error (px) |',
            '|---|---:|---:|---:|---:|---:|']
    def display(value):
        return 'n/a' if value is None else f'{value:.2f}'
    for path in paths:
        groups={r['subset']:r for r in summary if r['model']==path.stem}
        all_row=groups['all']
        text.append('| '+path.stem+' | '+' | '.join(display(v) for v in [all_row['recall_pct'],all_row['precision_pct'],
                    groups.get('hard',{}).get('recall_pct'),groups.get('other',{}).get('recall_pct'),all_row['mean_error_px']])+' |')
    text.extend(['','Repeated images are not independent samples. Counts are averaged across repeats; localisation error uses matched detections only.',
                 'The hard subset is the historical four-frame grouping, not an objective per-animal pile classification.',
                 'Historical mode uses mean-of-visible-keypoints targets for the four no-body-anchor models; body targets for the others. It reproduces the previous mixed-target comparison, not a common-target benchmark.',
                 'Training/test overlap has not been audited. Use --exclude-orders 11 for a sensitivity analysis of frame 41972.',
                 'Threshold sweeps only filter saved detections; lowering the inference threshold requires new predictions.'])
    (args.output/'report.md').write_text('\n'.join(text)+'\n')
    print(f'Wrote assessment to {args.output}')


if __name__=='__main__':
    main()
