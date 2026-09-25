#!/usr/bin/env python3
"""Tune the multi-anchor merge on held-back frames, then report the same settings on the test frames.

Tuning frames: labelled frames that none of the pipeline models (the seed-42 centroid and
centered-instance models) trained on; predictions made on them with `sleap predict -i
tuning_frames.pkg.slp`. Test frames: the 14 benchmark images (x10 repeats) in the test
video. Every merge configuration is scored on both; the configuration is chosen on the
tuning frames only (highest F1, then crowded recall), so its test score is unbiased by the
choice. The full grid on the test frames is also written, to show how sensitive the result
is to the settings.
"""
from __future__ import annotations

import argparse
import csv
import itertools
from pathlib import Path

import numpy as np
import sleap_io as sio

from assess_centroids import ROOT, load_benchmark, match_distances, write_csv
from assess_poses import mean_node_distance
from assess_poses_labels import frame_key, load_truth
from merge_poses import merge_frame

A = ROOT / 'outputs/model_assessment'


def score(truth, preds, tolerance=15.0):
    """truth: key -> (gt (n,5,2), crowded (n,)); preds: key -> array (m,5,2)."""
    tp = {'all': 0, 'crowded': 0}
    n = {'all': 0, 'crowded': 0}
    n_pred, errors = 0, []
    for key, (gt, crowded) in truth.items():
        pred = preds.get(key)
        if pred is None or not len(pred):
            pred = np.empty((0,) + gt.shape[1:])
        n_pred += len(pred)
        matches = match_distances(mean_node_distance(gt, pred), tolerance) if len(pred) else []
        matched = {i for i, _, _ in matches}
        n['all'] += len(gt); tp['all'] += len(matched)
        n['crowded'] += int(crowded.sum()); tp['crowded'] += sum(bool(crowded[i]) for i in matched)
        errors += [np.linalg.norm(gt[i] - pred[j], axis=1) for i, j, _ in matches]
    e = np.array(errors)
    recall = 100 * tp['all'] / n['all']
    precision = 100 * tp['all'] / n_pred if n_pred else 0.0
    return {'recall': recall, 'precision': precision,
            'f1': 2 * recall * precision / (recall + precision) if recall + precision else 0.0,
            'crowded_recall': 100 * tp['crowded'] / n['crowded'],
            'pck5': 100 * float((np.nan_to_num(e, nan=np.inf) <= 5).mean()) if len(e) else 0.0}


def load_pipelines(folder, names, key_fn):
    out = {}
    for name in names:
        by_key = {}
        for frame in sio.load_slp(str(folder / f'{name}.slp'), open_videos=False):
            by_key[key_fn(frame)] = [(name, i.numpy(), np.asarray(i.points['score'], float)) for i in frame.instances]
        out[name] = by_key
    return out


def test_truth():
    meta_rows = load_benchmark(Path(__file__).with_name('comparison_manifest.csv'),
                               ROOT/'data/combined_ground_truth_cleaned_visibility-fixed.pkg.slp', set(), 'body')
    labels = sio.load_slp(str(ROOT/'data/combined_ground_truth_cleaned_visibility-fixed.pkg.slp'), open_videos=False)
    lookup = {(labels.videos.index(f.video), int(f.frame_idx)): f for f in labels}
    truth = {}
    for row in meta_rows:
        gt = np.stack([i.numpy() for i in lookup[row['video_id'], row['frame_idx']].instances])
        for repeat in range(10):
            truth[(row['order'] - 1) * 10 + repeat] = (gt, row['crowded'])
    return truth


def configurations():
    for front, tail, kind, px, rep in itertools.product(('head', 'mouthhooks'), (False, True),
                                                        ('topdown', 'redrawn', 'consistent'),
                                                        (10, 15, 20, 25), ('fuse', 'priority')):
        # Always keeping body skeletons (trust_body) did not help in the first grid, so it is not searched.
        support_rules = [(1, None, False)] + [(2, s, False) for s in (None, 0.5, 0.6, 0.7, 0.8)]
        for min_support, single, trust_body in support_rules:
            yield {'front': front, 'tail': tail, 'kind': kind, 'match_px': px, 'representative': rep,
                   'min_support': min_support, 'single_min_score': single, 'trust_body': trust_body}


def inputs_for(cfg):
    kind = cfg['kind']
    names = [f"{kind}_{cfg['front']}", 'topdown_body'] + ([f'{kind}_tail'] if cfg['tail'] else [])
    return names


def run(cfg, truth, pipelines):
    names = inputs_for(cfg)
    keys = set(truth)
    preds = {}
    for key in keys:
        skeletons = [s for name in names for s in pipelines[name].get(key, [])]
        merged = merge_frame(skeletons, cfg['match_px'], cfg['representative'], ['topdown_body'] + names,
                             cfg['min_support'], cfg['single_min_score'],
                             trusted=['topdown_body'] if cfg['trust_body'] else [])
        preds[key] = np.stack([m[0] for m in merged]) if merged else np.empty((0, 5, 2))
    return score(truth, preds)


def label(cfg):
    anchors = '+'.join([cfg['front'], 'body'] + (['tail'] if cfg['tail'] else []))
    support = 'keep all' if cfg['min_support'] == 1 else ('>=2 agree' if cfg['single_min_score'] is None
                                                          else f">=2 or single >= {cfg['single_min_score']}")
    if cfg.get('trust_body'):
        support += ' + all body'
    kind = {'topdown': 'standard', 'redrawn': 'two-step', 'consistent': 'consistent'}[cfg['kind']]
    return f"{anchors}, {kind}, {cfg['match_px']} px, {support}, {cfg['representative']}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=A / 'merge_tuning')
    args = parser.parse_args()
    names = ['topdown_body', 'topdown_head', 'topdown_mouthhooks', 'topdown_tail', 'redrawn_head', 'redrawn_mouthhooks',
             'redrawn_tail', 'consistent_head', 'consistent_mouthhooks', 'consistent_tail']
    val_truth = load_truth(A / 'validation/tuning_frames.pkg.slp')
    val_pipes = load_pipelines(A / 'validation/predictions', names, frame_key)
    test_truth_ = test_truth()
    test_pipes = load_pipelines(A / 'topdown/predictions', names, lambda f: int(f.frame_idx))

    rows = []
    for name in ('topdown_body', 'topdown_head', 'topdown_mouthhooks', 'topdown_tail'):
        v = score(val_truth, {k: np.stack([s[1] for s in sk]) for k, sk in val_pipes[name].items() if sk})
        t = score(test_truth_, {k: np.stack([s[1] for s in sk]) for k, sk in test_pipes[name].items() if sk})
        blank = dict.fromkeys(next(configurations()))
        rows.append({'config': f'{name} alone', 'single_pipeline': True, **blank, **{f'val_{k}': x for k, x in v.items()},
                     **{f'test_{k}': x for k, x in t.items()}})
    for cfg in configurations():
        v, t = run(cfg, val_truth, val_pipes), run(cfg, test_truth_, test_pipes)
        rows.append({'config': label(cfg), 'single_pipeline': False, **cfg,
                     **{f'val_{k}': x for k, x in v.items()}, **{f'test_{k}': x for k, x in t.items()}})
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / 'grid.csv', rows)

    merges = [r for r in rows if not r['single_pipeline']]
    chosen = max(merges, key=lambda r: (round(r['val_f1'], 1), r['val_crowded_recall']))
    best_test = max(merges, key=lambda r: r['test_f1'])
    body = rows[0]
    print(f"{'':62s} {'--- tuning frames ---':>24s}   {'------ test frames ------':>31s}")
    print(f"{'configuration':62s} {'F1':>5s} {'rec':>5s} {'prec':>5s} {'crowd':>6s}   {'F1':>5s} {'rec':>5s} {'prec':>5s} {'crowd':>6s} {'PCK5':>5s}")
    def show(r, tag=''):
        print(f"{(tag + r['config'])[:62]:62s} {r['val_f1']:5.1f} {r['val_recall']:5.1f} {r['val_precision']:5.1f} {r['val_crowded_recall']:6.1f}   "
              f"{r['test_f1']:5.1f} {r['test_recall']:5.1f} {r['test_precision']:5.1f} {r['test_crowded_recall']:6.1f} {r['test_pck5']:5.1f}")
    for r in rows[:4]:
        show(r)
    show(chosen, 'CHOSEN ON TUNING: ')
    show(best_test, 'best on test (optimistic): ')
    test_f1 = np.array([r['test_f1'] for r in merges])
    print(f"\ntest F1 across all {len(merges)} merge configurations: median {np.median(test_f1):.1f}, "
          f"top 10% >= {np.percentile(test_f1, 90):.1f}; body alone {body['test_f1']:.1f}")
    print(f'Wrote {args.output / "grid.csv"}')


if __name__ == '__main__':
    main()
