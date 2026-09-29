#!/usr/bin/env python3
"""Backbone comparison (UNet, ConvNeXt pretrained / from scratch, Swin pretrained), seed 42 and seed 7 side by side.

Pose models (--group pose):
1. pose_known_centre: crops centred on each labelled larva's body point (assess_stage2.py output):
   wrong-larva rate and nodes within 5 px for crowded larvae, and complete skeletons.
2. pose_pipeline: the same pose models behind the UNet full-frame body detector (seed 42) on the
   test video, counting complete skeletons only: recall for crowded / other / all larvae, and precision.
3. pose_examples: crowded larvae that the seed-42 models draw differently (some right, some wrong).

Body detectors, all trained with the full-frame synthetic data (--group detector):
4. detector_detection: recall of body points (assess_centroids.py output in the backbones folder).
5. detector_pipeline: each detector followed by the UNet body pose model (seed 42), recall of complete skeletons.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np

from assess_centroids import ROOT, match_distances
from assess_poses import mean_node_distance
from plot_anchors import SEEDS, paired_bars
from plot_assessment import prism_style, read_csv
from plot_pipelines import load_preds
from plot_stage2 import examples_plot
from tune_merge import test_truth

B = ROOT / 'outputs/model_assessment/backbones'
BACKBONES = [('UNet', 'centered_instance_body_holdout'),
             ('ConvNeXt pretrained', 'centered_instance_body_convnext_pretrained_holdout'),
             ('ConvNeXt from scratch', 'centered_instance_body_convnext_scratch_holdout'),
             ('Swin pretrained', 'centered_instance_body_swint_pretrained_holdout')]
DETECTOR = 'centroid_body_synth_fullframe_darkoverlap_v1'
DETECTORS = [('UNet', f'{DETECTOR}_holdout'),
             ('ConvNeXt pretrained', f'{DETECTOR}_convnext_pretrained_holdout'),
             ('ConvNeXt from scratch', f'{DETECTOR}_convnext_scratch_holdout'),
             ('Swin pretrained', f'{DETECTOR}_swint_pretrained_holdout')]
RECALL_PANELS = [('crowded_recall', 'Crowded recall', 'Recall (%)'), ('others_recall', 'Other larvae recall', 'Recall (%)'),
                 ('total_recall', 'Overall recall', 'Recall (%)'), ('precision', 'Precision', 'Precision (%)')]


def missed(truth, preds, min_nodes=5):
    """Larvae missed (crowded, others, total; mean per repeat), recall (%) per subset and precision,
    counting skeletons with >= min_nodes."""
    tp = {'crowded': 0, 'others': 0}
    n = {'crowded': 0, 'others': 0}
    n_pred = 0
    for key, (gt, crowded) in truth.items():
        p = preds.get(key, np.empty((0,) + gt.shape[1:]))
        p = p[np.isfinite(p).all(axis=2).sum(axis=1) >= min_nodes]
        n_pred += len(p)
        found = {i for i, _, _ in match_distances(mean_node_distance(gt, p), 15)} if len(p) else set()
        for i in range(len(gt)):
            s = 'crowded' if crowded[i] else 'others'
            n[s] += 1
            tp[s] += i in found
    repeats = 10
    out = {s: (n[s] - tp[s]) / repeats for s in n}
    out['total'] = out['crowded'] + out['others']
    for s in n:
        out[f'{s}_recall'] = 100 * tp[s] / n[s]
    out['total_recall'] = 100 * sum(tp.values()) / sum(n.values())
    out['precision'] = 100 * sum(tp.values()) / n_pred if n_pred else 0.0
    return out


def recall_bars(output, labels, scores):
    """scores[seed][model] -> dict with *_recall and precision keys (percent)."""
    values = {k: [[s[k] for s in seed] for seed in scores] for k, _, _ in RECALL_PANELS}
    paired_bars(output, labels, RECALL_PANELS, values, {k: (100, True) for k, _, _ in RECALL_PANELS})


def pose_plots(folder, out, truth, n_examples, seed):
    labels = [label for label, _ in BACKBONES]
    summary = {(r['model'], r['subset']): r for r in read_csv(folder / 'stage2/stage2_summary.csv')}
    panels = [('wrong', 'Wrong larva, crowded', 'Larvae (%)'), ('pck5', 'Nodes within 5 px, crowded', 'Nodes (%)'),
              ('complete', 'Complete skeletons', 'Larvae (%)')]
    fields = {'wrong': ('wrong_animal_pct', 'crowded'), 'pck5': ('pck5_pct', 'crowded'), 'complete': ('complete_pct', 'all')}
    values = {k: [[float(summary[base + suffix, fields[k][1]][fields[k][0]]) for _, base in BACKBONES]
                  for suffix, _ in SEEDS] for k, _, _ in panels}
    paired_bars(out / 'pose_known_centre', labels, panels, values, {k: (100, True) for k, _, _ in panels})

    recall_bars(out / 'pose_pipeline', labels,
                [[missed(truth, load_preds(folder / 'pipeline/predictions' / f'{base}{suffix}.slp')) for _, base in BACKBONES]
                 for suffix, _ in SEEDS])

    stage2 = folder / 'stage2'
    meta = json.loads((stage2 / 'stage2_assessment.json').read_text())
    models = [base for _, base in BACKBONES]
    prism_style()
    examples_plot(read_csv(stage2 / 'stage2_per_animal.csv'), models, Path(meta['ground_truth']), stage2 / 'predictions',
                  n_examples, seed, out / 'pose_examples', labels=dict(zip(models, labels)),
                  anchors={m: 'body' for m in models}, disagree=True)
    print(f'Wrote {out / "pose_examples.png"}')


def detector_plots(folder, out, truth):
    labels = [label for label, _ in DETECTORS]
    summary = {(r['model'], r['subset']): r for r in read_csv(folder / 'summary.csv') if r['threshold'] == '0.2'}
    subsets = {'crowded_recall': 'crowded', 'others_recall': 'isolated', 'total_recall': 'all'}
    scores = []
    for suffix, _ in SEEDS:
        seed = []
        for _, base in DETECTORS:
            m = base + suffix
            s = {k: float(summary[m, subset]['recall_pct']) for k, subset in subsets.items()}
            s['precision'] = float(summary[m, 'all']['precision_pct'])
            seed.append(s)
        scores.append(seed)
    recall_bars(out / 'detector_detection', labels, scores)
    recall_bars(out / 'detector_pipeline', labels,
                [[missed(truth, load_preds(folder / 'pipeline/predictions' / f'detector__{base}{suffix}.slp'))
                  for _, base in DETECTORS] for suffix, _ in SEEDS])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backbones', type=Path, default=B)
    parser.add_argument('--group', choices=['pose', 'detector', 'all'], default='all')
    parser.add_argument('--examples', type=int, default=8)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    out = args.backbones / 'plots'
    out.mkdir(parents=True, exist_ok=True)
    truth = test_truth()
    if args.group in ('pose', 'all'):
        pose_plots(args.backbones, out, truth, args.examples, args.seed)
    if args.group in ('detector', 'all'):
        detector_plots(args.backbones, out, truth)


if __name__ == '__main__':
    main()
