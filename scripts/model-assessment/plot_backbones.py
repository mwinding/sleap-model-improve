#!/usr/bin/env python3
"""Body pose model with different backbones (UNet, ConvNeXt pretrained / from scratch, Swin pretrained).

1. pose_known_centre: crops centred on each labelled larva's body point (assess_stage2.py output):
   wrong-larva rate and nodes within 5 px for crowded larvae, and complete skeletons.
2. pose_pipeline: the same pose models behind the UNet full-frame body detector (seed 42) on the
   test video, counting complete skeletons only: larvae missed (crowded / all others / total) and precision.
3. pose_examples: crowded larvae that the seed-42 models draw differently (some right, some wrong).
Seed 42 and seed 7 are shown side by side.
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


def missed(truth, preds, min_nodes=5):
    """Larvae missed (crowded, others, total; mean per repeat) and precision, counting skeletons with >= min_nodes."""
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
    out['precision'] = 100 * sum(tp.values()) / n_pred if n_pred else 0.0
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backbones', type=Path, default=B)
    parser.add_argument('--examples', type=int, default=8)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    out = args.backbones / 'plots'
    out.mkdir(parents=True, exist_ok=True)
    labels = [label for label, _ in BACKBONES]

    summary = {(r['model'], r['subset']): r for r in read_csv(args.backbones / 'stage2/stage2_summary.csv')}
    panels = [('wrong', 'Wrong larva, crowded', 'Larvae (%)'), ('pck5', 'Nodes within 5 px, crowded', 'Nodes (%)'),
              ('complete', 'Complete skeletons', 'Larvae (%)')]
    fields = {'wrong': ('wrong_animal_pct', 'crowded'), 'pck5': ('pck5_pct', 'crowded'), 'complete': ('complete_pct', 'all')}
    values = {k: [[float(summary[base + suffix, fields[k][1]][fields[k][0]]) for _, base in BACKBONES]
                  for suffix, _ in SEEDS] for k, _, _ in panels}
    paired_bars(out / 'pose_known_centre', labels, panels, values, {k: (100, True) for k, _, _ in panels})

    truth = test_truth()
    scores = [[missed(truth, load_preds(args.backbones / 'pipeline/predictions' / f'{base}{suffix}.slp'))
               for _, base in BACKBONES] for suffix, _ in SEEDS]
    panels = [('crowded', 'Crowded larvae missed', 'Larvae (of 155)'), ('others', 'Other larvae missed', 'Larvae (of 213)'),
              ('total', 'All larvae missed', 'Larvae (of 368)'), ('precision', 'Precision', 'Skeletons (%)')]
    values = {k: [[s[k] for s in seed] for seed in scores] for k, _, _ in panels}
    limits = {k: (max(max(v) for v in values[k]), False) for k in ('crowded', 'others', 'total')}
    limits['precision'] = (100, True)
    paired_bars(out / 'pose_pipeline', labels, panels, values, limits)

    stage2 = args.backbones / 'stage2'
    meta = json.loads((stage2 / 'stage2_assessment.json').read_text())
    models = [base for _, base in BACKBONES]
    prism_style()
    examples_plot(read_csv(stage2 / 'stage2_per_animal.csv'), models, Path(meta['ground_truth']), stage2 / 'predictions',
                  args.examples, args.seed, out / 'pose_examples', labels=dict(zip(models, labels)),
                  anchors={m: 'body' for m in models}, disagree=True)
    print(f'Wrote {out / "pose_examples.png"}')


if __name__ == '__main__':
    main()
