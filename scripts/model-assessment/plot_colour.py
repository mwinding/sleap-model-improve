#!/usr/bin/env python3
"""Benchmark scores on the true-colour vs the red/blue-swapped benchmark video.

comparison_frames_plain.mp4, on which every earlier benchmark score was measured, has red and blue swapped, and so do
the synthetic training sets made before 2026-10-14; the real training frames and real videos are in true colour.
make_benchmark_video.py rebuilt the video both ways with the same encoder, and predict_benchmark.sh re-ran the
predictions on each (outputs/model_assessment/colour/{true,swapped}colour). Score the detectors first:

  python scripts/model-assessment/assess_centroids.py \
      --predictions-dir outputs/model_assessment/colour/truecolour/centroids \
      --comparison-video /Volumes/lab-windingm/home/shared/sleap/model-tests/comparison_frames_truecolour.mp4 \
      --output outputs/model_assessment/colour/truecolour/assessment
  (and the same for swappedcolour)

Detectors are scored on body points (assess_centroids.py), pipelines on complete skeletons (plot_backbones.missed).
Writes colour_summary.csv (larvae missed and recall for crowded / other / all larvae, and precision, per model, seed
and video) and plots: detector_colour and pipeline_colour (seed means, swapped vs true colour), detector_truecolour
and pipeline_truecolour (seed 42 vs seed 7 on the true-colour video, including the models retrained on the corrected
synthetic data once their predictions exist).
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np

from assess_centroids import ROOT, write_csv
from plot_anchors import SEEDS, paired_bars
from plot_assessment import read_csv
from plot_backbones import RECALL_PANELS, missed
from plot_pipelines import load_preds
from tune_merge import test_truth

C = ROOT / 'outputs/model_assessment/colour'
VIDEOS = [('swappedcolour', 'Swapped (old benchmark)'), ('truecolour', 'True colour')]
SYN = 'centroid_body_synth_fullframe_darkoverlap_v1'
FIX = 'centroid_body_synth_fullframe_truecolour'
POSE = 'centered_instance_body_holdout'
# (label, model); the seed suffix is appended to the model name
DETECTORS = [('UNet, real only', 'centroid_fullres_body_holdout'),
             ('UNet + synthetic', f'{SYN}_holdout'),
             ('ConvNeXt pretrained + synthetic', f'{SYN}_convnext_pretrained_holdout'),
             ('ConvNeXt from scratch + synthetic', f'{SYN}_convnext_scratch_holdout'),
             ('Swin pretrained + synthetic', f'{SYN}_swint_pretrained_holdout')]
RETRAINED = [('ConvNeXt pretrained, real only', 'centroid_fullres_body_convnext_pretrained_holdout'),
             ('UNet + corrected synthetic', f'{FIX}_holdout'),
             ('UNet colour input + corrected synthetic', f'{FIX}_unet_rgb_holdout'),
             ('ConvNeXt pretrained + corrected synthetic', f'{FIX}_convnext_pretrained_holdout')]
# (label, path relative to the video folder with {s} for the seed suffix and {n} for 42 / 7)
PIPELINES = [('UNet real only, UNet pose', 'merge_seed{n}/topdown_body.slp'),
             ('Two-step merge', 'merge_seed{n}/final_twostep_mhbt.slp')] + [
    (f'{label.removesuffix(" + synthetic")} + synth, UNet pose', f'pipeline/{model}{{s}}__{POSE}.slp')
    for label, model in DETECTORS[1:]] + [
    ('ConvNeXt pretrained + synth, ConvNeXt pose',
     f'pipeline/{SYN}_convnext_pretrained_holdout{{s}}__centered_instance_body_convnext_pretrained_holdout.slp')] + [
    # Pose backbones behind the seed-42 UNet + synthetic detector; here the seed is the pose model's
    (f'UNet + synth, {label} pose', f'pipeline/{SYN}_holdout__centered_instance_body_{name}_holdout{{s}}.slp')
    for label, name in (('ConvNeXt pretrained', 'convnext_pretrained'), ('ConvNeXt from scratch', 'convnext_scratch'),
                        ('Swin pretrained', 'swint_pretrained'))]
RETRAINED_PIPELINES = [(f'{label}, UNet pose', f'pipeline/{model}{{s}}__{POSE}.slp') for label, model in RETRAINED] + [
    ('ConvNeXt pretrained + corrected synth, ConvNeXt pose',
     f'pipeline/{FIX}_convnext_pretrained_holdout{{s}}__centered_instance_body_convnext_pretrained_holdout.slp')]


def detector_scores(video, model):
    summary = {(r['model'], r['subset']): r for r in read_csv(C / video / 'assessment/summary.csv')
               if r['threshold'] == '0.2'}
    if (model, 'all') not in summary:
        return None
    out = {}
    for key, subset in (('crowded', 'crowded'), ('others', 'isolated'), ('total', 'all')):
        out[key] = float(summary[model, subset]['fn_mean'])
        out[f'{key}_recall'] = float(summary[model, subset]['recall_pct'])
    out['precision'] = float(summary[model, 'all']['precision_pct'])
    return out


def pipeline_scores(truth, video, path):
    file = C / video / path
    return missed(truth, load_preds(file)) if file.exists() else None


def collect(truth, video, rows, kind):
    """[(label, [scores per seed])] for the rows whose predictions exist for both seeds."""
    out = []
    for label, item in rows:
        seeds = []
        for suffix, n in zip((s for s, _ in SEEDS), (42, 7)):
            seeds.append(detector_scores(video, item + suffix) if kind == 'detector'
                         else pipeline_scores(truth, video, item.format(s=suffix, n=n)))
        if all(seeds):
            out.append((label, seeds))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output', type=Path, default=C / 'plots')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    truth = test_truth()
    table = []
    results = {}
    for kind, rows, retrained in (('detector', DETECTORS, RETRAINED), ('pipeline', PIPELINES, RETRAINED_PIPELINES)):
        for video, _ in VIDEOS:
            results[kind, video] = collect(truth, video, rows + (retrained if video == 'truecolour' else []), kind)
            for label, seeds in results[kind, video]:
                for (_, seed_label), s in zip(SEEDS, seeds):
                    table.append({'kind': kind, 'video': video, 'model': label, 'seed': seed_label,
                                  **{k: round(v, 2) for k, v in s.items()}})
        # Swapped vs true colour, mean of the two seeds, for the models scored on both videos
        both = [label for label, _ in results[kind, 'swappedcolour']]
        means = {video: {label: {k: np.mean([s[k] for s in seeds]) for k in seeds[0]}
                         for label, seeds in results[kind, video]} for video, _ in VIDEOS}
        values = {k: [[means[video][label][k] for label in both] for video, _ in VIDEOS] for k, _, _ in RECALL_PANELS}
        paired_bars(args.output / f'{kind}_colour', both, RECALL_PANELS, values, {k: (100, True) for k, _, _ in RECALL_PANELS},
                    series=[name for _, name in VIDEOS])
        # True-colour video, seed 42 vs seed 7
        rows_true = results[kind, 'truecolour']
        values = {k: [[seeds[i][k] for _, seeds in rows_true] for i in range(2)] for k, _, _ in RECALL_PANELS}
        paired_bars(args.output / f'{kind}_truecolour', [label for label, _ in rows_true], RECALL_PANELS, values,
                    {k: (100, True) for k, _, _ in RECALL_PANELS})
    write_csv(C / 'colour_summary.csv', table)
    print(f'Wrote {C / "colour_summary.csv"}')


if __name__ == '__main__':
    main()
