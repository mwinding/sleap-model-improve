#!/usr/bin/env python3
"""Pipelines built with seed-42 and seed-7 detectors (same pose models), side by side.

Each entry names one prediction file per seed; bars show recall, precision, F1 and crowded
recall on the test frames. The merge settings are identical for both seeds, so the seed-7
bars are an independent replication of the choice made with seed-42 detectors.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np

from assess_centroids import ROOT
from plot_anchors import paired_bars
from plot_pipelines import load_preds
from tune_merge import score, test_truth

A = ROOT / 'outputs/model_assessment'
DEFAULT = [('Body alone', 'topdown_body'), ('Merge, standard', 'final_standard_mhbt'),
           ('Merge, two-step', 'final_twostep_mhbt')]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=A / 'topdown/plots/seed_replication')
    args = parser.parse_args()
    truth = test_truth()
    folders = [A / 'topdown/predictions', A / 'topdown_seed7/predictions']
    results = [[score(truth, load_preds(folder / f'{name}.slp')) for _, name in DEFAULT] for folder in folders]
    panels = [('recall', 'Recall', 'Larvae (%)'), ('precision', 'Precision', 'Skeletons (%)'),
              ('f1', 'F1', 'Score (%)'), ('crowded_recall', 'Crowded recall', 'Larvae (%)')]
    values = {k: [[r[k] for r in seed] for seed in results] for k, _, _ in panels}
    paired_bars(args.output, [label for label, _ in DEFAULT], panels, values, {k: (100, True) for k, _, _ in panels})


if __name__ == '__main__':
    main()
