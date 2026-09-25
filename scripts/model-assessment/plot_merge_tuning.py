#!/usr/bin/env python3
"""Merge configurations: F1 on the tuning frames against F1 on the test frames.

Each dot is one merge configuration from tune_merge.py (grid.csv); the dashed lines are the
body pipeline alone. Configurations above the horizontal line beat body alone on the test
frames; the spread along x shows how well 4 tuning frames can rank them.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from assess_centroids import ROOT
from plot_assessment import gradient_colors, prism_style, read_csv, save_plot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--grid', type=Path, default=ROOT / 'outputs/model_assessment/merge_tuning_round2/grid.csv')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/model_assessment/topdown/plots/merge_tuning')
    args = parser.parse_args()
    rows = read_csv(args.grid)
    merges = [r for r in rows if r['single_pipeline'] == 'False']
    body = next(r for r in rows if r['config'] == 'topdown_body alone')
    prism_style()
    kinds = [('topdown', 'Standard'), ('redrawn', 'Two-step'), ('consistent', 'Consistency-filtered')]
    colors = gradient_colors(len(kinds) + 1)[1:]
    fig, ax = plt.subplots(figsize=(7.5, 6.2))
    for (kind, name), color in zip(kinds, colors):
        sel = [r for r in merges if r['kind'] == kind]
        ax.scatter([float(r['val_f1']) for r in sel], [float(r['test_f1']) for r in sel], s=16, color=color,
                   alpha=0.75, edgecolors='none', label=name, zorder=3)
    ax.axvline(float(body['val_f1']), color='0.3', linestyle='--', linewidth=1.4, zorder=2)
    ax.axhline(float(body['test_f1']), color='0.3', linestyle='--', linewidth=1.4, zorder=2)
    x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
    ax.text(x1, float(body['test_f1']) + 0.12, 'body alone ', color='0.3', fontsize=11, ha='right', va='bottom')
    ax.text(float(body['val_f1']) + 0.15, y1, ' body alone', color='0.3', fontsize=11, ha='left', va='top')
    ax.set_xlabel('F1 on tuning frames (%)')
    ax.set_ylabel('F1 on test frames (%)')
    ax.set_title('Merge configurations')
    ax.legend(loc='upper left', fontsize=11, title='Inputs', title_fontsize=11)
    fig.tight_layout()
    save_plot(fig, args.output)
    print(f'Wrote {args.output.with_suffix(".png")}')


if __name__ == '__main__':
    main()
