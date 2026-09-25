#!/usr/bin/env python3
"""Recall of combinations of centroid models trained on different anchor nodes.

A labelled larva counts as found by a combination if any model in it detected the larva
at that model's own anchor node (from each assessment's matches.csv). This is the
detection ceiling before merging duplicates, so false positives are reported as the sum
over the member models (an upper bound). Crowded/isolated use one definition for every
anchor: body point within --crowded-px of another larva's skeleton.

Example (one seed at a time):
  python scripts/model-assessment/union_anchors.py \
    --entry body outputs/model_assessment/anchors_body centroid_fullres_body_holdout \
    --entry mouthhooks outputs/model_assessment/anchors_mouthhooks centroid_fullres_mouthhooks_holdout \
    --output outputs/model_assessment/union_seed42
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
from collections import defaultdict
from pathlib import Path

from assess_centroids import load_benchmark, write_csv


def load_hits(folder, model):
    """Per labelled larva and repeat, whether `model` detected it; plus its false-positive count."""
    folder = Path(folder)
    meta = json.loads((folder/'assessment.json').read_text())
    hits, fp = {}, 0
    with (folder/'matches.csv').open() as handle:
        for r in csv.DictReader(handle):
            if r['model'] != model:
                continue
            if r['status'] == 'FP':
                fp += 1
            else:
                hits[int(r['order']), int(r['gt_index']), int(r['repeat'])] = r['status'] == 'TP'
    if not hits:
        raise ValueError(f'No matches for {model} in {folder}')
    return hits, fp, meta


def benchmark_key(meta):
    return meta['manifest_sha256'], meta['ground_truth_sha256'], meta['repeats']


def common_subsets(meta, crowded_px=15.0):
    """Body-based crowded flag per larva and hard flag per image, the same for every anchor."""
    benchmark = load_benchmark(meta['manifest'], meta['ground_truth'], set(meta.get('excluded_orders', [])), 'body', crowded_px)
    crowded = {(r['order'], i): bool(flag) for r in benchmark for i, flag in enumerate(r['crowded'])}
    hard = {r['order']: r['subset'] == 'hard' for r in benchmark}
    return crowded, hard


def combined_recall(hit_sets, crowded, hard, repeats):
    """Recall per subset when a larva counts as found if any hit set found it; and larvae missed."""
    totals, tp = defaultdict(int), defaultdict(int)
    for order, i, repeat in hit_sets[0]:
        hit = any(h[order, i, repeat] for h in hit_sets)
        for s in ['all', 'crowded' if crowded[order, i] else 'isolated'] + (['hard'] if hard[order] else []):
            totals[s] += 1
            tp[s] += hit
    recall = {s: 100 * tp[s] / totals[s] for s in ('all', 'hard', 'crowded', 'isolated')}
    return recall, (totals['all'] - tp['all']) / repeats


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--entry', nargs=3, action='append', required=True, metavar=('LABEL', 'ASSESSMENT_DIR', 'MODEL'),
                        help='One anchor model: a short label, its assessment directory and model name')
    parser.add_argument('--crowded-px', type=float, default=15.0)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    labels = [e[0] for e in args.entry]
    if len(set(labels)) != len(labels):
        parser.error('Entry labels must be unique')

    found, fps, metadata = {}, {}, None
    for label, folder, model in args.entry:
        hits, fp, meta = load_hits(folder, model)
        if metadata is not None and benchmark_key(meta) != benchmark_key(metadata):
            raise ValueError(f'{folder} was assessed on a different benchmark')
        metadata = metadata or meta
        found[label], fps[label] = hits, fp
    keys = set.intersection(*(set(h) for h in found.values()))
    if any(len(h) != len(keys) for h in found.values()):
        raise ValueError('Assessments cover different larvae')
    repeats = metadata['repeats']
    crowded, hard = common_subsets(metadata, args.crowded_px)
    frames = len(hard) * repeats

    rows = []
    for size in range(1, len(labels) + 1):
        for combo in itertools.combinations(labels, size):
            recall, missed = combined_recall([found[label] for label in combo], crowded, hard, repeats)
            rows.append({'anchors': '+'.join(combo), 'n_anchors': size,
                         **{f'{s}_recall_pct': recall[s] for s in ('all', 'hard', 'crowded', 'isolated')},
                         'missed_larvae': missed,
                         'fp_per_frame_upper_bound': sum(fps[label] for label in combo) / frames})
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output/'union.csv', rows)
    print(f"{'anchors':40s} {'overall':>7s} {'hard':>6s} {'crowded':>7s} {'isolated':>8s} {'missed':>6s} {'FP/frame<=':>10s}")
    for size in range(1, len(labels) + 1):
        group = sorted((r for r in rows if r['n_anchors'] == size), key=lambda r: -r['all_recall_pct'])
        for r in (group if size == 1 else group[:3]):
            print(f"{r['anchors']:40s} {r['all_recall_pct']:7.1f} {r['hard_recall_pct']:6.1f} {r['crowded_recall_pct']:7.1f} "
                  f"{r['isolated_recall_pct']:8.1f} {r['missed_larvae']:6.1f} {r['fp_per_frame_upper_bound']:10.2f}")
    print(f'Wrote {args.output/"union.csv"}')


if __name__ == '__main__':
    main()
