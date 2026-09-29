#!/usr/bin/env python3
"""High-resolution before/after figures for the full-frame synthetic sets: each original training frame next to
its synthetic version (8 examples each, from as many source videos as possible). Full-frame set: frames with two
added larvae. Hard-crossing set: frames with 9-10 added larvae.

Both images are shown in the colours of the videos. Frames read from the ground-truth .pkg.slp with sleap-io come
out with red and blue swapped relative to the videos, so the originals are decoded with OpenCV here, as the
generators do; each pair is checked to differ only where larvae were added.

Writes <output>/full_frame_before_after.png and <output>/hard_crossings_before_after.png.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import cv2
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sleap_io as sio

from generate_full_frame_crossings import decode_frame

ROOT = Path(__file__).resolve().parents[2]
SYNTH = ROOT / 'outputs/synthetic_data_holdout'
GROUND_TRUTH = ROOT / 'data/combined_ground_truth_cleaned_visibility-fixed.pkg.slp'


def check_pair(before, after, label):
    diff = np.abs(before.astype(int) - after.astype(int)).max(axis=2)
    changed = float((diff > 30).mean())
    if changed > 0.01:
        raise ValueError(f'{label}: {100 * changed:.1f}% of pixels differ - frames are not a matching pair')
    return changed


def figure(pairs, output):
    """pairs: list of (before RGB, after RGB); one row per example, original left and synthetic right."""
    h, w = pairs[0][0].shape[:2]
    dpi = 100
    fig, axes = plt.subplots(len(pairs), 2, figsize=(2 * w / dpi + 0.6, len(pairs) * h / dpi * 1.02 + 1.2), dpi=dpi)
    axes = np.atleast_2d(axes)
    for r, (before, after) in enumerate(pairs):
        for c, image in enumerate((before, after)):
            ax = axes[r, c]
            ax.imshow(image)
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
    axes[0, 0].set_title('Original frame', fontsize=44, fontweight='bold', pad=18)
    axes[0, 1].set_title('Synthetic frame', fontsize=44, fontweight='bold', pad=18)
    fig.tight_layout(h_pad=1.0, w_pad=1.0)
    fig.savefig(output, dpi=dpi, bbox_inches='tight', pad_inches=0.2)
    plt.close(fig)
    print(f'Wrote {output}')


def pick(items, key_video, n, rng):
    """n items (seeded), from as many different source videos as possible."""
    order = [items[i] for i in rng.permutation(len(items))]
    chosen, videos = [], set()
    for item in order:
        if item[key_video] not in videos:
            chosen.append(item); videos.add(item[key_video])
    for item in order:                      # fewer videos than examples: fill up with other frames
        if len(chosen) >= n:
            break
        if not any(item is c for c in chosen):
            chosen.append(item)
    return chosen[:n]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--synthetic', type=Path, default=SYNTH)
    parser.add_argument('--ground-truth', type=Path, default=GROUND_TRUTH)
    parser.add_argument('--examples', type=int, default=8)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--output', type=Path, default=SYNTH / 'before_after')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    with h5py.File(args.ground_truth) as gt:
        def original(video, frame):
            return cv2.cvtColor(decode_frame(gt, int(video), int(frame)), cv2.COLOR_BGR2RGB)

        # Full-frame set (darken overlap): frames with two added larvae
        meta = json.loads((args.synthetic / 'crossings.json').read_text())
        synthetic = sio.load_slp(str(args.synthetic / 'crossings_full-frame.slp'))
        frames = {lf.frame_idx: lf for lf in synthetic.labeled_frames}
        pairs = []
        for c in pick([c for c in meta['crossings'] if len(c['events']) == 2], 'source_video', args.examples, rng):
            before, after = original(c['source_video'], c['source_frame']), np.asarray(frames[c['crossing_id']].image)
            check_pair(before, after, f"full-frame crossing {c['crossing_id']}")
            pairs.append((before, after))
        figure(pairs, args.output / 'full_frame_before_after.png')

        # Hard-crossing set: frames with the most added larvae
        meta = json.loads((args.synthetic / 'crossings_hard.json').read_text())
        pairs = []
        for r in pick([r for r in meta['records'] if len(r['events']) >= 9], 'source_video', args.examples, rng):
            before = original(r['source_video'], r['source_frame'])
            after = cv2.cvtColor(cv2.imread(str(args.synthetic / r['image'])), cv2.COLOR_BGR2RGB)
            check_pair(before, after, f"hard crossing {r['hard_id']}")
            pairs.append((before, after))
        figure(pairs, args.output / 'hard_crossings_before_after.png')


if __name__ == '__main__':
    main()
