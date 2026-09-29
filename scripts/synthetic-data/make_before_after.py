#!/usr/bin/env python3
"""High-resolution before/after figures for the full-frame synthetic sets: each original training frame next to
its synthetic version, with a zoom on the added larvae underneath (the zoom area is boxed on the full frames).

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


def zoom_window(before, after, width=480, aspect=1100 / 1800):
    """The width x (width * aspect) window containing the most changed pixels."""
    height = int(round(width * aspect))
    changed = (np.abs(before.astype(int) - after.astype(int)).max(axis=2) > 30).astype(np.float32)
    density = cv2.boxFilter(changed, -1, (width, height), normalize=False, borderType=cv2.BORDER_CONSTANT)
    h, w = changed.shape
    cy, cx = np.unravel_index(int(np.argmax(density)), density.shape)
    x0 = int(np.clip(cx - width // 2, 0, w - width))
    y0 = int(np.clip(cy - height // 2, 0, h - height))
    return x0, y0, width, height


def check_pair(before, after, label):
    diff = np.abs(before.astype(int) - after.astype(int)).max(axis=2)
    changed = float((diff > 30).mean())
    if changed > 0.01:
        raise ValueError(f'{label}: {100 * changed:.1f}% of pixels differ - frames are not a matching pair')
    return changed


def figure(pairs, output, title):
    """pairs: list of (before RGB, after RGB, after-title)."""
    rows = []
    for before, after, name in pairs:
        box = zoom_window(before, after)
        rows.append((before, after, name, box))
    h, w = pairs[0][0].shape[:2]
    dpi = 100
    fig, axes = plt.subplots(2 * len(rows), 2, figsize=(2 * w / dpi + 1, 2 * len(rows) * h / dpi * 1.1), dpi=dpi)
    for r, (before, after, name, (x0, y0, zw, zh)) in enumerate(rows):
        for c, (image, heading) in enumerate(((before, 'Original frame'), (after, name))):
            ax = axes[2 * r, c]
            ax.imshow(image)
            ax.add_patch(plt.Rectangle((x0, y0), zw, zh, fill=False, edgecolor='#ffdc00', linewidth=3))
            ax.set_title(heading, fontsize=30, fontweight='bold', pad=12)
            zoom = axes[2 * r + 1, c]
            zoom.imshow(cv2.resize(image[y0:y0 + zh, x0:x0 + zw], (w, h), interpolation=cv2.INTER_CUBIC))
            zoom.set_title(f'Zoom ({w / zw:.1f}×)', fontsize=26, pad=10)
            for a in (ax, zoom):
                a.set_xticks([]); a.set_yticks([])
                for spine in a.spines.values():
                    spine.set_visible(False)
    fig.suptitle(title, fontsize=36, fontweight='bold', y=0.995)
    fig.tight_layout(h_pad=2, w_pad=1.5)
    fig.savefig(output, dpi=dpi, bbox_inches='tight', pad_inches=0.3)
    plt.close(fig)
    print(f'Wrote {output}')


def pick(items, key_video, n, rng):
    """n items from different source videos, in random (seeded) order."""
    chosen, videos = [], set()
    for i in rng.permutation(len(items)):
        if items[i][key_video] not in videos:
            chosen.append(items[i]); videos.add(items[i][key_video])
        if len(chosen) == n:
            break
    return chosen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--synthetic', type=Path, default=SYNTH)
    parser.add_argument('--ground-truth', type=Path, default=GROUND_TRUTH)
    parser.add_argument('--examples', type=int, default=2)
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
            pairs.append((before, after, f"Synthetic: {len(c['events'])} larvae added"))
        figure(pairs, args.output / 'full_frame_before_after.png', 'Full-frame synthetic (darken overlap)')

        # Hard-crossing set: frames with the most added larvae
        meta = json.loads((args.synthetic / 'crossings_hard.json').read_text())
        most = max(len(r['events']) for r in meta['records'])
        pairs = []
        for r in pick([r for r in meta['records'] if len(r['events']) >= most - 1], 'source_video', args.examples, rng):
            before = original(r['source_video'], r['source_frame'])
            after = cv2.cvtColor(cv2.imread(str(args.synthetic / r['image'])), cv2.COLOR_BGR2RGB)
            check_pair(before, after, f"hard crossing {r['hard_id']}")
            pairs.append((before, after, f"Synthetic: {len(r['events'])} larvae added"))
        figure(pairs, args.output / 'hard_crossings_before_after.png', 'Crowded full-frame synthetic (hard crossings)')


if __name__ == '__main__':
    main()
