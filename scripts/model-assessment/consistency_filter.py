#!/usr/bin/env python3
"""Drop top-down skeletons whose anchor node is far from the point they were cropped around.

A centered-instance model should put its anchor node at the crop centre. When it doesn't,
it has usually drawn a neighbouring larva (see assess_stage2.py). To know every skeleton's
crop centre, run the pose model on the detector's points (body_points_for_redraw.py writes
them as single-node labels; sleap-nn then crops exactly there and keeps their order), then:

  python scripts/model-assessment/consistency_filter.py preds/crops_tail.slp preds/known_tail.slp \
      preds/consistent_tail.slp --node tail --max-px 10
"""
from __future__ import annotations

import argparse

import numpy as np
import sleap_io as sio

from assess_stage2 import pair_with_larvae


def key(frame):
    return str(frame.video.filename), int(frame.frame_idx)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('crops', help='Single-node labels the pose model was run on')
    parser.add_argument('skeletons', help='Pose-model predictions on those labels')
    parser.add_argument('output')
    parser.add_argument('--node', required=True)
    parser.add_argument('--max-px', type=float, default=10.0)
    args = parser.parse_args()
    crops = sio.load_slp(args.crops, open_videos=False)
    preds = sio.load_slp(args.skeletons, open_videos=False)
    skeleton = preds.skeletons[0]
    k = skeleton.node_names.index(args.node)
    by_key = {key(f): f for f in preds}
    frames, kept, dropped = [], 0, 0
    for frame in crops:
        centres = np.stack([i.numpy() for i in frame.instances])
        pf = by_key.get(key(frame))
        if pf is None or not len(pf.instances):
            continue
        instances = list(pf.instances)
        aligned = pair_with_larvae(centres, np.stack([i.numpy() for i in instances]), key(frame), args.skeletons)
        # map aligned rows back to prediction objects (same order, gaps where a crop gave nothing)
        rows = [i for i in range(len(aligned)) if np.isfinite(aligned[i]).any()]
        keep = []
        for row, inst in zip(rows, instances):
            offset = np.linalg.norm(aligned[row, k] - centres[row, k])
            if np.isfinite(offset) and offset <= args.max_px:
                keep.append(inst)
            else:
                dropped += 1
        kept += len(keep)
        if keep:
            frames.append(sio.LabeledFrame(pf.video, pf.frame_idx, keep))
    sio.Labels(labeled_frames=frames, videos=preds.videos, skeletons=[skeleton]).save(args.output)
    print(f'{args.output}: kept {kept} skeletons, dropped {dropped} with {args.node} > {args.max_px:g} px from its crop centre')


if __name__ == '__main__':
    main()
