#!/usr/bin/env python3
"""Turn top-down predictions into body-point-only labels, so the body pose model can redraw them.

For each predicted skeleton, keep only its body node and write it as a user instance on the
same video. Running the body centered-instance model on the result with no centroid model
makes sleap-nn crop around exactly those body points (ground-truth-centroid mode):

  python scripts/model-assessment/body_points_for_redraw.py preds/topdown_head.slp preds/body_points_head.slp
  sleap predict -i preds/body_points_head.slp -m <centered_instance_body_holdout> -o preds/redrawn_head.slp

Skeletons without a body node are dropped (reported).
"""
from __future__ import annotations

import argparse

import numpy as np
import sleap_io as sio


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('predictions')
    parser.add_argument('output')
    parser.add_argument('--node', default='body')
    args = parser.parse_args()
    labels = sio.load_slp(args.predictions, open_videos=False)
    skeleton = labels.skeletons[0]
    k = skeleton.node_names.index(args.node)
    frames, kept, dropped = [], 0, 0
    for frame in labels:
        instances = []
        for inst in frame.instances:
            point = inst.numpy()[k]
            if not np.isfinite(point).all():
                dropped += 1
                continue
            points = np.full((len(skeleton.node_names), 2), np.nan)
            points[k] = point
            instances.append(sio.Instance.from_numpy(points, skeleton))
            kept += 1
        if instances:
            frames.append(sio.LabeledFrame(labels.videos[0], frame.frame_idx, instances))
    sio.Labels(labeled_frames=frames, videos=labels.videos, skeletons=[skeleton]).save(args.output)
    print(f'{args.output}: {kept} {args.node} points on {len(frames)} frames ({dropped} skeletons without a {args.node} node dropped)')


if __name__ == '__main__':
    main()
