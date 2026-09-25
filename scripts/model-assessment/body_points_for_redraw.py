#!/usr/bin/env python3
"""Write points as single-node labels, so a centered-instance model crops exactly around them.

Input is either full-pose predictions (the --node of each skeleton is kept) or centroid-only
predictions (their single point is placed at --node; pass --skeleton-from to supply the
full skeleton). Running a centered-instance model on the output with no centroid model makes
sleap-nn crop around exactly those points (ground-truth-centroid mode), in the same order:

  # redraw another pipeline's larvae with the body pose model
  python scripts/model-assessment/body_points_for_redraw.py preds/topdown_head.slp preds/body_points_head.slp
  sleap predict -i preds/body_points_head.slp -m <centered_instance_body_holdout> -o preds/redrawn_head.slp

  # reproduce top-down with known crop centres
  python scripts/model-assessment/body_points_for_redraw.py centroid_fullres_tail_holdout.slp preds/crops_tail.slp \
      --node tail --skeleton-from outputs/benchmark/benchmark_frames.pkg.slp

Skeletons without the node are dropped (reported).
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
    parser.add_argument('--skeleton-from', default=None, help='Labels file supplying the full skeleton for centroid-only input')
    parser.add_argument('--video', default=None, help='Replace the (single) video path, e.g. with a locally mounted copy')
    args = parser.parse_args()
    labels = sio.load_slp(args.predictions, open_videos=False)
    if args.video:
        if len(labels.videos) != 1:
            raise ValueError('--video only applies to single-video predictions')
        labels.replace_filenames(new_filenames=[args.video])
    source = labels.skeletons[0]
    skeleton = sio.load_slp(args.skeleton_from, open_videos=False).skeletons[0] if args.skeleton_from else source
    k = skeleton.node_names.index(args.node)
    k_in = 0 if source.node_names == ['centroid'] else source.node_names.index(args.node)
    frames, kept, dropped = [], 0, 0
    for frame in labels:
        instances = []
        for inst in frame.instances:
            point = inst.numpy()[k_in]
            if not np.isfinite(point).all():
                dropped += 1
                continue
            points = np.full((len(skeleton.node_names), 2), np.nan)
            points[k] = point
            instances.append(sio.Instance.from_numpy(points, skeleton))
            kept += 1
        if instances:
            frames.append(sio.LabeledFrame(frame.video, frame.frame_idx, instances))
    sio.Labels(labeled_frames=frames, videos=labels.videos, skeletons=[skeleton]).save(args.output)
    print(f'{args.output}: {kept} {args.node} points on {len(frames)} frames ({dropped} without a {args.node} node dropped)')


if __name__ == '__main__':
    main()
