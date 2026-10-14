#!/usr/bin/env python3
"""Write a true-colour copy of a synthetic crossings SLP.

The ground-truth .pkg.slp frames are PNGs stored with red and blue swapped and tagged
channel_order=BGR, which sleap-io honours. The generators decoded them with cv2.imdecode, which
ignores the tag and so returns an RGB array, then saved that array with cv2.imwrite as if it were
BGR. Every synthetic frame therefore reaches sleap-io / sleap-nn with red and blue swapped.

This copies the labels unchanged, reverses the channels of every frame and embeds the result in
<stem>_truecolour.slp. Each corrected frame is checked against its source frame: away from the added
larvae the pixels must be identical.
"""
import argparse
import json
import tempfile
from pathlib import Path

import cv2
import numpy as np
import sleap_io


def identical_fraction(image, source):
    return float(np.mean(np.all(image == source, axis=-1)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('slp', type=Path)
    parser.add_argument('--records', type=Path, required=True, help='crossings.json or crossings_hard.json')
    parser.add_argument('--source', type=Path, help='ground truth the set was generated from (default: from the records)')
    parser.add_argument('--output', type=Path, help='default: <stem>_truecolour.slp next to the input')
    parser.add_argument('--min-identical', type=float, default=0.8,
                        help='minimum share of pixels identical to the source frame after the fix')
    args = parser.parse_args()
    output = args.output or args.slp.with_name(f'{args.slp.stem}_truecolour.slp')
    metadata = json.loads(args.records.read_text())
    records = metadata.get('crossings', metadata.get('records'))
    source = sleap_io.load_slp(str(args.source or metadata['source']))
    labels = sleap_io.load_slp(str(args.slp))
    if len(labels.videos) != 1 or [f.frame_idx for f in labels] != list(range(len(labels))):
        raise ValueError('Expected one video with frames 0..N-1')
    if len(records) != len(labels):
        raise ValueError(f'{len(records)} records for {len(labels)} frames')
    old_images = labels.videos[0]

    with tempfile.TemporaryDirectory() as tmp:
        paths = []
        for frame in labels:
            path = Path(tmp) / f'frame_{frame.frame_idx:05d}.png'
            # The swapped RGB frame from sleap-io is the true frame in BGR order, as cv2.imwrite expects.
            cv2.imwrite(str(path), np.asarray(old_images[frame.frame_idx]))
            paths.append(str(path))
        labels.replace_videos(old_videos=[old_images], new_videos=[sleap_io.Video(filename=paths)])
        labels.save(str(output), embed=True, verbose=False)

    original = sleap_io.load_slp(str(args.slp))
    fixed = sleap_io.load_slp(str(output))
    before, after = [], []
    for old, new in zip(original, fixed):
        if old.frame_idx != new.frame_idx or len(old.instances) != len(new.instances):
            raise ValueError(f'Frame {old.frame_idx}: labels differ')
        for a, b in zip(old.instances, new.instances):
            if not np.array_equal(a.numpy(), b.numpy(), equal_nan=True):
                raise ValueError(f'Frame {old.frame_idx}: points differ')
        old_image, new_image = np.asarray(old.image), np.asarray(new.image)
        if not np.array_equal(new_image, old_image[..., ::-1]):
            raise ValueError(f'Frame {old.frame_idx}: corrected frame is not the channel-reversed original')
        record = records[old.frame_idx]
        reference = np.asarray(source.videos[record['source_video']][record['source_frame']]).reshape(new_image.shape)
        before.append(identical_fraction(old_image, reference))
        after.append(identical_fraction(new_image, reference))
    before, after = np.array(before), np.array(after)
    print(f'{output}: {len(fixed)} frames, labels unchanged')
    print(f'pixels identical to the source frame: before median {np.median(before):.3f}, '
          f'after median {np.median(after):.3f} (min {after.min():.3f})')
    if after.min() < args.min_identical:
        raise ValueError(f'Frame {int(after.argmin())} matches its source on only {after.min():.3f} of pixels')


if __name__ == '__main__':
    main()
