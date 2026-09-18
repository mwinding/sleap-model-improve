#!/usr/bin/env python3
"""Remove crossings whose target background includes another annotated animal.

Uses other instances' keypoint bounding boxes with a 12 px body margin.
This is conservative and depends on completeness of the source annotations.
The pasted donor is already isolated by its reviewed SAM2 mask.
"""
import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import sleap_io


def extra_instances(donor, frame, margin=12):
    x0, y0, x1, y1 = donor['crop_xyxy']
    extras = []
    for index, instance in enumerate(frame.instances):
        if index == donor['source_instance']:
            continue
        points = instance.numpy()
        points = points[np.isfinite(points).all(axis=1)]
        if (len(points) and np.all(points.max(axis=0) + margin >= [x0, y0])
                and np.all(points.min(axis=0) - margin < [x1, y1])):
            extras.append(index)
    return extras


def montage(tiles, path, columns=8, width=192):
    if not tiles:
        path.unlink(missing_ok=True)
        return
    canvas = np.full((int(np.ceil(len(tiles) / columns)) * 192, columns * width, 3), 255, np.uint8)
    for index, tile in enumerate(tiles):
        row, col = divmod(index, columns)
        canvas[row*192:(row+1)*192, col*width:(col+1)*width] = tile
    cv2.imwrite(str(path), canvas)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--crossings', type=Path, default=Path('outputs/synthetic_data'))
    parser.add_argument('--library', type=Path, default=Path('outputs/donor_library_sam2'))
    parser.add_argument('--source', type=Path, default=Path('data/combined_ground_truth_cleaned_visibility-fixed.pkg.slp'))
    args = parser.parse_args()
    source = sleap_io.load_slp(str(args.source), open_videos=False)
    frames = {(source.videos.index(f.video), f.frame_idx): f for f in source}
    donors = json.loads((args.library/'donors.json').read_text())['donors']
    extras = {d['donor_id']: extra_instances(d, frames[d['source_video'], d['source_frame']]) for d in donors}
    path = args.crossings
    metadata = json.loads((path/'crossings.json').read_text())
    old = metadata['crossings']
    kept = [r for r in old if not extras[r['target_donor']]]
    removed = [r for r in old if extras[r['target_donor']]]
    if not kept:
        raise ValueError('No crossings would remain')
    if not removed:
        print(f'No additional exclusions; {len(kept)} crossings remain.')
        return
    video = sleap_io.Video(filename=[str((path/r['image']).resolve()) for r in kept])
    labeled = []
    qc, flagged = [], []
    for index, r in enumerate(kept):
        instances = []
        tile = cv2.imread(str(path/r['image']))
        if tile is None:
            raise ValueError(f"Missing image: {r['image']}")
        for key, color, prefix in [('points_xy', (0,180,0), 'T'), ('occluder_points_xy', (255,120,0), 'O')]:
            points = np.asarray(r[key])
            instance = sleap_io.Instance.from_numpy(points, source.skeletons[0])
            instance.points['visible'] = True
            instance.points['complete'] = True
            instances.append(instance)
            for node, point in enumerate(points, 1):
                xy = tuple(np.rint(point).astype(int))
                cv2.circle(tile, xy, 3, color, -1)
                cv2.putText(tile, f'{prefix}{node}', (xy[0]+4,xy[1]-4), 0, .28, color, 1, cv2.LINE_AA)
        cv2.putText(tile, str(r['crossing_id']).zfill(5), (5,15), 0, .4, (0,0,0), 1)
        if len(qc)<64: qc.append(tile)
        if r['review_reasons']: flagged.append(tile)
        labeled.append(sleap_io.LabeledFrame(video,index,instances))
    sleap_io.Labels(labeled_frames=labeled, videos=[video], skeletons=source.skeletons).save(str(path/'crossings.slp'),embed=True,verbose=False)
    check = sleap_io.load_slp(str(path/'crossings.slp'))
    assert len(check.labeled_frames)==len(kept)
    assert all(len(f.instances)==2 and all(i.points['visible'].all() for i in f.instances) for f in check)
    metadata['crossings'] = kept
    metadata['extra_animal_filter'] = {'source':str(args.source), 'body_margin_px':12, 'method':'other source instance bounding box intersects target crop'}
    (path/'crossings.json').write_text(json.dumps(metadata,indent=2))
    (path/'excluded_extra_animals.json').write_text(json.dumps({'removed_count':len(removed),'crossings':[{**r,'extra_source_instances':extras[r['target_donor']]} for r in removed]},indent=2))
    # The comparison montage contains the original first 16 crossing IDs.
    comparison_path = path/'appearance_comparison.png'
    comparison = cv2.imread(str(comparison_path)) if comparison_path.exists() else None
    if comparison is not None:
        kept_ids = {r['crossing_id'] for r in kept}
        pairs = []
        for i,r in enumerate(old[:16]):
            if r['crossing_id'] in kept_ids:
                row,col=divmod(i,2)
                pairs.append(comparison[row*192:(row+1)*192,col*384:(col+1)*384])
        montage(pairs,comparison_path,columns=2,width=384)
    montage(qc,path/'qc_montage.png')
    montage(flagged,path/'flagged_montage.png',columns=4)
    review_path=path/'appearance_review.csv'
    with review_path.open(newline='') as handle:
        reader=csv.DictReader(handle); fields=reader.fieldnames; rows=list(reader)
    kept_ids={r['crossing_id'] for r in kept}
    with review_path.open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader()
        writer.writerows(r for r in rows if int(r['crossing_id']) in kept_ids)
    for r in removed:
        (path/r['image']).unlink()
    print(f'Removed {len(removed)} crossings; retained {len(kept)} with original IDs. Labels and previews refreshed.')


if __name__=='__main__':
    main()
